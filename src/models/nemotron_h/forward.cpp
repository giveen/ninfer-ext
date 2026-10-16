#include "models/nemotron_h/forward.h"

#include "artifact/views.h" // bind_view
#include "core/layout.h"
#include "core/weight_view.h" // as_dense, native_weight, weight_tensor
#include "models/nemotron_h/experts.h"
#include "models/nemotron_h/load.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_group_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mamba2_scan.h"
#include "ninfer/ops/nemotron_h_moe_experts.h"
#include "ninfer/ops/nemotron_h_moe_router.h"
#include "ninfer/ops/relu2.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::nemotron_h {
namespace {

// The scan's key extent, which is a constant of `mamba2_scan`'s kernel (128 keys, the checkpoint's
// own ssm_state_size, four consecutive keys per lane). It is repeated here rather than taken from
// the Op's private launch header because the composition has to *extend* to it, and the Op takes no
// scale argument that a shorter state would need compensating for.
constexpr std::int32_t kScanKeyExtent = 128;

// The head widths `ops::sliding_causal_attention` admits -- and therefore the widths a config's
// head is stored at, since the state keeps the rows at the carried width -- live with the state
// that stores them (`carried_attention_head`).

// The widest visible span a *zero-state* pass may have. Such a pass hands the Op its own tokens as
// the only key set, and the Op's finite window is 4096 wide, so a wider one would silently attend
// causally over only its last 4096 tokens: it is refused instead. A state-driven pass has no such
// bound -- it attends with the Op's unbounded causal route (`kUnboundedWindow`, `window == 0`),
// which is the route this model's own context needs.
constexpr std::int32_t kAttentionWindowMaximum = 4096;

[[nodiscard]] std::string extent_text(std::int32_t first, std::int32_t second) {
    return std::to_string(first) + "x" + std::to_string(second);
}

[[nodiscard]] const char* dtype_text(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "BF16";
    case DType::FP32:
        return "FP32";
    case DType::I32:
        return "I32";
    default:
        return "a foreign type";
    }
}

[[nodiscard]] std::string shape_text(const Tensor& tensor) {
    return extent_text(tensor.ne[0], tensor.ne[1]);
}

// One parameter against the config's own numbers. The extents the config derives *are* the
// checkpoint's shapes, so a parameter that disagrees with them is a wrong model rather than a
// tolerance question, and it is refused here rather than read out of range by a kernel.
void require_tensor(const BoundWeight& parameter, DType dtype,
                    std::initializer_list<std::int32_t> shape) {
    const std::int32_t rows    = shape.size() > 0 ? *shape.begin() : 1;
    const std::int32_t columns = shape.size() > 1 ? *(shape.begin() + 1) : 1;
    if (parameter.tensor.dtype != dtype || parameter.tensor.ne[0] != rows ||
        (shape.size() > 1 ? parameter.tensor.ne[1] != columns : parameter.tensor.numel() != rows)) {
        throw std::invalid_argument("nemotron_h forward: parameter '" + parameter.name + "' is " +
                                    shape_text(parameter.tensor) + " " +
                                    dtype_text(parameter.tensor.dtype) + " where the config says " +
                                    extent_text(rows, columns) + " " + dtype_text(dtype));
    }
}

// A projection or a bank: the resident tensor is the logical [rows, columns] matrix and the Op
// weight has to describe the same problem, because a weight registered at another shape is read
// out of range rather than refused by the Op that consumes it.
void require_matrix(const BoundWeight& parameter, std::int32_t rows, std::int32_t columns) {
    // A quantized matrix has its planes in `weight` and no single-typed tensor, so its resident
    // view is unset by design. The logical shape is then the Weight's own, and it is the same
    // check: a parameter bound one expert short or at another layer's width is refused either way.
    if (parameter.tensor.data == nullptr) {
        if (parameter.weight.payload == nullptr || parameter.weight.n != rows ||
            parameter.weight.k != columns) {
            throw std::invalid_argument("nemotron_h forward: parameter '" + parameter.name +
                                        "' is not bound as a " + extent_text(rows, columns) +
                                        " matrix");
        }
        return;
    }
    require_tensor(parameter, DType::BF16, {rows, columns});
    if (parameter.weight.qtype != QType::BF16 || parameter.weight.n != rows ||
        parameter.weight.k != columns) {
        throw std::invalid_argument("nemotron_h forward: parameter '" + parameter.name +
                                    "' is not bound as a " + extent_text(rows, columns) +
                                    " BF16 matrix");
    }
}

// A state plane, against the extents the Op that consumes it declares. A state bound at another
// shape is a wrong model rather than a tolerance question -- the kernel would read it out of range
// -- so it is refused by name, before any token is consumed. State storage is the caller's and has
// no default: an unbound plane is an error, not a zero state.
void require_state_tensor(const Tensor& tensor, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* role,
                          std::size_t layer) {
    std::int32_t want[4] = {1, 1, 1, 1};
    std::size_t axis     = 0;
    for (const std::int32_t extent : shape) { want[axis++] = extent; }
    bool matches = tensor.data != nullptr && tensor.dtype == dtype && tensor.is_contiguous();
    for (int index = 0; index < 4 && matches; ++index) {
        matches = tensor.ne[index] == want[index];
    }
    if (matches) { return; }
    std::string got = "unset";
    if (tensor.data != nullptr) {
        got = extent_text(tensor.ne[0], tensor.ne[1]);
        if (tensor.ne[2] != 1 || tensor.ne[3] != 1) {
            got += "x" + std::to_string(tensor.ne[2]) + "x" + std::to_string(tensor.ne[3]);
        }
        got += " " + std::string(dtype_text(tensor.dtype));
        if (!tensor.is_contiguous()) { got += " and not contiguous"; }
    }
    throw std::invalid_argument("nemotron_h forward: block " + std::to_string(layer) + "'s " +
                                std::string(role) + " is " + got + " where the Op consuming it " +
                                "declares " + extent_text(want[0], want[1]));
}

// The expert bank's own width, checked before its shape because a bank bound at the block's hidden
// width instead of the latent width is the mistake this family invites and the message has to say
// which of the two numbers is wrong.
void require_bank_latent(const BoundWeight& parameter, std::int32_t latent_width) {
    const std::int32_t bound =
        parameter.tensor.data == nullptr ? parameter.weight.k : parameter.tensor.ne[1];
    if (bound != latent_width) {
        throw std::invalid_argument("nemotron_h forward: the expert bank '" + parameter.name +
                                    "' is bound at " + std::to_string(bound) +
                                    " columns where the config's MoE layer runs its experts at " +
                                    std::to_string(latent_width));
    }
}

void observe(const BlockObserver* observer, std::size_t layer, BlockStage stage,
             const Tensor& tensor) {
    if (observer != nullptr) { observer->on_stage(layer, stage, tensor); }
}

// `count` tokens' worth of `block` contiguous BF16 elements each: the form every regrouping of a
// projection's or a convolution's rows takes, because both sides hold a token's channels
// contiguously and differ only in how many channels a token has. A row rectangle whose two sides
// advance by different pitches -- which is what regroups a projection's rows into the operands the
// Ops consume and what scatters a narrow head into a padded one -- is the same call with the
// channels as the block and the *rows* as the count: these tensors keep the row axis fastest, so
// the token count is what is strided, never the run inside a token.
void copy_blocks(const Tensor& destination, std::int32_t destination_first,
                 std::int32_t destination_pitch, const Tensor& source, std::int32_t source_first,
                 std::int32_t source_pitch, std::int32_t block, std::int32_t count,
                 cudaStream_t stream) {
    if (block <= 0 || count <= 0) { return; }
    constexpr std::size_t kElement = sizeof(std::uint16_t);
    auto* to                       = static_cast<std::byte*>(destination.data) +
                                     static_cast<std::size_t>(destination_first) * kElement;
    const auto* from               = static_cast<const std::byte*>(source.data) +
                                     static_cast<std::size_t>(source_first) * kElement;
    CUDA_CHECK(cudaMemcpy2DAsync(to, static_cast<std::size_t>(destination_pitch) * kElement, from,
                                 static_cast<std::size_t>(source_pitch) * kElement,
                                 static_cast<std::size_t>(block) * kElement,
                                 static_cast<std::size_t>(count), cudaMemcpyDeviceToDevice,
                                 stream));
}

void zero_fill(const Tensor& tensor, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), stream));
}

// One grouped operand of the convolution (x, B or C: `groups` runs of `state_size` rows per token,
// in the convolution's own channel order) regrouped into the scan's own [key, group, token] layout,
// zero-extended to the scan's fixed key extent. The extension is exact: a zero key contributes
// `dt*x*0` to the update and `0*0` to the readout, and `exp(dt*A)*0` stays exactly zero, so the
// padded tail never moves and needs no compensation -- which is only possible because `mamba2_scan`
// takes no scale argument at all.
void copy_grouped_rows(const Tensor& destination, const Tensor& source, std::int32_t source_first,
                       std::int32_t source_pitch, std::int32_t groups, std::int32_t state_size,
                       std::int32_t key_extent, std::int32_t tokens, cudaStream_t stream) {
    const std::int32_t pitch = groups * key_extent;
    if (state_size < key_extent) { zero_fill(destination, stream); }
    for (std::int32_t group = 0; group < groups; ++group) {
        copy_blocks(destination, group * key_extent, pitch, source,
                    source_first + group * state_size, source_pitch, state_size, tokens, stream);
    }
}

// The head widths `ops::sliding_causal_attention` admits, and the width this config's head is
// carried at, live with the state that stores the rows (`carried_attention_head`): the carry is a
// property of the state rather than of one pass, so the composition asks the state module for it.
// Nothing here rotates: this model has no rotation to carry (see forward.h), so the only property
// the carried width has to preserve is the Op's own domain, and the Op's `scale` argument carries
// the config's own `1/sqrt(head_dim)`.

// ---------------------------------------------------------------------------------------------
// Workspace. One planner, called with the real arena by the pass and with a dry-run layout builder
// by the byte queries, so what a caller reserves and what a pass allocates are the same sequence.

struct MambaScratch {
    Tensor projection, z, conv_in, conv_out, dt, conv_state, scan_x, scan_b, scan_c, scan_out,
        scan_state;
};

struct MoeScratch {
    Tensor router_logits, expert_ids, expert_weights, latent_in, experts_out, shared_hidden,
        shared_out, experts_workspace;
    // The gather route's own scratch, unused by a staged-width call: the routed experts' cache
    // slots, and the pager's miss list -- a count followed by (slot, expert) pairs.
    Tensor slot_ids, misses;
};

struct AttentionScratch {
    Tensor query, key, value, projected, positions;
    Tensor wide_query, wide_key, wide_value, wide_projected;
};

struct Scratch {
    Tensor normed;
    MambaScratch mamba;
    MoeScratch moe;
    AttentionScratch attention;
};

struct PassBuffers {
    Tensor hidden_a, hidden_b, normed;
};

template <class Allocator>
Scratch allocate_block(const TextConfig& config, const MoeBankConfig* bank, std::int32_t tokens,
                       BlockKind kind, Allocator& allocator) {
    Scratch scratch;
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
    scratch.normed            = allocator.alloc(DType::BF16, {hidden, tokens});

    if (kind == BlockKind::Mamba) {
        const MambaConfig& mamba    = config.mamba.value();
        const std::int32_t inner    = static_cast<std::int32_t>(mamba.inner_size());
        const std::int32_t channels = static_cast<std::int32_t>(mamba.conv_channels());
        const std::int32_t heads    = static_cast<std::int32_t>(mamba.num_heads);
        const std::int32_t groups   = static_cast<std::int32_t>(mamba.n_groups);
        const std::int32_t values   = static_cast<std::int32_t>(mamba.head_dim);
        const std::int32_t width    = static_cast<std::int32_t>(mamba.in_projection_size());
        scratch.mamba.projection    = allocator.alloc(DType::BF16, {width, tokens});
        scratch.mamba.z             = allocator.alloc(DType::BF16, {inner, tokens});
        scratch.mamba.conv_in       = allocator.alloc(DType::BF16, {channels, tokens});
        scratch.mamba.conv_out      = allocator.alloc(DType::BF16, {channels, tokens});
        scratch.mamba.dt            = allocator.alloc(DType::BF16, {heads, tokens});
        // The history window is the kernel minus one, oldest to newest, which is the width the
        // convolution's state contract names.
        scratch.mamba.conv_state = allocator.alloc(
            DType::BF16, {channels, static_cast<std::int32_t>(mamba.conv_kernel) - 1});
        scratch.mamba.scan_x     = allocator.alloc(DType::BF16, {inner, tokens});
        scratch.mamba.scan_b     = allocator.alloc(DType::BF16, {kScanKeyExtent * groups, tokens});
        scratch.mamba.scan_c     = allocator.alloc(DType::BF16, {kScanKeyExtent * groups, tokens});
        scratch.mamba.scan_out   = allocator.alloc(DType::BF16, {inner, tokens});
        scratch.mamba.scan_state = allocator.alloc(DType::FP32, {kScanKeyExtent, values, heads});
    }

    if (kind == BlockKind::Moe) {
        const MoeBankConfig& layer = *bank;
        const std::int32_t experts = static_cast<std::int32_t>(layer.num_experts);
        const std::int32_t ranks   = static_cast<std::int32_t>(layer.experts_per_token);
        const std::int32_t latent =
            static_cast<std::int32_t>(layer.expert_input_width(config.hidden_size));
        const std::int32_t intermediate = static_cast<std::int32_t>(layer.intermediate_size);
        const std::int32_t shared       = static_cast<std::int32_t>(layer.shared_intermediate_size);
        // These three are the router Op's own layout, which is expert- or rank-major with the token
        // innermost (`logits[expert * tokens + token]`, `ids[rank * tokens + token]`), while the
        // extents declared here are the shapes the Op's wrapper validates. They therefore describe
        // the Op's requirement rather than its memory order, so a consumer that reads the reported
        // stage element-wise has to index it the Op's way; the logits' workspace is the Op's, which
        // is why it is allocated here rather than left to the Op.
        scratch.moe.router_logits  = allocator.alloc(DType::FP32, {experts, tokens});
        scratch.moe.expert_ids     = allocator.alloc(DType::I32, {ranks, tokens});
        scratch.moe.expert_weights = allocator.alloc(DType::FP32, {ranks, tokens});
        scratch.moe.latent_in      = allocator.alloc(DType::BF16, {latent, tokens});
        scratch.moe.experts_out    = allocator.alloc(DType::BF16, {latent, tokens});
        scratch.moe.shared_hidden  = allocator.alloc(DType::BF16, {shared, tokens});
        scratch.moe.shared_out     = allocator.alloc(DType::BF16, {hidden, tokens});
        // The gather route's scratch: one slot id per routed assignment, and a miss list of that
        // many (slot, expert) pairs plus its count, which is what `ExpertPager::resolve` writes.
        // Both are allocated for every MoE block -- a width is a property of the pass, not of the
        // layer -- and a staged-width pass simply leaves them alone.
        scratch.moe.slot_ids = allocator.alloc(DType::I32, {ranks, tokens});
        scratch.moe.misses   = allocator.alloc(DType::I32, {2 * ranks * tokens + 1});
        // The experts' Op keeps one token tile's selected (token, rank) activation products and
        // their raw down partials here: the row-tiled route, which takes every call this family's
        // scheduler issues, walks a call's token axis in tiles through them, and the per-token
        // route a wider call takes reads none of it. The registered form is the one that reports a
        // size, and it is asked at the pass's own width: the query never falls as T does and never
        // exceeds a tile's, so the storage a prefill-sized pass reserves covers the T=1 calls that
        // run through it.
        const ops::NemotronHMoeExpertGeometry geometry{latent, experts, ranks, intermediate};
        scratch.moe.experts_workspace = allocator.alloc(
            DType::U8, {static_cast<std::int32_t>(ops::nemotron_h_moe_experts_workspace_bytes(
                           geometry, QType::NVFP4, tokens))});
    }

    if (kind == BlockKind::Attention) {
        const AttentionConfig& attention = config.attention.value();
        const std::int32_t head          = static_cast<std::int32_t>(attention.head_dim);
        const std::int32_t queries       = static_cast<std::int32_t>(attention.query_width());
        const std::int32_t keys          = static_cast<std::int32_t>(attention.key_width());
        scratch.attention.query          = allocator.alloc(DType::BF16, {queries, tokens});
        scratch.attention.key            = allocator.alloc(DType::BF16, {keys, tokens});
        scratch.attention.value          = allocator.alloc(DType::BF16, {keys, tokens});
        scratch.attention.projected      = allocator.alloc(DType::BF16, {queries, tokens});
        scratch.attention.positions      = allocator.alloc(DType::I32, {tokens});
        if (carried_attention_head(attention) != head) {
            const std::int32_t carried   = carried_attention_head(attention);
            const std::int32_t heads     = static_cast<std::int32_t>(attention.num_attention_heads);
            const std::int32_t key_heads = static_cast<std::int32_t>(attention.num_key_value_heads);
            scratch.attention.wide_query = allocator.alloc(DType::BF16, {carried * heads, tokens});
            scratch.attention.wide_key =
                allocator.alloc(DType::BF16, {carried * key_heads, tokens});
            scratch.attention.wide_value =
                allocator.alloc(DType::BF16, {carried * key_heads, tokens});
            scratch.attention.wide_projected =
                allocator.alloc(DType::BF16, {carried * heads, tokens});
        }
    }
    return scratch;
}

template <class Allocator>
PassBuffers allocate_pass(const TextConfig& config, std::int32_t tokens, Allocator& allocator) {
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
    PassBuffers buffers;
    buffers.hidden_a = allocator.alloc(DType::BF16, {hidden, tokens});
    buffers.hidden_b = allocator.alloc(DType::BF16, {hidden, tokens});
    buffers.normed   = allocator.alloc(DType::BF16, {hidden, tokens});
    return buffers;
}

template <class Allocator>
void plan_pass(const TextConfig& config, std::int32_t tokens, Allocator& allocator) {
    const PassBuffers buffers = allocate_pass(config, tokens, allocator);
    (void)buffers;
    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        const BlockKind kind = config.block_kind(layer);
        const MoeBankConfig* bank =
            kind == BlockKind::Moe ? &config.moe_bank(static_cast<std::uint32_t>(layer)) : nullptr;
        const auto scope   = allocator.scope();
        const Scratch kept = allocate_block(config, bank, tokens, kind, allocator);
        (void)scope;
        (void)kept;
    }
}

} // namespace

Model::Model(TextConfig config, ModelWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {}

Model::Model(TextConfig config, ModelWeights weights, artifact::MaterializedArtifact backing,
             ExpertResidencyPlan residency)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      residency_(std::move(residency)) {}

void Model::bind(WeightId id, std::string name, Tensor tensor, Weight weight) {
    if (id.index == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("nemotron_h model: parameter '" + name +
                                    "' carries no id, so nothing can address it");
    }
    if (id.index >= parameters_.size()) {
        parameters_.resize(id.index + 1);
        bound_.resize(id.index + 1, false);
    }
    if (bound_[id.index]) {
        throw std::invalid_argument("nemotron_h model: parameter '" + name + "' reuses id " +
                                    std::to_string(id.index) + ", already bound as '" +
                                    parameters_[id.index].name + "'");
    }
    BoundWeight parameter;
    parameter.name        = std::move(name);
    parameter.tensor      = tensor;
    parameter.weight      = weight;
    parameters_[id.index] = std::move(parameter);
    bound_[id.index]      = true;
}

const BoundWeight& Model::bound(WeightId id) const {
    if (id.index >= parameters_.size() || !bound_[id.index]) {
        throw std::invalid_argument("nemotron_h model: parameter " + std::to_string(id.index) +
                                    " is not bound, so the block cannot run");
    }
    return parameters_[id.index];
}

const Weight& Model::matrix(WeightId id) const {
    const BoundWeight& parameter = bound(id);
    if (parameter.weight.qdata == nullptr) {
        throw std::invalid_argument("nemotron_h model: parameter '" + parameter.name +
                                    "' is bound as a tensor rather than as a matrix");
    }
    return parameter.weight;
}

namespace {

// A dense parameter's payload is also its tensor. A quantized matrix has no single-typed tensor at
// all -- its codes, scales and high plane are the Weight's own -- so it stays unset rather than
// being misrepresented.
Tensor matrix_tensor(const Weight& weight) {
    if (weight.qtype == QType::FP32 || weight.qtype == QType::BF16) { return as_dense(weight); }
    return Tensor{};
}

// A dense parameter's resident tensor, in the rank-1 or rank-2 form the Ops read.
Tensor direct_tensor(const WeightView& view) {
    if (view.shape.empty() || view.shape.size() > 2) {
        throw artifact::ArtifactError("a dense parameter must have rank 1 or 2");
    }
    const auto extent = [](std::uint64_t value) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            throw artifact::ArtifactError("a parameter extent exceeds int32");
        }
        return static_cast<std::int32_t>(value);
    };
    if (view.shape.size() == 1) { return weight_tensor(view, {extent(view.shape[0]), 1, 1, 1}); }
    return weight_tensor(view, {extent(view.shape[0]), extent(view.shape[1]), 1, 1});
}

// Every id the config's block list implies, read back through the model's own `bound()`, so a hole
// -- a layer whose kind and whose bound set disagree -- is refused here rather than by a kernel
// reading an unbound id at the first token.
void verify_complete(const Model& model) {
    const ModelWeights& weights = model.weights();
    const TextConfig& config    = model.config();
    const auto require          = [&model](WeightId id) { (void)model.bound(id); };

    require(weights.text.token_embedding);
    require(weights.text.output_head);
    require(weights.text.final_norm);
    if (weights.text.layers.size() != config.num_hidden_layers()) {
        throw artifact::ArtifactError("the bound layer count differs from the config's block list");
    }
    for (std::size_t layer = 0; layer < weights.text.layers.size(); ++layer) {
        const LayerWeights& bound = weights.text.layers[layer];
        const BlockKind kind      = config.block_kind(static_cast<std::uint32_t>(layer));
        require(bound.block_norm);
        if (bound.kind != kind) {
            throw artifact::ArtifactError(
                "layer " + std::to_string(layer) +
                " is bound as a different kind than the config's block list");
        }
        if (kind == BlockKind::Mamba && bound.mamba) {
            const MambaWeights& mamba = *bound.mamba;
            for (const WeightId id : {mamba.in_proj, mamba.out_proj, mamba.conv, mamba.conv_bias,
                                      mamba.a_log, mamba.d_skip, mamba.dt_bias, mamba.norm}) {
                require(id);
            }
        } else if (kind == BlockKind::Attention && bound.attention) {
            const AttentionWeights& attention = *bound.attention;
            for (const WeightId id :
                 {attention.query, attention.key, attention.value, attention.output}) {
                require(id);
            }
        } else if (kind == BlockKind::Moe && bound.moe) {
            const MoeWeights& moe = *bound.moe;
            for (const WeightId id :
                 {moe.router, moe.correction_bias, moe.fc1_latent, moe.fc2_latent, moe.shared_up,
                  moe.shared_down, moe.experts_up, moe.experts_down}) {
                require(id);
            }
        } else {
            throw artifact::ArtifactError("layer " + std::to_string(layer) +
                                          " is bound without the parameters its config's kind "
                                          "implies");
        }
    }
}

} // namespace

// The materializer lives on this side because it constructs `Model`: the loading library plans and
// declares, this side uploads the bytes those declarations name and binds them to the model's own
// ids. The plan hands its content over and is left spent, so the 45 GB it resolved is uploaded
// once.
std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    // The split the loader decided travels with the model: it is what the pool stages the
    // non-resident banks against, and it is not recoverable from the bound weights alone.
    ExpertResidencyPlan residency  = plan.expert_residency();
    auto content                   = plan.release();
    const artifact::Reader& source = *content.materialization.source;
    auto backing =
        artifact::materialize(source, std::move(content.materialization), device, observer);
    auto model =
        std::unique_ptr<Model>(new Model(std::move(content.config), std::move(content.weights),
                                         std::move(backing), std::move(residency)));
    for (std::size_t index = 0; index < content.declarations.size(); ++index) {
        const DeclaredParameter& declared = content.declarations[index];
        const WeightId id{index};
        const WeightView view = artifact::bind_view(declared.reference, model->backing_);
        if (!declared.matrix) {
            model->bind(id, declared.name, direct_tensor(view));
            continue;
        }
        // The declaration is zero when the artifact declares no activation input divisor, which is
        // this family's A16 case: the kernel reads the raw activation and quantizes nothing. The
        // native NVFP4 ABI validates that field even when nothing reads it, so an undeclared
        // divisor is bound as the neutral 1 rather than as 0 -- the same mapping the Qwen path
        // applies where it turns an artifact's optional use auxiliaries into a Weight
        // (src/ops/weight_input.cpp).
        const float input_divisor =
            declared.activation_divisor == 0.0F ? 1.0F : declared.activation_divisor;
        const Weight weight = native_weight(view, input_divisor);
        model->bind(id, declared.name, matrix_tensor(weight), weight);
    }
    verify_complete(*model);
    return model;
}

std::size_t block_workspace_bytes(const TextConfig& config, std::int32_t tokens) {
    std::size_t bytes = 0;
    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        const BlockKind kind = config.block_kind(layer);
        const MoeBankConfig* bank =
            kind == BlockKind::Moe ? &config.moe_bank(static_cast<std::uint32_t>(layer)) : nullptr;
        WorkspaceLayoutBuilder builder;
        {
            const auto scope   = builder.scope();
            const Scratch kept = allocate_block(config, bank, tokens, kind, builder);
            (void)scope;
            (void)kept;
        }
        bytes = std::max(bytes, builder.peak_bytes(256));
    }
    return bytes;
}

std::size_t model_workspace_bytes(const TextConfig& config, std::int32_t tokens) {
    WorkspaceLayoutBuilder builder;
    {
        const auto scope = builder.scope();
        plan_pass(config, tokens, builder);
        (void)scope;
    }
    return builder.peak_bytes(256);
}

namespace {

// The Mamba-2 mixer. `mixer_input` is the block norm's output and `mixer_out` receives the mixer's
// output, which is `[hidden, tokens]` and is added to the residual by the caller. `state` is this
// layer's state, or null for a zero-state pass; either way the two stateplanes `ops::` consume are
// handed to the Ops' own single-state overloads, which read and write them in place, so a
// state-driven pass leaves the frontier the next pass starts from.
void forward_mamba(const Model& model, const MambaWeights& weights, const Tensor& mixer_input,
                   Scratch& scratch, Tensor& mixer_out, DeviceExecutionView execution,
                   BlockState* state, const BlockObserver* observer, std::size_t layer) {
    const TextConfig& config    = model.config();
    const MambaConfig& mamba    = config.mamba.value();
    const std::int32_t hidden   = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t tokens   = mixer_input.ne[1];
    const std::int32_t inner    = static_cast<std::int32_t>(mamba.inner_size());
    const std::int32_t channels = static_cast<std::int32_t>(mamba.conv_channels());
    const std::int32_t heads    = static_cast<std::int32_t>(mamba.num_heads);
    const std::int32_t groups   = static_cast<std::int32_t>(mamba.n_groups);
    const std::int32_t values   = static_cast<std::int32_t>(mamba.head_dim);
    const std::int32_t width    = static_cast<std::int32_t>(mamba.in_projection_size());
    const std::int32_t kernel   = static_cast<std::int32_t>(mamba.conv_kernel);
    const cudaStream_t stream   = execution.stream;

    // Every parameter is checked against the config's own numbers before an Op reads it: a
    // projection bound at another width would otherwise be read out of range by the kernel's own
    // indexing, and the checkpoint's shapes are exactly what the config derives.
    require_matrix(model.bound(weights.in_proj), width, hidden);
    require_matrix(model.bound(weights.out_proj), hidden, inner);
    // The declared shape only; the payload has to be in the Op's tap-major order, which no shape
    // check can see -- see `MambaWeights` in weights.h for where the transpose belongs.
    require_tensor(model.bound(weights.conv), DType::BF16, {channels, kernel});
    if (mamba.conv_bias) {
        require_tensor(model.bound(weights.conv_bias), DType::BF16, {channels});
    }
    require_tensor(model.bound(weights.a_log), DType::FP32, {heads});
    require_tensor(model.bound(weights.d_skip), DType::FP32, {heads});
    require_tensor(model.bound(weights.dt_bias), DType::FP32, {heads});
    require_tensor(model.bound(weights.norm), DType::BF16, {inner});

    // 1. The projection, then the split the reference performs on it: z, x/B/C and dt, in that
    //    order. The Op writes one contiguous [width, tokens] block and the three operands the rest
    //    of the mixer needs are row ranges of it, so each is copied into a contiguous buffer rather
    //    than sliced: the convolution and the scan both read their operands as [rows, tokens] with
    //    the row axis fastest.
    ops::linear(mixer_input, model.matrix(weights.in_proj), scratch.mamba.projection, stream);
    observe(observer, layer, BlockStage::MambaInProj, scratch.mamba.projection);
    copy_blocks(scratch.mamba.z, 0, inner, scratch.mamba.projection, 0, width, inner, tokens,
                stream);
    copy_blocks(scratch.mamba.conv_in, 0, channels, scratch.mamba.projection, inner, width,
                channels, tokens, stream);
    copy_blocks(scratch.mamba.dt, 0, heads, scratch.mamba.projection, inner + channels, width,
                heads, tokens, stream);

    // 2. The depthwise convolution over x, B and C together, with its bias and its activation
    //    fused. The activation applies to B and C as well as to x because the reference splits them
    //    *after* the convolution, so the split above is the projection's, not the convolution's.
    //    The history is the lane's own slot when a state is engaged, and an otherwise-unread
    //    scratch buffer the Op starts from zero when it is not (a prefill's first pass): the Op
    //    always reads the state it is handed, so the zero-state route must hand it zeros rather
    //    than whatever the previous pass left in the block's scratch.
    const Tensor* bias = mamba.conv_bias ? &model.bound(weights.conv_bias).tensor : nullptr;
    if (state != nullptr) {
        require_state_tensor(state->conv, DType::BF16,
                             {channels, static_cast<std::int32_t>(mamba.conv_kernel) - 1},
                             "the convolution history", layer);
        ops::causal_conv1d_silu(scratch.mamba.conv_in, model.bound(weights.conv).tensor, bias,
                                state->conv, scratch.mamba.conv_out, stream);
    } else {
        zero_fill(scratch.mamba.conv_state, stream);
        ops::causal_conv1d_silu(scratch.mamba.conv_in, model.bound(weights.conv).tensor, bias,
                                scratch.mamba.conv_state, scratch.mamba.conv_out, stream);
    }
    observe(observer, layer, BlockStage::MambaConv, scratch.mamba.conv_out);

    // 3. The scan's operands: the convolution's rows regrouped, and B and C zero-extended along the
    //    key axis to the Op's fixed extent (see kScanKeyExtent above).
    copy_blocks(scratch.mamba.scan_x, 0, inner, scratch.mamba.conv_out, 0, channels, inner, tokens,
                stream);
    copy_grouped_rows(scratch.mamba.scan_b, scratch.mamba.conv_out, inner, channels, groups,
                      static_cast<std::int32_t>(mamba.state_size), kScanKeyExtent, tokens, stream);
    copy_grouped_rows(scratch.mamba.scan_c, scratch.mamba.conv_out,
                      inner + groups * static_cast<std::int32_t>(mamba.state_size), channels,
                      groups, static_cast<std::int32_t>(mamba.state_size), kScanKeyExtent, tokens,
                      stream);

    // 4. The scan itself, from this layer's frontier (or from a zero state, which is what the scan
    //    starts a sequence from), adding `D*x` to its own readout. The operands are views of the
    //    contiguous buffers above; the state and the output are the Op's own [key, value, head] and
    //    [value, head, token] layouts.
    Tensor scan_out = scratch.mamba.scan_out.view({values, heads, tokens});
    if (state != nullptr) {
        require_state_tensor(state->recurrent, DType::FP32, {kScanKeyExtent, values, heads},
                             "the recurrent plane", layer);
        ops::mamba2_scan(scratch.mamba.scan_x.view({values, heads, tokens}),
                         scratch.mamba.scan_b.view({kScanKeyExtent, groups, tokens}),
                         scratch.mamba.scan_c.view({kScanKeyExtent, groups, tokens}),
                         scratch.mamba.dt, model.tensor(weights.dt_bias),
                         model.tensor(weights.a_log), model.tensor(weights.d_skip),
                         mamba.time_step_min, state->recurrent, scan_out, stream);
    } else {
        zero_fill(scratch.mamba.scan_state, stream);
        ops::mamba2_scan(scratch.mamba.scan_x.view({values, heads, tokens}),
                         scratch.mamba.scan_b.view({kScanKeyExtent, groups, tokens}),
                         scratch.mamba.scan_c.view({kScanKeyExtent, groups, tokens}),
                         scratch.mamba.dt, model.tensor(weights.dt_bias),
                         model.tensor(weights.a_log), model.tensor(weights.d_skip),
                         mamba.time_step_min, scratch.mamba.scan_state, scan_out, stream);
    }

    // 5. The gated grouped norm, then the output projection. The gate is z and it is applied
    //    *before* the group reduction, which is what `ops::gated_group_rmsnorm` does and
    //    `ops::gated_rmsnorm` does not. The norm's output reuses the projection's storage: every
    //    operand it read was copied out of it above.
    // A fresh view over the projection's storage rather than `view()`: the buffer is the
    // projection's [width, tokens] allocation, which is larger than the norm's output, so the
    // extents have to be declared rather than re-derived.
    Tensor normed(scratch.mamba.projection.data, DType::BF16, {inner, tokens});
    ops::gated_group_rmsnorm(scratch.mamba.scan_out, model.tensor(weights.norm), scratch.mamba.z,
                             inner / groups, config.rms_norm_eps, normed, stream);
    observe(observer, layer, BlockStage::MambaNorm, normed);
    ops::linear(normed, model.matrix(weights.out_proj), mixer_out, stream);
    observe(observer, layer, BlockStage::MambaOutProj, mixer_out);
    observe(observer, layer, BlockStage::MambaMixer, mixer_out);
}

// The latent-MoE mixer.
void forward_moe(const Model& model, const MoeBankConfig& bank, const MoeWeights& weights,
                 const Tensor& mixer_input, Scratch& scratch, Tensor& mixer_out,
                 DeviceExecutionView execution, const BlockObserver* observer, std::size_t layer,
                 RoutedExpertPool* pool) {
    const TextConfig& config   = model.config();
    const std::int32_t hidden  = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t tokens  = mixer_input.ne[1];
    const std::int32_t experts = static_cast<std::int32_t>(bank.num_experts);
    const std::int32_t ranks   = static_cast<std::int32_t>(bank.experts_per_token);
    const std::int32_t latent =
        static_cast<std::int32_t>(bank.expert_input_width(config.hidden_size));
    const std::int32_t intermediate = static_cast<std::int32_t>(bank.intermediate_size);
    const std::int32_t shared       = static_cast<std::int32_t>(bank.shared_intermediate_size);
    const cudaStream_t stream       = execution.stream;

    require_matrix(model.bound(weights.router), experts, hidden);
    require_tensor(model.bound(weights.correction_bias), DType::FP32, {experts});
    require_bank_latent(model.bound(weights.experts_up), latent);
    require_matrix(model.bound(weights.experts_up), experts * intermediate, latent);
    require_matrix(model.bound(weights.experts_down), experts * latent, intermediate);
    if (bank.latent()) { require_matrix(model.bound(weights.fc1_latent), latent, hidden); }
    if (bank.latent()) { require_matrix(model.bound(weights.fc2_latent), hidden, latent); }
    require_matrix(model.bound(weights.shared_up), shared, hidden);
    require_matrix(model.bound(weights.shared_down), hidden, shared);

    // 1. The router runs on the norm's output -- the mixer's own input -- and the correction bias
    //    is FP32 because that is the precision the selection is made of. Its workspace *is* the
    //    logits the reference exposes, so it is also the stage a harness compares, and its memory
    //    order is the Op's own (expert-major, token innermost) rather than this tree's axes
    //    convention: see the three allocations in `allocate_block`.
    const ops::NemotronHMoeRouterWeights router_weights{model.matrix(weights.router),
                                                        model.tensor(weights.correction_bias)};
    const ops::NemotronHMoeRouterGeometry router_geometry{
        hidden,
        experts,
        ranks,
        static_cast<std::int32_t>(config.router.groups),
        static_cast<std::int32_t>(config.router.groups_used),
        config.router.normalize_top_k,
        config.router.scaling_factor};
    observe(observer, layer, BlockStage::MoeRouterInput, mixer_input);
    ops::nemotron_h_moe_router(mixer_input, router_weights, router_geometry, scratch.moe.expert_ids,
                               scratch.moe.expert_weights, scratch.moe.router_logits, stream);
    observe(observer, layer, BlockStage::MoeRouterLogits, scratch.moe.router_logits);
    observe(observer, layer, BlockStage::MoeRouterIds, scratch.moe.expert_ids);
    observe(observer, layer, BlockStage::MoeRouterWeights, scratch.moe.expert_weights);

    // 2. The latent round trip: the experts run between the two projections, in the latent space.
    //    A bank with no latent projection is the reference's own identity case, where the experts
    //    read and write the hidden width directly.
    if (bank.latent()) {
        ops::linear(mixer_input, model.matrix(weights.fc1_latent), scratch.moe.latent_in, stream);
    }
    const Tensor& latent_in = bank.latent() ? scratch.moe.latent_in : mixer_input;
    observe(observer, layer, BlockStage::MoeLatentIn, latent_in);

    const ops::NemotronHMoeExpertGeometry expert_geometry{latent, experts, ranks, intermediate};
    Tensor& routed = bank.latent() ? scratch.moe.experts_out : mixer_out;
    // The bank's representation decides the route: a BF16 bank is read whole and in place, and a
    // block-scaled NVFP4 one is read through its planes, either where it was bound (host-pinned
    // over UVA, or device-resident), from the pool's staged copy of it, or from the pool's slot
    // cache of the experts this call routed to -- the pool's width rule picks between the last two.
    const Weight& up   = model.matrix(weights.experts_up);
    const Weight& down = model.matrix(weights.experts_down);
    if (up.qtype == QType::BF16) {
        const ops::NemotronHMoeExpertWeights banks{up, down};
        ops::nemotron_h_moe_experts(latent_in, scratch.moe.expert_ids, scratch.moe.expert_weights,
                                    banks, expert_geometry, routed, scratch.moe.experts_workspace,
                                    stream);
    } else {
        const std::int32_t pool_layer =
            expert_pool_layer(config, static_cast<std::uint32_t>(layer));
        if (pool_layer < 0) {
            throw std::invalid_argument("nemotron_h forward: block " + std::to_string(layer) +
                                        " has a registered-format bank but no pool layer");
        }
        const ops::ExpertWeights stored =
            expert_bank_addressing(up, down, experts, latent, intermediate, "routed experts");
        // The route is the *width's*: a call narrow enough to route a fraction of the bank gathers
        // exactly the experts it routed to through the pager's slot cache, and a call wide enough
        // to touch every expert of the layer stages its whole bank. A resident layer is neither --
        // it was uploaded once and is read where it lives.
        if (pool != nullptr && !pool->resident(pool_layer) && !pool->stages(tokens)) {
            const RoutedExpertPool::Cached cached =
                pool->acquire_cached(pool_layer, scratch.moe.expert_ids, scratch.moe.slot_ids,
                                     scratch.moe.misses, stream);
            // The gather runs on the pager's own stream, so the Op's stream waits for it. Nothing
            // is released: the slots outlive the call and the next resolve owns their eviction.
            CUDA_CHECK(cudaStreamWaitEvent(stream, cached.fetched));
            ops::nemotron_h_moe_experts(latent_in, scratch.moe.slot_ids, scratch.moe.expert_weights,
                                        cached.weights, cached.span, expert_geometry, routed,
                                        scratch.moe.experts_workspace, stream);
        } else if (pool != nullptr) {
            const RoutedExpertPool::Placed placed = pool->acquire(pool_layer, stream);
            ops::nemotron_h_moe_experts(
                latent_in, scratch.moe.expert_ids, scratch.moe.expert_weights, placed.weights,
                placed.span, expert_geometry, routed, scratch.moe.experts_workspace, stream);
            // The release is recorded even on the way out of a throw: the bank's next fill waits on
            // it, so a pass that abandons a layer must not leave the bank looking busy.
            if (placed.staged) { pool->release(pool_layer, stream); }
        } else {
            ops::nemotron_h_moe_experts(latent_in, scratch.moe.expert_ids,
                                        scratch.moe.expert_weights, stored,
                                        ops::ExpertBankSpan{0, experts}, expert_geometry, routed,
                                        scratch.moe.experts_workspace, stream);
        }
    }
    observe(observer, layer, BlockStage::MoeExperts, routed);

    // 3. Back out of the latent space, straight into the mixer's output buffer.
    if (bank.latent()) {
        ops::linear(scratch.moe.experts_out, model.matrix(weights.fc2_latent), mixer_out, stream);
    }
    observe(observer, layer, BlockStage::MoeLatentOut, mixer_out);

    // 4. The shared expert is a relu2 MLP on the *mixer's input*, not on the latent, and its output
    //    is added to the routed result.
    ops::linear(mixer_input, model.matrix(weights.shared_up), scratch.moe.shared_hidden, stream);
    ops::relu2(scratch.moe.shared_hidden, stream);
    ops::linear(scratch.moe.shared_hidden, model.matrix(weights.shared_down),
                scratch.moe.shared_out, stream);
    observe(observer, layer, BlockStage::MoeShared, scratch.moe.shared_out);
    ops::residual_add(scratch.moe.shared_out, mixer_out, stream);
    observe(observer, layer, BlockStage::MoeMixer, mixer_out);
}

// The GQA attention mixer. `state` is this block's binding into the lane's attention state, or null
// for a zero-state block. A state-driven block appends its own K/V to the state *before* its query
// runs -- the reference's order, so a token's own key and value are in its own output -- and
// attends over the lane's whole state with the Op's unbounded causal route; a zero-state block
// attends over its own tokens only.
void forward_attention(const Model& model, const AttentionWeights& weights,
                       const Tensor& mixer_input, Scratch& scratch, Tensor& mixer_out,
                       DeviceExecutionView execution, BlockState* state,
                       const BlockObserver* observer, std::size_t layer) {
    const TextConfig& config         = model.config();
    const AttentionConfig& attention = config.attention.value();
    const std::int32_t hidden        = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t tokens        = mixer_input.ne[1];
    const std::int32_t head          = static_cast<std::int32_t>(attention.head_dim);
    const std::int32_t heads         = static_cast<std::int32_t>(attention.num_attention_heads);
    const std::int32_t key_heads     = static_cast<std::int32_t>(attention.num_key_value_heads);
    const std::int32_t queries       = static_cast<std::int32_t>(attention.query_width());
    const std::int32_t keys          = static_cast<std::int32_t>(attention.key_width());
    const cudaStream_t stream        = execution.stream;

    require_matrix(model.bound(weights.query), queries, hidden);
    require_matrix(model.bound(weights.key), keys, hidden);
    require_matrix(model.bound(weights.value), keys, hidden);
    require_matrix(model.bound(weights.output), hidden, queries);
    // The route the head takes, resolved before any work: it decides the width of the operands the
    // attention Op is handed, and it is the width the state stores its rows at, so a state bound at
    // another width is refused rather than read out of range.
    const std::int32_t carried = carried_attention_head(attention);
    AttentionState* const kv   = state == nullptr ? nullptr : state->attention.state;
    if (state != nullptr && state->attention.state != nullptr &&
        state->attention.state->carried_head() != carried) {
        throw std::invalid_argument(
            "nemotron_h forward: block " + std::to_string(layer) +
            "'s attention state stores its rows at a different head width than this config's " +
            "carry (" + std::to_string(state->attention.state->carried_head()) + " against " +
            std::to_string(carried) + ")");
    }
    if (kv == nullptr && tokens > kAttentionWindowMaximum) {
        throw std::invalid_argument(
            "nemotron_h forward: a zero-state pass of " + std::to_string(tokens) +
            " tokens exceeds the landed attention Op's 4096-wide visible span, and this model has "
            "no window of its own: a longer pass needs the lane's attention state");
    }

    // 1. The three projections, whose outputs are the stages the reference exposes and are the
    //    mixer's operands as they are: nothing rotates them (see forward.h).
    ops::linear(mixer_input, model.matrix(weights.query), scratch.attention.query, stream);
    ops::linear(mixer_input, model.matrix(weights.key), scratch.attention.key, stream);
    ops::linear(mixer_input, model.matrix(weights.value), scratch.attention.value, stream);
    observe(observer, layer, BlockStage::AttentionQuery, scratch.attention.query);
    observe(observer, layer, BlockStage::AttentionKey, scratch.attention.key);
    observe(observer, layer, BlockStage::AttentionValue, scratch.attention.value);

    // The operands the Op takes, at the width it admits. The Op can neither admit this config's own
    // width nor bake a scale into its kernel, so for a carried head each head's real dims stay
    // contiguous at its carried head's start and the rest of the head is zero. That is exact -- a
    // zero query or key dim adds nothing to a score and a zero value dim contributes
    // `probability * 0` to every output -- and the config's own `scale`, which the Op takes as an
    // argument, is the only thing the carry has to preserve. The zeros are what the state's rows
    // are appended from, so a stored row's pad dims are exactly zero for the life of the state.
    if (carried != head) {
        zero_fill(scratch.attention.wide_query, stream);
        zero_fill(scratch.attention.wide_key, stream);
        zero_fill(scratch.attention.wide_value, stream);
        for (std::int32_t index = 0; index < heads; ++index) {
            copy_blocks(scratch.attention.wide_query, index * carried, carried * heads,
                        scratch.attention.query, index * head, queries, head, tokens, stream);
        }
        for (std::int32_t index = 0; index < key_heads; ++index) {
            copy_blocks(scratch.attention.wide_key, index * carried, carried * key_heads,
                        scratch.attention.key, index * head, keys, head, tokens, stream);
            copy_blocks(scratch.attention.wide_value, index * carried, carried * key_heads,
                        scratch.attention.value, index * head, keys, head, tokens, stream);
        }
    }
    const Tensor query_operand =
        carried == head ? scratch.attention.query.view({head, heads, tokens, 1})
                        : scratch.attention.wide_query.view({carried, heads, tokens, 1});
    const Tensor key_operand =
        carried == head ? scratch.attention.key.view({head, key_heads, tokens, 1})
                        : scratch.attention.wide_key.view({carried, key_heads, tokens, 1});
    const Tensor value_operand =
        carried == head ? scratch.attention.value.view({head, key_heads, tokens, 1})
                        : scratch.attention.wide_value.view({carried, key_heads, tokens, 1});
    Tensor projected_operand =
        carried == head ? scratch.attention.projected.view({head, heads, tokens, 1})
                        : scratch.attention.wide_projected.view({carried, heads, tokens, 1});

    // The width the landed attention Op is given, and this config's own scale.
    const ops::AttentionHeadGeometry geometry{carried, heads, key_heads};
    const float scale = 1.0F / std::sqrt(static_cast<float>(head));

    if (kv != nullptr) {
        // The lane's state, which is the whole (absolute) causal context this token attends over:
        // the append comes first, so the step's own key and value participate in its own output,
        // and the query's absolute position is the state's own frontier -- the same counter the
        // keys' positions are written from, so the two cannot disagree.
        kv->append(state->attention.lane, state->attention.layer, key_operand, value_operand,
                   tokens, stream);
        kv->attend(state->attention.lane, state->attention.layer, query_operand, projected_operand,
                   state->attention.workspace, execution);
    } else {
        // The positions are this pass's own: a zero-state pass is the sequence's first `tokens`
        // tokens. They are what bounds the Op causally -- with `window` at the pass's own token
        // count, every key at or before a query is visible and no later one is, which is plain
        // causal attention without a window of its own.
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) { positions[token] = token; }
        CUDA_CHECK(cudaMemcpyAsync(scratch.attention.positions.data, positions.data(),
                                   static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        ops::sliding_causal_attention(
            query_operand, key_operand, value_operand, scratch.attention.positions, Tensor{},
            scratch.attention.positions, Tensor{}, Tensor{}, Tensor{}, geometry,
            static_cast<std::uint32_t>(tokens), scale, projected_operand, Tensor{}, execution);
    }
    if (carried != head) {
        for (std::int32_t index = 0; index < heads; ++index) {
            copy_blocks(scratch.attention.projected, index * head, queries,
                        scratch.attention.wide_projected, index * carried, carried * heads, head,
                        tokens, stream);
        }
    }
    observe(observer, layer, BlockStage::AttentionProjected, scratch.attention.projected);

    // 2. The output projection.
    ops::linear(scratch.attention.projected, model.matrix(weights.output), mixer_out, stream);
    observe(observer, layer, BlockStage::AttentionOutProj, mixer_out);
    observe(observer, layer, BlockStage::AttentionMixer, mixer_out);
}

} // namespace

void forward_block(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   BlockState* state, const BlockObserver* observer,
                   RoutedExpertPool* expert_pool) {
    const TextConfig& config  = model.config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
    if (layer >= config.num_hidden_layers()) {
        throw std::invalid_argument("nemotron_h forward: the config has " +
                                    std::to_string(config.num_hidden_layers()) + " blocks, so " +
                                    std::to_string(layer) + " is not one of them");
    }
    if (model.weights().text.layers.size() != config.num_hidden_layers()) {
        throw std::invalid_argument("nemotron_h forward: the parameter set describes " +
                                    std::to_string(model.weights().text.layers.size()) +
                                    " blocks where the config has " +
                                    std::to_string(config.num_hidden_layers()));
    }
    if (hidden_in.dtype != DType::BF16 || hidden_in.ne[0] != hidden || !hidden_in.is_contiguous()) {
        throw std::invalid_argument(
            "nemotron_h forward: the block's input must be contiguous BF16 " +
            extent_text(hidden, hidden_in.ne[1]));
    }
    if (hidden_out.dtype != DType::BF16 || hidden_out.ne[0] != hidden ||
        hidden_out.ne[1] != hidden_in.ne[1] || !hidden_out.is_contiguous() ||
        hidden_out.data == hidden_in.data) {
        throw std::invalid_argument("nemotron_h forward: the block's output must be a disjoint "
                                    "contiguous BF16 buffer of the input's shape");
    }
    const std::int32_t tokens   = hidden_in.ne[1];
    const BlockKind kind        = config.block_kind(static_cast<std::uint32_t>(layer));
    const LayerWeights& weights = model.weights().text.layers[layer];

    // The config's kind and the layer's bound parameter set have to agree. A layer whose kind says
    // Mamba and whose parameters hold an attention mixer is not a shape mismatch to be absorbed:
    // the two compute different models.
    const char* expected = kind == BlockKind::Mamba       ? "Mamba"
                           : kind == BlockKind::Attention ? "attention"
                                                          : "MoE";
    const bool present   = kind == BlockKind::Mamba       ? weights.mamba.has_value()
                           : kind == BlockKind::Attention ? weights.attention.has_value()
                                                          : weights.moe.has_value();
    if (weights.kind != kind || !present) {
        throw std::invalid_argument(std::string("nemotron_h forward: block ") +
                                    std::to_string(layer) + " is a " + expected +
                                    " block in the config, but its parameters hold a mixer it is "
                                    "not");
    }
    require_tensor(model.bound(weights.block_norm), DType::BF16, {hidden});

    const MoeBankConfig* bank =
        kind == BlockKind::Moe ? &config.moe_bank(static_cast<std::uint32_t>(layer)) : nullptr;
    DeviceArena::Scope scope = arena.scope();
    Scratch scratch          = allocate_block(config, bank, tokens, kind, arena);

    // The block norm: one plain RMSNorm over the whole row. It is *not* the gated grouped norm a
    // Mamba mixer carries, which normalises within groups and gates first.
    ops::rmsnorm(hidden_in, model.tensor(weights.block_norm), config.rms_norm_eps, false,
                 scratch.normed, execution.stream);
    observe(observer, layer, BlockStage::BlockNorm, scratch.normed);

    switch (kind) {
    case BlockKind::Mamba:
        forward_mamba(model, weights.mamba.value(), scratch.normed, scratch, hidden_out, execution,
                      state, observer, layer);
        break;
    case BlockKind::Moe:
        forward_moe(model, *bank, weights.moe.value(), scratch.normed, scratch, hidden_out,
                    execution, observer, layer, expert_pool);
        break;
    case BlockKind::Attention:
        forward_attention(model, weights.attention.value(), scratch.normed, scratch, hidden_out,
                          execution, state, observer, layer);
        break;
    }

    // `h = h + mixer(norm(h))`. The mixer wrote its own output into `hidden_out`, so the residual
    // is added in place and no fourth buffer is needed.
    ops::residual_add(hidden_in, hidden_out, execution.stream);
    observe(observer, layer, BlockStage::Block, hidden_out);
}

void forward_model(const Model& model, const Tensor& token_ids, DeviceArena& arena, Tensor& logits,
                   DeviceExecutionView execution, const PassState* state, HeadColumns columns,
                   const BlockObserver* observer, RoutedExpertPool* expert_pool) {
    const TextConfig& config       = model.config();
    const std::int32_t hidden      = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t tokens      = token_ids.ne[0];
    const std::int32_t vocabulary  = static_cast<std::int32_t>(config.vocab_size);
    const std::int32_t columns_out = columns == HeadColumns::Every ? tokens : 1;
    if (token_ids.dtype != DType::I32 || token_ids.ne[1] != 1 || !token_ids.is_contiguous() ||
        tokens < 1) {
        throw std::invalid_argument("nemotron_h forward: token ids must be a contiguous I32 "
                                    "[tokens] vector");
    }
    if (logits.dtype != DType::BF16 || logits.ne[0] != vocabulary || logits.ne[1] != columns_out ||
        !logits.is_contiguous()) {
        throw std::invalid_argument("nemotron_h forward: logits must be contiguous BF16 " +
                                    extent_text(vocabulary, columns_out) + " for " +
                                    (columns == HeadColumns::Every ? "every column of the pass"
                                                                   : "the pass's last column"));
    }
    // A state-driven pass states where its first token sits, and the attention state's own frontier
    // is what the keys' and queries' positions are actually written from. The two must agree, or
    // the pass would attend at positions no token of the lane ever occupied: a caller whose own
    // cursor is off by a token is caught here rather than producing plausible logits at the wrong
    // position.
    if (state != nullptr && !state->per_layer.empty()) {
        if (state->per_layer.size() != config.num_hidden_layers()) {
            throw std::invalid_argument(
                "nemotron_h forward: a pass's state has one entry per block, and this config has " +
                std::to_string(config.num_hidden_layers()) + " blocks where the state has " +
                std::to_string(state->per_layer.size()));
        }
        if (state->first_position < 0) {
            throw std::invalid_argument("nemotron_h forward: a pass's first position is the tokens "
                                        "the lane already consumed, so it cannot be negative");
        }
        for (std::size_t layer = 0; layer < state->per_layer.size(); ++layer) {
            const BlockState& entry = state->per_layer[layer];
            if (config.block_kind(static_cast<std::uint32_t>(layer)) != BlockKind::Attention) {
                continue;
            }
            if (entry.attention.state == nullptr) {
                throw std::invalid_argument(
                    "nemotron_h forward: block " + std::to_string(layer) +
                    " is an attention block and a state-driven pass must bind its K/V state");
            }
            if (entry.attention.state->filled(entry.attention.lane) != state->first_position) {
                throw std::invalid_argument(
                    "nemotron_h forward: block " + std::to_string(layer) +
                    "'s attention state "
                    "holds " +
                    std::to_string(entry.attention.state->filled(entry.attention.lane)) +
                    " tokens where the pass says its first token is at position " +
                    std::to_string(state->first_position) +
                    ", so the two disagree about where the "
                    "pass starts");
            }
        }
    }
    BlockState* const per_layer =
        state != nullptr && !state->per_layer.empty() ? state->per_layer.data() : nullptr;

    DeviceArena::Scope scope = arena.scope();
    PassBuffers buffers      = allocate_pass(config, tokens, arena);
    // A pass opens the pool once, before its first expert layer: a staged pass reclaims the staged
    // banks there, so whatever the previous pass left in them -- staged bytes or the slot cache a
    // narrow pass gathered into, which is the same region -- is not this one's data.
    if (expert_pool != nullptr) { expert_pool->begin_forward(); }

    ops::embedding(token_ids, model.matrix(model.weights().text.token_embedding), 1.0F,
                   buffers.hidden_a, execution.stream);
    observe(observer, 0, BlockStage::Embedding, buffers.hidden_a);

    Tensor current = buffers.hidden_a;
    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        Tensor next = current.data == buffers.hidden_a.data ? buffers.hidden_b : buffers.hidden_a;
        forward_block(model, layer, current, arena, next, execution,
                      per_layer == nullptr ? nullptr : per_layer + layer, observer, expert_pool);
        current = next;
    }

    // The head over the pass's last position, or over every position. The state the layers advanced
    // is this pass's whole contribution: the head reads the residual stream and touches none of it.
    Tensor head_input = current;
    if (columns == HeadColumns::LastOnly && tokens > 1) {
        // The last column of a `[hidden, tokens]` buffer, which is contiguous on its own because
        // the row axis is the slow one.
        head_input = Tensor(static_cast<std::uint8_t*>(current.data) +
                                static_cast<std::size_t>(tokens - 1) *
                                    static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                            DType::BF16, {hidden, 1});
    }
    Tensor normed_out = columns == HeadColumns::LastOnly && tokens > 1
                            ? Tensor(buffers.normed.data, DType::BF16, {hidden, 1})
                            : buffers.normed;
    ops::rmsnorm(head_input, model.tensor(model.weights().text.final_norm), config.rms_norm_eps,
                 false, normed_out, execution.stream);
    observe(observer, 0, BlockStage::FinalNorm, normed_out);
    ops::linear(normed_out, model.matrix(model.weights().text.output_head), logits,
                execution.stream);
    observe(observer, 0, BlockStage::Logits, logits);
}

} // namespace ninfer::models::nemotron_h
