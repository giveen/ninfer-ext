#include "models/nemotron_h/execution/draft.h"

#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/grouped_tap_apply.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <cuda_runtime.h>

#include "core/decode_graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::nemotron_h {
namespace {

// The draft's weights are dense BF16, so the only compute profile any projection can take is the
// dense A16 one; asking for a quantized profile would be asking for a route the artifact does not
// hold codes for.
constexpr ops::LinearPolicy kDraftPolicy = ops::LinearPolicy::A16Only;

// `feature_norm` is an nn.LayerNorm(5) whose epsilon is the library default 1e-5, not the draft's
// own RMSNorm epsilon: the reference notes both sources hardcode it and neither documents a
// different value, so it is a property of the head rather than of the body.
constexpr float kFeatureNormEps = 1e-5F;

// A position no token has written, far enough in the past that `position_q - position_k < window`
// fails for every query row of any sequence this engine admits, and far enough from INT32_MIN that
// the difference cannot overflow either sign. It is deliberately not `attention_state.h`'s
// `kUnwrittenPosition`: that one marks a slot *ahead* of the frontier for a causal window, and this
// draft's window is symmetric, so an empty slot is excluded from the far side instead.
constexpr std::int32_t kDraftUnwrittenPosition = -1'000'000;

std::int32_t narrow(std::uint64_t value, const char* what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("nemotron_h draft: ") + what +
                                  " exceeds the int32 extents the state is declared with");
    }
    return static_cast<std::int32_t>(value);
}

std::int32_t positive(std::uint64_t value, const char* what) {
    const std::int32_t narrowed = narrow(value, what);
    if (narrowed <= 0) {
        throw std::invalid_argument(std::string("nemotron_h draft: ") + what + " must be positive");
    }
    return narrowed;
}

void require_matrix(const Model& model, WeightId id, std::int32_t rows, std::int32_t columns,
                    const char* what) {
    const Weight& weight = model.matrix(id);
    if (weight.n != rows || weight.k != columns) {
        throw std::invalid_argument(std::string("nemotron_h draft: ") + what + " is bound as [" +
                                    std::to_string(weight.n) + ", " + std::to_string(weight.k) +
                                    "] where the spec derives [" + std::to_string(rows) + ", " +
                                    std::to_string(columns) + "]");
    }
}

void require_tensor(const Model& model, WeightId id, DType dtype,
                    std::initializer_list<std::int32_t> shape, const char* what) {
    const Tensor& tensor = model.bound(id).tensor;
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("nemotron_h draft: ") + what +
                                    " is bound with a different dtype than the spec derives");
    }
    std::size_t extent = 0;
    for (const std::int32_t expected : shape) {
        if (extent >= 4 || tensor.ne[extent] != expected) {
            throw std::invalid_argument(std::string("nemotron_h draft: ") + what +
                                        " is bound with a different extent than the spec derives");
        }
        ++extent;
    }
    for (; extent < 4; ++extent) {
        if (tensor.ne[extent] != 1) {
            throw std::invalid_argument(std::string("nemotron_h draft: ") + what +
                                        " is bound with trailing dimensions the spec does not "
                                        "derive");
        }
    }
}

void observe(const DraftObserver* observer, const char* stage, const Tensor& tensor) {
    if (observer != nullptr) { observer->on_stage(stage, tensor); }
}

// The bounding box of the two staging helpers below: both cross the host/device boundary for the
// head's index arithmetic, which has no Op at this tree's scale.
void copy_to_host(const Tensor& source, void* destination, cudaStream_t stream) {
    CUDA_CHECK(
        cudaMemcpyAsync(destination, source.data, source.bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void copy_from_host(const void* source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value                 = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) {
        return static_cast<std::uint16_t>((bits >> 16) | ((bits & 0xFFFFU) != 0U ? 1U : 0U));
    }
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

// ---- The plan ---------------------------------------------------------------------------------

struct Geometry {
    models::draft::Spec::Flavor flavor = models::draft::Spec::Flavor::Lilicorr;
    std::int32_t hidden      = 0; // the tower's residual width
    std::int32_t vocab       = 0;
    std::int32_t layers      = 0;
    std::int32_t heads       = 0;
    std::int32_t kv_heads    = 0;
    std::int32_t head_dim    = 0;
    std::int32_t inner       = 0;
    std::int32_t taps        = 0;
    std::int32_t conv_taps   = 0;
    std::int32_t group_width = 0;
    std::int32_t block_size  = 0;
    std::int32_t window      = 0;
    // The token every scored column carries beyond the anchor, as the artifact stores it.
    std::int32_t mask_token  = 0;
    float eps                = 0.0F;
    float theta              = 0.0F;

    std::int32_t head_width  = 0;
    std::int32_t head_layers = 0;
    std::int32_t head_heads  = 0;
    std::int32_t lattice_dim = 0;
    std::int32_t top_k       = 0;
    std::int32_t factor_dim  = 0;
    std::int32_t feature_dim = 0;
    std::int32_t head_mlp    = 0;
    float vector_eps         = 0.0F;
    float logit_scale        = 0.0F;

    [[nodiscard]] std::int32_t query_width() const noexcept { return heads * head_dim; }

    [[nodiscard]] std::int32_t key_width() const noexcept { return kv_heads * head_dim; }

    [[nodiscard]] std::int32_t groups() const noexcept { return hidden / group_width; }

    [[nodiscard]] std::int32_t taps_rows() const noexcept { return taps * hidden; }

    [[nodiscard]] std::int32_t coefficients() const noexcept { return 2 * conv_taps * groups(); }

    [[nodiscard]] bool lilicorr() const noexcept {
        return flavor == models::draft::Spec::Flavor::Lilicorr;
    }
};

// A plain DFlash's proposal state: the tower's `[vocabulary, block]` logits, one greedy sampler
// configuration per column, the columns' absolute positions, the drawn ids, and the sampler's
// workspace. Zero for a LiLiCorr draft, whose head allocates its own logits plane per round.
std::size_t proposal_logits_bytes(const Geometry& g) {
    if (g.lilicorr()) { return 0; }
    return static_cast<std::size_t>(g.vocab) * static_cast<std::size_t>(g.block_size) *
           sizeof(std::uint16_t);
}

std::size_t proposal_columns_bytes(const Geometry& g, std::size_t element) {
    if (g.lilicorr()) { return 0; }
    return static_cast<std::size_t>(g.block_size) * element;
}

std::size_t proposal_workspace_bytes(const Geometry& g) {
    if (g.lilicorr()) { return 0; }
    return std::max<std::size_t>(ops::sampling_workspace_capacity_bytes(g.vocab, 1, g.block_size),
                                 1);
}

Geometry derive_geometry(const TextConfig& text, const models::draft::Spec& spec,
                         std::int32_t block_width, std::int32_t context_window) {
    Geometry g;
    g.flavor   = spec.flavor;
    g.hidden   = positive(spec.hidden_size, "the spec's hidden width");
    g.vocab    = static_cast<std::int32_t>(text.vocab_size);
    g.layers   = positive(spec.num_hidden_layers, "the spec's layer count");
    g.heads    = positive(spec.num_attention_heads, "the spec's query head count");
    g.kv_heads = positive(spec.num_key_value_heads, "the spec's KV head count");
    g.head_dim = positive(spec.head_dim, "the spec's head width");
    g.inner    = positive(spec.intermediate_size, "the spec's intermediate width");
    g.taps     = std::max<std::int32_t>(1, static_cast<std::int32_t>(spec.target_layer_ids.size()));
    g.mask_token = static_cast<std::int32_t>(spec.mask_token_id);
    if (g.lilicorr()) {
        // LiLiCorr stores its own geometry: the convolution, the block length it was trained at and
        // the sliding window its ring holds. A caller's width is not an override -- a round wider
        // than the trained block is refused by `propose`, not re-planned.
        g.conv_taps   = positive(spec.conv_kernel_size, "the spec's convolution tap count");
        g.group_width = positive(spec.conv_group_size, "the spec's convolution group width");
        g.block_size  = positive(spec.block_size, "the spec's block size");
        g.window      = positive(spec.sliding_window, "the spec's sliding window");
    } else {
        // A plain DFlash stores neither: the round chooses the block, and its attention is full over
        // the committed prefix, so the ring holds the request's own capacity. Both are the caller's
        // to state -- defaulting the window to the checkpoint's positional capacity would plan a
        // ring nobody can afford.
        g.conv_taps   = 0;
        g.group_width = 1;
        g.block_size  = positive(block_width, "the draft's block width");
        g.window      = positive(context_window, "the draft's context window");
    }
    g.eps         = spec.rms_norm_eps;
    g.theta       = spec.rope_theta;
    if (g.lilicorr()) {
        g.head_width  = positive(spec.head.hidden_size, "the head's width");
        g.head_layers = positive(spec.head.num_layers, "the head's layer count");
        g.head_heads  = positive(spec.head.num_heads, "the head's head count");
        g.top_k       = positive(spec.head.candidate_top_k, "the head's candidate count");
        g.factor_dim  = positive(spec.head.factor_dim, "the head's factor width");
        g.vector_eps  = spec.head.vector_eps;
        g.logit_scale = spec.head.logit_scale;
        g.feature_dim = 5;
        g.head_mlp    = static_cast<std::int32_t>(std::llround(
            static_cast<double>(g.head_width) * static_cast<double>(spec.head.mlp_ratio)));
    }

    if (g.hidden != static_cast<std::int32_t>(text.hidden_size)) {
        throw std::invalid_argument("nemotron_h draft: the draft's width is not the tower's");
    }
    if (g.hidden % g.group_width != 0) {
        throw std::invalid_argument("nemotron_h draft: the convolution group does not divide the "
                                    "draft's width");
    }
    if (g.lilicorr()) {
        if (g.head_width % g.head_heads != 0) {
            throw std::invalid_argument("nemotron_h draft: the head's width is not a multiple of "
                                        "its head count");
        }
        g.lattice_dim = g.head_width / g.head_heads;
    }
    if (g.block_size < 2) {
        throw std::invalid_argument("nemotron_h draft: a block is one anchor row and at least one "
                                    "scored slot, so its size must be at least two");
    }
    // The taps are absolute block indices of the tower.
    for (const std::uint32_t layer : spec.target_layer_ids) {
        if (layer >= text.num_hidden_layers()) {
            throw std::invalid_argument("nemotron_h draft: tap block " + std::to_string(layer) +
                                        " is past the tower's " +
                                        std::to_string(text.num_hidden_layers()) + " blocks");
        }
    }
    if (spec.mask_token_id >= text.vocab_size) {
        throw std::invalid_argument("nemotron_h draft: the mask token is outside the tower's "
                                    "vocabulary");
    }

    return g;
}

// The geometry plus every bound parameter checked against it. The composition is what needs this
// form; a caller planning bytes from the artifact's config has no parameters to check yet and takes
// `derive_geometry`.
Geometry derive(const Model& model, const models::draft::Spec& spec, std::int32_t block_width,
                std::int32_t context_window) {
    Geometry g = derive_geometry(model.config(), spec, block_width, context_window);
    const DraftWeights& weights = *model.weights().draft;
    if (static_cast<std::int32_t>(weights.layers.size()) != g.layers) {
        throw std::invalid_argument("nemotron_h draft: the bound layer count is not the spec's");
    }
    require_matrix(model, weights.feature_projection, g.hidden, g.taps_rows(), "the tap fusion");
    require_tensor(model, weights.context_norm, DType::BF16, {g.hidden}, "the context norm");
    require_tensor(model, weights.final_norm, DType::BF16, {g.hidden}, "the final norm");
    if (!g.lilicorr()) {
        require_tensor(model, weights.mask_embedding, DType::BF16, {1, g.hidden},
                       "the mask-token embedding");
    }
    for (const DraftLayerWeights& layer : weights.layers) {
        require_tensor(model, layer.input_norm, DType::BF16, {g.hidden}, "a layer input norm");
        require_tensor(model, layer.post_attention_norm, DType::BF16, {g.hidden},
                       "a layer post-attention norm");
        require_matrix(model, layer.query, g.query_width(), g.hidden, "a layer query");
        require_matrix(model, layer.key, g.key_width(), g.hidden, "a layer key");
        require_matrix(model, layer.value, g.key_width(), g.hidden, "a layer value");
        if (!g.lilicorr()) {
            // A plain DFlash stores the context projections separately; LiLiCorr reuses the block's
            // key/value parameters for both roles (see `context`).
            require_matrix(model, layer.context_key, g.key_width(), g.hidden,
                           "a layer context key");
            require_matrix(model, layer.context_value, g.key_width(), g.hidden,
                           "a layer context value");
        }
        require_matrix(model, layer.output, g.hidden, g.query_width(), "a layer output");
        require_tensor(model, layer.query_norm, DType::BF16, {g.head_dim}, "a layer query norm");
        require_tensor(model, layer.key_norm, DType::BF16, {g.head_dim}, "a layer key norm");
        if (g.lilicorr()) {
            require_tensor(model, layer.sink_bias, DType::BF16, {g.heads}, "a layer sink bias");
            require_tensor(model, layer.attention_conv_base, DType::BF16,
                           {2, g.conv_taps * g.hidden}, "a layer attention convolution base");
            require_matrix(model, layer.attention_conv_projection, g.coefficients(), g.hidden,
                           "a layer attention convolution projection");
        }
        require_matrix(model, layer.mlp_gate, g.inner, g.hidden, "a layer gate");
        require_matrix(model, layer.mlp_up, g.inner, g.hidden, "a layer up");
        require_matrix(model, layer.mlp_down, g.hidden, g.inner, "a layer down");
        if (g.lilicorr()) {
            require_tensor(model, layer.mlp_conv_base, DType::BF16, {2, g.conv_taps * g.hidden},
                           "a layer mlp convolution base");
            require_matrix(model, layer.mlp_conv_projection, g.coefficients(), g.hidden,
                           "a layer mlp convolution projection");
        }
    }
    // A plain DFlash has no head of its own: its proposal is the tower's output head over the mask
    // columns, which the body's final norm already feeds.
    if (!g.lilicorr()) { return g; }
    const DraftHeadWeights& head = weights.head;
    require_tensor(model, head.anchor_norm, DType::BF16, {g.head_width}, "the anchor norm");
    require_tensor(model, head.output_norm, DType::BF16, {g.head_width}, "the head output norm");
    require_tensor(model, head.feature_norm_weight, DType::BF16, {g.feature_dim},
                   "the feature norm gain");
    require_tensor(model, head.feature_norm_bias, DType::BF16, {g.feature_dim},
                   "the feature norm bias");
    require_matrix(model, head.in_head, g.head_width, g.head_width, "the incoming head");
    require_matrix(model, head.out_head, g.head_width, g.head_width, "the outgoing head");
    require_matrix(model, head.anchor_out_head, g.head_width, g.head_width,
                   "the anchor output head");
    require_tensor(model, head.in_head_bias, DType::BF16, {g.head_width}, "the incoming head bias");
    require_tensor(model, head.out_head_bias, DType::BF16, {g.head_width},
                   "the outgoing head bias");
    require_tensor(model, head.anchor_out_head_bias, DType::BF16, {g.head_width},
                   "the anchor output head bias");
    require_matrix(model, head.token_projection, g.head_width, g.hidden, "the token projection");
    require_matrix(model, head.context_projection, g.head_width, g.hidden,
                   "the context projection");
    require_matrix(model, head.pass_hidden_projection, g.head_width, g.hidden,
                   "the pass-hidden projection");
    require_tensor(model, head.token_bias, DType::BF16, {g.head_width}, "the token bias");
    require_tensor(model, head.context_bias, DType::BF16, {g.head_width}, "the context bias");
    require_tensor(model, head.pass_hidden_bias, DType::BF16, {g.head_width},
                   "the pass-hidden bias");
    require_matrix(model, head.feature_mlp_up, g.head_width, g.feature_dim, "the feature MLP up");
    require_matrix(model, head.feature_mlp_down, g.head_width, g.head_width,
                   "the feature MLP down");
    require_tensor(model, head.feature_mlp_up_bias, DType::BF16, {g.head_width},
                   "the feature MLP up bias");
    require_tensor(model, head.feature_mlp_down_bias, DType::BF16, {g.head_width},
                   "the feature MLP down bias");
    require_matrix(model, head.factor_input_projection, g.factor_dim, 3 * g.head_width,
                   "the factor input projection");
    require_tensor(model, head.factor_input_bias, DType::BF16, {g.factor_dim},
                   "the factor input bias");
    require_tensor(model, head.rank_embedding, DType::BF16, {g.block_size, g.head_width},
                   "the rank embedding");
    require_tensor(model, head.slot_embedding, DType::BF16, {g.block_size - 1, g.head_width},
                   "the slot embedding");
    require_tensor(model, head.relative_slot_bias, DType::BF16,
                   {g.head_heads, 2 * g.block_size - 1}, "the relative slot bias");
    require_tensor(model, head.same_slot_bias, DType::BF16, {g.head_heads}, "the same-slot bias");
    if (static_cast<std::int32_t>(head.layers.size()) != g.head_layers) {
        throw std::invalid_argument("nemotron_h draft: the head's bound layer count is not the "
                                    "spec's");
    }
    for (const DraftHeadLayerWeights& layer : head.layers) {
        require_tensor(model, layer.norm, DType::BF16, {g.head_width}, "a head layer norm");
        require_tensor(model, layer.mlp_norm, DType::BF16, {g.head_width}, "a head layer mlp norm");
        require_matrix(model, layer.attention_in, 3 * g.head_width, g.head_width,
                       "a head layer attention input");
        require_matrix(model, layer.attention_out, g.head_width, g.head_width,
                       "a head layer attention output");
        require_matrix(model, layer.mlp_up, g.head_mlp, g.head_width, "a head layer mlp up");
        require_matrix(model, layer.mlp_down, g.head_width, g.head_mlp, "a head layer mlp down");
        require_tensor(model, layer.attention_in_bias, DType::BF16, {3 * g.head_width},
                       "a head layer attention input bias");
        require_tensor(model, layer.attention_out_bias, DType::BF16, {g.head_width},
                       "a head layer attention output bias");
        require_tensor(model, layer.mlp_up_bias, DType::BF16, {g.head_mlp},
                       "a head layer mlp up bias");
        require_tensor(model, layer.mlp_down_bias, DType::BF16, {g.head_width},
                       "a head layer mlp down bias");
    }
    return g;
}

// ---- The head's index arithmetic, on the host ------------------------------------------------
//
// Everything below is `[56, 1024]`-scale or smaller and has no Op at this tree: the only top-k is
// pinned to the Qwen head's geometry, there is no attention with an additive per-head bias, and
// there is no elementwise product of two tensors. The values cross the boundary as BF16 because
// that is what the Ops produced; the arithmetic below is FP32, which is the reference's own.

struct HeadScratch {
    std::vector<std::uint16_t> packed;     // one stage's bytes, converted below
    std::vector<float> values;             // the same stage as floats
    std::vector<std::uint16_t> logits;     // [vocab * slots]
    std::vector<std::int32_t> ids;         // [nodes]
    std::vector<float> candidate_logits;   // [nodes]
    std::vector<float> candidate_logprobs; // [nodes]
    std::vector<float> candidate_probs;    // [nodes]
    std::vector<float> features;           // [feature_dim * nodes]
    std::vector<float> pass_states;        // [head_width * slots]
    std::vector<float> bias_table;         // [head_width * nodes]
    std::vector<float> qkv;                // [3 * head_width * nodes]
    std::vector<float> query;              // [nodes * lattice_dim]
    std::vector<float> key;                // [nodes * lattice_dim]
    std::vector<float> value;              // [nodes * lattice_dim]
    std::vector<float> weights;            // [nodes * nodes]
    std::vector<float> attention;          // [head_width * nodes]
    std::vector<float> node;               // [head_width * nodes]
    std::vector<float> anchor;             // [head_width]
    std::vector<float> factor_input;       // [3 * head_width * nodes]
    std::vector<float> edges;              // [2 * head_width * nodes]
    std::vector<float> out_vec;            // [head_width * nodes]
    std::vector<float> in_vec;             // [head_width * nodes]
    std::vector<float> lattice_bias_table; // [heads * nodes * nodes]
    std::vector<float> start;              // [top_k]
    std::vector<float> pairs;              // [(slots-1) * top_k * top_k]
    std::vector<float> scores;             // [slots * top_k * top_k]
    std::vector<float> rows;               // [slots * top_k]
    std::vector<std::int32_t> walk;        // [slots]
    std::vector<std::int32_t> indices;     // [slots]
};

void load_bf16(Tensor& source, std::vector<std::uint16_t>& packed, std::vector<float>& values,
               cudaStream_t stream) {
    packed.resize(static_cast<std::size_t>(source.numel()));
    copy_to_host(source, packed.data(), stream);
    values.resize(packed.size());
    for (std::size_t at = 0; at < packed.size(); ++at) { values[at] = bf16_to_float(packed[at]); }
}

// The slot's top `top_k` logits, its full-vocabulary log-partition, and the five scalars the head's
// feature MLP reads from them. Rank is by descending logit with exact ties resolved by the lower
// token id, which is the reference's `topk` and its `argsort` stability.
void select_candidates(const std::uint16_t* logits, std::int32_t vocab, std::int32_t top_k,
                       std::int32_t* ids, float* picked, float* log_probs, float* probs) {
    float peak = -std::numeric_limits<float>::infinity();
    for (std::int32_t v = 0; v < vocab; ++v) { peak = std::max(peak, bf16_to_float(logits[v])); }
    // The partition in float: the double-precision `std::exp` this replaces was 3.7 ms per column
    // -- 26 ms of the round's 40 ms forward -- and the five features cannot resolve 1e-7 nats.
    float total = 0.0F;
    for (std::int32_t v = 0; v < vocab; ++v) { total += std::exp(bf16_to_float(logits[v]) - peak); }
    const float log_z = peak + std::log(total);

    std::vector<float> best(static_cast<std::size_t>(top_k),
                            -std::numeric_limits<float>::infinity());
    std::vector<std::int32_t> best_id(static_cast<std::size_t>(top_k), 0);
    std::int32_t filled = 0;
    for (std::int32_t v = 0; v < vocab; ++v) {
        const float value = bf16_to_float(logits[v]);
        if (filled == top_k && value <= best[static_cast<std::size_t>(top_k - 1)]) { continue; }
        std::int32_t at = std::min(filled, top_k - 1);
        while (at > 0 && best[static_cast<std::size_t>(at - 1)] < value) {
            best[static_cast<std::size_t>(at)]    = best[static_cast<std::size_t>(at - 1)];
            best_id[static_cast<std::size_t>(at)] = best_id[static_cast<std::size_t>(at - 1)];
            --at;
        }
        best[static_cast<std::size_t>(at)]    = value;
        best_id[static_cast<std::size_t>(at)] = v;
        if (filled < top_k) { ++filled; }
    }
    for (std::int32_t k = 0; k < top_k; ++k) {
        const std::size_t at = static_cast<std::size_t>(k);
        ids[k]               = best_id[at];
        picked[k]            = best[at];
        log_probs[k]         = static_cast<float>(static_cast<double>(best[at]) - log_z);
        probs[k]             = std::exp(log_probs[k]);
    }
}

// `rank` is 0-based, so feature 4 is `rank / (top_k - 1)` and feature 5 is `rank == 0`. The
// candidate axis is the *node* axis (`slots * top_k`), because features are one row of five per
// candidate; the rank of a node is its position within its slot.
void build_features(const float* log_probs, const float* probs, std::int32_t top_k,
                    std::int32_t feature_dim, std::int32_t nodes, float* features) {
    const float denominator = top_k > 1 ? static_cast<float>(top_k - 1) : 1.0F;
    for (std::int32_t node = 0; node < nodes; ++node) {
        const std::size_t at    = static_cast<std::size_t>(node);
        const std::int32_t rank = node % top_k;
        const std::size_t top =
            static_cast<std::size_t>(node / top_k) * static_cast<std::size_t>(top_k);
        const std::size_t base = static_cast<std::size_t>(feature_dim) * at;
        features[base + 0]     = log_probs[at];
        features[base + 1]     = probs[at];
        features[base + 2]     = log_probs[at] - log_probs[top];
        features[base + 3]     = static_cast<float>(rank) / denominator;
        features[base + 4]     = rank == 0 ? 1.0F : 0.0F;
    }
}

// LayerNorm over the feature axis, one row per candidate. `features` and `out` are
// `[feature_dim, nodes]` with the feature axis fastest, which is the engine's own order.
void normalise_features(const float* features, const float* gain, const float* bias,
                        std::int32_t feature_dim, std::int32_t nodes, float eps, float* out) {
    for (std::int32_t node = 0; node < nodes; ++node) {
        double mean = 0.0;
        for (std::int32_t f = 0; f < feature_dim; ++f) {
            mean += features[static_cast<std::size_t>(f) * nodes + node];
        }
        mean /= feature_dim;
        double variance = 0.0;
        for (std::int32_t f = 0; f < feature_dim; ++f) {
            const double centred = features[static_cast<std::size_t>(f) * nodes + node] - mean;
            variance += centred * centred;
        }
        variance /= feature_dim;
        const double inv = 1.0 / std::sqrt(variance + eps);
        for (std::int32_t f = 0; f < feature_dim; ++f) {
            out[static_cast<std::size_t>(f) * nodes + node] = static_cast<float>(
                (features[static_cast<std::size_t>(f) * nodes + node] - mean) * inv * gain[f] +
                bias[f]);
        }
    }
}

// `relative_slot_bias[head, slot_q - slot_k + block_size - 1] + same_slot_bias[head] * same slot`.
// The offset is the *trained* `block_size - 1` even when a shorter block is scored, which is what
// keeps each relative distance on the weight it was trained with.
void lattice_bias(const float* relative, const float* same, std::int32_t heads,
                  std::int32_t block_size, std::int32_t slots, std::int32_t top_k,
                  std::vector<float>& bias) {
    const std::int32_t nodes = slots * top_k;
    bias.assign(static_cast<std::size_t>(heads) * static_cast<std::size_t>(nodes) *
                    static_cast<std::size_t>(nodes),
                0.0F);
    const std::int32_t span = 2 * block_size - 1;
    for (std::int32_t head = 0; head < heads; ++head) {
        for (std::int32_t query = 0; query < nodes; ++query) {
            const std::int32_t slot_query = query / top_k;
            for (std::int32_t key = 0; key < nodes; ++key) {
                const std::int32_t slot_key = key / top_k;
                const std::int32_t offset   = slot_query - slot_key + block_size - 1;
                const float relative_term =
                    relative[static_cast<std::size_t>(head) * span + offset];
                const float same_term = slot_query == slot_key ? same[head] : 0.0F;
                const std::size_t at =
                    static_cast<std::size_t>(head) +
                    static_cast<std::size_t>(heads) *
                        (static_cast<std::size_t>(query) * nodes + static_cast<std::size_t>(key));
                bias[at] = relative_term + same_term;
            }
        }
    }
}

// The head's own attention: bidirectional over the block's `slots * top_k` nodes, with no position
// encoding and the additive bias as the entire mask. `qkv` is `[3 * head_width, nodes]` with the
// feature axis fastest, which is the linear's own output order.
void lattice_attention(const std::vector<float>& qkv, std::int32_t head_width, std::int32_t heads,
                       std::int32_t head_dim, std::int32_t nodes, const std::vector<float>& bias,
                       std::vector<float>& out, HeadScratch& host) {
    const std::int32_t width = 3 * head_width;
    const float scale        = 1.0F / std::sqrt(static_cast<float>(head_dim));
    host.query.assign(static_cast<std::size_t>(nodes) * head_dim, 0.0F);
    host.key.assign(host.query.size(), 0.0F);
    host.value.assign(host.query.size(), 0.0F);
    host.weights.assign(static_cast<std::size_t>(nodes) * nodes, 0.0F);
    out.assign(static_cast<std::size_t>(head_width) * nodes, 0.0F);

    for (std::int32_t head = 0; head < heads; ++head) {
        for (std::int32_t token = 0; token < nodes; ++token) {
            for (std::int32_t d = 0; d < head_dim; ++d) {
                const std::size_t source =
                    static_cast<std::size_t>(head * head_dim + d) +
                    static_cast<std::size_t>(width) * static_cast<std::size_t>(token);
                const std::size_t at =
                    static_cast<std::size_t>(token) * head_dim + static_cast<std::size_t>(d);
                host.query[at] = qkv[source];
                host.key[at]   = qkv[source + static_cast<std::size_t>(head_width)];
                host.value[at] = qkv[source + 2 * static_cast<std::size_t>(head_width)];
            }
        }
        const float* const bias_head = bias.data() + static_cast<std::size_t>(head) * nodes * nodes;
        for (std::int32_t query = 0; query < nodes; ++query) {
            float peak = -std::numeric_limits<float>::infinity();
            float* row = host.weights.data() + static_cast<std::size_t>(query) * nodes;
            for (std::int32_t key = 0; key < nodes; ++key) {
                float sum      = 0.0F;
                const float* q = host.query.data() + static_cast<std::size_t>(query) * head_dim;
                const float* k = host.key.data() + static_cast<std::size_t>(key) * head_dim;
                for (std::int32_t d = 0; d < head_dim; ++d) { sum += q[d] * k[d]; }
                row[key] = sum * scale + bias_head[static_cast<std::size_t>(query) * nodes + key];
                peak     = std::max(peak, row[key]);
            }
            double total = 0.0;
            for (std::int32_t key = 0; key < nodes; ++key) {
                row[key] = std::exp(row[key] - peak);
                total += static_cast<double>(row[key]);
            }
            const float inverse = static_cast<float>(1.0 / total);
            for (std::int32_t key = 0; key < nodes; ++key) { row[key] *= inverse; }
        }
        for (std::int32_t token = 0; token < nodes; ++token) {
            float* destination =
                out.data() + static_cast<std::size_t>(head * head_dim) +
                static_cast<std::size_t>(head_width) * static_cast<std::size_t>(token);
            for (std::int32_t d = 0; d < head_dim; ++d) {
                float sum = 0.0F;
                for (std::int32_t key = 0; key < nodes; ++key) {
                    sum += host.weights[static_cast<std::size_t>(token) * nodes +
                                        static_cast<std::size_t>(key)] *
                           host.value[static_cast<std::size_t>(key) * head_dim +
                                      static_cast<std::size_t>(d)];
                }
                destination[d] = sum;
            }
        }
    }
}

// `ggml_l2_norm`'s row normalisation: the norm is *floored* at eps rather than added to, which is
// `torch.nn.functional.normalize`'s default and the reference's own choice.
void l2_normalise(std::vector<float>& values, std::int32_t rows, std::int32_t dim, float eps) {
    for (std::int32_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::int32_t d = 0; d < dim; ++d) {
            const double value =
                values[static_cast<std::size_t>(d) +
                       static_cast<std::size_t>(dim) * static_cast<std::size_t>(row)];
            sum += value * value;
        }
        const float norm    = std::max(std::sqrt(static_cast<float>(sum)), eps);
        const float inverse = 1.0F / norm;
        for (std::int32_t d = 0; d < dim; ++d) {
            values[static_cast<std::size_t>(d) +
                   static_cast<std::size_t>(dim) * static_cast<std::size_t>(row)] *= inverse;
        }
    }
}

float dot(const std::vector<float>& left, std::size_t left_at, const std::vector<float>& right,
          std::size_t right_at, std::int32_t dim) {
    float sum = 0.0F;
    for (std::int32_t d = 0; d < dim; ++d) { sum += left[left_at + d] * right[right_at + d]; }
    return sum;
}

// The greedy walk: `previous = argmax_k scores[slot, previous, k]` over the row the committed
// predecessor selects, chained left to right. The first maximal index wins, which is the
// reference's `argmax`.
void walk_scores(const std::vector<float>& scores, const std::vector<std::int32_t>& ids,
                 std::int32_t slots, std::int32_t top_k, std::vector<std::int32_t>& chosen,
                 std::vector<std::int32_t>& indices, std::vector<float>& rows) {
    chosen.assign(static_cast<std::size_t>(slots), 0);
    indices.assign(static_cast<std::size_t>(slots), 0);
    rows.assign(static_cast<std::size_t>(slots) * top_k, 0.0F);
    std::int32_t previous = 0;
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        const float* row =
            scores.data() + static_cast<std::size_t>(slot * top_k + previous) * top_k;
        std::int32_t best = 0;
        for (std::int32_t k = 1; k < top_k; ++k) {
            if (row[k] > row[best]) { best = k; }
        }
        indices[static_cast<std::size_t>(slot)] = best;
        chosen[static_cast<std::size_t>(slot)] =
            ids[static_cast<std::size_t>(slot) * top_k + static_cast<std::size_t>(best)];
        std::copy(row, row + top_k, rows.begin() + static_cast<std::ptrdiff_t>(slot) * top_k);
        previous = best;
    }
}

} // namespace

struct DraftContext::Impl {
    const Model* model = nullptr;
    Geometry g;
    std::int32_t lanes     = 0;
    std::int32_t capacity  = 0; // context slots per lane
    std::int32_t max_nodes = 0;

    // The context ring, per lane: [lane][layer][side][key_width][capacity], where `side` is key 0
    // and value 1 and a slot is contiguous key_width elements.
    DeviceBuffer ring;
    // The absolute position each slot holds, per lane: [lane][capacity]. Slots no token has written
    // carry a position every query's window excludes.
    DeviceBuffer ring_positions;
    // One all-ones BF16 plane, read by the silu-through-`silu_mul` calls the head's two MLPs need
    // (the tree's only SiLU Op is the gated one). Every element is one, so any view of it is the
    // ones tensor of that shape.
    DeviceBuffer ones_plane;
    // One forward's transient storage.
    DeviceArena arena;
    // The body's own arena: `context()` allocates with a round-varying width, so the shared arena's
    // bump position is not round-stable and a captured body would write at varying offsets. The
    // body allocates only from here, rewound per round.
    DeviceArena body_arena;
    // The anchor state, one column: `context()` fills it in place so the captured body reads one
    // address instead of a fresh arena slice every round.
    DeviceBuffer anchor_plane;
    // A plain DFlash's proposal: the tower's logits over the mask columns, one greedy configuration
    // per column, the columns' positions, the drawn ids, and the sampler's workspace. Empty for a
    // LiLiCorr draft, whose head owns its own plane.
    DeviceBuffer proposal_logits;
    DeviceBuffer proposal_configs;
    DeviceBuffer proposal_positions;
    DeviceBuffer proposal_drawn;
    DeviceBuffer proposal_workspace_backing;
    std::optional<WorkspaceArena> proposal_workspace;
    std::vector<std::int32_t> proposal_drawn_host;
    // One captured body per lane (the ring pointers are lane-specific) and its executable. The head
    // stays outside: its full-vocabulary select runs on the host and cannot be captured.
    std::vector<DecodeGraphDefinition> body_graphs;
    std::vector<DecodeGraphExecutable> body_execs;

    // Per lane: the last absolute position appended to the ring, so a round appends only what is
    // new; -1 before the first.
    std::vector<std::int32_t> appended_through;
    std::vector<std::vector<std::int32_t>> walk;
    std::vector<std::vector<std::int32_t>> candidates;

    // The plan-time constants read once from the artifact, in the checkpoint's own row-major order.
    std::vector<float> slot_table;    // [block_size - 1][head_width]
    std::vector<float> rank_table;    // [block_size][head_width]
    std::vector<float> relative_bias; // [heads][2 * block_size - 1]
    std::vector<float> same_bias;     // [heads]
    std::vector<float> feature_gain;  // [feature_dim]
    std::vector<float> feature_bias;  // [feature_dim]
    HeadScratch host;

    [[nodiscard]] std::size_t ring_offset(std::int32_t lane, std::int32_t layer,
                                          std::int32_t side) const noexcept {
        const std::size_t per_side =
            static_cast<std::size_t>(g.key_width()) * static_cast<std::size_t>(capacity);
        return ((static_cast<std::size_t>(lane) * g.layers + static_cast<std::size_t>(layer)) * 2 +
                static_cast<std::size_t>(side)) *
               per_side;
    }

    [[nodiscard]] Tensor ring_view(std::int32_t lane, std::int32_t layer, std::int32_t side) const {
        auto* base = static_cast<std::uint8_t*>(ring.p) +
                     ring_offset(lane, layer, side) * sizeof(std::uint16_t);
        return Tensor(base, DType::BF16, {g.head_dim, g.kv_heads, capacity, 1});
    }

    [[nodiscard]] Tensor ring_positions_view(std::int32_t lane) const {
        auto* base = static_cast<std::uint8_t*>(ring_positions.p) +
                     static_cast<std::size_t>(lane) * static_cast<std::size_t>(capacity) *
                         sizeof(std::int32_t);
        return Tensor(base, DType::I32, {capacity, 1});
    }

    [[nodiscard]] Tensor ones(std::int32_t width, std::int32_t columns) const {
        return Tensor(ones_plane.p, DType::BF16, {width, columns});
    }

    [[nodiscard]] Tensor report_fp32(const float* source, std::int32_t d0, std::int32_t d1,
                                     cudaStream_t stream) {
        Tensor tensor = arena.alloc(DType::FP32, {d0, d1});
        copy_from_host(source, tensor, stream);
        return tensor;
    }

    [[nodiscard]] Tensor report_i32(const std::int32_t* source, std::int32_t d0, std::int32_t d1,
                                    cudaStream_t stream) {
        Tensor tensor = arena.alloc(DType::I32, {d0, d1});
        copy_from_host(source, tensor, stream);
        return tensor;
    }

    // Uploads `[dim, columns]` floats as the BF16 operand an Op takes.
    [[nodiscard]] Tensor upload_bf16(const float* source, std::int32_t dim, std::int32_t columns,
                                     cudaStream_t stream) {
        const std::size_t count = static_cast<std::size_t>(dim) * static_cast<std::size_t>(columns);
        host.packed.resize(count);
        for (std::size_t at = 0; at < count; ++at) { host.packed[at] = float_to_bf16(source[at]); }
        Tensor tensor = arena.alloc(DType::BF16, {dim, columns});
        copy_from_host(host.packed.data(), tensor, stream);
        return tensor;
    }

    Impl(const Model& model_source, const Geometry& geometry, std::int32_t lane_count,
         const DraftMemory& memory);
    void reset(std::int32_t lane, cudaStream_t stream);
    std::vector<std::int32_t> propose(std::int32_t lane, const DraftRound& round,
                                      DeviceExecutionView execution, const DraftObserver* observer);
    // The context phase: the fused target feature over `[ctx_first, ctx_last]`, its norm, and every
    // layer's context K/V appended to the ring. `anchor_hidden` is the fused feature's last column,
    // which is the state the head's anchor branch reads. Both the round and `condition` are this
    // one function, so a position is materialized by exactly one piece of arithmetic whichever path
    // brings it in.
    void context(std::int32_t lane, const Tensor& taps, std::int32_t ctx_first,
                 std::int32_t ctx_last, Tensor& anchor_hidden, const DraftObserver* observer,
                 DeviceExecutionView execution);
    void condition(std::int32_t lane, const Tensor& taps, std::int32_t first_position,
                   DeviceExecutionView execution);
    void body(std::int32_t lane, const DraftRound& round, const Tensor& id_staging,
              const Tensor& position_staging, const Tensor& position_high_staging,
              Tensor& pass_hidden, Tensor& anchor_hidden, const DraftObserver* observer,
              DeviceExecutionView execution);
    std::vector<std::int32_t> head(const DraftRound& round, const Tensor& pass_hidden,
                                   const Tensor& anchor_hidden, const DraftObserver* observer,
                                   DeviceExecutionView execution);
    std::vector<std::int32_t> head_dflash(const DraftRound& round, const Tensor& pass_hidden,
                                          const DraftObserver* observer,
                                          DeviceExecutionView execution);
};

void DraftContext::Impl::reset(std::int32_t lane, cudaStream_t stream) {
    const std::size_t per_lane = static_cast<std::size_t>(g.layers) * 2 *
                                 static_cast<std::size_t>(g.key_width()) *
                                 static_cast<std::size_t>(capacity) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(ring.p) +
                                   static_cast<std::size_t>(lane) * per_lane,
                               0, per_lane, stream));
    std::vector<std::int32_t> unwritten(static_cast<std::size_t>(capacity),
                                        kDraftUnwrittenPosition);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(ring_positions.p) +
                                   static_cast<std::size_t>(lane) *
                                       static_cast<std::size_t>(capacity) * sizeof(std::int32_t),
                               unwritten.data(), unwritten.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    appended_through[static_cast<std::size_t>(lane)] = -1;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void DraftContext::Impl::context(std::int32_t lane, const Tensor& taps, std::int32_t ctx_first,
                                 std::int32_t ctx_last, Tensor& anchor_hidden,
                                 const DraftObserver* observer, DeviceExecutionView execution) {
    const cudaStream_t stream    = execution.stream;
    const DraftWeights& weights  = *model->weights().draft;
    const std::int32_t hidden    = g.hidden;
    const std::int32_t appended  = appended_through[static_cast<std::size_t>(lane)];
    const std::int32_t new_first = std::max(ctx_first, appended + 1);
    const std::int32_t new_count = ctx_last - new_first + 1;
    // The columns the fused feature must cover: every new one, and the last one even when it is
    // already in the ring, because the anchor state is the fused feature at the last committed
    // position and the head reads it from here.
    const std::int32_t work_first   = std::min(new_first, ctx_last);
    const std::int32_t work_columns = ctx_last - work_first + 1;

    // ---- The fused target feature, and the context K/V it projects -----------------------------
    Tensor tap_operand = arena.alloc(DType::BF16, {g.taps_rows(), work_columns});
    {
        const std::size_t width = static_cast<std::size_t>(g.taps_rows()) * sizeof(std::uint16_t);
        const auto* source =
            static_cast<const std::uint8_t*>(taps.data) +
            static_cast<std::size_t>(work_first - ctx_first) * static_cast<std::size_t>(taps.nb[1]);
        CUDA_CHECK(cudaMemcpy2DAsync(
            tap_operand.data, width, source, static_cast<std::size_t>(taps.nb[1]), width,
            static_cast<std::size_t>(work_columns), cudaMemcpyDeviceToDevice, stream));
    }
    observe(observer, "body.tap_operand", tap_operand);
    Tensor fused = arena.alloc(DType::BF16, {hidden, work_columns});
    ops::linear(tap_operand, model->matrix(weights.feature_projection), fused, stream);
    observe(observer, "body.fc_out", fused);
    Tensor normed = arena.alloc(DType::BF16, {hidden, work_columns});
    ops::rmsnorm(fused, model->tensor(weights.context_norm), g.eps, false, normed, stream);
    observe(observer, "body.hidden_norm_out", normed);
    anchor_hidden = normed.slice(1, work_columns - 1, 1);
    // The anchor state moves into its fixed plane, so the body (and a captured body) reads one
    // address. The slice is one column of `hidden` contiguous BF16 channels.
    CUDA_CHECK(cudaMemcpyAsync(anchor_plane.p, anchor_hidden.data,
                               static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
    anchor_hidden = Tensor(anchor_plane.p, DType::BF16, {hidden, 1});
    observe(observer, "body.anchor_hidden", anchor_hidden);

    Tensor positions = arena.alloc(DType::I32, {work_columns});
    ops::fill_i32_positions(positions, work_first, stream);
    for (std::int32_t layer = 0; layer < g.layers; ++layer) {
        const DraftLayerWeights& w = weights.layers[static_cast<std::size_t>(layer)];
        Tensor key_raw             = arena.alloc(DType::BF16, {g.key_width(), work_columns});
        Tensor key       = arena.alloc(DType::BF16, {g.head_dim, g.kv_heads, work_columns});
        Tensor value_raw = arena.alloc(DType::BF16, {g.key_width(), work_columns});
        Tensor value     = value_raw.view({g.head_dim, g.kv_heads, work_columns});
        // LiLiCorr reuses the block's key/value parameters for the cached context; a plain DFlash
        // stores its own `context_key`/`context_value`, and reading the block's would attend over a
        // projection the drafter never trained for its context.
        const WeightId context_key   = g.lilicorr() ? w.key : w.context_key;
        const WeightId context_value = g.lilicorr() ? w.value : w.context_value;
        ops::linear(normed, model->matrix(context_key), key_raw, stream);
        ops::rmsnorm(key_raw.view({g.head_dim, g.kv_heads, work_columns}),
                     model->tensor(w.key_norm), g.eps, false, key, stream);
        ops::rope(positions, g.head_dim, g.head_dim / 2, g.theta, key, execution);
        ops::linear(normed, model->matrix(context_value), value_raw, stream);
        const std::string prefix = "body.layer" + std::to_string(layer);
        observe(observer, (prefix + ".k_ctx_post_rope").c_str(), key);
        observe(observer, (prefix + ".v_ctx").c_str(), value);
        if (new_count <= 0) { continue; }
        // The ring is at most `capacity` slots deep and the appended positions are consecutive, so
        // a slot holds a contiguous `key_width` window: one copy, or two if the run wraps.
        const std::size_t per_slot = static_cast<std::size_t>(g.key_width());
        const std::int32_t within  = new_first - work_first;
        for (std::int32_t side = 0; side < 2; ++side) {
            const auto* source =
                static_cast<const std::uint16_t*>(side == 0 ? key.data : value.data) +
                static_cast<std::size_t>(within) * per_slot;
            std::int32_t slot = new_first % capacity;
            std::int32_t done = 0;
            while (done < new_count) {
                const std::int32_t run = std::min(new_count - done, capacity - slot);
                const std::size_t offset =
                    ring_offset(lane, layer, side) + static_cast<std::size_t>(slot) * per_slot;
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint16_t*>(ring.p) + offset,
                                           source + static_cast<std::size_t>(done) * per_slot,
                                           static_cast<std::size_t>(run) * per_slot *
                                               sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToDevice, stream));
                done += run;
                slot = (slot + run) % capacity;
            }
        }
    }
    // A stage diff cannot reconstruct the attention without the operands the ring holds, so the
    // ring itself is a stage: one view per layer and side, addressed exactly as the body reads it.
    observe(observer, "ring.positions", ring_positions_view(lane));
    for (std::int32_t layer = 0; layer < g.layers; ++layer) {
        const std::string prefix = "ring.layer" + std::to_string(layer);
        observe(observer, (prefix + ".k").c_str(), ring_view(lane, layer, 0));
        observe(observer, (prefix + ".v").c_str(), ring_view(lane, layer, 1));
    }
    if (new_count > 0) {
        std::int32_t slot = new_first % capacity;
        std::int32_t done = 0;
        while (done < new_count) {
            const std::int32_t run = std::min(new_count - done, capacity - slot);
            auto* base = static_cast<std::uint8_t*>(ring_positions.p) +
                         static_cast<std::size_t>(lane) * static_cast<std::size_t>(capacity) *
                             sizeof(std::int32_t) +
                         static_cast<std::size_t>(slot) * sizeof(std::int32_t);
            Tensor run_view(base, DType::I32, {run});
            ops::fill_i32_positions(run_view, new_first + done, stream);
            done += run;
            slot = (slot + run) % capacity;
        }
    }
    appended_through[static_cast<std::size_t>(lane)] = ctx_last;
}

// The conditioning entry a committed pass uses: every column the pass wrote becomes context, in the
// pass's own order. Only the tail the ring can hold is materialized, because a position older than
// the draft's window is invisible to every query the draft asks -- `propose` clips the same way, so
// the two paths agree about which columns the ring covers.
void DraftContext::Impl::condition(std::int32_t lane, const Tensor& taps,
                                   std::int32_t first_position, DeviceExecutionView execution) {
    if (taps.dtype != DType::BF16 || taps.ne[0] != g.taps_rows() || taps.ne[2] != 1 ||
        taps.ne[3] != 1) {
        throw std::invalid_argument("nemotron_h draft: the conditioning taps must be BF16 [" +
                                    std::to_string(g.taps_rows()) + ", columns]");
    }
    const std::int32_t available = taps.ne[1];
    if (available <= 0 || first_position < 0) { return; }
    const std::int32_t columns = std::min(available, capacity);
    const std::int32_t first   = first_position + available - columns;
    Tensor window              = taps;
    if (columns != available) {
        window = Tensor(static_cast<std::uint8_t*>(taps.data) +
                            static_cast<std::size_t>(available - columns) *
                                static_cast<std::size_t>(taps.nb[1]),
                        DType::BF16, {g.taps_rows(), columns});
    }
    DeviceArena::Scope scope = arena.scope();
    Tensor unused;
    context(lane, window, first, first + columns - 1, unused, nullptr, execution);
}

void DraftContext::Impl::body(std::int32_t lane, const DraftRound& round, const Tensor& id_staging,
                              const Tensor& position_staging, const Tensor& position_high_staging,
                              Tensor& pass_hidden, Tensor& anchor_hidden,
                              const DraftObserver* observer, DeviceExecutionView execution) {
    const cudaStream_t stream   = execution.stream;
    const DraftWeights& weights = *model->weights().draft;
    const std::int32_t hidden   = g.hidden;
    const std::int32_t rows     = g.block_size;
    // Every allocation below comes from the body's own rewound arena (see `body_arena`), so the
    // offsets are identical every round -- the precondition for capturing this sequence.
    DeviceArena& arena = body_arena;

    // ---- The query block -----------------------------------------------------------------------
    // The residual stream stays FP32 for the whole body. The body's accumulated BF16 rounding at
    // every residual add and norm is the measured acceptance term (oracle head on the engine's
    // BF16 hidden caps rank 0 at 2.062 against 2.938 on the oracle's FP32 hidden, same taps); the
    // BF16-only operands (linears, grouped convolution, attention, embedding) get a widened cast.
    Tensor residual = arena.alloc(DType::FP32, {hidden, rows});
    {
        // The block's ids arrive in a fixed staging buffer the caller fills before the round, so
        // the captured body reads one address (no per-round upload inside the forward).
        Tensor embedded = arena.alloc(DType::BF16, {hidden, rows});
        ops::embedding(id_staging, model->matrix(model->weights().text.token_embedding), 1.0F,
                       embedded, stream);
        if (!g.lilicorr()) {
            // Every proposal row is the mask token, and the companion carries that token's embedding
            // itself -- the one row of its `embed_tokens` that differs from the tower's. The anchor
            // row keeps the tower's embedding of a real token, which the two stores share.
            const Tensor& mask = model->bound(weights.mask_embedding).tensor;
            const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
            for (std::int32_t row = 1; row < rows; ++row) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedded.data) +
                                               static_cast<std::size_t>(row) * embedded.nb[1],
                                           mask.data, row_bytes, cudaMemcpyDeviceToDevice,
                                           stream));
            }
        }
        ops::cast_bf16_to_fp32(embedded, residual, stream);
    }
    observe(observer, "body.embeddings", residual);
    const Tensor position_q    = position_staging.view({rows, 1});
    const Tensor position_high = position_high_staging.view({rows, 1});

    const std::size_t workspace_bytes = ops::sliding_causal_attention_workspace_bytes(
        ops::AttentionHeadGeometry{g.head_dim, g.heads, g.kv_heads}, rows, capacity + rows, 1);
    Tensor attention_workspace;
    if (workspace_bytes != 0) {
        attention_workspace = arena.alloc(DType::U8, {static_cast<std::int32_t>(workspace_bytes)});
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(g.head_dim));

    // One side of one branch's grouped convolution. The Op indexes its payloads dim0-fastest --
    // `x`/`out` as `[hidden, rows, 1]` and `delta` in the kernel projection's own row order -- so
    // every operand is the tensor the linear produced, with `delta` shared by the branch's two
    // sides.
    const auto convolve = [&](const Tensor& input, const Tensor& base, const Tensor& delta,
                              std::int32_t side, Tensor& into) {
        Tensor input_block = input.view({hidden, rows, 1});
        Tensor delta_block = delta.view({g.coefficients(), rows, 1});
        Tensor out_block   = into.view({hidden, rows, 1});
        ops::grouped_tap_apply(input_block, base, delta_block, out_block, side, g.group_width, rows,
                               execution);
    };

    for (std::int32_t layer = 0; layer < g.layers; ++layer) {
        const DraftLayerWeights& w = weights.layers[static_cast<std::size_t>(layer)];
        const std::string prefix   = "body.layer" + std::to_string(layer);
        observe(observer, (prefix + ".residual_in").c_str(), residual);

        Tensor normed_in = arena.alloc(DType::FP32, {hidden, rows});
        ops::rmsnorm(residual, model->tensor(w.input_norm), g.eps, false, normed_in, stream);
        observe(observer, (prefix + ".norm_in").c_str(), normed_in);
        Tensor normed_in_bf16 = arena.alloc(DType::BF16, {hidden, rows});
        ops::cast_fp32_to_bf16(normed_in, normed_in_bf16, stream);
        // A plain DFlash has no convolution: the normalized input *is* the branch input, and the
        // delta plane exists only where a convolution consumes it.
        Tensor prepared = normed_in_bf16;
        Tensor delta;
        if (g.lilicorr()) {
            delta = arena.alloc(DType::BF16, {g.coefficients(), rows});
            ops::linear(normed_in_bf16, model->matrix(w.attention_conv_projection), delta, stream);
            observe(observer, (prefix + ".conv_attn_delta").c_str(), delta);
            prepared = arena.alloc(DType::BF16, {hidden, rows});
            convolve(normed_in_bf16, model->tensor(w.attention_conv_base), delta, 0, prepared);
            observe(observer, (prefix + ".conv_attn_pre").c_str(), prepared);
        }

        Tensor query      = arena.alloc(DType::BF16, {g.head_dim, g.heads, rows});
        Tensor key        = arena.alloc(DType::BF16, {g.head_dim, g.kv_heads, rows});
        Tensor value      = arena.alloc(DType::BF16, {g.head_dim, g.kv_heads, rows});
        Tensor query_flat = query.view({g.query_width(), rows});
        Tensor key_flat   = key.view({g.key_width(), rows});
        Tensor value_flat = value.view({g.key_width(), rows});
        ops::linear(prepared, model->matrix(w.query), query_flat, stream);
        ops::linear(prepared, model->matrix(w.key), key_flat, stream);
        ops::linear(prepared, model->matrix(w.value), value_flat, stream);
        observe(observer, (prefix + ".q_pre_norm").c_str(), query);
        observe(observer, (prefix + ".k_pre_norm").c_str(), key);
        observe(observer, (prefix + ".v_pre_norm").c_str(), value);
        Tensor query_normed = arena.alloc(DType::BF16, {g.head_dim, g.heads, rows});
        Tensor key_normed   = arena.alloc(DType::BF16, {g.head_dim, g.kv_heads, rows});
        ops::rmsnorm(query, model->tensor(w.query_norm), g.eps, false, query_normed, stream);
        ops::rmsnorm(key, model->tensor(w.key_norm), g.eps, false, key_normed, stream);
        observe(observer, (prefix + ".q_post_norm").c_str(), query_normed);
        observe(observer, (prefix + ".k_post_norm").c_str(), key_normed);
        ops::rope(position_staging, g.head_dim, g.head_dim / 2, g.theta, query_normed, key_normed,
                  execution);
        observe(observer, (prefix + ".q_post_rope").c_str(), query_normed);
        observe(observer, (prefix + ".k_post_rope").c_str(), key_normed);

        Tensor attention     = arena.alloc(DType::BF16, {g.head_dim, g.heads, rows});
        Tensor query_heads   = query_normed.view({g.head_dim, g.heads, rows, 1});
        Tensor key_rows      = key_normed.view({g.head_dim, g.kv_heads, rows, 1});
        Tensor value_rows    = value.view({g.head_dim, g.kv_heads, rows, 1});
        Tensor attention_out = attention.view({g.head_dim, g.heads, rows, 1});
        ops::sliding_causal_attention(
            query_heads, ring_view(lane, layer, 0), ring_view(lane, layer, 1), position_q,
            position_high, ring_positions_view(lane), key_rows, value_rows, position_q,
            ops::AttentionHeadGeometry{g.head_dim, g.heads, g.kv_heads}, g.window, scale,
            attention_out, attention_workspace, execution,
            g.lilicorr() ? model->tensor(w.sink_bias) : Tensor{});
        observe(observer, (prefix + ".attn_out").c_str(), attention);
        Tensor projected      = arena.alloc(DType::BF16, {hidden, rows});
        Tensor attention_flat = attention.view({g.query_width(), rows});
        ops::linear(attention_flat, model->matrix(w.output), projected, stream);
        observe(observer, (prefix + ".attn_out_proj").c_str(), projected);
        Tensor convolved = projected;
        if (g.lilicorr()) {
            convolved = arena.alloc(DType::BF16, {hidden, rows});
            convolve(projected, model->tensor(w.attention_conv_base), delta, 1, convolved);
            observe(observer, (prefix + ".attn_out_conv").c_str(), convolved);
        }
        Tensor convolved_wide = arena.alloc(DType::FP32, {hidden, rows});
        ops::cast_bf16_to_fp32(convolved, convolved_wide, stream);
        ops::residual_add(convolved_wide, residual, stream);
        observe(observer, (prefix + ".hidden_post_attn").c_str(), residual);

        Tensor normed_mlp = arena.alloc(DType::FP32, {hidden, rows});
        ops::rmsnorm(residual, model->tensor(w.post_attention_norm), g.eps, false, normed_mlp,
                     stream);
        observe(observer, (prefix + ".norm_mlp_in").c_str(), normed_mlp);
        Tensor normed_mlp_bf16 = arena.alloc(DType::BF16, {hidden, rows});
        ops::cast_fp32_to_bf16(normed_mlp, normed_mlp_bf16, stream);
        Tensor prepared_mlp = normed_mlp_bf16;
        Tensor delta_mlp;
        if (g.lilicorr()) {
            delta_mlp = arena.alloc(DType::BF16, {g.coefficients(), rows});
            ops::linear(normed_mlp_bf16, model->matrix(w.mlp_conv_projection), delta_mlp, stream);
            observe(observer, (prefix + ".conv_mlp_delta").c_str(), delta_mlp);
            prepared_mlp = arena.alloc(DType::BF16, {hidden, rows});
            convolve(normed_mlp_bf16, model->tensor(w.mlp_conv_base), delta_mlp, 0, prepared_mlp);
            observe(observer, (prefix + ".pre_mlp").c_str(), prepared_mlp);
        }
        Tensor gate = arena.alloc(DType::BF16, {g.inner, rows});
        Tensor up   = arena.alloc(DType::BF16, {g.inner, rows});
        ops::linear(prepared_mlp, model->matrix(w.mlp_gate), gate, stream);
        ops::linear(prepared_mlp, model->matrix(w.mlp_up), up, stream);
        observe(observer, (prefix + ".mlp_gate").c_str(), gate);
        observe(observer, (prefix + ".mlp_up").c_str(), up);
        Tensor activated = arena.alloc(DType::BF16, {g.inner, rows});
        ops::silu_mul(gate, up, activated, stream);
        observe(observer, (prefix + ".mlp_act").c_str(), activated);
        Tensor down = arena.alloc(DType::BF16, {hidden, rows});
        ops::linear(activated, model->matrix(w.mlp_down), down, stream);
        observe(observer, (prefix + ".mlp_out").c_str(), down);
        Tensor convolved_mlp = down;
        if (g.lilicorr()) {
            convolved_mlp = arena.alloc(DType::BF16, {hidden, rows});
            convolve(down, model->tensor(w.mlp_conv_base), delta_mlp, 1, convolved_mlp);
            observe(observer, (prefix + ".post_mlp").c_str(), convolved_mlp);
        }
        Tensor convolved_mlp_wide = arena.alloc(DType::FP32, {hidden, rows});
        ops::cast_bf16_to_fp32(convolved_mlp, convolved_mlp_wide, stream);
        ops::residual_add(convolved_mlp_wide, residual, stream);
        observe(observer, (prefix + ".out").c_str(), residual);
    }
    Tensor final_normed = arena.alloc(DType::FP32, {hidden, rows});
    ops::rmsnorm(residual, model->tensor(weights.final_norm), g.eps, false, final_normed, stream);
    observe(observer, "body.norm_out", final_normed);
    // The head's linears are BF16 operands; this cast is the head's boundary and nothing else.
    Tensor final_bf16 = arena.alloc(DType::BF16, {hidden, rows});
    ops::cast_fp32_to_bf16(final_normed, final_bf16, stream);
    // `pass_hidden` is a fixed caller plane: a captured body writes the same address every round,
    // so this is a 2-D device copy of columns [1, slots] rather than a fresh slice.
    const Tensor pass_source = final_bf16.slice(1, 1, g.block_size - 1);
    CUDA_CHECK(cudaMemcpy2DAsync(pass_hidden.data, static_cast<std::size_t>(pass_hidden.nb[1]),
                                 pass_source.data, static_cast<std::size_t>(pass_source.nb[1]),
                                 static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                                 static_cast<std::size_t>(g.block_size - 1),
                                 cudaMemcpyDeviceToDevice, stream));
    observe(observer, "head.pass_hidden", pass_hidden);
}

std::vector<std::int32_t> DraftContext::Impl::head(const DraftRound& round,
                                                   const Tensor& pass_hidden,
                                                   const Tensor& anchor_hidden,
                                                   const DraftObserver* observer,
                                                   DeviceExecutionView execution) {
    if (!g.lilicorr()) { return head_dflash(round, pass_hidden, observer, execution); }
    const cudaStream_t stream  = execution.stream;
    const DraftHeadWeights& hd = model->weights().draft->head;
    const std::int32_t slots   = round.slots;
    const std::int32_t top_k   = g.top_k;
    const std::int32_t nodes   = slots * top_k;
    const std::int32_t width   = g.head_width;

    // ---- Candidates: the tower's own head, then this head's selection of it --------------------
    Tensor logits = arena.alloc(DType::BF16, {g.vocab, slots});
    ops::linear(pass_hidden, model->matrix(model->weights().text.output_head), logits, stream);
    observe(observer, "head.logits", logits);
    host.logits.resize(static_cast<std::size_t>(g.vocab) * static_cast<std::size_t>(slots));
    copy_to_host(logits, host.logits.data(), stream);

    host.ids.assign(static_cast<std::size_t>(nodes), 0);
    host.candidate_logits.assign(static_cast<std::size_t>(nodes), 0.0F);
    host.candidate_logprobs.assign(static_cast<std::size_t>(nodes), 0.0F);
    host.candidate_probs.assign(static_cast<std::size_t>(nodes), 0.0F);
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        const std::size_t at = static_cast<std::size_t>(slot) * static_cast<std::size_t>(top_k);
        select_candidates(&host.logits[static_cast<std::size_t>(g.vocab) * slot], g.vocab, top_k,
                          &host.ids[at], &host.candidate_logits[at], &host.candidate_logprobs[at],
                          &host.candidate_probs[at]);
    }
    host.features.assign(static_cast<std::size_t>(g.feature_dim) * static_cast<std::size_t>(nodes),
                         0.0F);
    build_features(host.candidate_logprobs.data(), host.candidate_probs.data(), top_k,
                   g.feature_dim, nodes, host.features.data());
    if (observer != nullptr) {
        observe(observer, "head.candidate_ids", report_i32(host.ids.data(), top_k, slots, stream));
        observe(observer, "head.candidate_logits",
                report_fp32(host.candidate_logits.data(), top_k, slots, stream));
        observe(observer, "head.candidate_logprobs",
                report_fp32(host.candidate_logprobs.data(), top_k, slots, stream));
        observe(observer, "head.candidate_probs",
                report_fp32(host.candidate_probs.data(), top_k, slots, stream));
        observe(observer, "head.features",
                report_fp32(host.features.data(), g.feature_dim, nodes, stream));
    }
    std::vector<float> normalised(
        static_cast<std::size_t>(g.feature_dim) * static_cast<std::size_t>(nodes), 0.0F);
    normalise_features(host.features.data(), feature_gain.data(), feature_bias.data(),
                       g.feature_dim, nodes, kFeatureNormEps, normalised.data());
    if (observer != nullptr) {
        observe(observer, "head.feature_norm_out",
                report_fp32(normalised.data(), g.feature_dim, nodes, stream));
    }
    Tensor feature_input  = upload_bf16(normalised.data(), g.feature_dim, nodes, stream);
    Tensor feature_up     = arena.alloc(DType::BF16, {width, nodes});
    Tensor feature_act    = arena.alloc(DType::BF16, {width, nodes});
    Tensor feature_states = arena.alloc(DType::BF16, {width, nodes});
    ops::linear(feature_input, model->matrix(hd.feature_mlp_up), feature_up, stream);
    ops::add_bias(model->tensor(hd.feature_mlp_up_bias), feature_up, stream);
    ops::silu_mul(feature_up, ones(width, nodes), feature_act, stream);
    ops::linear(feature_act, model->matrix(hd.feature_mlp_down), feature_states, stream);
    ops::add_bias(model->tensor(hd.feature_mlp_down_bias), feature_states, stream);
    observe(observer, "head.feature_mlp_out", feature_states);

    // ---- The node table: the candidate token, the slot's hidden state, the features ------------
    Tensor candidate_ids = arena.alloc(DType::I32, {nodes});
    copy_from_host(host.ids.data(), candidate_ids, stream);
    observe(observer, "head.candidate_ids_device", candidate_ids);
    Tensor embeddings = arena.alloc(DType::BF16, {g.hidden, nodes});
    ops::embedding(candidate_ids, model->matrix(model->weights().text.token_embedding), 1.0F,
                   embeddings, stream);
    Tensor node = arena.alloc(DType::BF16, {width, nodes});
    ops::linear(embeddings, model->matrix(hd.token_projection), node, stream);
    ops::add_bias(model->tensor(hd.token_bias), node, stream);
    observe(observer, "head.node_token", node);

    Tensor pass_states = arena.alloc(DType::BF16, {width, slots});
    ops::linear(pass_hidden, model->matrix(hd.pass_hidden_projection), pass_states, stream);
    ops::add_bias(model->tensor(hd.pass_hidden_bias), pass_states, stream);
    observe(observer, "head.node_pass", pass_states);
    host.pass_states.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(slots));
    load_bf16(pass_states, host.packed, host.pass_states, stream);
    // The pass state is one vector per slot and the two structural embeddings one per slot and one
    // per rank, so their sum is a `[width, nodes]` table the node adds once.
    host.bias_table.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(nodes), 0.0F);
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t column = static_cast<std::size_t>(slot * top_k + rank);
            for (std::int32_t channel = 0; channel < width; ++channel) {
                const std::size_t key =
                    static_cast<std::size_t>(channel) + static_cast<std::size_t>(width) * column;
                host.bias_table[key] = host.pass_states[static_cast<std::size_t>(channel) +
                                                        static_cast<std::size_t>(width) *
                                                            static_cast<std::size_t>(slot)] +
                                       slot_table[static_cast<std::size_t>(slot) * width +
                                                  static_cast<std::size_t>(channel)] +
                                       rank_table[static_cast<std::size_t>(rank) * width +
                                                  static_cast<std::size_t>(channel)];
            }
        }
    }
    Tensor bias_table = upload_bf16(host.bias_table.data(), width, nodes, stream);
    ops::residual_add(bias_table, node, stream);
    ops::residual_add(feature_states, node, stream);
    observe(observer, "head.node", node);

    // ---- The lattice: two pre-norm bidirectional layers with the bias as their only mask --------
    lattice_bias(relative_bias.data(), same_bias.data(), g.head_heads, g.block_size, slots, top_k,
                 host.lattice_bias_table);
    if (observer != nullptr) {
        observe(observer, "head.lattice_bias",
                report_fp32(host.lattice_bias_table.data(), g.head_heads, nodes * nodes, stream));
    }
    for (std::int32_t layer = 0; layer < g.head_layers; ++layer) {
        const DraftHeadLayerWeights& w = hd.layers[static_cast<std::size_t>(layer)];
        const std::string prefix       = "head.lattice" + std::to_string(layer);
        Tensor normed                  = arena.alloc(DType::BF16, {width, nodes});
        ops::rmsnorm(node, model->tensor(w.norm), g.eps, false, normed, stream);
        observe(observer, (prefix + "_attn_in").c_str(), normed);
        Tensor qkv = arena.alloc(DType::BF16, {3 * width, nodes});
        ops::linear(normed, model->matrix(w.attention_in), qkv, stream);
        ops::add_bias(model->tensor(w.attention_in_bias), qkv, stream);
        if (observer != nullptr) {
            // A dim-0 slice of `[3 * width, nodes]` keeps the parent's row pitch, so each half is
            // its own contiguous view over the fused buffer rather than a slice of it.
            const std::size_t half = sizeof(std::uint16_t) * static_cast<std::size_t>(width) *
                                     static_cast<std::size_t>(nodes);
            for (std::int32_t part = 0; part < 3; ++part) {
                Tensor heads(static_cast<std::uint8_t*>(qkv.data) +
                                 static_cast<std::size_t>(part) * half,
                             DType::BF16, {g.lattice_dim, g.head_heads, nodes});
                const char* const suffix = part == 0 ? "_q" : part == 1 ? "_k" : "_v";
                observe(observer, (prefix + suffix).c_str(), heads);
            }
        }
        load_bf16(qkv, host.packed, host.qkv, stream);
        if (host.qkv.size() != static_cast<std::size_t>(3) * static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(nodes)) {
            throw std::logic_error("nemotron_h draft: the lattice's fused projection is not the "
                                   "shape its attention expects");
        }
        lattice_attention(host.qkv, width, g.head_heads, g.lattice_dim, nodes,
                          host.lattice_bias_table, host.attention, host);
        Tensor attention = upload_bf16(host.attention.data(), width, nodes, stream);
        Tensor projected = arena.alloc(DType::BF16, {width, nodes});
        ops::linear(attention, model->matrix(w.attention_out), projected, stream);
        ops::add_bias(model->tensor(w.attention_out_bias), projected, stream);
        observe(observer, (prefix + "_attn_out").c_str(), projected);
        ops::residual_add(projected, node, stream);
        observe(observer, (prefix + "_resid1").c_str(), node);

        Tensor mlp_in = arena.alloc(DType::BF16, {width, nodes});
        ops::rmsnorm(node, model->tensor(w.mlp_norm), g.eps, false, mlp_in, stream);
        Tensor mlp_up   = arena.alloc(DType::BF16, {g.head_mlp, nodes});
        Tensor mlp_act  = arena.alloc(DType::BF16, {g.head_mlp, nodes});
        Tensor mlp_down = arena.alloc(DType::BF16, {width, nodes});
        ops::linear(mlp_in, model->matrix(w.mlp_up), mlp_up, stream);
        ops::add_bias(model->tensor(w.mlp_up_bias), mlp_up, stream);
        ops::silu_mul(mlp_up, ones(g.head_mlp, nodes), mlp_act, stream);
        ops::linear(mlp_act, model->matrix(w.mlp_down), mlp_down, stream);
        ops::add_bias(model->tensor(w.mlp_down_bias), mlp_down, stream);
        observe(observer, (prefix + "_mlp_out").c_str(), mlp_down);
        ops::residual_add(mlp_down, node, stream);
        observe(observer, (prefix + "_out").c_str(), node);
    }
    Tensor normed_nodes = arena.alloc(DType::BF16, {width, nodes});
    ops::rmsnorm(node, model->tensor(hd.output_norm), g.eps, false, normed_nodes, stream);
    observe(observer, "head.lattice_out", normed_nodes);

    // ---- The anchor, outside the lattice and built from the fused target state only -------------
    Tensor anchor_pre = arena.alloc(DType::BF16, {width, 1});
    ops::linear(anchor_hidden, model->matrix(hd.context_projection), anchor_pre, stream);
    ops::add_bias(model->tensor(hd.context_bias), anchor_pre, stream);
    observe(observer, "head.anchor_state_pre_mask", anchor_pre);
    Tensor anchor_state = arena.alloc(DType::BF16, {width, 1});
    ops::rmsnorm(anchor_pre, model->tensor(hd.anchor_norm), g.eps, false, anchor_state, stream);
    observe(observer, "head.anchor_state", anchor_state);
    Tensor anchor_raw = arena.alloc(DType::BF16, {width, 1});
    ops::linear(anchor_state, model->matrix(hd.anchor_out_head), anchor_raw, stream);
    ops::add_bias(model->tensor(hd.anchor_out_head_bias), anchor_raw, stream);
    load_bf16(anchor_raw, host.packed, host.anchor, stream);
    l2_normalise(host.anchor, 1, width, g.vector_eps);
    if (observer != nullptr) {
        observe(observer, "head.anchor_out", report_fp32(host.anchor.data(), width, 1, stream));
    }

    // ---- The edge factors and the walk --------------------------------------------------------
    load_bf16(normed_nodes, host.packed, host.node, stream);
    host.factor_input.assign(static_cast<std::size_t>(3) * static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(nodes),
                             0.0F);
    for (std::size_t at = 0; at < static_cast<std::size_t>(width) * nodes; ++at) {
        const std::int32_t channel = static_cast<std::int32_t>(at % width);
        const float node_value     = host.node[at];
        host.factor_input[at]      = node_value;
        host.factor_input[at + static_cast<std::size_t>(width) * nodes] = host.anchor[channel];
        host.factor_input[at + 2 * static_cast<std::size_t>(width) * nodes] =
            node_value * host.anchor[channel];
    }
    if (observer != nullptr) {
        observe(observer, "head.factor_input",
                report_fp32(host.factor_input.data(), 3 * width, nodes, stream));
    }
    Tensor factor_input_operand = upload_bf16(host.factor_input.data(), 3 * width, nodes, stream);
    Tensor pre                  = arena.alloc(DType::BF16, {width, nodes});
    ops::linear(factor_input_operand, model->matrix(hd.factor_input_projection), pre, stream);
    ops::add_bias(model->tensor(hd.factor_input_bias), pre, stream);
    observe(observer, "head.factor_pre", pre);
    Tensor factor = arena.alloc(DType::BF16, {width, nodes});
    ops::silu_mul(pre, ones(width, nodes), factor, stream);
    observe(observer, "head.factor", factor);

    Tensor out_edges = arena.alloc(DType::BF16, {width, nodes});
    Tensor in_edges  = arena.alloc(DType::BF16, {width, nodes});
    ops::linear(factor, model->matrix(hd.out_head), out_edges, stream);
    ops::add_bias(model->tensor(hd.out_head_bias), out_edges, stream);
    ops::linear(factor, model->matrix(hd.in_head), in_edges, stream);
    ops::add_bias(model->tensor(hd.in_head_bias), in_edges, stream);
    host.edges.assign(static_cast<std::size_t>(2) * static_cast<std::size_t>(width) *
                          static_cast<std::size_t>(nodes),
                      0.0F);
    load_bf16(out_edges, host.packed, host.values, stream);
    std::copy(host.values.begin(), host.values.end(), host.edges.begin());
    load_bf16(in_edges, host.packed, host.values, stream);
    std::copy(host.values.begin(), host.values.end(),
              host.edges.begin() + static_cast<std::ptrdiff_t>(width) * nodes);
    if (observer != nullptr) {
        observe(observer, "head.edges", report_fp32(host.edges.data(), 2 * width, nodes, stream));
    }

    host.out_vec.assign(host.edges.begin(),
                        host.edges.begin() + static_cast<std::ptrdiff_t>(width) * nodes);
    host.in_vec.assign(host.edges.begin() + static_cast<std::ptrdiff_t>(width) * nodes,
                       host.edges.end());
    l2_normalise(host.out_vec, nodes, width, g.vector_eps);
    l2_normalise(host.in_vec, nodes, width, g.vector_eps);
    if (observer != nullptr) {
        observe(observer, "head.out_vec", report_fp32(host.out_vec.data(), width, nodes, stream));
        observe(observer, "head.in_vec", report_fp32(host.in_vec.data(), width, nodes, stream));
    }

    // `start[k]` is the anchor's outgoing vector against slot 0's incoming vectors; `pairs` is one
    // slot's outgoing against the next slot's incoming; the score is the scaled cosine.
    host.start.assign(static_cast<std::size_t>(top_k), 0.0F);
    for (std::int32_t k = 0; k < top_k; ++k) {
        host.start[static_cast<std::size_t>(k)] =
            dot(host.anchor, 0, host.in_vec, static_cast<std::size_t>(k) * width, width);
    }
    host.scores.assign(static_cast<std::size_t>(slots) * top_k * top_k, 0.0F);
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        for (std::int32_t previous = 0; previous < top_k; ++previous) {
            for (std::int32_t next = 0; next < top_k; ++next) {
                float value = 0.0F;
                if (slot == 0) {
                    value = host.start[static_cast<std::size_t>(next)];
                } else {
                    value = dot(host.out_vec,
                                (static_cast<std::size_t>((slot - 1) * top_k + previous)) * width,
                                host.in_vec, static_cast<std::size_t>(slot * top_k + next) * width,
                                width);
                }
                host.scores[static_cast<std::size_t>((slot * top_k + previous) * top_k + next)] =
                    g.logit_scale * value;
            }
        }
    }
    host.pairs.assign(static_cast<std::size_t>(std::max(0, slots - 1)) * top_k * top_k, 0.0F);
    for (std::int32_t slot = 1; slot < slots; ++slot) {
        for (std::int32_t previous = 0; previous < top_k; ++previous) {
            for (std::int32_t next = 0; next < top_k; ++next) {
                host.pairs[static_cast<std::size_t>(((slot - 1) * top_k + previous) * top_k +
                                                    next)] =
                    host.scores[static_cast<std::size_t>((slot * top_k + previous) * top_k +
                                                         next)] /
                    g.logit_scale;
            }
        }
    }
    if (observer != nullptr) {
        observe(observer, "head.start_scores", report_fp32(host.start.data(), top_k, 1, stream));
        observe(observer, "head.pair_scores",
                report_fp32(host.pairs.data(), top_k, std::max(1, slots - 1) * top_k, stream));
        observe(observer, "head.score_tensor",
                report_fp32(host.scores.data(), top_k * top_k, slots, stream));
    }
    walk_scores(host.scores, host.ids, slots, top_k, host.walk, host.indices, host.rows);
    if (observer != nullptr) {
        observe(observer, "head.walk_ids", report_i32(host.walk.data(), slots, 1, stream));
        observe(observer, "head.walk_indices", report_i32(host.indices.data(), slots, 1, stream));
        observe(observer, "head.walk_rows", report_fp32(host.rows.data(), top_k, slots, stream));
    }
    return host.walk;
}

// A plain DFlash's whole head: the tower's own output head over the mask columns, one greedy argmax
// per column. The companion carries no head weights, no candidate set and no lattice, so this is
// the proposal -- and it is why the block length is a round parameter: the columns scored are
// exactly the slots the round will verify.
std::vector<std::int32_t> DraftContext::Impl::head_dflash(const DraftRound& round,
                                                          const Tensor& pass_hidden,
                                                          const DraftObserver* observer,
                                                          DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const std::int32_t slots  = round.slots;
    Tensor logits(proposal_logits.p, DType::BF16, {g.vocab, slots});
    ops::linear(pass_hidden, model->matrix(model->weights().text.output_head), logits, stream);
    observe(observer, "head.logits", logits);

    std::vector<std::int32_t> positions(static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        positions[static_cast<std::size_t>(slot)] = round.first_block_position + 1 + slot;
    }
    Tensor logical(proposal_positions.p, DType::I32, {slots});
    copy_from_host(positions.data(), logical, stream);
    Tensor drawn(proposal_drawn.p, DType::I32, {slots});
    proposal_workspace->reset();
    ops::sample(logits, drawn, g.vocab,
                static_cast<const ops::SamplingConfig*>(proposal_configs.p), logical,
                ops::kSamplePurposeSpeculativeAccept, *proposal_workspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(proposal_drawn_host.data(), proposal_drawn.p,
                               static_cast<std::size_t>(slots) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {proposal_drawn_host.begin(),
            proposal_drawn_host.begin() + static_cast<std::ptrdiff_t>(slots)};
}

std::vector<std::int32_t> DraftContext::Impl::propose(std::int32_t lane, const DraftRound& round,
                                                      DeviceExecutionView execution,
                                                      const DraftObserver* observer) {
    const cudaStream_t stream = execution.stream;
    if (round.taps.dtype != DType::BF16 || round.taps.ne[0] != g.taps_rows() ||
        round.taps.ne[2] != 1 || round.taps.ne[3] != 1) {
        throw std::invalid_argument(
            "nemotron_h draft: the round's taps must be BF16 [" + std::to_string(g.taps_rows()) +
            ", columns] and are " + std::to_string(round.taps.ne[0]) + "x" +
            std::to_string(round.taps.ne[1]) + "x" + std::to_string(round.taps.ne[2]) + "x" +
            std::to_string(round.taps.ne[3]) + " at " +
            (round.taps.dtype == DType::BF16 ? "BF16" : "another dtype"));
    }
    const std::int32_t available = round.taps.ne[1];
    if (available <= 0) {
        throw std::invalid_argument("nemotron_h draft: a round conditions on at least one context "
                                    "column");
    }
    if (round.slots < 1 || round.slots >= g.block_size) {
        throw std::invalid_argument("nemotron_h draft: a round scores 1 to " +
                                    std::to_string(g.block_size - 1) +
                                    " slots, and this one names " + std::to_string(round.slots));
    }
    if (round.block_ids == nullptr) {
        throw std::invalid_argument("nemotron_h draft: a round needs its block's ids");
    }
    if (round.first_context_position < 0 || round.first_block_position < 0) {
        throw std::invalid_argument("nemotron_h draft: positions are non-negative");
    }
    if (round.first_block_position != round.first_context_position + available) {
        throw std::invalid_argument("nemotron_h draft: the block must begin where the context "
                                    "ends, at " +
                                    std::to_string(round.first_context_position + available) +
                                    ", and the round names " +
                                    std::to_string(round.first_block_position));
    }
    // Only the tail of the context the ring can hold is materialized: a position older than the
    // window is invisible to every query the block asks, so passing it would cost a projection and
    // change nothing. `condition` clips the same way, which is what keeps the two paths' notion of
    // which positions the ring covers identical.
    const std::int32_t columns       = std::min(available, capacity);
    const std::int32_t first_context = round.first_block_position - columns;
    if (first_context < 0) {
        throw std::invalid_argument("nemotron_h draft: a round's block starts at " +
                                    std::to_string(round.first_block_position) +
                                    " with no room for its context");
    }
    DraftRound clipped             = round;
    clipped.first_context_position = first_context;
    if (columns != available) {
        clipped.taps = Tensor(static_cast<std::uint8_t*>(round.taps.data) +
                                  static_cast<std::size_t>(available - columns) *
                                      static_cast<std::size_t>(round.taps.nb[1]),
                              DType::BF16, {g.taps_rows(), columns});
    }
    const std::int32_t appended = appended_through[static_cast<std::size_t>(lane)];
    if (appended >= 0 && first_context > appended + 1) {
        throw std::invalid_argument(
            "nemotron_h draft: the round's context starts at " + std::to_string(first_context) +
            " and the ring holds up to " + std::to_string(appended) +
            "; the positions between them were never conditioned on and the ring cannot invent "
            "them");
    }
    for (std::int32_t row = 0; row < 1 + round.slots; ++row) {
        if (round.block_ids[row] < 0 || round.block_ids[row] >= g.vocab) {
            throw std::invalid_argument("nemotron_h draft: block id " +
                                        std::to_string(round.block_ids[row]) +
                                        " is outside the vocabulary");
        }
    }

    DeviceArena::Scope scope      = arena.scope();
    DeviceArena::Scope body_scope = body_arena.scope();
    // The round-varying inputs live in fixed staging planes: the arena rewinds deterministically,
    // so these addresses are the same every round and a captured body can read them.
    // The body runs the *widest* block every round, exactly as the reference does: the rows past
    // the round's slots carry the mask token too and their hidden columns are simply not scored.
    // That is what keeps the captured body's width -- and so the extent of every staging plane it
    // reads -- the same from round to round; a narrower round staging fewer ids would leave the
    // graph's tail rows reading whatever the next allocation wrote there.
    const std::int32_t rows      = g.block_size;
    Tensor id_staging            = arena.alloc(DType::I32, {rows});
    Tensor position_staging      = arena.alloc(DType::I32, {rows});
    Tensor position_high_staging = arena.alloc(DType::I32, {rows});
    std::vector<std::int32_t> block_ids(static_cast<std::size_t>(rows), g.mask_token);
    for (std::int32_t row = 0; row < 1 + clipped.slots; ++row) {
        block_ids[static_cast<std::size_t>(row)] = clipped.block_ids[row];
    }
    CUDA_CHECK(cudaMemcpyAsync(id_staging.data, block_ids.data(),
                               static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(rows));
        for (std::int32_t row = 0; row < rows; ++row) {
            positions[static_cast<std::size_t>(row)] = clipped.first_block_position + row;
        }
        CUDA_CHECK(cudaMemcpyAsync(position_staging.data, positions.data(),
                                   positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream));
        for (std::int32_t row = 0; row < rows; ++row) {
            positions[static_cast<std::size_t>(row)] =
                clipped.first_block_position + g.window - 1 + row;
        }
        CUDA_CHECK(cudaMemcpyAsync(position_high_staging.data, positions.data(),
                                   positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream));
    }
    Tensor pass_hidden = arena.alloc(DType::BF16, {g.hidden, g.block_size - 1});
    Tensor anchor_hidden;
    // The context phase stays outside the body (its tap copy has a round-varying offset and extent,
    // which a capture would bake); the body alone is the capture candidate.
    context(lane, clipped.taps, clipped.first_context_position,
            clipped.first_context_position + clipped.taps.ne[1] - 1, anchor_hidden, observer,
            execution);
    const char* no_graph = std::getenv("NINFER_DRAFT_NO_GRAPH");
    const bool graph_on =
        observer == nullptr && (no_graph == nullptr || std::string(no_graph) != "1");
    if (graph_on) {
        auto& definition = body_graphs[static_cast<std::size_t>(lane)];
        auto& executable = body_execs[static_cast<std::size_t>(lane)];
        if (!executable.ready()) {
            definition.capture(stream, [&] {
                body(lane, clipped, id_staging, position_staging, position_high_staging,
                     pass_hidden, anchor_hidden, nullptr, execution);
            });
            executable.instantiate(definition);
            executable.upload(stream);
        }
        executable.launch(stream);
    } else {
        body(lane, clipped, id_staging, position_staging, position_high_staging, pass_hidden,
             anchor_hidden, observer, execution);
    }
    // The body filled the widest plane; the head scores the round's slots, which are its leading
    // columns (hidden fastest, so the prefix is contiguous).
    const Tensor scored(pass_hidden.data, DType::BF16, {g.hidden, round.slots});
    std::vector<std::int32_t> proposed =
        head(round, scored, anchor_hidden, observer, execution);
    walk[static_cast<std::size_t>(lane)]       = proposed;
    candidates[static_cast<std::size_t>(lane)] = host.ids;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return proposed;
}

// ---- The plan, and the state itself -----------------------------------------------------------

namespace {

// The forward's transient arena: every tensor `body` and `head` allocate, at the widest block the
// spec admits, plus the Ops' own workspaces. It is planned rather than grown because everything in
// it is a function of the spec -- a round wider than the sliding window is refused, not grown into.
std::size_t scratch_bytes(const Geometry& g, std::int32_t capacity, std::int32_t rows,
                          std::int32_t nodes) {
    const std::size_t element = sizeof(std::uint16_t);
    const std::size_t hidden  = static_cast<std::size_t>(g.hidden);
    const std::size_t width   = static_cast<std::size_t>(g.head_width);
    const std::size_t inner   = static_cast<std::size_t>(g.inner);
    const std::size_t key     = static_cast<std::size_t>(g.key_width());
    const std::size_t query   = static_cast<std::size_t>(g.query_width());
    const std::size_t coeff   = static_cast<std::size_t>(g.coefficients());
    const std::size_t columns = static_cast<std::size_t>(capacity);
    const std::size_t block   = static_cast<std::size_t>(rows);
    const std::size_t count   = static_cast<std::size_t>(nodes);

    std::size_t bytes = 0;
    // The context phase: the tap operand, the fused feature, its norm, and one layer's K/V.
    bytes += static_cast<std::size_t>(g.taps_rows()) * columns * element;
    bytes += 2 * hidden * columns * element;
    bytes += 3 * key * columns * element * static_cast<std::size_t>(g.layers);
    bytes += columns * sizeof(std::int32_t);
    // The block phase: eight hidden-wide planes, two delta planes and three query/key/value planes
    // per layer, plus the residual and the final norm.
    bytes += (9 * hidden + 3 * query + 3 * key + 2 * coeff + 3 * inner) * block * element *
             static_cast<std::size_t>(g.layers);
    // The FP32 residual stream the body runs on: per layer a 4-byte normed plane, its 2-byte
    // widened copy (twice), and the two 4-byte widened branch outputs; plus the residual, the final
    // norm, its widened copy and the widened embedding plane.
    bytes += (16 * hidden) * block * static_cast<std::size_t>(g.layers);
    bytes += 8 * hidden * block;
    bytes += hidden * block * element;
    bytes += 3 * block * sizeof(std::int32_t);
    // The head: the tower's logits, the gather, the node table, the lattice's planes and the two
    // edge projections.
    bytes +=
        static_cast<std::size_t>(g.vocab) * static_cast<std::size_t>(g.block_size - 1) * element;
    bytes += (hidden + 2 * width) * count * element;
    bytes += 3 * width * count * element;
    bytes += 3 * width * count * element * static_cast<std::size_t>(g.head_layers);
    bytes += 2 * static_cast<std::size_t>(g.head_mlp) * count * element *
             static_cast<std::size_t>(g.head_layers);
    bytes += 2 * width * count * element * static_cast<std::size_t>(g.head_layers);
    bytes += 6 * width * count * element;
    bytes += 4 * width * count * element;
    bytes += width * static_cast<std::size_t>(g.block_size - 1) * element;
    bytes += static_cast<std::size_t>(g.feature_dim) * count * element;
    bytes += 5 * width * element;
    // The stages a diff reads back, which are FP32 and uploaded on demand.
    bytes +=
        4 * (static_cast<std::size_t>(g.feature_dim) * count * 2 +
             4 * static_cast<std::size_t>(g.top_k) * static_cast<std::size_t>(g.block_size - 1) +
             2 * static_cast<std::size_t>(g.head_heads) * count * count + 4 * width * count +
             3 * static_cast<std::size_t>(g.top_k) * g.top_k *
                 static_cast<std::size_t>(g.block_size - 1));
    // The Ops' workspaces.
    bytes += ops::sliding_causal_attention_workspace_bytes(
        ops::AttentionHeadGeometry{g.head_dim, g.heads, g.kv_heads}, rows, capacity + rows, 1);
    bytes += 1U << 20U;
    return bytes;
}

} // namespace

DraftMemory DraftContext::plan(const TextConfig& config, const models::draft::Spec& spec,
                               std::int32_t lanes, std::int32_t block_width,
                               std::int32_t context_window) {
    if (lanes <= 0) {
        throw std::invalid_argument("nemotron_h draft: the state serves a positive lane count");
    }
    const Geometry g            = derive_geometry(config, spec, block_width, context_window);
    const std::int32_t capacity = g.window + g.block_size;
    const std::int32_t rows     = g.block_size;
    const std::int32_t nodes    = (g.block_size - 1) * g.top_k;
    DraftMemory memory;
    memory.lanes      = lanes;
    memory.ring_bytes = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(g.layers) * 2 *
                            static_cast<std::size_t>(g.key_width()) *
                            static_cast<std::size_t>(capacity) * sizeof(std::uint16_t) +
                        static_cast<std::size_t>(lanes) * static_cast<std::size_t>(capacity) *
                            sizeof(std::int32_t) +
                        static_cast<std::size_t>(g.head_mlp) * static_cast<std::size_t>(nodes) *
                            sizeof(std::uint16_t);
    memory.scratch_bytes = scratch_bytes(g, capacity, rows, nodes);
    memory.anchor_bytes  = static_cast<std::size_t>(g.hidden) * sizeof(std::uint16_t);
    memory.proposal_bytes =
        proposal_logits_bytes(g) +
        2 * proposal_columns_bytes(g, sizeof(std::int32_t)) +
        proposal_columns_bytes(g, sizeof(ops::SamplingConfig)) + proposal_workspace_bytes(g);
    return memory;
}

DraftMemory DraftContext::plan(const Model& model, const models::draft::Spec& spec,
                               std::int32_t lanes, std::int32_t block_width,
                               std::int32_t context_window) {
    // The model's form also checks every bound parameter against the geometry, so it cannot go
    // through the config-only plan: that is the whole point of having both.
    (void)derive(model, spec, block_width, context_window);
    return plan(model.config(), spec, lanes, block_width, context_window);
}

DraftContext::Impl::Impl(const Model& model_source, const Geometry& geometry,
                         std::int32_t lane_count, const DraftMemory& memory)
    : model(&model_source), g(geometry), lanes(lane_count),
      capacity(geometry.window + geometry.block_size),
      max_nodes((geometry.block_size - 1) * geometry.top_k), ring(memory.ring_bytes),
      ring_positions(static_cast<std::size_t>(lane_count) * static_cast<std::size_t>(capacity) *
                     sizeof(std::int32_t)),
      ones_plane(static_cast<std::size_t>(geometry.head_mlp) *
                 static_cast<std::size_t>((geometry.block_size - 1) * geometry.top_k) *
                 sizeof(std::uint16_t)),
      arena(memory.scratch_bytes), body_arena(memory.scratch_bytes),
      anchor_plane(static_cast<std::size_t>(geometry.hidden) * sizeof(std::uint16_t)),
      proposal_logits(proposal_logits_bytes(geometry)),
      proposal_configs(proposal_columns_bytes(geometry, sizeof(ops::SamplingConfig))),
      proposal_positions(proposal_columns_bytes(geometry, sizeof(std::int32_t))),
      proposal_drawn(proposal_columns_bytes(geometry, sizeof(std::int32_t))),
      proposal_workspace_backing(proposal_workspace_bytes(geometry)),
      body_graphs(static_cast<std::size_t>(lane_count)),
      body_execs(static_cast<std::size_t>(lane_count)),
      appended_through(static_cast<std::size_t>(lane_count), -1),
      walk(static_cast<std::size_t>(lane_count)), candidates(static_cast<std::size_t>(lane_count)) {
    CUDA_CHECK(cudaMemset(ring.p, 0, ring.bytes));
    std::vector<std::int32_t> unwritten(static_cast<std::size_t>(capacity),
                                        kDraftUnwrittenPosition);
    for (std::int32_t lane = 0; lane < lane_count; ++lane) {
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(ring_positions.p) +
                                  static_cast<std::size_t>(lane) *
                                      static_cast<std::size_t>(capacity) * sizeof(std::int32_t),
                              unwritten.data(), unwritten.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    }
    std::vector<std::uint16_t> ones_values(
        static_cast<std::size_t>(geometry.head_mlp) *
            static_cast<std::size_t>((geometry.block_size - 1) * geometry.top_k),
        float_to_bf16(1.0F));
    CUDA_CHECK(cudaMemcpy(ones_plane.p, ones_values.data(),
                          ones_values.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    if (!g.lilicorr()) {
        // The plain proposal's operands are constant across rounds -- every column is a greedy
        // argmax with no counts array and no penalties -- so they are written once here rather than
        // re-uploaded per round.
        const std::size_t columns = static_cast<std::size_t>(g.block_size);
        std::vector<ops::SamplingConfig> configs(columns);
        CUDA_CHECK(cudaMemcpy(proposal_configs.p, configs.data(),
                              columns * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice));
        proposal_workspace.emplace(
            DeviceSpan{proposal_workspace_backing.p, proposal_workspace_backing.bytes});
        proposal_drawn_host.resize(columns);
        return;
    }
    const DraftHeadWeights& hd = model->weights().draft->head;
    const auto read            = [&](WeightId id, std::vector<float>& into) {
        const Tensor& source = model->bound(id).tensor;
        std::vector<std::uint16_t> packed(source.numel());
        CUDA_CHECK(cudaMemcpy(packed.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        into.resize(packed.size());
        for (std::size_t at = 0; at < packed.size(); ++at) { into[at] = bf16_to_float(packed[at]); }
    };
    read(hd.slot_embedding, slot_table);
    read(hd.rank_embedding, rank_table);
    read(hd.relative_slot_bias, relative_bias);
    read(hd.same_slot_bias, same_bias);
    read(hd.feature_norm_weight, feature_gain);
    read(hd.feature_norm_bias, feature_bias);
}

DraftContext::DraftContext(const Model& model, const models::draft::Spec& spec, std::int32_t lanes,
                           std::int32_t block_width, std::int32_t context_window)
    : spec_(spec), lanes_(lanes) {
    if (!model.weights().draft.has_value() || model.weights().draft->layers.empty()) {
        throw std::invalid_argument("nemotron_h draft: the model carries no draft to execute");
    }
    if (lanes <= 0) {
        throw std::invalid_argument("nemotron_h draft: the state serves a positive lane count");
    }
    const Geometry geometry = derive(model, spec, block_width, context_window);
    memory_                 = plan(model, spec, lanes, block_width, context_window);
    context_capacity_       = geometry.window + geometry.block_size;
    impl_                   = std::make_unique<Impl>(model, geometry, lanes, memory_);
}

DraftContext::~DraftContext() = default;

void DraftContext::reset(std::int32_t lane, DeviceExecutionView execution) {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h draft: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes_) + " lanes");
    }
    impl_->reset(lane, execution.stream);
}

std::vector<std::int32_t> DraftContext::propose(std::int32_t lane, const DraftRound& round,
                                                DeviceExecutionView execution,
                                                const DraftObserver* observer) {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h draft: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes_) + " lanes");
    }
    return impl_->propose(lane, round, execution, observer);
}

void DraftContext::condition(std::int32_t lane, const Tensor& taps, std::int32_t first_position,
                             DeviceExecutionView execution) {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h draft: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes_) + " lanes");
    }
    impl_->condition(lane, taps, first_position, execution);
}

const std::vector<std::int32_t>& DraftContext::walk(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h draft: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes_) + " lanes");
    }
    return impl_->walk[static_cast<std::size_t>(lane)];
}

const std::vector<std::int32_t>& DraftContext::candidates(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h draft: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes_) + " lanes");
    }
    return impl_->candidates[static_cast<std::size_t>(lane)];
}

} // namespace ninfer::models::nemotron_h
