// The composed Nemotron-H block against the canonical implementation's own dump.
//
// The dump (`tools/verify/nemotron_h_reference.py dump --tiny --out DIR --weights`) is the oracle
// and it is exact: it holds every parameter under the layers *and* every stage, so this walks the
// model the composition builds stage by stage rather than only at the logits. A mismatch localizes
// to a block and then to a mixer stage, which is the only way a composition this wide is
// debuggable.
//
// Five things about the walk are deliberate:
//
//  * The dump is FP32 (the reference's own dtype) and the composition's operands are the BF16 the
//    Ops consume, so the floor is the operand representation rather than zero. Every stage reports
//    its own relative L2 and maximum absolute difference and the loosest stage is named, rather
//    than one summary number hiding it.
//  * The dump's memory order is the composition's for every stage but one: a module's output is
//    `[batch, tokens, channels]` there and `[channels, tokens]` here, and both keep the channels
//    fastest, so the flattened sequences are the same numbers. The convolution is the exception --
//    its dump is the `nn.Conv1d` module's raw output with its bias over the left-padded sequence,
//    before the fused activation -- so the comparator crops it to the pass's own length and applies
//    SiLU, which is what makes the *fused* activation checked rather than assumed.
//  * The reference's top-k is `torch.topk(sorted=False)`, whose rank order it does not define, so
//    the router's two rank tensors are matched as a set per token -- with each weight paired to the
//    expert id it was emitted beside -- and an order difference is counted rather than failed. The
//    logits beside them are compared element by element.
//  * The carried attention head. This config's head is 16 wide, which
//    `ops::sliding_causal_attention` does not admit (it takes 32, 64, 128, 256), so
//    `forward_attention` carries it at 32 with the real dims contiguous and zeros elsewhere, and
//    passes the config's own `1/sqrt(16)` as the Op's `scale` (the Op takes it as an argument
//    instead of deriving it from the operand width). Both halves are load-bearing and the probes
//    below show the Op refusing the head's own width.
//  * No rotation, and none is composed: transformers 5.14.1's `modeling_nemotron_h.py` defines
//    `apply_rotary_pos_emb` but never calls it -- no `position_embeddings` argument reaches the
//    attention body -- and its `NemotronHConfig` has no `rope_theta` field, so the dump carries no
//    positional information at all. A rotation would show here as an attention mixer and a block
//    output that disagree while the projections still match, which is the only place it could show:
//    the projections are dumped before any rotation could be applied.
//
// The refusals the composition must make are exercised too: a block index the config does not have,
// a layer whose kind disagrees with its parameters, a layer whose parameters hold no mixer at all,
// a parameter bound at a shape the config does not derive, and an expert bank bound at the wrong
// latent width.
#include "core/arena.h"
#include "core/device.h"
#include "models/nemotron_h/forward.h"
#include "models/nemotron_h/weights.h"
#include "ninfer/ops/mamba2_scan.h"
#include "ninfer/ops/sliding_causal_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
namespace nemotron = ninfer::models::nemotron_h;

constexpr std::int32_t kTokens = 130;
// The operand representation's own floor, not a performance margin: the dump is FP32 (the
// reference's own dtype) and every operand this composition hands an Op is BF16, so a stage that
// agrees to a few parts in a thousand is exact to the representation it was given. The measured
// floor of this walk is 0.0016 to 0.008 relative L2 with a largest absolute difference of 2.8e-2,
// and block 1's routed tail reaches 0.05 because two of its tokens sit on a near-tie selection (see
// the router reporting at the end). The criterion keeps headroom for those two and nothing else: a
// defect that changes an operand, an order or a layout lands one to two orders above it rather than
// inside it -- the depthwise convolution's weight order alone read 1.4 relative L2 while its values
// stayed shape-compatible.
constexpr double kStageRelL2  = 0.08;
constexpr double kStageMaxAbs = 0.05;

// ------------------------------------------------------------------------------------- the dump

struct DumpTensor {
    std::vector<float> values;
    std::vector<std::int32_t> shape;
};

using Dump = std::map<std::string, DumpTensor>;

std::vector<char> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("missing " + path); }
    return std::vector<char>((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
}

Dump read_dump(const std::string& directory) {
    std::ifstream manifest(directory + "/manifest.txt");
    if (!manifest) { throw std::runtime_error("no manifest.txt in " + directory); }
    Dump tensors;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty()) { continue; }
        const std::size_t name_end  = line.find('\t');
        const std::size_t shape_end = line.find('\t', name_end + 1);
        if (name_end == std::string::npos || shape_end == std::string::npos) {
            throw std::runtime_error("malformed manifest line: " + line);
        }
        DumpTensor tensor;
        std::istringstream extents(line.substr(name_end + 1, shape_end - name_end - 1));
        std::string extent;
        std::int64_t elements = 1;
        while (std::getline(extents, extent, 'x')) {
            const std::int32_t dimension = std::stoi(extent);
            tensor.shape.push_back(dimension);
            elements *= dimension;
        }
        const std::vector<char> bytes = read_bytes(directory + "/" + line.substr(shape_end + 1));
        if (static_cast<std::int64_t>(bytes.size()) != elements * sizeof(float)) {
            throw std::runtime_error("dump file does not match its manifest shape: " + line);
        }
        tensor.values.resize(bytes.size() / sizeof(float));
        std::memcpy(tensor.values.data(), bytes.data(), bytes.size());
        tensors.emplace(line.substr(0, name_end), std::move(tensor));
    }
    return tensors;
}

const DumpTensor& need(const Dump& dump, const std::string& name) {
    const auto found = dump.find(name);
    if (found == dump.end()) {
        throw std::runtime_error("the dump has no " + name +
                                 ": regenerate it with tools/verify/nemotron_h_reference.py");
    }
    return found->second;
}

std::vector<std::int32_t> read_tokens(const std::string& directory) {
    const std::vector<char> bytes = read_bytes(directory + "/tokens.i32");
    std::vector<std::int32_t> tokens(bytes.size() / sizeof(std::int32_t));
    std::memcpy(tokens.data(), bytes.data(), bytes.size());
    return tokens;
}

// ----------------------------------------------------------------------------------- the config

// The scaled-down configuration the harness builds, whose every derived extent the dump's own
// shapes then confirm.
nemotron::TextConfig tiny_config() {
    nemotron::TextConfig config;
    config.hidden_size             = 64;
    config.vocab_size              = 256;
    config.max_position_embeddings = 4096;
    config.tie_word_embeddings     = false;
    config.rms_norm_eps            = 1.0e-5F;
    config.block_kinds             = {nemotron::BlockKind::Mamba, nemotron::BlockKind::Moe,
                                      nemotron::BlockKind::Attention, nemotron::BlockKind::Moe};
    nemotron::MambaConfig mamba;
    mamba.num_heads     = 4;
    mamba.head_dim      = 16;
    mamba.state_size    = 8;
    mamba.n_groups      = 2;
    mamba.conv_kernel   = 4;
    mamba.chunk_size    = 128;
    mamba.time_step_min = 0.001F;
    mamba.conv_bias     = true;
    config.mamba        = mamba;
    nemotron::AttentionConfig attention;
    attention.num_attention_heads = 4;
    attention.num_key_value_heads = 2;
    attention.head_dim            = 16;
    // The reference's config declares no rotation, its attention applies none, and the composition
    // reads no positional frequency at all: the field stays here because it is the config's own,
    // and zero is what the checkpoint's own config carries (it defines no `rope_theta`).
    attention.rope_theta = 0.0F;
    config.attention     = attention;
    nemotron::RouterConfig router;
    router.scaling_factor  = 1.0F;
    router.normalize_top_k = true;
    router.correction_bias = true;
    router.groups          = 1;
    router.groups_used     = 1;
    config.router          = router;
    nemotron::MoeBankConfig bank;
    bank.num_experts              = 8;
    bank.experts_per_token        = 2;
    bank.intermediate_size        = 32;
    bank.shared_intermediate_size = 32;
    bank.latent_size              = 16;
    config.moe_banks              = {std::nullopt, bank, std::nullopt, bank};
    return config;
}

void require_dump_shape(const Dump& dump, const std::string& name,
                        std::initializer_list<std::int32_t> shape) {
    const DumpTensor& tensor = need(dump, name);
    if (tensor.shape != std::vector<std::int32_t>(shape)) {
        std::ostringstream got;
        for (const std::int32_t extent : tensor.shape) { got << extent << 'x'; }
        throw std::runtime_error("the dump's " + name + " is " + got.str() +
                                 " rather than the tiny config's shape");
    }
}

void check_config_against_dump(const nemotron::TextConfig& config, const Dump& dump) {
    const auto hidden = static_cast<std::int32_t>(config.hidden_size);
    const auto vocab  = static_cast<std::int32_t>(config.vocab_size);
    require_dump_shape(dump, "model.embeddings.weight", {vocab, hidden});
    require_dump_shape(dump, "model.norm_f.weight", {hidden});
    require_dump_shape(dump, "lm_head.weight", {vocab, hidden});
    require_dump_shape(dump, "model.layers.0.norm.weight", {hidden});
    require_dump_shape(dump, "model.layers.0.mixer.in_proj.weight",
                       {static_cast<std::int32_t>(config.mamba->in_projection_size()), hidden});
    require_dump_shape(dump, "model.layers.0.mixer.conv1d.weight",
                       {static_cast<std::int32_t>(config.mamba->conv_channels()), 1,
                        static_cast<std::int32_t>(config.mamba->conv_kernel)});
    require_dump_shape(dump, "model.layers.0.mixer.norm.weight",
                       {static_cast<std::int32_t>(config.mamba->inner_size())});
    require_dump_shape(dump, "model.layers.2.mixer.q_proj.weight",
                       {static_cast<std::int32_t>(config.attention->query_width()), hidden});
    require_dump_shape(dump, "model.layers.2.mixer.k_proj.weight",
                       {static_cast<std::int32_t>(config.attention->key_width()), hidden});
    for (const std::int32_t layer : {1, 3}) {
        const std::string prefix            = "model.layers." + std::to_string(layer) + ".";
        const nemotron::MoeBankConfig& bank = config.moe_bank(static_cast<std::uint32_t>(layer));
        require_dump_shape(dump, prefix + "mixer.gate.weight",
                           {static_cast<std::int32_t>(bank.num_experts), hidden});
        require_dump_shape(dump, prefix + "mixer.experts.up_proj",
                           {static_cast<std::int32_t>(bank.num_experts),
                            static_cast<std::int32_t>(bank.intermediate_size),
                            static_cast<std::int32_t>(bank.latent_size)});
    }
}

// ---------------------------------------------------------------------------------- the fixture

// What one uploaded parameter was bound as, kept until the model exists: `Model::bind` needs the
// resident views, and the model is constructed from the finished parameter list.
struct Pending {
    nemotron::WeightId id;
    std::string name;
    Tensor tensor;
    Weight weight;
};

struct Uploaded {
    std::vector<DeviceBuffer> rows;
    std::vector<std::unique_ptr<DeviceBuffer>> matrices;
    std::vector<Pending> pending;
};

Weight dense_weight(std::int32_t rows, std::int32_t columns, const void* data) {
    Weight weight{};
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    weight.qtype           = QType::BF16;
    weight.shape[0]        = rows;
    weight.shape[1]        = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.ndim            = 2;
    weight.qdata           = data;
    weight.n               = rows;
    weight.k               = columns;
    weight.layout          = QuantLayout::Contiguous;
    return weight;
}

void bind_dense(Uploaded& uploaded, nemotron::ModelWeights& weights, nemotron::WeightId id,
                const std::string& name, const Dump& dump, std::int32_t rows, std::int32_t columns,
                nemotron::WeightId& field) {
    const DumpTensor& source = need(dump, name);
    std::vector<std::uint16_t> bits(source.values.size());
    for (std::size_t index = 0; index < source.values.size(); ++index) {
        bits[index] = f32_to_bf16(source.values[index]);
    }
    uploaded.matrices.emplace_back(
        std::make_unique<DeviceBuffer>(bits.size() * sizeof(std::uint16_t)));
    DeviceBuffer& buffer = *uploaded.matrices.back();
    buffer.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    uploaded.pending.push_back(Pending{id, name, Tensor(buffer.p, DType::BF16, {rows, columns}),
                                       dense_weight(rows, columns, buffer.p)});
    field = id;
    (void)weights;
}

// A 1-D BF16 parameter: a norm gain or a bias.
nemotron::WeightId upload_row(Uploaded& uploaded, const Dump& dump, const std::string& name,
                              std::size_t id, std::int32_t extent) {
    const DumpTensor& source = need(dump, name);
    std::vector<std::uint16_t> bits(source.values.size());
    for (std::size_t index = 0; index < source.values.size(); ++index) {
        bits[index] = f32_to_bf16(source.values[index]);
    }
    uploaded.rows.emplace_back(bits.size() * sizeof(std::uint16_t));
    DeviceBuffer& buffer = uploaded.rows.back();
    buffer.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    uploaded.pending.push_back(
        Pending{nemotron::WeightId{id}, name, Tensor(buffer.p, DType::BF16, {extent}), Weight{}});
    return nemotron::WeightId{id};
}

// An FP32 per-head or per-expert parameter, which stays FP32 because that is the precision the
// reference's own state arithmetic and selection use.
nemotron::WeightId upload_floats(Uploaded& uploaded, const Dump& dump, const std::string& name,
                                 std::size_t id, std::int32_t extent) {
    const DumpTensor& source = need(dump, name);
    const std::size_t bytes  = source.values.size() * sizeof(float);
    uploaded.rows.emplace_back(bytes);
    DeviceBuffer& buffer = uploaded.rows.back();
    buffer.copy_from_host(source.values.data(), bytes);
    uploaded.pending.push_back(
        Pending{nemotron::WeightId{id}, name, Tensor(buffer.p, DType::FP32, {extent}), Weight{}});
    return nemotron::WeightId{id};
}

// The depthwise convolution's weight. The checkpoint stores it as [C, 1, K], whose flat order is
// `c * K + j`, and the Op's kernels read it tap-major -- the tap `j` of channel `c` at `j * C + c`
// -- so this is a transpose rather than a relabelling: binding the checkpoint's own order is a
// shape-valid tensor whose taps belong to other channels, i.e. a silently wrong model. The declared
// shape stays [C, K], which is the one the Op validates.
nemotron::WeightId upload_conv_weight(Uploaded& uploaded, const Dump& dump, const std::string& name,
                                      std::size_t id, std::int32_t channels, std::int32_t kernel) {
    const DumpTensor& source = need(dump, name);
    std::vector<std::uint16_t> bits(source.values.size());
    for (std::int32_t c = 0; c < channels; ++c) {
        for (std::int32_t j = 0; j < kernel; ++j) {
            bits[static_cast<std::size_t>(j) * channels + c] = f32_to_bf16(
                source.values[static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel) + j]);
        }
    }
    uploaded.rows.emplace_back(bits.size() * sizeof(std::uint16_t));
    DeviceBuffer& buffer = uploaded.rows.back();
    buffer.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    uploaded.pending.push_back(Pending{
        nemotron::WeightId{id}, name, Tensor(buffer.p, DType::BF16, {channels, kernel}), Weight{}});
    return nemotron::WeightId{id};
}

// What the refusal cases perturb about an otherwise valid construction.
enum class Perturbation {
    None,
    WrongBlockKind,
    MissingMixer,
    WrongParameterShape,
    WrongBankLatent
};

struct Built {
    Uploaded uploaded;
    std::unique_ptr<nemotron::Model> model;
};

constexpr std::size_t kTextBase    = 1;
constexpr std::size_t kLayerBase   = 16;
constexpr std::size_t kLayerStride = 24;

enum : std::size_t { kEmbedding = kTextBase, kFinalNorm, kOutputHead };

Built build(const nemotron::TextConfig& config, const Dump& dump, Perturbation perturbation) {
    Built built;
    Uploaded& uploaded = built.uploaded;
    nemotron::ModelWeights weights;
    weights.text.layers.resize(config.num_hidden_layers());
    nemotron::WeightId embedding;
    nemotron::WeightId final_norm;
    nemotron::WeightId output_head;
    bind_dense(uploaded, weights, nemotron::WeightId{kEmbedding}, "model.embeddings.weight", dump,
               static_cast<std::int32_t>(config.vocab_size),
               static_cast<std::int32_t>(config.hidden_size), embedding);
    final_norm = upload_row(uploaded, dump, "model.norm_f.weight", kFinalNorm,
                            static_cast<std::int32_t>(config.hidden_size));
    bind_dense(uploaded, weights, nemotron::WeightId{kOutputHead}, "lm_head.weight", dump,
               static_cast<std::int32_t>(config.vocab_size),
               static_cast<std::int32_t>(config.hidden_size), output_head);
    (void)final_norm;
    weights.text.token_embedding = embedding;
    weights.text.final_norm      = final_norm;
    weights.text.output_head     = output_head;

    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        const std::size_t base                = kLayerBase + layer * kLayerStride;
        const std::string prefix              = "model.layers." + std::to_string(layer) + ".";
        nemotron::LayerWeights& layer_weights = weights.text.layers[layer];
        layer_weights.kind       = config.block_kind(static_cast<std::uint32_t>(layer));
        layer_weights.block_norm = upload_row(uploaded, dump, prefix + "norm.weight", base,
                                              static_cast<std::int32_t>(config.hidden_size));
        if (layer_weights.kind == nemotron::BlockKind::Mamba) {
            nemotron::MambaWeights mamba;
            bind_dense(uploaded, weights, nemotron::WeightId{base + 1},
                       prefix + "mixer.in_proj.weight", dump,
                       static_cast<std::int32_t>(config.mamba->in_projection_size()),
                       static_cast<std::int32_t>(config.hidden_size), mamba.in_proj);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 2},
                       prefix + "mixer.out_proj.weight", dump,
                       static_cast<std::int32_t>(config.hidden_size),
                       static_cast<std::int32_t>(config.mamba->inner_size()), mamba.out_proj);
            mamba.conv =
                upload_conv_weight(uploaded, dump, prefix + "mixer.conv1d.weight", base + 3,
                                   static_cast<std::int32_t>(config.mamba->conv_channels()),
                                   static_cast<std::int32_t>(config.mamba->conv_kernel));
            mamba.conv_bias = upload_row(uploaded, dump, prefix + "mixer.conv1d.bias", base + 4,
                                         static_cast<std::int32_t>(config.mamba->conv_channels()));
            mamba.a_log     = upload_floats(uploaded, dump, prefix + "mixer.A_log", base + 5,
                                            static_cast<std::int32_t>(config.mamba->num_heads));
            mamba.d_skip    = upload_floats(uploaded, dump, prefix + "mixer.D", base + 6,
                                            static_cast<std::int32_t>(config.mamba->num_heads));
            mamba.dt_bias   = upload_floats(uploaded, dump, prefix + "mixer.dt_bias", base + 7,
                                            static_cast<std::int32_t>(config.mamba->num_heads));
            mamba.norm      = upload_row(uploaded, dump, prefix + "mixer.norm.weight", base + 8,
                                         static_cast<std::int32_t>(config.mamba->inner_size()));
            layer_weights.mamba = mamba;
        } else if (layer_weights.kind == nemotron::BlockKind::Attention) {
            nemotron::AttentionWeights attention;
            bind_dense(uploaded, weights, nemotron::WeightId{base + 1},
                       prefix + "mixer.q_proj.weight", dump,
                       static_cast<std::int32_t>(config.attention->query_width()),
                       static_cast<std::int32_t>(config.hidden_size), attention.query);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 2},
                       prefix + "mixer.k_proj.weight", dump,
                       static_cast<std::int32_t>(config.attention->key_width()),
                       static_cast<std::int32_t>(config.hidden_size), attention.key);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 3},
                       prefix + "mixer.v_proj.weight", dump,
                       static_cast<std::int32_t>(config.attention->key_width()),
                       static_cast<std::int32_t>(config.hidden_size), attention.value);
            bind_dense(
                uploaded, weights, nemotron::WeightId{base + 4}, prefix + "mixer.o_proj.weight",
                dump, static_cast<std::int32_t>(config.hidden_size),
                static_cast<std::int32_t>(config.attention->query_width()), attention.output);
            layer_weights.attention = attention;
        } else {
            const nemotron::MoeBankConfig& bank =
                config.moe_bank(static_cast<std::uint32_t>(layer));
            nemotron::MoeWeights moe;
            bind_dense(uploaded, weights, nemotron::WeightId{base + 1},
                       prefix + "mixer.gate.weight", dump,
                       static_cast<std::int32_t>(bank.num_experts),
                       static_cast<std::int32_t>(config.hidden_size), moe.router);
            moe.correction_bias =
                upload_floats(uploaded, dump, prefix + "mixer.gate.e_score_correction_bias",
                              base + 2, static_cast<std::int32_t>(bank.num_experts));
            bind_dense(uploaded, weights, nemotron::WeightId{base + 3},
                       prefix + "mixer.fc1_latent_proj.weight", dump,
                       static_cast<std::int32_t>(bank.latent_size),
                       static_cast<std::int32_t>(config.hidden_size), moe.fc1_latent);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 4},
                       prefix + "mixer.fc2_latent_proj.weight", dump,
                       static_cast<std::int32_t>(config.hidden_size),
                       static_cast<std::int32_t>(bank.latent_size), moe.fc2_latent);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 5},
                       prefix + "mixer.shared_experts.up_proj.weight", dump,
                       static_cast<std::int32_t>(bank.shared_intermediate_size),
                       static_cast<std::int32_t>(config.hidden_size), moe.shared_up);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 6},
                       prefix + "mixer.shared_experts.down_proj.weight", dump,
                       static_cast<std::int32_t>(config.hidden_size),
                       static_cast<std::int32_t>(bank.shared_intermediate_size), moe.shared_down);
            // The bank is stored expert-major as [E, rows, columns] and the Op reads that order
            // flattened, so the extent is E * rows. A perturbed bank keeps the same bytes and lies
            // about its columns, which is exactly the mistake the composition has to refuse: the
            // hidden width is what a bank looks like when someone forgets the latent projection.
            const std::int32_t latent = static_cast<std::int32_t>(bank.latent_size);
            const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
            const std::int32_t declared =
                perturbation == Perturbation::WrongBankLatent ? hidden : latent;
            bind_dense(uploaded, weights, nemotron::WeightId{base + 7},
                       prefix + "mixer.experts.up_proj", dump,
                       static_cast<std::int32_t>(bank.num_experts) *
                           static_cast<std::int32_t>(bank.intermediate_size),
                       declared, moe.experts_up);
            bind_dense(uploaded, weights, nemotron::WeightId{base + 8},
                       prefix + "mixer.experts.down_proj", dump,
                       static_cast<std::int32_t>(bank.num_experts) * latent,
                       static_cast<std::int32_t>(bank.intermediate_size), moe.experts_down);
            layer_weights.moe = moe;
        }
        if (perturbation == Perturbation::WrongBlockKind && layer == 0) {
            layer_weights.kind = nemotron::BlockKind::Attention;
        }
        // The parameters a Mamba block needs are bound, but the layer holds none of them: a config
        // whose kind and whose bound set agree about nothing is not a smaller model.
        if (perturbation == Perturbation::MissingMixer && layer == 0) {
            layer_weights.mamba.reset();
        }
    }
    if (perturbation == Perturbation::WrongParameterShape) {
        // The same bytes, declared one row short: the composition must refuse the width the config
        // derives rather than let a kernel read the row it claims.
        for (Pending& pending : uploaded.pending) {
            if (pending.name == "model.layers.0.mixer.in_proj.weight") {
                pending.tensor.ne[0] = pending.tensor.ne[0] - 4;
            }
        }
    }
    built.model = std::make_unique<nemotron::Model>(config, weights);
    for (const Pending& pending : uploaded.pending) {
        built.model->bind(pending.id, pending.name, pending.tensor, pending.weight);
    }
    return built;
}

// ------------------------------------------------------------------------------- the comparison

struct StageStat {
    std::string name;
    double relative_l2 = 0.0;
    double max_abs     = 0.0;
};

double silu(double value) { return value / (1.0 + std::exp(-value)); }

class Comparator : public nemotron::BlockObserver {
public:
    Comparator(const Dump& dump, const nemotron::TextConfig& config)
        : dump_(dump), config_(config) {}

    void on_stage(std::size_t layer, nemotron::BlockStage stage,
                  const Tensor& tensor) const override {
        const std::string name = dump_name(layer, stage);
        if (name.empty()) {
            not_dumped_.push_back(stage_text(stage) + " (block " + std::to_string(layer) + ")");
            return;
        }
        const DumpTensor& dumped         = need(dump_, name);
        const std::vector<double> actual = host_values(tensor);
        if (stage == nemotron::BlockStage::MoeRouterLogits) {
            // The workspace's own perturbation, per token: the largest logit difference between the
            // composition's BF16 operands and the dump's FP32 ones, which the sigmoid carries
            // one-to-one. It is what a selection difference has to be inside to be this
            // composition's precision rather than its routing, and the ids callback beside this one
            // reads it.
            const std::vector<double> want = dumped_values(stage, dumped, actual.size(), name);
            record(name, want, actual);
            const std::int32_t tokens  = dumped.shape[0];
            const std::int32_t experts = dumped.shape[1];
            std::vector<double> perturbation(static_cast<std::size_t>(tokens), 0.0);
            for (std::int32_t token = 0; token < tokens; ++token) {
                for (std::int32_t expert = 0; expert < experts; ++expert) {
                    const std::size_t at = static_cast<std::size_t>(expert) * tokens + token;
                    perturbation[static_cast<std::size_t>(token)] =
                        std::max(perturbation[static_cast<std::size_t>(token)],
                                 std::abs(actual[at] - want[at]));
                }
            }
            logits_error_[layer] = std::move(perturbation);
            return;
        }
        if (stage == nemotron::BlockStage::MoeRouterIds) {
            compare_rank_sets(layer, dumped, actual, tensor);
            return;
        }
        if (stage == nemotron::BlockStage::MoeRouterWeights) {
            compare_rank_weights(layer, dumped, actual, tensor);
            return;
        }
        record(name, dumped_values(stage, dumped, actual.size(), name), actual);
    }

    [[nodiscard]] const std::vector<StageStat>& stats() const { return stats_; }

    [[nodiscard]] const std::vector<std::string>& not_dumped() const { return not_dumped_; }

    // How many tokens of the router's rank stages the composition emitted in another order than the
    // reference. The reference's `sorted=False` top-k defines no order, so this is reported rather
    // than failed; a set difference inside the token's own logit perturbation is reported too.
    [[nodiscard]] int reordered() const { return reordered_; }

    [[nodiscard]] int set_mismatch() const { return set_mismatch_; }

    [[nodiscard]] int unexplained() const { return unexplained_; }

    // The tightest gap/perturbation ratio among the tokens whose expert set moved: how close the
    // reference's own top-k boundary was to the movement the composition's operands introduced.
    [[nodiscard]] double tightest_ratio() const { return tightest_ratio_; }

    [[nodiscard]] int failures() const { return failures_; }

private:
    static std::string stage_text(nemotron::BlockStage stage) {
        switch (stage) {
        case nemotron::BlockStage::BlockNorm:
            return "BlockNorm";
        case nemotron::BlockStage::MambaInProj:
            return "MambaInProj";
        case nemotron::BlockStage::MambaConv:
            return "MambaConv";
        case nemotron::BlockStage::MambaNorm:
            return "MambaNorm";
        case nemotron::BlockStage::MambaOutProj:
            return "MambaOutProj";
        case nemotron::BlockStage::MambaMixer:
            return "MambaMixer";
        case nemotron::BlockStage::MoeRouterInput:
            return "MoeRouterInput";
        case nemotron::BlockStage::MoeRouterLogits:
            return "MoeRouterLogits";
        case nemotron::BlockStage::MoeRouterIds:
            return "MoeRouterIds";
        case nemotron::BlockStage::MoeRouterWeights:
            return "MoeRouterWeights";
        case nemotron::BlockStage::MoeLatentIn:
            return "MoeLatentIn";
        case nemotron::BlockStage::MoeExperts:
            return "MoeExperts";
        case nemotron::BlockStage::MoeLatentOut:
            return "MoeLatentOut";
        case nemotron::BlockStage::MoeShared:
            return "MoeShared";
        case nemotron::BlockStage::MoeMixer:
            return "MoeMixer";
        case nemotron::BlockStage::AttentionQuery:
            return "AttentionQuery";
        case nemotron::BlockStage::AttentionKey:
            return "AttentionKey";
        case nemotron::BlockStage::AttentionValue:
            return "AttentionValue";
        case nemotron::BlockStage::AttentionProjected:
            return "AttentionProjected";
        case nemotron::BlockStage::AttentionOutProj:
            return "AttentionOutProj";
        case nemotron::BlockStage::AttentionMixer:
            return "AttentionMixer";
        case nemotron::BlockStage::Block:
            return "Block";
        case nemotron::BlockStage::Embedding:
            return "Embedding";
        case nemotron::BlockStage::FinalNorm:
            return "FinalNorm";
        case nemotron::BlockStage::Logits:
            return "Logits";
        }
        return "?";
    }

    // The dumped stage a reported one corresponds to, or the empty string when the reference does
    // not expose it. Every other stage's memory order is the dump's own, because both are
    // feature-fastest with the tokens outside.
    [[nodiscard]] std::string dump_name(std::size_t layer, nemotron::BlockStage stage) const {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        switch (stage) {
        case nemotron::BlockStage::Embedding:
            return "model.embeddings";
        case nemotron::BlockStage::FinalNorm:
            return "model.norm_f";
        case nemotron::BlockStage::Logits:
            return "lm_head";
        // The block norm's own output: the reference dumps the router's input, which is that
        // tensor, only for a MoE layer. The other two blocks' inputs are covered transitively by
        // the projections that consume them.
        case nemotron::BlockStage::BlockNorm:
            return config_.block_kind(static_cast<std::uint32_t>(layer)) == nemotron::BlockKind::Moe
                       ? prefix + "mixer.gate.input"
                       : std::string{};
        case nemotron::BlockStage::MambaInProj:
            return prefix + "mixer.in_proj";
        case nemotron::BlockStage::MambaConv:
            return prefix + "mixer.conv1d";
        case nemotron::BlockStage::MambaNorm:
            return prefix + "mixer.norm";
        case nemotron::BlockStage::MambaOutProj:
            return prefix + "mixer.out_proj";
        case nemotron::BlockStage::MambaMixer:
            return prefix + "mixer";
        case nemotron::BlockStage::MoeRouterInput:
            return prefix + "mixer.gate.input";
        case nemotron::BlockStage::MoeRouterLogits:
            return prefix + "mixer.gate.0";
        case nemotron::BlockStage::MoeRouterIds:
            return prefix + "mixer.gate.2";
        case nemotron::BlockStage::MoeRouterWeights:
            return prefix + "mixer.gate.1";
        case nemotron::BlockStage::MoeLatentIn:
            return prefix + "mixer.fc1_latent_proj";
        case nemotron::BlockStage::MoeExperts:
            return prefix + "mixer.experts";
        case nemotron::BlockStage::MoeLatentOut:
            return prefix + "mixer.fc2_latent_proj";
        case nemotron::BlockStage::MoeShared:
            return prefix + "mixer.shared_experts";
        case nemotron::BlockStage::MoeMixer:
            return prefix + "mixer";
        case nemotron::BlockStage::AttentionQuery:
            return prefix + "mixer.q_proj";
        case nemotron::BlockStage::AttentionKey:
            return prefix + "mixer.k_proj";
        case nemotron::BlockStage::AttentionValue:
            return prefix + "mixer.v_proj";
        case nemotron::BlockStage::AttentionOutProj:
            return prefix + "mixer.o_proj";
        // The attention module returns `(attn_output, attn_weights)`, so the reference's dump holds
        // its first element as `mixer.0` beside the `o_proj` module's own output; the two are the
        // same tensor, which is why both stages compare against their own file.
        case nemotron::BlockStage::AttentionMixer:
            return prefix + "mixer.0";
        case nemotron::BlockStage::Block:
            return "model.layers." + std::to_string(layer);
        default:
            return {};
        }
    }

    // The dump's values in the composition's own flat order. The reference and the composition both
    // keep a token's channels contiguous, so the two orders flatten to the same sequence of numbers
    // for every stage but two: the convolution, which the reference dumps raw over a padded
    // sequence, and the router's logits workspace, whose order is the router Op's own.
    [[nodiscard]] std::vector<double> dumped_values(nemotron::BlockStage stage,
                                                    const DumpTensor& dumped, std::size_t count,
                                                    const std::string& name) const {
        if (stage == nemotron::BlockStage::MoeRouterLogits) {
            // The Op writes its workspace expert-major -- one expert's tokens contiguous -- while
            // the dump holds `[tokens, experts]` with the experts contiguous. Both are the same
            // matrix, so the expectation is rearranged rather than refused.
            if (dumped.shape.size() != 2 || dumped.shape[0] <= 0 || dumped.shape[1] <= 0) {
                throw std::runtime_error("the dump's " + name +
                                         " is not a [tokens, experts] stage");
            }
            const std::int32_t tokens  = dumped.shape[0];
            const std::int32_t experts = dumped.shape[1];
            if (static_cast<std::size_t>(tokens) * static_cast<std::size_t>(experts) != count) {
                throw std::runtime_error("the dump's " + name +
                                         " is not the same router stage the composition produced");
            }
            std::vector<double> want(count, 0.0);
            for (std::int32_t token = 0; token < tokens; ++token) {
                for (std::int32_t expert = 0; expert < experts; ++expert) {
                    want[static_cast<std::size_t>(expert) * tokens + token] =
                        dumped.values[static_cast<std::size_t>(token) * experts + expert];
                }
            }
            return want;
        }
        if (stage != nemotron::BlockStage::MambaConv) {
            if (dumped.values.size() != count) {
                throw std::runtime_error(
                    "the dump's " + name + " holds " + std::to_string(dumped.values.size()) +
                    " values where the composition's stage has " + std::to_string(count));
            }
            return std::vector<double>(dumped.values.begin(), dumped.values.end());
        }
        // The convolution: `nn.Conv1d(padding = kernel - 1)` followed by `[..., :L]`, so the dumped
        // module output is the raw convolution with its bias over the left-padded sequence, three
        // columns longer than the pass and without the fused activation. The crop is the *first* L
        // columns -- each output column's window ends at its own token, which is the causal one --
        // and SiLU is applied here, so the fusion is checked rather than assumed. The dumped shape
        // is `[batch, channels, padded]`; the composition's stage is `[channels, tokens]`.
        if (dumped.shape.size() != 3 || dumped.shape[0] != 1 || dumped.shape[1] <= 0) {
            throw std::runtime_error("the dump's " + name +
                                     " is not a [1, channels, padded] stage");
        }
        const std::int32_t channels = dumped.shape[1];
        const std::int32_t padded   = dumped.shape[2];
        const std::int32_t tokens   = static_cast<std::int32_t>(count) / channels;
        if (static_cast<std::size_t>(channels) * static_cast<std::size_t>(tokens) != count ||
            padded < tokens) {
            throw std::runtime_error("the dump's " + name +
                                     " is not a padded convolution over the same pass");
        }
        std::vector<double> want(count, 0.0);
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t channel = 0; channel < channels; ++channel) {
                want[static_cast<std::size_t>(token) * channels + channel] =
                    silu(dumped.values[static_cast<std::size_t>(channel) * padded + token]);
            }
        }
        return want;
    }

    void record(const std::string& name, const std::vector<double>& want,
                const std::vector<double>& actual) const {
        if (want.size() != actual.size()) {
            throw std::runtime_error("the dump's " + name +
                                     " and the composition's stage differ in "
                                     "element count");
        }
        double worst         = 0.0;
        double energy_actual = 0.0;
        double energy_error  = 0.0;
        for (std::size_t index = 0; index < want.size(); ++index) {
            const double difference = std::abs(actual[index] - want[index]);
            worst                   = std::max(worst, difference);
            energy_actual += actual[index] * actual[index];
            energy_error += difference * difference;
        }
        const double relative_l2 = std::sqrt(energy_error / std::max(energy_actual, 1e-30));
        stats_.push_back(StageStat{name, relative_l2, worst});
        if (relative_l2 > kStageRelL2 || worst > kStageMaxAbs) {
            ++failures_;
            std::cerr << name << ": rel_l2=" << relative_l2 << " max_abs=" << worst << '\n';
        }
    }

    [[nodiscard]] static std::int32_t round_to_id(double value) {
        return static_cast<std::int32_t>(std::lround(value));
    }

    // The router's expert ids, as a set per token. A token whose set differs is a failure unless
    // the reference's own boundary gap there -- how far apart its K-th and (K+1)-th corrected
    // scores are
    // -- is inside the perturbation this composition's operands introduced at that token, which is
    // the only way rounding the logits' operands can move a selection. A set difference outside it
    // is this composition's routing being wrong rather than its precision, and the router Op's own
    // test draws the line in the same place. The composition's `ids` is in the Op's layout, [ranks,
    // tokens] with the token innermost, and the dump's is [tokens, ranks].
    void compare_rank_sets(std::size_t layer, const DumpTensor& dumped,
                           const std::vector<double>& actual, const Tensor& tensor) const {
        const std::int32_t ranks  = tensor.ne[0];
        const std::int32_t tokens = tensor.ne[1];
        std::vector<std::int32_t> ids(actual.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            ids[index] = round_to_id(actual[index]);
        }
        ids_[layer] = ids;

        const auto perturbation = logits_error_.find(layer);
        if (perturbation == logits_error_.end()) {
            throw std::runtime_error("the router's ids arrived before its logits");
        }
        const std::string prefix   = "model.layers." + std::to_string(layer) + ".";
        const DumpTensor& logits   = need(dump_, prefix + "mixer.gate.0");
        const DumpTensor& bias     = need(dump_, prefix + "mixer.gate.e_score_correction_bias");
        const std::int32_t experts = logits.shape[1];

        int mismatched  = 0;
        int unexplained = 0;
        double worst_id = 0.0;
        std::vector<double> corrected(static_cast<std::size_t>(experts));
        for (std::int32_t token = 0; token < tokens; ++token) {
            std::vector<std::int32_t> want(static_cast<std::size_t>(ranks));
            std::vector<std::int32_t> got(static_cast<std::size_t>(ranks));
            for (std::int32_t rank = 0; rank < ranks; ++rank) {
                want[static_cast<std::size_t>(rank)] =
                    round_to_id(dumped.values[static_cast<std::size_t>(token) * ranks + rank]);
                got[static_cast<std::size_t>(rank)] =
                    ids[static_cast<std::size_t>(rank) * tokens + token];
            }
            if (want != got) { ++reordered_; }
            for (std::int32_t rank = 0; rank < ranks; ++rank) {
                worst_id = std::max(
                    worst_id, std::abs(static_cast<double>(want[static_cast<std::size_t>(rank)]) -
                                       static_cast<double>(got[static_cast<std::size_t>(rank)])));
            }
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());
            if (want == got) { continue; }
            ++mismatched;
            for (std::int32_t expert = 0; expert < experts; ++expert) {
                corrected[static_cast<std::size_t>(expert)] =
                    1.0 /
                        (1.0 +
                         std::exp(
                             -logits.values[static_cast<std::size_t>(token) * experts + expert])) +
                    bias.values[static_cast<std::size_t>(expert)];
            }
            std::sort(corrected.begin(), corrected.end());
            const double gap   = corrected[static_cast<std::size_t>(experts - ranks)] -
                                 corrected[static_cast<std::size_t>(experts - ranks - 1)];
            const double limit = perturbation->second[static_cast<std::size_t>(token)];
            if (gap > limit) {
                ++unexplained;
            } else if (limit > 0.0) {
                const double ratio = gap / limit;
                tightest_ratio_ = tightest_ratio_ == 0.0 ? ratio : std::min(tightest_ratio_, ratio);
            }
        }
        set_mismatch_ += mismatched;
        unexplained_ += unexplained;
        const double relative = static_cast<double>(mismatched) / std::max(tokens, 1);
        stats_.push_back(
            StageStat{dump_name(layer, nemotron::BlockStage::MoeRouterIds), relative, worst_id});
        if (unexplained != 0) {
            ++failures_;
            std::cerr << "model.layers." << layer << ".mixer.gate.2: " << unexplained << "/"
                      << tokens << " tokens select a different expert set than the reference by "
                      << "more than this composition's own logit perturbation\n";
        }
    }

    // The router's weights, paired with the id each was emitted beside: the *set* of (id, weight)
    // pairs is what the reference defines, so both sides are sorted by id before the weights are
    // compared. The ids the composition emitted arrive in the callback beside this one, which is
    // why they are kept.
    void compare_rank_weights(std::size_t layer, const DumpTensor& dumped,
                              const std::vector<double>& actual, const Tensor& tensor) const {
        const std::int32_t ranks  = tensor.ne[0];
        const std::int32_t tokens = tensor.ne[1];
        const auto found          = ids_.find(layer);
        if (found == ids_.end()) {
            throw std::runtime_error("the router's weights arrived before its ids");
        }
        const std::vector<std::int32_t>& ids = found->second;
        const DumpTensor& dumped_ids =
            need(dump_, dump_name(layer, nemotron::BlockStage::MoeRouterIds));

        std::vector<double> want(actual.size(), 0.0);
        std::vector<double> got(actual.size(), 0.0);
        for (std::int32_t token = 0; token < tokens; ++token) {
            std::vector<std::pair<std::int32_t, double>> wanted(static_cast<std::size_t>(ranks));
            std::vector<std::pair<std::int32_t, double>> emitted(static_cast<std::size_t>(ranks));
            for (std::int32_t rank = 0; rank < ranks; ++rank) {
                const std::size_t at        = static_cast<std::size_t>(rank) * tokens + token;
                const std::size_t dumped_at = static_cast<std::size_t>(token) * ranks + rank;
                wanted[static_cast<std::size_t>(rank)] = {round_to_id(dumped_ids.values[dumped_at]),
                                                          dumped.values[dumped_at]};
                emitted[static_cast<std::size_t>(rank)] = {ids[at], actual[at]};
            }
            std::sort(wanted.begin(), wanted.end());
            std::sort(emitted.begin(), emitted.end());
            for (std::int32_t rank = 0; rank < ranks; ++rank) {
                const std::size_t at = static_cast<std::size_t>(token) * ranks + rank;
                want[at]             = wanted[static_cast<std::size_t>(rank)].second;
                got[at]              = emitted[static_cast<std::size_t>(rank)].second;
            }
        }
        record(dump_name(layer, nemotron::BlockStage::MoeRouterWeights), want, got);
    }

    // The stages are read back while the caller's own stream is still running them, so the read has
    // to be ordered after the work already enqueued: this is the one place the test may not rely on
    // the default stream's implicit ordering.
    [[nodiscard]] static std::vector<double> host_values(const Tensor& tensor) {
        cuda_synchronize();
        const std::int64_t count = tensor.numel();
        std::vector<double> values(static_cast<std::size_t>(count));
        if (tensor.dtype == DType::BF16) {
            const auto bits = from_device<std::uint16_t>(tensor.data, count);
            for (std::int64_t index = 0; index < count; ++index) {
                values[static_cast<std::size_t>(index)] =
                    bf16_to_f32(bits[static_cast<std::size_t>(index)]);
            }
        } else if (tensor.dtype == DType::FP32) {
            const auto host = from_device<float>(tensor.data, count);
            for (std::int64_t index = 0; index < count; ++index) {
                values[static_cast<std::size_t>(index)] = host[static_cast<std::size_t>(index)];
            }
        } else {
            const auto host = from_device<std::int32_t>(tensor.data, count);
            for (std::int64_t index = 0; index < count; ++index) {
                values[static_cast<std::size_t>(index)] =
                    static_cast<double>(host[static_cast<std::size_t>(index)]);
            }
        }
        return values;
    }

    const Dump& dump_;
    const nemotron::TextConfig& config_;
    mutable std::vector<StageStat> stats_;
    mutable std::vector<std::string> not_dumped_;
    mutable std::map<std::size_t, std::vector<std::int32_t>> ids_;
    mutable std::map<std::size_t, std::vector<double>> logits_error_;
    mutable int failures_          = 0;
    mutable int reordered_         = 0;
    mutable int set_mismatch_      = 0;
    mutable int unexplained_       = 0;
    mutable double tightest_ratio_ = 0.0;
};

// ------------------------------------------------------------------------- the op domain probes

// The two facts the padding rests on, asserted rather than reasoned about: `ops::mamba2_scan`'s key
// extent is a constant of its kernel, so this config's 8-wide state has to be zero-extended to it;
// and `ops::sliding_causal_attention` admits neither this config's 16-wide head nor a scale baked
// into the kernel, so the head is carried and the scale is passed.
int probe_op_domains(DeviceExecutionView execution) {
    int failures              = 0;
    const std::int32_t tokens = 4;
    DeviceBuffer x(16 * 4 * tokens * sizeof(std::uint16_t));
    DeviceBuffer b(128 * 2 * tokens * sizeof(std::uint16_t));
    DeviceBuffer c(128 * 2 * tokens * sizeof(std::uint16_t));
    DeviceBuffer dt(4 * tokens * sizeof(std::uint16_t));
    DeviceBuffer dt_bias(4 * sizeof(float));
    DeviceBuffer a_log(4 * sizeof(float));
    DeviceBuffer d_skip(4 * sizeof(float));
    DeviceBuffer state(128 * 16 * 4 * sizeof(float));
    DeviceBuffer scan_out(16 * 4 * tokens * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(x.p, 0, x.bytes));
    CUDA_CHECK(cudaMemset(b.p, 0, b.bytes));
    CUDA_CHECK(cudaMemset(c.p, 0, c.bytes));
    CUDA_CHECK(cudaMemset(dt.p, 0, dt.bytes));
    CUDA_CHECK(cudaMemset(dt_bias.p, 0, dt_bias.bytes));
    CUDA_CHECK(cudaMemset(a_log.p, 0, a_log.bytes));
    CUDA_CHECK(cudaMemset(d_skip.p, 0, d_skip.bytes));
    CUDA_CHECK(cudaMemset(state.p, 0, state.bytes));
    CUDA_CHECK(cudaMemset(scan_out.p, 0, scan_out.bytes));
    const auto scan = [&](std::int32_t key_extent) {
        Tensor x_t(x.p, DType::BF16, {16, 4, tokens});
        Tensor b_t(b.p, DType::BF16, {key_extent, 2, tokens});
        Tensor c_t(c.p, DType::BF16, {key_extent, 2, tokens});
        Tensor dt_t(dt.p, DType::BF16, {4, tokens});
        Tensor dt_bias_t(dt_bias.p, DType::FP32, {4});
        Tensor a_log_t(a_log.p, DType::FP32, {4});
        Tensor d_skip_t(d_skip.p, DType::FP32, {4});
        Tensor state_t(state.p, DType::FP32, {128, 16, 4});
        Tensor out_t(scan_out.p, DType::BF16, {16, 4, tokens});
        ops::mamba2_scan(x_t, b_t, c_t, dt_t, dt_bias_t, a_log_t, d_skip_t, 0.001F, state_t, out_t,
                         execution.stream);
    };
    try {
        scan(8);
        std::cerr << "probe: ops::mamba2_scan accepted this config's 8-wide state, so the "
                     "zero-extended key axis this composition uses would be unnecessary\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        scan(128);
    } catch (const std::exception& error) {
        std::cerr << "probe: ops::mamba2_scan refused the padded 128-wide state: " << error.what()
                  << '\n';
        ++failures;
    }

    DeviceBuffer q(16 * 4 * tokens * sizeof(std::uint16_t));
    DeviceBuffer kv(16 * 2 * tokens * sizeof(std::uint16_t));
    DeviceBuffer positions(tokens * sizeof(std::int32_t));
    DeviceBuffer attention_out(16 * 4 * tokens * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(q.p, 0, q.bytes));
    CUDA_CHECK(cudaMemset(kv.p, 0, kv.bytes));
    CUDA_CHECK(cudaMemset(positions.p, 0, positions.bytes));
    CUDA_CHECK(cudaMemset(attention_out.p, 0, attention_out.bytes));
    const auto attention = [&](std::int32_t head) {
        Tensor q_t(q.p, DType::BF16, {head, 4, tokens, 1});
        Tensor k_t(kv.p, DType::BF16, {head, 2, tokens, 1});
        Tensor v_t(kv.p, DType::BF16, {head, 2, tokens, 1});
        Tensor positions_t(positions.p, DType::I32, {tokens});
        Tensor out_t(attention_out.p, DType::BF16, {head, 4, tokens, 1});
        ops::sliding_causal_attention(q_t, k_t, v_t, positions_t, Tensor{}, positions_t, Tensor{},
                                      Tensor{}, Tensor{}, ops::AttentionHeadGeometry{head, 4, 2},
                                      static_cast<std::uint32_t>(tokens), 0.25F, out_t, Tensor{},
                                      execution);
    };
    try {
        attention(16);
        std::cerr << "probe: ops::sliding_causal_attention accepted a 16-wide head, so the carried "
                     "head this composition uses would be unnecessary\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        attention(32);
    } catch (const std::exception& error) {
        std::cerr << "probe: ops::sliding_causal_attention refused the carried 32-wide head: "
                  << error.what() << '\n';
        ++failures;
    }

    cuda_synchronize();
    return failures;
}

// The refusals, none of which needs the dump's stages to be right -- only its parameters.
int refusal_cases(const nemotron::TextConfig& config, const Dump& dump) {
    int failures         = 0;
    const int32_t tokens = 130;
    DeviceBuffer input(static_cast<std::size_t>(64) * tokens * sizeof(std::uint16_t));
    DeviceBuffer output(static_cast<std::size_t>(64) * tokens * sizeof(std::uint16_t));
    DeviceExecutionView execution{nullptr, 1};
    const auto expect_refusal = [&](const char* label, const nemotron::Model& model,
                                    std::size_t layer, const char* expected_text) {
        int result = 0;
        try {
            DeviceArena arena(nemotron::model_workspace_bytes(config, tokens));
            Tensor input_t(input.p, DType::BF16, {64, tokens});
            Tensor output_t(output.p, DType::BF16, {64, tokens});
            nemotron::forward_block(model, layer, input_t, arena, output_t, execution);
            std::cerr << label << ": the composition accepted the call\n";
            result = 1;
        } catch (const std::invalid_argument& error) {
            if (expected_text != nullptr &&
                std::string(error.what()).find(expected_text) == std::string::npos) {
                std::cerr << label << ": refused with another reason: " << error.what() << '\n';
                result = 1;
            }
        }
        return result;
    };
    {
        Built built = build(config, dump, Perturbation::None);
        failures += expect_refusal("block index beyond the config", *built.model, 4, "blocks");
    }
    {
        Built built = build(config, dump, Perturbation::MissingMixer);
        failures += expect_refusal("a block whose parameters hold no mixer", *built.model, 0,
                                   "mixer it is not");
    }
    {
        Built built = build(config, dump, Perturbation::WrongParameterShape);
        failures += expect_refusal("a parameter at another shape", *built.model, 0,
                                   "where the "
                                   "config says");
    }
    {
        Built built = build(config, dump, Perturbation::WrongBankLatent);
        failures += expect_refusal("an expert bank at the wrong latent width", *built.model, 1,
                                   "expert bank");
    }
    {
        Built built = build(config, dump, Perturbation::WrongBlockKind);
        failures += expect_refusal("a block kind the config does not have", *built.model, 0,
                                   "mixer it is not");
    }
    return failures;
}

} // namespace

int main() {
    const char* directory = std::getenv("NINFER_NEMOTRON_REF");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "nemotron_h forward: set NINFER_NEMOTRON_REF to a "
                     "tools/verify/nemotron_h_reference.py dump (with --weights) to compare the "
                     "composition against the reference\n";
        return 77;
    }
    if (cuda_unavailable()) {
        std::cout << "nemotron_h forward: no CUDA device, skipping\n";
        return 77;
    }
    try {
        const Dump dump                        = read_dump(directory);
        const std::vector<std::int32_t> tokens = read_tokens(directory);
        const nemotron::TextConfig config      = tiny_config();
        check_config_against_dump(config, dump);
        if (static_cast<std::int32_t>(tokens.size()) != kTokens) {
            throw std::runtime_error("the dump's token count is not the tiny config's");
        }

        const std::int32_t multiprocessors = [] {
            int count = 0;
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0);
            return count;
        }();
        DeviceContext context;
        const DeviceExecutionView execution{context.stream, multiprocessors};

        int failures = probe_op_domains(execution);
        failures += refusal_cases(config, dump);

        Built built = build(config, dump, Perturbation::None);
        // The token ids and the logits the pass writes.
        DeviceBuffer ids(tokens.size() * sizeof(std::int32_t));
        ids.copy_from_host(tokens.data(), ids.bytes);
        GuardedDeviceBuffer logits(static_cast<std::size_t>(config.vocab_size) * kTokens *
                                   sizeof(std::uint16_t));
        const std::size_t capacity = nemotron::model_workspace_bytes(config, kTokens);
        DeviceArena arena(capacity);
        Comparator comparator(dump, config);
        Tensor ids_t(ids.p, DType::I32, {kTokens});
        Tensor logits_t(logits.data(), DType::BF16,
                        {static_cast<std::int32_t>(config.vocab_size), kTokens});
        nemotron::forward_model(*built.model, ids_t, arena, logits_t, execution, &comparator);
        cuda_synchronize();
        failures += comparator.failures();
        if (arena.used() != 0) {
            std::cerr << "the pass left " << arena.used() << " arena bytes allocated\n";
            ++failures;
        }
        if (arena.peak_used() > capacity) {
            std::cerr << "the pass used more arena than model_workspace_bytes reported\n";
            ++failures;
        }

        std::printf(
            "nemotron_h forward: %zu dumped stages compared, %zu reported stages not in the "
            "dump\n",
            comparator.stats().size(), comparator.not_dumped().size());
        for (const std::string& name : comparator.not_dumped()) {
            std::printf("  not in the dump: %s\n", name.c_str());
        }
        // Coverage is asserted rather than described, so a stage that silently loses its dump -- or
        // a comparison that stops being reached -- is a failure instead of a shorter report. 37 is
        // the number of stages this config's four blocks plus the head produce, and 3 is the number
        // of them the reference does not expose: the block norm of the two non-MoE blocks, whose
        // norm is not a dumped module (the same Op is exercised by `model.norm_f`, and each block's
        // *input* is dumped) and the attention's output before its own projection.
        //
        // That leaves no dumped *stage* unaccounted for: the reference's own `model.layers.N.input`
        // entries are byte-identical to stages this walk compares (layer 0's to `model.embeddings`
        // and every other to the previous layer's output, checked with `cmp`), and
        // `model.layers.2.mixer.0` is byte-identical to `mixer.o_proj`'s output.
        constexpr std::size_t kComparedStages     = 37;
        constexpr std::size_t kStagesWithoutADump = 3;
        if (comparator.stats().size() != kComparedStages ||
            comparator.not_dumped().size() != kStagesWithoutADump) {
            std::cerr << "the walk compared " << comparator.stats().size() << " stages and left "
                      << comparator.not_dumped().size()
                      << " unreported, where this config's walk is " << kComparedStages << " and "
                      << kStagesWithoutADump << '\n';
            ++failures;
        }
        const StageStat* loosest = nullptr;
        for (const StageStat& stat : comparator.stats()) {
            std::printf("  %-32s rel_l2=%.6g max_abs=%.6g\n", stat.name.c_str(), stat.relative_l2,
                        stat.max_abs);
            if (loosest == nullptr || stat.relative_l2 > loosest->relative_l2) { loosest = &stat; }
        }
        if (loosest != nullptr) {
            std::printf("nemotron_h forward: the loosest stage is %s (rel_l2=%.6g, max_abs=%.6g)\n",
                        loosest->name.c_str(), loosest->relative_l2, loosest->max_abs);
        }
        // The reference's top-k order is `sorted=False` and therefore undefined, so a reordering is
        // reported; a token whose expert *set* moved is reported with the reference's own boundary
        // gap against the movement this composition's operands introduced, and only a move outside
        // that gap counts as a failure -- the ones inside it are counted and reported here.
        std::printf(
            "nemotron_h forward: router ranks: %d tokens reordered, %d with a moved expert set, "
            "%d outside the token's own logit perturbation (tightest gap/perturbation %.3f)\n",
            comparator.reordered(), comparator.set_mismatch(), comparator.unexplained(),
            comparator.tightest_ratio());
        if (failures != 0) {
            std::cerr << "nemotron_h forward: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "nemotron_h forward: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h forward: " << error.what() << '\n';
        return 1;
    }
}
