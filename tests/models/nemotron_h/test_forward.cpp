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
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "models/nemotron_h/attention_state.h"
#include "models/nemotron_h/forward.h"
#include "models/nemotron_h/program.h"
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
#include <span>
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

// The i32 vector one file holds, exactly as the reference wrote it.
std::vector<std::int32_t> read_tokens_file(const std::string& path) {
    const std::vector<char> bytes = read_bytes(path);
    std::vector<std::int32_t> tokens(bytes.size() / sizeof(std::int32_t));
    std::memcpy(tokens.data(), bytes.data(), bytes.size());
    return tokens;
}

std::vector<std::int32_t> read_tokens(const std::string& directory) {
    return read_tokens_file(directory + "/tokens.i32");
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

// One stage's name, as every report below writes it: the identity the composition reports a stage
// by rather than the reference's dotted path, so this harness carries no reference naming outside
// the `dump_name` mapping further down.
std::string stage_text(nemotron::BlockStage stage) {
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

// A tensor's values on the host, read back while the caller's own stream is still running them, so
// the read has to be ordered after the work already enqueued: this is the one place this harness
// may not rely on the default stream's implicit ordering.
std::vector<double> host_values(const Tensor& tensor) {
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

// ------------------------------------------------------------------------------- the decode oracle
//
// The walk above is a *zero-state* prefill and it is the oracle for decode: a stateful decode of a
// prefix is the same computation from the state the prefill left, so it has to report the same
// stages, at the same absolute positions, with the same logits. Four runs establish that, and each
// one exists for a failure the others cannot see:
//
//  * decode ≡ prefill, one token at a time (`K = 8`, and `K = 1`, one step being what
//    `Program` drives per generated token): the state is read and written per token, and a
//    comparison of the logits alone would let a wrong mixer stage be absorbed by a later
//    norm before it shows;
//  * two chunks versus one pass: a state that is *written* but never *read*, or read but never
//    written, agrees with a single-call test and fails only at a boundary -- so every stage of the
//    second chunk is compared *including its first tokens*, where a chunk that started from a zero
//    state differs, and the same tokens are then decoded one at a time from the state two chunks
//    left, which is the property `Program` relies on;
//  * a lane reused across the reset `Program::create` performs: the convolution Op always reads the
//    state it is handed, so a frontier that was not emptied makes the *next* sequence's first pass
//    silently wrong rather than merely different -- and a reset is only worth performing if it
//    reproduces the run exactly, which is why that comparison is bit for bit rather than within the
//    walk's criterion;
//  * the reference's own post-decode state, when the dump's second mode wrote one: a state that is
//    subtly wrong (an axis transposed, a key tail that is not exactly zero) is invisible in the
//    logits until it propagates.
//
// Every comparison is against the *same* stage's column at the same absolute position. Every stage
// this composition reports keeps its innermost extent -- the channels, the experts, the ranks --
// fastest with the token axis strided, so column `t` is `columns` contiguous values at
// `t * columns`, which is also what a one-token pass writes as its only column.

// The floors the state itself is compared at, from the decode design
// (local://nemotron-decode-design.md §3.2): the convolution window is a copy of the composition's
// own BF16 operands (the walk's measured floor is 0.0016 to 0.008, so 0.01 is the operand
// representation with headroom), and the recurrent plane is FP32 on both sides but *built* from
// BF16 operands here and FP32 ones there, so its difference is the representation (~1e-3) rather
// than FP32 drift. The decisive part of the comparison is not its upper bound but its lower one: a
// state image of the wrong width or dtype is far outside both.
constexpr double kStateConvRelL2      = 0.01;
constexpr double kStateRecurrentRelL2 = 1e-3;

// One pass's stage, read back to the host: `columns` is what one token owns and `values` holds
// `columns` per column, in the tensor's own memory order.
struct CapturedStage {
    std::int32_t columns = 0;
    std::vector<double> values;

    [[nodiscard]] std::int32_t tokens() const {
        return columns == 0
                   ? 0
                   : static_cast<std::int32_t>(values.size() / static_cast<std::size_t>(columns));
    }
};

using StageKey = std::pair<std::size_t, nemotron::BlockStage>;
using Capture  = std::map<StageKey, CapturedStage>;

std::string stage_label(const StageKey& key) {
    return stage_text(key.second) + " (block " + std::to_string(key.first) + ")";
}

// Every stage one pass writes, read back as it is reported: a stage borrows the pass's arena or
// workspace and is valid for the callback only, so the read has to happen inside it.
class CaptureObserver : public nemotron::BlockObserver {
public:
    explicit CaptureObserver(Capture& capture) : capture_(&capture) {}

    void on_stage(std::size_t layer, nemotron::BlockStage stage,
                  const Tensor& tensor) const override {
        CapturedStage captured;
        captured.columns = tensor.ne[0];
        captured.values  = host_values(tensor);
        if (captured.columns <= 0 ||
            captured.values.size() % static_cast<std::size_t>(captured.columns) != 0) {
            throw std::runtime_error("nemotron_h decode: block " + std::to_string(layer) + "'s " +
                                     stage_text(stage) +
                                     " stage has no per-token column: its innermost extent does "
                                     "not divide its element count");
        }
        if (!capture_->emplace(StageKey{layer, stage}, std::move(captured)).second) {
            throw std::runtime_error("nemotron_h decode: block " + std::to_string(layer) + "'s " +
                                     stage_text(stage) + " stage was reported twice in one pass");
        }
    }

private:
    Capture* capture_;
};

// One pass over a slice of the fixture's own tokens, with every stage it writes captured. The
// logits are one of those stages, so a comparison covers them with the rest rather than beside
// them.
Capture run_pass(const nemotron::Model& model, const Tensor& ids, Tensor& logits,
                 const nemotron::PassState* state, DeviceArena& arena,
                 DeviceExecutionView execution) {
    Capture capture;
    const CaptureObserver observer(capture);
    nemotron::forward_model(model, ids, arena, logits, execution, state,
                            nemotron::HeadColumns::Every, &observer);
    cuda_synchronize();
    return capture;
}

// The decode steps arrive one token at a time, so their captures are single-column passes; a
// comparison against a multi-token run wants those columns beside each other, which is what the
// same tokens as one pass would have written.
void append_capture(Capture& into, const Capture& one) {
    for (const auto& [key, stage] : one) {
        if (stage.tokens() != 1) {
            throw std::runtime_error("nemotron_h decode: " + stage_label(key) + " reported " +
                                     std::to_string(stage.tokens()) +
                                     " columns for a one-token pass");
        }
        auto found = into.find(key);
        if (found == into.end()) {
            found = into.emplace(key, CapturedStage{stage.columns, {}}).first;
        }
        if (found->second.columns != stage.columns) {
            throw std::runtime_error("nemotron_h decode: " + stage_label(key) +
                                     " changed its innermost extent between decode steps");
        }
        found->second.values.insert(found->second.values.end(), stage.values.begin(),
                                    stage.values.end());
    }
    if (into.size() != one.size()) {
        throw std::runtime_error("nemotron_h decode: a decode step did not report every stage the "
                                 "others did");
    }
}

// One stage's column against another's, in the same shape the walk above reports a whole stage in:
// relative L2 against the actual column's own energy, and the largest absolute difference.
struct ColumnStat {
    std::string name;
    double relative_l2 = 0.0;
    double max_abs     = 0.0;
};

ColumnStat compare_column(const StageKey& key, const CapturedStage& reference,
                          std::int32_t reference_token, const CapturedStage& actual,
                          std::int32_t actual_token) {
    if (reference.columns != actual.columns) {
        throw std::runtime_error("nemotron_h decode: " + stage_label(key) + " is " +
                                 std::to_string(reference.columns) +
                                 " values wide in the reference run and " +
                                 std::to_string(actual.columns) + " in the state-driven one");
    }
    if (reference_token < 0 || reference_token >= reference.tokens() || actual_token < 0 ||
        actual_token >= actual.tokens()) {
        throw std::runtime_error("nemotron_h decode: " + stage_label(key) +
                                 " was asked for a column it does not hold");
    }
    const std::size_t columns = static_cast<std::size_t>(reference.columns);
    const double* const want =
        reference.values.data() + static_cast<std::size_t>(reference_token) * columns;
    const double* const got =
        actual.values.data() + static_cast<std::size_t>(actual_token) * columns;
    double worst         = 0.0;
    double energy_actual = 0.0;
    double energy_error  = 0.0;
    for (std::size_t index = 0; index < columns; ++index) {
        const double difference = std::abs(got[index] - want[index]);
        worst                   = std::max(worst, difference);
        energy_actual += got[index] * got[index];
        energy_error += difference * difference;
    }
    return ColumnStat{stage_label(key), std::sqrt(energy_error / std::max(energy_actual, 1e-30)),
                      worst};
}

// What one comparison found. `loosest` is the stage with the largest relative L2 among those
// compared and `logits` is the head's own stage beside it: the logits are one stage among many, and
// naming them separately is what says whether a difference is the model's own arithmetic or the
// state's.
struct OracleVerdict {
    int failures            = 0;
    std::size_t comparisons = 0;
    ColumnStat loosest;
    ColumnStat logits;
};

// Every stage a state-driven run wrote, against the reference run's own column at the same absolute
// position, for `count` positions starting at `reference_first`. Fatal on a stage outside the
// walk's criterion, in the walk's own reporting style.
OracleVerdict compare_positions(const Capture& reference, std::int32_t reference_first,
                                const Capture& actual, std::int32_t count,
                                const std::string& label) {
    OracleVerdict verdict;
    if (actual.size() != reference.size()) {
        throw std::runtime_error(
            "nemotron_h decode: " + label + " reported " + std::to_string(actual.size()) +
            " stages where the reference run reported " + std::to_string(reference.size()));
    }
    for (const auto& [key, stage] : actual) {
        const auto found = reference.find(key);
        if (found == reference.end()) {
            throw std::runtime_error("nemotron_h decode: " + label + " reported " +
                                     stage_label(key) + ", which the reference run did not");
        }
        for (std::int32_t token = 0; token < count; ++token) {
            const ColumnStat stat =
                compare_column(key, found->second, reference_first + token, stage, token);
            ++verdict.comparisons;
            if (verdict.loosest.name.empty() || stat.relative_l2 > verdict.loosest.relative_l2) {
                verdict.loosest = stat;
            }
            if (key.second == nemotron::BlockStage::Logits) { verdict.logits = stat; }
            if (stat.relative_l2 > kStageRelL2 || stat.max_abs > kStageMaxAbs) {
                ++verdict.failures;
                std::cerr << label << ": " << stat.name << " at position "
                          << (reference_first + token) << ": rel_l2=" << stat.relative_l2
                          << " max_abs=" << stat.max_abs << '\n';
            }
        }
    }
    if (verdict.logits.name.empty()) {
        throw std::runtime_error("nemotron_h decode: " + label + " reported no logits stage");
    }
    return verdict;
}

// Two captures of the same computation, value for value. The reported doubles are the operands
// themselves (a BF16 or FP32 bit pattern maps to one double), so equality here is equality of the
// computation's own output rather than a tolerance on it.
int compare_exact(const Capture& first, const Capture& second, const std::string& label) {
    if (first.size() != second.size()) {
        throw std::runtime_error("nemotron_h decode: " + label + " reported " +
                                 std::to_string(second.size()) +
                                 " stages where the first run of "
                                 "the same passes reported " +
                                 std::to_string(first.size()));
    }
    int failures = 0;
    for (const auto& [key, stage] : first) {
        const auto found = second.find(key);
        if (found == second.end()) {
            throw std::runtime_error("nemotron_h decode: " + label + " did not report " +
                                     stage_label(key));
        }
        if (found->second.values.size() != stage.values.size()) {
            throw std::runtime_error("nemotron_h decode: " + stage_label(key) + " holds " +
                                     std::to_string(found->second.values.size()) +
                                     " values in one run and " +
                                     std::to_string(stage.values.size()) + " in the other");
        }
        for (std::size_t index = 0; index < stage.values.size(); ++index) {
            if (found->second.values[index] != stage.values[index]) {
                ++failures;
                std::cerr << label << ": " << stage_label(key) << " value " << index << " is "
                          << stage.values[index] << " before the reset and "
                          << found->second.values[index] << " after it\n";
                break;
            }
        }
    }
    return failures;
}

// The family's state over storage this walk owns: the Mamba pool (one slot), the attention state
// (one lane) and the per-block binding a pass takes, which is the same three pieces `Program`
// assembles. Nothing here is a second layout: the pool's extents are the config's own except the
// scan's key axis, which is a constant of that Op's kernel (program.h) and carries a narrower
// configured state by zero extension.
struct FamilyState {
    DeviceBuffer backing;
    LinearAttentionStatePoolLayout layout;
    std::unique_ptr<nemotron::AttentionState> attention;
    std::unique_ptr<LinearAttentionStatePool> pool;
    std::vector<nemotron::BlockState> blocks;

    // The empty state a lane starts a request from, which is `Program::create`'s own pair of calls:
    // the Mamba frontier zeroed -- the zero history and plane a sequence's first pass begins from
    // -- and the attention positions made unwritten, because an unwritten slot is invisible while
    // an *initialized* one is what makes it so. A lane whose step has not been advanced is refused
    // by the attention state, so this follows the pass it resets after.
    void reset(DeviceExecutionView execution) {
        pool->zero_slot(0, execution.stream);
        attention->reset(0, execution.stream);
    }

    // A pass appends every attention block's K/V before its query runs and does not advance the
    // frontier; the caller advances once between passes, so a step's own key and value participate
    // in its own output and the next pass starts where this one ended.
    void advance() { attention->advance(0); }

    [[nodiscard]] nemotron::PassState pass(std::int32_t first_position) {
        return nemotron::PassState{blocks, first_position};
    }
};

std::uint32_t count_kind(const nemotron::TextConfig& config, nemotron::BlockKind kind) {
    std::uint32_t count = 0;
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        if (config.block_kind(layer) == kind) { ++count; }
    }
    return count;
}

FamilyState make_family_state(const nemotron::TextConfig& config, std::int32_t capacity) {
    const nemotron::MambaConfig& mamba = *config.mamba;
    FamilyState state;
    LinearAttentionStatePoolSpec spec{};
    spec.layers         = count_kind(config, nemotron::BlockKind::Mamba);
    spec.conv_channels  = static_cast<std::int32_t>(mamba.conv_channels());
    spec.conv_width     = static_cast<std::int32_t>(mamba.conv_kernel) - 1;
    spec.value_heads    = static_cast<std::int32_t>(mamba.num_heads);
    spec.value_head_dim = static_cast<std::int32_t>(mamba.head_dim);
    spec.key_head_dim   = nemotron::kMambaScanStateExtent;
    spec.slot_count     = 1;
    spec.conv_dtype     = DType::BF16;
    LayoutBuilder builder;
    state.layout  = plan_linear_attention_state_pool(builder, spec);
    state.backing = DeviceBuffer(builder.finish(256, "nemotron_h decode state"));
    state.pool    = std::make_unique<LinearAttentionStatePool>(
        DeviceSpan{state.backing.p, state.backing.bytes}, state.layout);
    state.attention = std::make_unique<nemotron::AttentionState>(
        *config.attention, count_kind(config, nemotron::BlockKind::Attention), capacity, 1);
    // One entry per block, as `Program::bind_state` binds it: the pool's layers are the Mamba
    // blocks in their own order and the attention state's are the attention blocks', because a
    // block index handed to either pool unchanged would bind another block's state on a config
    // whose block kinds interleave -- which this one's do.
    state.blocks.assign(config.num_hidden_layers(), nemotron::BlockState{});
    std::uint32_t mamba_layer     = 0;
    std::uint32_t attention_layer = 0;
    for (std::size_t layer = 0; layer < state.blocks.size(); ++layer) {
        switch (config.block_kind(static_cast<std::uint32_t>(layer))) {
        case nemotron::BlockKind::Mamba:
            state.blocks[layer].conv      = state.pool->conv_slot(mamba_layer, 0);
            state.blocks[layer].recurrent = state.pool->recurrent_slot(mamba_layer, 0);
            ++mamba_layer;
            break;
        case nemotron::BlockKind::Attention:
            state.blocks[layer].attention.state = state.attention.get();
            state.blocks[layer].attention.lane  = 0;
            state.blocks[layer].attention.layer = attention_layer++;
            // The Op's key-split workspace stays unset: its one-pass route computes the same result
            // (forward.h), and a walk that needs no second allocation to say so should not make
            // one.
            break;
        case nemotron::BlockKind::Moe:
            break;
        }
    }
    return state;
}

// The reference's own post-decode state, from the dump's second mode (`<dump>/decode/`,
// tools/verify/nemotron_h_reference.py). That mode states the sequence it is of --
// `prefix_tokens.i32` consumed from position zero and `decode_token.i32` stepped at the prefix's
// length -- which is the same call this walk makes through the state. The check is guarded because
// the mode is a second dump, and reported either way: a comparison that did not run is a fact about
// this run, not a silent pass. What it adds over the stage comparisons is the stored state itself,
// which no stage exposes -- an axis transposed, a width carried wrongly or a key tail that is not
// exactly zero is invisible in the logits until it propagates.
int check_decoded_state(const std::string& directory, const nemotron::Model& model,
                        DeviceArena& arena, DeviceExecutionView execution) {
    const nemotron::TextConfig& config = model.config();
    const std::string decode_directory = directory + "/decode";
    if (!std::ifstream(decode_directory + "/manifest.txt")) {
        std::printf("nemotron_h decode: no %s/manifest.txt, so the reference's own post-decode "
                    "state was not compared\n",
                    decode_directory.c_str());
        return 0;
    }
    const Dump reference          = read_dump(decode_directory);
    const std::string prefix_file = decode_directory + "/prefix_tokens.i32";
    const std::string step_file   = decode_directory + "/decode_token.i32";
    // The sequence the reference decodes, from the dump rather than from this fixture: the state
    // compared is then the state of the sequence the reference ran, whatever the fixture beside it
    // holds.
    const std::vector<std::int32_t> prefix = read_tokens_file(prefix_file);
    const std::vector<std::int32_t> step   = read_tokens_file(step_file);
    // The reference's own length -- its attention layer's cached length -- has to be what its
    // prefix and its step together account for, or the dump states no state of a sequence this walk
    // can reach and there is nothing to compare against.
    const auto length_entry = reference.find("state.attention_length.0");
    if (prefix.empty() || step.empty() || length_entry == reference.end() ||
        length_entry->second.values.size() != 1) {
        std::printf("nemotron_h decode: %s does not state a prefix, a step and a length together, "
                    "so its state was not compared\n",
                    decode_directory.c_str());
        return 0;
    }
    const auto length = static_cast<std::int32_t>(std::lround(length_entry->second.values.front()));
    if (length != static_cast<std::int32_t>(prefix.size() + step.size())) {
        std::printf(
            "nemotron_h decode: the decode dump's length of %d is neither its prefix of %zu nor "
            "that prefix plus its step of %zu, so its state was not compared\n",
            length, prefix.size(), step.size());
        return 0;
    }
    const auto require = [&](const std::string& name,
                             std::initializer_list<std::int32_t> shape) -> const DumpTensor& {
        const DumpTensor& tensor = need(reference, name);
        if (tensor.shape != std::vector<std::int32_t>(shape)) {
            std::ostringstream got;
            for (const std::int32_t extent : tensor.shape) { got << extent << 'x'; }
            throw std::runtime_error("nemotron_h decode: the decode dump's " + name + " is " +
                                     got.str() + " rather than this config's state shape");
        }
        return tensor;
    };

    const nemotron::MambaConfig& mamba = *config.mamba;
    const std::int32_t channels        = static_cast<std::int32_t>(mamba.conv_channels());
    const std::int32_t width           = static_cast<std::int32_t>(mamba.conv_kernel) - 1;
    const std::int32_t heads           = static_cast<std::int32_t>(mamba.num_heads);
    const std::int32_t head_dim        = static_cast<std::int32_t>(mamba.head_dim);
    const std::int32_t state_size      = static_cast<std::int32_t>(mamba.state_size);
    const std::int32_t keys            = nemotron::kMambaScanStateExtent;
    const std::int32_t vocabulary      = static_cast<std::int32_t>(config.vocab_size);

    // The state the reference's own decode left: its prefix as one state-driven pass from position
    // zero, then its step at the prefix's length, which is the call the dump's second mode makes.
    const std::int32_t prompt_tokens = static_cast<std::int32_t>(prefix.size());
    const std::int32_t step_tokens   = static_cast<std::int32_t>(step.size());
    FamilyState state                = make_family_state(config, length);
    state.reset(execution);
    std::vector<std::int32_t> ids;
    ids.reserve(prefix.size() + step.size());
    ids.insert(ids.end(), prefix.begin(), prefix.end());
    ids.insert(ids.end(), step.begin(), step.end());
    DeviceBuffer ids_buffer(ids.size() * sizeof(std::int32_t));
    ids_buffer.copy_from_host(ids.data(), ids_buffer.bytes);
    DeviceBuffer logits(static_cast<std::size_t>(vocabulary) * static_cast<std::size_t>(length) *
                        sizeof(std::uint16_t));
    const auto ids_view = [&ids_buffer](std::int32_t first, std::int32_t count) {
        return Tensor(static_cast<std::int32_t*>(ids_buffer.p) + first, DType::I32, {count});
    };
    Tensor prompt_logits(logits.p, DType::BF16, {vocabulary, prompt_tokens});
    nemotron::PassState prompt_pass = state.pass(0);
    (void)run_pass(model, ids_view(0, prompt_tokens), prompt_logits, &prompt_pass, arena,
                   execution);
    state.advance();
    Tensor step_logits_buffer(logits.p, DType::BF16, {vocabulary, step_tokens});
    nemotron::PassState step_pass = state.pass(prompt_tokens);
    (void)run_pass(model, ids_view(prompt_tokens, step_tokens), step_logits_buffer, &step_pass,
                   arena, execution);
    state.advance();

    int failures = 0;
    if (state.attention->filled(0) != length) {
        std::cerr << "nemotron_h decode: the attention state holds " << state.attention->filled(0)
                  << " tokens where the reference's own counts " << length << '\n';
        ++failures;
    }
    // The two planes the pool holds, per Mamba block, against the reference's own post-decode
    // state. The attention planes' *values* are covered by the stage comparison above; what this
    // adds is the stored state itself, which no stage exposes.
    std::uint32_t mamba_layer = 0;
    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        if (config.block_kind(static_cast<std::uint32_t>(layer)) != nemotron::BlockKind::Mamba) {
            continue;
        }
        const std::string suffix = std::to_string(mamba_layer++);

        // The convolution window: the reference stores `[1, conv_dim, conv_kernel]` with the taps
        // innermost and keeps one tap more than the convolution reads, so the tree's `[C,
        // W-column]`
        // -- its own order is taps outermost, `w * C + c` -- is the reference's last `W` taps
        // transposed rather than relabelled. Both are the same raw convolution inputs, so the
        // difference between them is this composition's operand representation.
        const DumpTensor& want_conv    = require("state.conv." + suffix, {1, channels, width + 1});
        const std::vector<double> conv = host_values(state.blocks[layer].conv);
        double conv_worst              = 0.0;
        double conv_energy             = 0.0;
        double conv_error              = 0.0;
        for (std::int32_t channel = 0; channel < channels; ++channel) {
            for (std::int32_t tap = 0; tap < width; ++tap) {
                const double got = conv[static_cast<std::size_t>(tap) * channels + channel];
                const double want =
                    want_conv.values[static_cast<std::size_t>(channel) * (width + 1) + tap + 1];
                const double difference = std::abs(got - want);
                conv_worst              = std::max(conv_worst, difference);
                conv_energy += got * got;
                conv_error += difference * difference;
            }
        }
        const double conv_rel = std::sqrt(conv_error / std::max(conv_energy, 1e-30));

        // The recurrent plane: the reference stores `[1, H, head_dim, state]` with the state
        // innermost, the tree `[key, head_dim, H]` with the key axis fastest, so tree
        // `(key, d, h)` is the reference's `[0, h, d, key]`. The tree's plane is wider than this
        // config's state -- the scan's key extent is a constant of its kernel and a narrower state
        // is carried by zero-extending the axis -- so every key past the configured state is
        // required to be *exactly* zero: the carry is the same computation only while a zero key
        // stays zero.
        const DumpTensor& want_plane =
            require("state.recurrent." + suffix, {1, heads, head_dim, state_size});
        const std::vector<double> plane = host_values(state.blocks[layer].recurrent);
        double plane_worst              = 0.0;
        double plane_energy             = 0.0;
        double plane_error              = 0.0;
        std::int64_t tail_nonzero       = 0;
        for (std::int32_t head = 0; head < heads; ++head) {
            for (std::int32_t dim = 0; dim < head_dim; ++dim) {
                const std::size_t row = static_cast<std::size_t>(head) * head_dim * keys +
                                        static_cast<std::size_t>(dim) * keys;
                for (std::int32_t key = 0; key < keys; ++key) {
                    const double got = plane[row + static_cast<std::size_t>(key)];
                    if (key >= state_size) {
                        if (got != 0.0) { ++tail_nonzero; }
                        continue;
                    }
                    const double want =
                        want_plane.values[static_cast<std::size_t>(head) * head_dim * state_size +
                                          static_cast<std::size_t>(dim) * state_size + key];
                    const double difference = std::abs(got - want);
                    plane_worst             = std::max(plane_worst, difference);
                    plane_energy += got * got;
                    plane_error += difference * difference;
                }
            }
        }
        const double plane_rel = std::sqrt(plane_error / std::max(plane_energy, 1e-30));

        std::printf(
            "nemotron_h decode: the reference's state at %d tokens, Mamba block %zu: conv "
            "rel_l2=%.6g max_abs=%.6g (limit %.1e), recurrent rel_l2=%.6g max_abs=%.6g over %d of "
            "%d keys (limit %.1e), %lld of %d zero-tail keys not exactly zero\n",
            length, layer, conv_rel, conv_worst, kStateConvRelL2, plane_rel, plane_worst,
            state_size, keys, kStateRecurrentRelL2, static_cast<long long>(tail_nonzero),
            keys - state_size);
        if (conv_rel > kStateConvRelL2) {
            ++failures;
            std::cerr << "nemotron_h decode: Mamba block " << layer << "'s convolution window at "
                      << conv_rel << " relative L2 against the reference's own\n";
        }
        if (plane_rel > kStateRecurrentRelL2) {
            ++failures;
            std::cerr << "nemotron_h decode: Mamba block " << layer << "'s recurrent plane at "
                      << plane_rel << " relative L2 against the reference's own\n";
        }
        if (tail_nonzero != 0) {
            ++failures;
            std::cerr << "nemotron_h decode: Mamba block " << layer << "'s recurrent plane holds "
                      << tail_nonzero
                      << " nonzero key values past the configured state, where the "
                         "scan's zero-extended carry requires exactly zero\n";
        }
    }
    return failures;
}

// The family's own entry point over the same hand-bound model: a `Program` driven through the
// request lifecycle and compared against the same zero-state columns. The walk above proves the
// composition's state arithmetic; this proves the Program *binds* that state -- the pool's slots,
// the block-to-pool-layer mapping, the attention layer binding and the lane's first position -- and
// that it drives it. A pass leaves the logits of the token it consumed, so a prefill of `M` tokens
// leaves column `M - 1` and the one-token decode after it leaves column `M` of the fixture.
int program_check(DeviceContext& context, DeviceExecutionView execution,
                  const nemotron::Model& model, const std::vector<std::int32_t>& tokens,
                  const Capture& reference) {
    const nemotron::TextConfig& config = model.config();
    const std::int32_t vocabulary      = static_cast<std::int32_t>(config.vocab_size);
    const std::int32_t total           = static_cast<std::int32_t>(tokens.size());
    const std::int32_t prefix          = total - 1;
    const StageKey logits_key{0, nemotron::BlockStage::Logits};
    const auto logits_stage = reference.find(logits_key);
    if (logits_stage == reference.end()) {
        throw std::runtime_error("nemotron_h decode: the zero-state prefill reported no logits "
                                 "stage, so the Program has no column to be compared against");
    }
    const auto as_column = [](const Tensor& tensor) {
        CapturedStage stage;
        stage.columns = tensor.ne[0];
        stage.values  = host_values(tensor);
        return stage;
    };

    EngineOptions options;
    options.max_context     = static_cast<std::uint32_t>(total);
    options.max_concurrency = 1;
    options.kv_cache        = KvCacheStorage::BFloat16;
    nemotron::Program program(model, options, context);

    int failures = 0;
    program.create(0, execution);
    if (program.position(0) != 0) {
        std::cerr << "nemotron_h Program: a lane just created is at position "
                  << program.position(0) << " rather than the sequence's first\n";
        ++failures;
    }
    program.prefill(0,
                    std::span<const std::int32_t>(tokens.data(), static_cast<std::size_t>(prefix)),
                    execution);
    if (program.position(0) != prefix) {
        std::cerr << "nemotron_h Program: a " << prefix << "-token prefill left the lane at "
                  << program.position(0) << '\n';
        ++failures;
    }
    const ColumnStat prefilled = compare_column(logits_key, logits_stage->second, prefix - 1,
                                                as_column(program.logits(0)), 0);
    program.decode(0, tokens[static_cast<std::size_t>(prefix)], execution);
    if (program.position(0) != total) {
        std::cerr << "nemotron_h Program: a decode step left the lane at " << program.position(0)
                  << " rather than " << total << '\n';
        ++failures;
    }
    const ColumnStat decoded =
        compare_column(logits_key, logits_stage->second, prefix, as_column(program.logits(0)), 0);
    // A draw is a token, and the greedy sampler `create` installs draws the column's own maximum --
    // reported beside it rather than asserted, because how a sampler breaks a tie is the sampling
    // Op's business and its own test's.
    const std::int32_t drawn = program.sample(0, execution);
    const std::size_t column =
        static_cast<std::size_t>(prefix) * static_cast<std::size_t>(vocabulary);
    std::int32_t argmax = 0;
    for (std::int32_t id = 1; id < vocabulary; ++id) {
        if (logits_stage->second.values[column + static_cast<std::size_t>(id)] >
            logits_stage->second.values[column + static_cast<std::size_t>(argmax)]) {
            argmax = id;
        }
    }
    if (drawn < 0 || drawn >= vocabulary) {
        std::cerr << "nemotron_h Program: a draw returned " << drawn
                  << ", which is outside the model's vocabulary of " << vocabulary << '\n';
        ++failures;
    }
    program.finish(0);

    std::printf(
        "nemotron_h Program: prefill %d then decode 1, against the forward's columns %d and %d: "
        "rel_l2=%.6g max_abs=%.6g and rel_l2=%.6g max_abs=%.6g; the draw is %d, %s the decoded "
        "column's argmax %d\n",
        prefix, prefix - 1, prefix, prefilled.relative_l2, prefilled.max_abs, decoded.relative_l2,
        decoded.max_abs, drawn, drawn == argmax ? "which is" : "which is not", argmax);
    return failures;
}

// The decode oracle, over `K = 8` and `K = 1`, plus the reused-lane, reference-state and Program
// checks.
int decode_oracle(const std::string& directory, const nemotron::Model& model,
                  const std::vector<std::int32_t>& tokens, DeviceArena& arena,
                  DeviceExecutionView execution, DeviceContext& context) {
    const nemotron::TextConfig& config = model.config();
    const std::int32_t vocabulary      = static_cast<std::int32_t>(config.vocab_size);
    const std::int32_t total           = static_cast<std::int32_t>(tokens.size());
    int failures                       = 0;

    // The fixture's tokens once, and every pass a view of its own slice: a one-token decode is the
    // same call with a one-wide view, so no step copies the prompt or the prefix.
    DeviceBuffer ids(tokens.size() * sizeof(std::int32_t));
    ids.copy_from_host(tokens.data(), ids.bytes);
    const auto ids_view = [&ids](std::int32_t first, std::int32_t count) {
        return Tensor(static_cast<std::int32_t*>(ids.p) + first, DType::I32, {count});
    };
    DeviceBuffer wide_logits(static_cast<std::size_t>(vocabulary) *
                             static_cast<std::size_t>(total) * sizeof(std::uint16_t));
    DeviceBuffer step_logits(static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t));

    // Run 1: the walk above's own call -- a zero-state pass over the whole fixture with the head on
    // every column. Every comparison below is against one of its columns, so it is the oracle for
    // every position and every stage at once.
    Tensor wide(wide_logits.p, DType::BF16, {vocabulary, total});
    const Capture reference = run_pass(model, ids_view(0, total), wide, nullptr, arena, execution);
    std::printf("nemotron_h decode: the zero-state prefill reported %zu stages over %d positions\n",
                reference.size(), total);

    for (const std::int32_t decoded : {8, 1}) {
        const std::int32_t prefix = total - decoded;
        {
            // Run 2: the prefix as one state-driven pass, then one token at a time -- the decode
            // `Program` drives per generated token.
            FamilyState state = make_family_state(config, total);
            state.reset(execution);
            Tensor chunk(wide_logits.p, DType::BF16, {vocabulary, prefix});
            nemotron::PassState chunk_pass = state.pass(0);
            (void)run_pass(model, ids_view(0, prefix), chunk, &chunk_pass, arena, execution);
            state.advance();
            Capture decoded_run;
            Tensor step(step_logits.p, DType::BF16, {vocabulary, 1});
            for (std::int32_t token = prefix; token < total; ++token) {
                nemotron::PassState step_pass = state.pass(token);
                append_capture(decoded_run, run_pass(model, ids_view(token, 1), step, &step_pass,
                                                     arena, execution));
                state.advance();
            }
            const std::string label = "the decode of " + std::to_string(decoded) + " token(s)";
            const OracleVerdict verdict =
                compare_positions(reference, prefix, decoded_run, decoded, label);
            failures += verdict.failures;
            std::printf(
                "nemotron_h decode: %s one at a time from a %d-token prefill: %zu comparisons, "
                "loosest %s (rel_l2=%.6g, max_abs=%.6g), logits rel_l2=%.6g max_abs=%.6g\n",
                label.c_str(), prefix, verdict.comparisons, verdict.loosest.name.c_str(),
                verdict.loosest.relative_l2, verdict.loosest.max_abs, verdict.logits.relative_l2,
                verdict.logits.max_abs);
        }
        {
            // Run 3: the same split as two *passes*, then the same tokens decoded one at a time
            // from the state those chunks left.
            FamilyState state = make_family_state(config, total);
            state.reset(execution);
            Tensor first(wide_logits.p, DType::BF16, {vocabulary, prefix});
            nemotron::PassState first_pass = state.pass(0);
            (void)run_pass(model, ids_view(0, prefix), first, &first_pass, arena, execution);
            state.advance();
            Tensor second_logits(wide_logits.p, DType::BF16, {vocabulary, decoded});
            nemotron::PassState second_pass = state.pass(prefix);
            const Capture second = run_pass(model, ids_view(prefix, decoded), second_logits,
                                            &second_pass, arena, execution);
            state.advance();
            // The second chunk's own columns, *including its first tokens*: that is where a chunk
            // that started from a zero state differs from the one-shot run, so a comparison of the
            // last column alone would not see the state it is supposed to be reading.
            const OracleVerdict chunk_verdict = compare_positions(
                reference, prefix, second, decoded, "the second half of two chunks");
            failures += chunk_verdict.failures;
            std::printf(
                "nemotron_h decode: the second chunk of %d+%d against the one-shot run: %zu "
                "comparisons, loosest %s (rel_l2=%.6g, max_abs=%.6g), logits rel_l2=%.6g "
                "max_abs=%.6g\n",
                prefix, decoded, chunk_verdict.comparisons, chunk_verdict.loosest.name.c_str(),
                chunk_verdict.loosest.relative_l2, chunk_verdict.loosest.max_abs,
                chunk_verdict.logits.relative_l2, chunk_verdict.logits.max_abs);

            Capture decoded_run;
            Tensor step(step_logits.p, DType::BF16, {vocabulary, 1});
            for (std::int32_t token = prefix; token < total; ++token) {
                nemotron::PassState step_pass = state.pass(token);
                append_capture(decoded_run, run_pass(model, ids_view(token, 1), step, &step_pass,
                                                     arena, execution));
                state.advance();
            }
            const std::string label = "the decode from the two-chunk state";
            const OracleVerdict verdict =
                compare_positions(reference, prefix, decoded_run, decoded, label);
            failures += verdict.failures;
            std::printf(
                "nemotron_h decode: %s: %zu comparisons, loosest %s (rel_l2=%.6g, max_abs=%.6g), "
                "logits rel_l2=%.6g max_abs=%.6g\n",
                label.c_str(), verdict.comparisons, verdict.loosest.name.c_str(),
                verdict.loosest.relative_l2, verdict.loosest.max_abs, verdict.logits.relative_l2,
                verdict.logits.max_abs);
        }
    }

    {
        // A lane reused for a new sequence: the same prefix and the same one-token step twice, with
        // the reset `Program::create` performs between them. The convolution Op always reads the
        // state it is handed, so a frontier that was not emptied would make the second sequence's
        // first pass silently wrong -- and it is only emptied if the reset reproduces the run,
        // which is why this comparison is exact rather than within the walk's criterion.
        FamilyState state        = make_family_state(config, total);
        const std::int32_t reuse = total - 1;
        const auto run_twice     = [&] {
            state.reset(execution);
            Tensor prompt(wide_logits.p, DType::BF16, {vocabulary, reuse});
            nemotron::PassState prompt_pass = state.pass(0);
            Capture run =
                run_pass(model, ids_view(0, reuse), prompt, &prompt_pass, arena, execution);
            state.advance();
            Tensor step(step_logits.p, DType::BF16, {vocabulary, 1});
            nemotron::PassState step_pass = state.pass(reuse);
            append_capture(run,
                           run_pass(model, ids_view(reuse, 1), step, &step_pass, arena, execution));
            state.advance();
            return run;
        };
        const Capture before = run_twice();
        const Capture after  = run_twice();
        const int mismatches = compare_exact(before, after, "a lane reused after its reset");
        failures += mismatches;
        std::printf("nemotron_h decode: a %d-token prefix plus one step, twice across the reset "
                    "Program::create performs: %zu stages, %s\n",
                    reuse, before.size(),
                    mismatches == 0 ? "identical value for value" : "not identical");
    }

    failures += program_check(context, execution, model, tokens, reference);
    failures += check_decoded_state(directory, model, arena, execution);
    return failures;
}

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
        nemotron::forward_model(*built.model, ids_t, arena, logits_t, execution, nullptr,
                                nemotron::HeadColumns::Every, &comparator);
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
        // The oracle above is a zero-state pass; this is the same composition driven with the state
        // that pass leaves, against the same columns. It runs after the walk so a state defect is
        // reported beside the walk's own verdict rather than instead of it.
        failures += decode_oracle(directory, *built.model, tokens, arena, execution, context);
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
