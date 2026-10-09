#include "models/gemma4/config.h"

#include "artifact/schema.h"
#include "models/registry.h"

#include <cmath>
#include <set>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

[[noreturn]] void fail(std::string message) {
    throw artifact::ArtifactError("Gemma 4 text config: " + std::move(message));
}

std::uint32_t positive(const artifact::Json& value, const char* name) {
    if (!value.contains(name) || !value.at(name).is_number_integer() ||
        value.at(name).get<std::int64_t>() <= 0) {
        fail(std::string(name) + " must be a positive integer");
    }
    return static_cast<std::uint32_t>(value.at(name).get<std::int64_t>());
}

float number(const artifact::Json& value, const char* name) {
    if (!value.contains(name) || !value.at(name).is_number()) {
        fail(std::string(name) + " must be a number");
    }
    return value.at(name).get<float>();
}

bool boolean(const artifact::Json& value, const char* name) {
    if (!value.contains(name) || !value.at(name).is_boolean()) {
        fail(std::string(name) + " must be a boolean");
    }
    return value.at(name).get<bool>();
}

// The rotary block of one layer kind, and the two invariants that make it well defined.
AttentionGeometry geometry(const artifact::Json& rope, AttentionGeometry sliding, bool global) {
    if (!rope.is_object()) { fail("rope_parameters entries must be objects"); }
    const std::string expect = global ? "proportional" : "default";
    if (rope.value("rope_type", std::string{}) != expect) {
        fail("rope_parameters must be 'default' for sliding attention and 'proportional' for global");
    }
    sliding.rope_theta  = rope.contains("rope_theta") ? rope.at("rope_theta").get<float>() : 0.0F;
    sliding.rotary_dim  = positive(rope, "rotary_dim");
    if (sliding.rotary_dim % 2 != 0) { fail("rope_parameters.rotary_dim must be even"); }
    return sliding;
}

} // namespace

TextConfig parse_text_config(const artifact::Json& value) {
    if (!value.is_object()) { fail("config must be an object"); }
    static constexpr const char* kMembers[] = {
        "architectures", "model_type", "hidden_act", "hidden_size", "intermediate_size",
        "vocab_size", "num_hidden_layers", "max_position_embeddings", "num_attention_heads",
        "num_key_value_heads", "num_global_key_value_heads", "head_dim", "global_head_dim",
        "sliding_window", "attention_k_eq_v", "attention_bias", "attention_scale",
        "final_logit_softcapping", "rms_norm_eps", "norm_unit_offset", "embedding_scale",
        "tie_word_embeddings", "layer_scalar", "layer_types", "global_layer_count",
        "rope_parameters"};
    for (const auto& member : value.items()) {
        bool known = false;
        for (const char* allowed : kMembers) { known = known || member.key() == allowed; }
        if (!known) { fail("unknown member " + member.key()); }
    }

    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures.at(0) != "Gemma4ForCausalLM") {
        fail("architectures must be [Gemma4ForCausalLM]");
    }
    if (value.at("model_type") != "gemma4_text") { fail("model_type must be gemma4_text"); }
    if (resolve_architecture("Gemma4ForCausalLM", "gemma4_text") != Architecture::Gemma4) {
        fail("the registry does not resolve this architecture");
    }

    TextConfig out;
    out.hidden_size             = positive(value, "hidden_size");
    out.intermediate_size       = positive(value, "intermediate_size");
    out.vocab_size              = positive(value, "vocab_size");
    out.num_hidden_layers       = positive(value, "num_hidden_layers");
    out.max_position_embeddings = positive(value, "max_position_embeddings");
    out.sliding_window          = positive(value, "sliding_window");
    out.rms_norm_eps            = number(value, "rms_norm_eps");
    out.embedding_scale         = number(value, "embedding_scale");
    out.attention_scale         = number(value, "attention_scale");
    out.final_logit_softcapping = number(value, "final_logit_softcapping");
    out.tie_word_embeddings     = boolean(value, "tie_word_embeddings");
    out.hidden_act              = value.at("hidden_act").get<std::string>();

    // The invariants the mathematics depends on. Each one, if violated, would silently change the
    // model rather than fail it.
    if (out.hidden_act != "gelu_pytorch_tanh") { fail("hidden_act must be gelu_pytorch_tanh"); }
    if (boolean(value, "attention_bias")) { fail("attention_bias must be false"); }
    if (!boolean(value, "attention_k_eq_v")) { fail("attention_k_eq_v must be true"); }
    if (boolean(value, "norm_unit_offset")) {
        fail("norm_unit_offset must be false: Gemma 4 norms carry no 1 + w offset");
    }
    if (!boolean(value, "layer_scalar")) { fail("layer_scalar must be true"); }
    if (!out.tie_word_embeddings) { fail("tie_word_embeddings must be true: the head is the embedding"); }
    if (out.attention_scale != 1.0F) { fail("attention_scale must be exactly 1.0"); }
    const float expected_embed = std::sqrt(static_cast<float>(out.hidden_size));
    if (std::fabs(out.embedding_scale - expected_embed) > 1e-3F * expected_embed) {
        fail("embedding_scale must be sqrt(hidden_size)");
    }

    const std::uint32_t heads = positive(value, "num_attention_heads");
    const std::uint32_t kv    = positive(value, "num_key_value_heads");
    const std::uint32_t gkv   = positive(value, "num_global_key_value_heads");
    if (heads % kv != 0 || heads % gkv != 0) {
        fail("query heads must divide into both KV head counts");
    }
    out.sliding.num_key_value_heads = kv;
    out.sliding.head_dim            = positive(value, "head_dim");
    out.global.shared.num_key_value_heads = gkv;
    out.global.shared.head_dim            = positive(value, "global_head_dim");

    // Global attention on every sixth layer, the last included: this is a checkpoint fact, and the
    // token budget of each mixer follows from it.
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        fail("layer_types must cover every layer");
    }
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const bool global = (index + 1) % 6 == 0;
        const std::string want = global ? "full_attention" : "sliding_attention";
        if (layers.at(index) != want) {
            fail("layer_types must place global attention on every sixth layer, last included");
        }
        out.layer_types.push_back(global ? MixerKind::FullAttention : MixerKind::SlidingAttention);
        out.global_layers += global ? 1 : 0;
    }
    if (positive(value, "global_layer_count") != out.global_layers) {
        fail("global_layer_count must agree with layer_types");
    }

    const auto& rope = value.at("rope_parameters");
    if (!rope.is_object() || !rope.contains("sliding_attention") || !rope.contains("full_attention")) {
        fail("rope_parameters must describe both layer kinds");
    }
    out.sliding = geometry(rope.at("sliding_attention"), out.sliding, false);
    if (out.sliding.rotary_dim != out.sliding.head_dim) {
        fail("sliding RoPE must rotate the whole head");
    }
    out.global.shared = geometry(rope.at("full_attention"), out.global.shared, true);
    const auto& full  = rope.at("full_attention");
    out.global.rope_angles = positive(full, "rope_angles");
    out.global.denominator_head_dim = positive(full, "denominator_head_dim");
    if (out.global.shared.rotary_dim != 2 * out.global.rope_angles) {
        fail("proportional RoPE rotates two dims per angle");
    }
    if (out.global.denominator_head_dim != out.global.shared.head_dim) {
        fail("proportional RoPE takes its frequencies against the full head width");
    }
    return out;
}

} // namespace ninfer::models::gemma4
