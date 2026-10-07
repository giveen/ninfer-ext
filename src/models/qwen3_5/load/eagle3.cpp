#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

// EAGLE3 draft: one autoregressive decoder layer conditioned on three concatenated target hidden
// states. Unlike DFlash it has no masked-context projections, its attention input is the 2h-wide
// concatenation of the token embedding and the fused feature, and it carries its own draft head and
// the integer draft-to-target token-id map.
DraftWeights bind_eagle3(Bindings& b, const DraftConfig& config, const TextConfig& target,
                         const TextWeights& weights, const std::string& component) {
    const auto& eagle = *config.eagle3;
    const auto h      = target.hidden_size;
    const auto q      = config.attention.query_width();
    const auto k      = config.attention.key_width();
    DraftWeights out;
    out.feature_projection =
        b.parameter(component + "/feature_projection", {h, eagle.fused_input_size()},
                    {component + "/target_features"});
    out.final_norm      = b.direct(component + "/final_norm", {h});
    out.token_embedding = weights.token_embedding;
    out.output_head =
        b.parameter(component + "/output_head", {eagle.draft_vocab_size, h},
                    {component + "/final_hidden"});
    out.d2t = b.direct(component + "/d2t", {eagle.draft_vocab_size}, QType::INT32);
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = component + "/layers/" + std::to_string(i) + "/";
        DraftBlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.hidden_norm         = b.direct(p + "hidden_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.attention.query =
            b.parameter(p + "attention/query", {q, 2 * h}, {p + "attention_input"});
        layer.attention.key =
            b.parameter(p + "attention/key", {k, 2 * h}, {p + "attention_input"});
        layer.attention.value =
            b.parameter(p + "attention/value", {k, 2 * h}, {p + "attention_input"});
        layer.attention.output =
            b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});
        layer.mlp = bind_dense(b, h, config.intermediate_size, p, true);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading