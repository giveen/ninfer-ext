#include "models/gemma4/measurement.h"

#include "models/qwen3_5/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

// The artifact's own word for a layer kind, so the signature reads against its config.
const char* mixer_name(MixerKind kind) noexcept {
    return kind == MixerKind::FullAttention ? "full_attention" : "sliding_attention";
}

} // namespace

std::string prefill_signature(const Model& model) {
    using Json       = nlohmann::json;
    const auto& t    = model.config();
    Json layer_types = Json::array();
    for (const MixerKind kind : t.layer_types) { layer_types.push_back(mixer_name(kind)); }
    // The two attention geometries, as the config stores them: the sliding one rotates the whole
    // head, the global one stores no value (attention_k_eq_v) and rotates two dims per angle.
    const Json attention{
        {"num_attention_heads", t.num_attention_heads},
        {"sliding",
         {t.sliding.num_key_value_heads, t.sliding.head_dim, t.sliding.rope_theta,
          t.sliding.rotary_dim}},
        {"global",
         {t.global.shared.num_key_value_heads, t.global.shared.head_dim, t.global.shared.rope_theta,
          t.global.shared.rotary_dim, t.global.rope_angles, t.global.denominator_head_dim}}};
    Json facts{{"implementation", "gemma4-prefill-2"},
               {"architecture", architecture_name(t.architecture)},
               {"hidden_size", t.hidden_size},
               {"vocab_size", t.vocab_size},
               {"num_hidden_layers", t.num_hidden_layers},
               {"rms_norm_eps", t.rms_norm_eps},
               {"sliding_window", t.sliding_window},
               {"global_layers", t.global_layers},
               // The loader admits only attention_k_eq_v == true, so this fact is constant here.
               {"attention_k_eq_v", true},
               {"attention", attention}};
    facts["layer_types"] = std::move(layer_types);
    // The dense MLP width and the MoE block. A dense-only artifact carries zeros for the four MoE
    // members, which the applicability fact reports rather than hides.
    facts["ffn"] = {t.intermediate_size, t.enable_moe_block, t.num_experts, t.top_k_experts,
                    t.moe_intermediate_size};
    if (const auto& vision = model.vision_config()) {
        facts["vision"] = {vision->hidden_size,
                           vision->intermediate_size,
                           vision->num_hidden_layers,
                           vision->num_attention_heads,
                           vision->head_dim,
                           vision->patch_size,
                           vision->pooling_kernel_size,
                           vision->position_embedding_size,
                           vision->soft_tokens_per_image,
                           vision->output_hidden_size,
                           vision->rope_theta};
    }
    // The binding structure: every materialized Text/Vision parameter's role, logical shape and the
    // encoded parents its regions address. Two artifacts whose configs agree but whose weights are
    // laid out differently (another format, another group or padding) do different prefill work, so
    // the signature has to separate them. Gemma binds every input A16-only, so the Use dimension
    // the signature folds for other models is constant here and carries no discrimination.
    std::vector<const BoundWeight*> weights;
    // PrefillWork prices primary Text/Vision reconstruction; an optional drafter does not change
    // that work's applicability, so its weights stay out of the key.
    for (const auto& weight : model.weight_data()) {
        if (weight.name.starts_with("text/") || weight.name.starts_with("vision/")) {
            weights.push_back(&weight);
        }
    }
    std::ranges::sort(weights, [](const auto* a, const auto* b) { return a->name < b->name; });
    std::map<const WeightParent*, std::size_t> parents;
    auto& inventory = facts["weights"] = Json::array();
    for (const auto* weight : weights) {
        // A bound Gemma weight always carries resident regions: the loader admits only Device and
        // pinned-host parameters, and bind_view proves they cover the logical shape. So unlike
        // Qwen there is no file-mapped case whose geometry would sit outside `parts`, and a weight
        // that lost its regions would otherwise contribute only its shape.
        if (weight->view.parts.empty()) {
            throw artifact::ArtifactError(weight->name + ": bound weight has no resident regions");
        }
        Json item{{"role", weight->name}, {"shape", weight->view.shape}, {"parts", Json::array()}};
        for (const auto& part : weight->view.parts) {
            const auto [it, inserted] = parents.emplace(part.parent, parents.size());
            const auto& geometry      = part.parent->geometry;
            item["parts"].push_back({it->second, part.begin, part.end, geometry.format,
                                     geometry.layout, geometry.shape, geometry.padded_columns,
                                     geometry.group_size});
        }
        inventory.push_back(std::move(item));
    }
    return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(facts.dump()));
}

} // namespace ninfer::models::gemma4
