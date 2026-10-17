#include "models/draft/draft_spec.h"

#include <cmath>
#include <set>
#include <string>
#include <string_view>

namespace ninfer::models::draft {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t dimension(const Json& value, std::string_view name) {
    const auto& member = value.at(name);
    if (!member.is_number_unsigned() || member.get<std::uint64_t>() == 0 ||
        member.get<std::uint64_t>() > 0xffffffffULL) {
        throw ArtifactError(std::string(name) + " must be a positive width");
    }
    return member.get<std::uint32_t>();
}

float positive_float(const Json& value, std::string_view label) {
    const auto& member = value.at(label);
    if (!member.is_number()) { throw ArtifactError(std::string(label) + " must be a real value"); }
    const float result = member.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(label) + " must be positive finite FP32");
    }
    return result;
}

std::string architecture(const Json& value) {
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

} // namespace

Spec parse_lilicorr(const Json& value, std::uint32_t target_hidden, std::uint32_t target_vocab,
                    std::uint32_t target_blocks) {
    artifact::require_members(value,
                              {"architectures", "model_type", "block_size", "hidden_size",
                               "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                               "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps",
                               "rope_parameters", "max_position_embeddings", "layer_types",
                               "sliding_window", "dflash_config"},
                              {"hidden_act", "tie_word_embeddings"}, "LiLiCorr draft");
    if (architecture(value) != "LiLiCorrDraftModel" || value.at("model_type") != "qwen3") {
        throw ArtifactError("LiLiCorr draft architecture/model_type mismatch");
    }
    Spec out;
    out.flavor                  = Spec::Flavor::Lilicorr;
    out.block_size              = dimension(value, "block_size");
    out.hidden_size             = dimension(value, "hidden_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.num_attention_heads     = dimension(value, "num_attention_heads");
    out.num_key_value_heads     = dimension(value, "num_key_value_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.sliding_window          = dimension(value, "sliding_window");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");

    // The draft conditions on the tower's residual width and vocabulary, so a draft stored against
    // another tower is a refusal rather than a silently wrong conditioning.
    if (out.hidden_size != target_hidden) {
        throw ArtifactError("draft width differs from the tower");
    }
    if (out.vocab_size != target_vocab) {
        throw ArtifactError("draft vocabulary differs from the tower");
    }
    if (out.num_attention_heads % out.num_key_value_heads != 0) {
        throw ArtifactError("draft query heads must be divisible by KV heads");
    }
    if (out.head_dim % 2 != 0) { throw ArtifactError("draft head dimension must be even"); }
    if (out.max_position_embeddings < out.sliding_window) {
        throw ArtifactError("draft positional capacity is below its window");
    }

    artifact::require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "draft RoPE");
    out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");

    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("draft layer_types length mismatch");
    }
    for (const auto& layer : layers) {
        if (layer != "sliding_attention") { throw ArtifactError("unknown draft layer_type"); }
    }

    const auto& config = value.at("dflash_config");
    artifact::require_members(
        config,
        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
         "attention_sink_bias", "projector_type", "lilicorr_enabled", "lilicorr_candidate_topk",
         "lilicorr_hidden_size", "lilicorr_num_layers", "lilicorr_num_heads", "lilicorr_mlp_ratio",
         "lilicorr_factor_dim", "lilicorr_vector_eps", "lilicorr_logit_scale"},
        {"use_swa", "swa_window_size", "causal"}, "LiLiCorr config");
    const auto& taps = config.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("target_layer_ids must be nonempty");
    }
    std::set<std::uint32_t> unique;
    for (const auto& tap : taps) {
        if (!tap.is_number_unsigned() || tap.get<std::uint64_t>() >= target_blocks ||
            !unique.insert(tap.get<std::uint32_t>()).second) {
            throw ArtifactError("invalid target layer index");
        }
        out.target_layer_ids.push_back(tap.get<std::uint32_t>());
    }
    out.mask_token_id    = dimension(config, "mask_token_id");
    out.conv_kernel_size = dimension(config, "conv_kernel_size");
    out.conv_group_size  = dimension(config, "conv_group_size");
    if (out.mask_token_id >= out.vocab_size) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    if (out.hidden_size % out.conv_group_size != 0) {
        throw ArtifactError("draft conv group does not divide the draft width");
    }
    // The sink and the projector are what make this draft LiLiCorr rather than plain DFlash2; a
    // stored flag that clears either is a different model, not a variant.
    if (!config.at("attention_sink_bias").is_boolean() ||
        !config.at("attention_sink_bias").get<bool>()) {
        throw ArtifactError("LiLiCorr draft requires its attention sink");
    }
    if (!config.at("projector_type").is_string() || config.at("projector_type") != "lilicorr") {
        throw ArtifactError("draft projector must be lilicorr");
    }
    if (!config.at("lilicorr_enabled").is_boolean() || !config.at("lilicorr_enabled").get<bool>()) {
        throw ArtifactError("LiLiCorr draft requires its projector enabled");
    }
    out.head.candidate_top_k = dimension(config, "lilicorr_candidate_topk");
    out.head.hidden_size     = dimension(config, "lilicorr_hidden_size");
    out.head.num_layers      = dimension(config, "lilicorr_num_layers");
    out.head.num_heads       = dimension(config, "lilicorr_num_heads");
    out.head.mlp_ratio       = positive_float(config, "lilicorr_mlp_ratio");
    out.head.factor_dim      = dimension(config, "lilicorr_factor_dim");
    out.head.vector_eps      = positive_float(config, "lilicorr_vector_eps");
    out.head.logit_scale     = positive_float(config, "lilicorr_logit_scale");
    if (out.head.hidden_size % out.head.num_heads != 0) {
        throw ArtifactError("head width must be divisible by its head count");
    }
    return out;
}


Spec parse_dflash(const Json& value, std::uint32_t target_hidden, std::uint32_t target_vocab,
                  std::uint32_t target_blocks) {
    // The converter validates the companion's width and vocabulary against the tower and does not
    // repeat them in the stored config, so they come from the tower here.
    artifact::require_members(value,
                              {"architectures", "model_type", "intermediate_size",
                               "num_attention_heads", "num_key_value_heads", "head_dim",
                               "num_hidden_layers", "max_position_embeddings", "rms_norm_eps",
                               "rope_parameters", "layer_types", "dflash_config"},
                              {"sliding_window", "hidden_size", "vocab_size"}, "DFlash draft");
    if (architecture(value) != "DFlashDraftModel" || value.at("model_type") != "qwen3") {
        throw ArtifactError("DFlash draft architecture/model_type mismatch");
    }
    Spec out;
    out.flavor                  = Spec::Flavor::Dflash;
    out.hidden_size             = target_hidden;
    out.vocab_size              = target_vocab;
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_attention_heads     = dimension(value, "num_attention_heads");
    out.num_key_value_heads     = dimension(value, "num_key_value_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    // A plain DFlash carries no stored block length; the round chooses it.
    out.block_size = 0;

    if (out.num_attention_heads % out.num_key_value_heads != 0 || out.head_dim % 2 != 0) {
        throw ArtifactError("DFlash attention geometry is invalid");
    }

    artifact::require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "DFlash RoPE");
    out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");

    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("DFlash layer_types length mismatch");
    }
    for (const auto& kind : layers) {
        if (!kind.is_string() || kind.get<std::string>() != "full_attention") {
            throw ArtifactError("DFlash layer_types must all be full_attention");
        }
    }

    const auto& draft = value.at("dflash_config");
    artifact::require_members(draft, {"target_layer_ids", "mask_token_id"}, {}, "DFlash taps");
    const auto& taps = draft.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("DFlash target_layer_ids must be a non-empty array");
    }
    std::set<std::uint32_t> seen;
    for (const auto& tap : taps) {
        if (!tap.is_number_unsigned()) {
            throw ArtifactError("DFlash target_layer_ids must be block indices");
        }
        const auto index = tap.get<std::uint32_t>();
        if (index >= target_blocks) {
            throw ArtifactError("DFlash target_layer_ids reach past the tower's blocks");
        }
        if (!seen.insert(index).second) {
            throw ArtifactError("DFlash target_layer_ids repeat a block");
        }
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = dimension(draft, "mask_token_id");
    if (out.mask_token_id >= target_vocab) {
        throw ArtifactError("DFlash mask token is outside the tower's vocabulary");
    }
    return out;
}

} // namespace ninfer::models::draft
