#include "models/nemotron_h/config.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace ninfer::models::nemotron_h {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_id;
using artifact::require_members;
using artifact::require_u64;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

BlockKind block_kind(const std::string& name) {
    if (name == "mamba") { return BlockKind::Mamba; }
    if (name == "moe") { return BlockKind::Moe; }
    if (name == "attention") { return BlockKind::Attention; }
    throw ArtifactError("block_configs: unsupported block type " + name);
}

MambaConfig mamba(const Json& config) {
    require_members(config,
                    {"mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups",
                     "conv_kernel", "chunk_size", "time_step_min", "use_conv_bias"},
                    {}, "text config (mamba)");

    MambaConfig out;
    out.num_heads     = dimension(config, "mamba_num_heads");
    out.head_dim      = dimension(config, "mamba_head_dim");
    out.state_size    = dimension(config, "ssm_state_size");
    out.n_groups      = dimension(config, "n_groups");
    out.conv_kernel   = dimension(config, "conv_kernel");
    out.chunk_size    = dimension(config, "chunk_size");
    out.time_step_min = positive_float(config, "time_step_min");
    out.conv_bias     = config.at("use_conv_bias").get<bool>();

    // The gated norm normalises within groups of inner_size / n_groups, so that must be integral.
    if (out.inner_size() % out.n_groups != 0) {
        throw ArtifactError("mamba_num_heads * mamba_head_dim must divide by n_groups");
    }
    return out;
}

AttentionConfig attention(const Json& config) {
    require_members(config, {"num_attention_heads", "num_key_value_heads", "head_dim"},
                    {"rope_theta"}, "text config (attention)");

    AttentionConfig out;
    out.num_attention_heads = dimension(config, "num_attention_heads");
    out.num_key_value_heads = dimension(config, "num_key_value_heads");
    out.head_dim            = dimension(config, "head_dim");
    out.rope_theta =
        config.contains("rope_theta") && config.at("rope_theta").is_number()
            ? positive_float(config, "rope_theta")
            : 10000.0F; // the checkpoint leaves it unset; the reference then uses the default
    if (out.num_attention_heads % out.num_key_value_heads != 0) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

RouterConfig router(const Json& config) {
    require_members(config, {"routed_scaling_factor", "norm_topk_prob"},
                    {"n_group", "topk_group", "e_score_correction_bias"}, "text config (router)");

    RouterConfig out;
    out.scaling_factor  = positive_float(config, "routed_scaling_factor");
    out.normalize_top_k = config.at("norm_topk_prob").get<bool>();
    // Emitted by the converter when the checkpoint carries the buffer, because a routed bank's
    // selection depends on it and a silent default of zero would change which experts fire.
    out.correction_bias = config.contains("e_score_correction_bias");
    out.groups          = config.contains("n_group") ? dimension(config, "n_group") : 1;
    out.groups_used     = config.contains("topk_group") ? dimension(config, "topk_group") : 1;
    if (out.groups_used > out.groups) { throw ArtifactError("topk_group cannot exceed n_group"); }
    return out;
}

// The per-block MoE fields live on the block entry; the latent and shared-expert widths are global
// in the checkpoint's config, so they are taken from the model config and copied onto every bank.
MoeBankConfig moe_bank(const Json& block, const Json& config) {
    require_members(
        block, {"block_type", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"},
        {}, "block_configs entry (moe)");
    // The latent and shared widths are model-wide in the checkpoint, so they sit at the text level
    // rather than inside each bank. They are read here because a bank is what needs them, and their
    // presence is checked rather than assumed: `require_members` at the text level permits them
    // without requiring them, since a model with no MoE blocks has no use for either.
    if (!config.contains("moe_latent_size") ||
        !config.contains("moe_shared_expert_intermediate_size")) {
        throw ArtifactError(
            "a moe block requires moe_latent_size and moe_shared_expert_intermediate_size");
    }

    MoeBankConfig out;
    out.num_experts              = dimension(block, "n_routed_experts");
    out.experts_per_token        = dimension(block, "num_experts_per_tok");
    out.intermediate_size        = dimension(block, "moe_intermediate_size");
    out.latent_size              = integer(config.at("moe_latent_size"), "moe_latent_size", false);
    out.shared_intermediate_size = dimension(config, "moe_shared_expert_intermediate_size");
    // A bank must be wide enough to route to the experts it asks for.
    if (out.experts_per_token > out.num_experts) {
        throw ArtifactError("num_experts_per_tok cannot exceed the layer's expert count");
    }
    return out;
}

TextConfig parse_text(const Json& config) {
    // `require_members` validates the exact key set of the object it is given, so each level
    // declares precisely what belongs to it: the model-wide numbers here, and the mixer, attention
    // and router shapes in their own objects. A field from another level is a mistake worth hearing
    // about.
    require_members(
        config,
        {"hidden_size", "vocab_size", "max_position_embeddings", "tie_word_embeddings",
         "layer_norm_epsilon", "block_configs"},
        {"mamba", "attention", "router", "moe_latent_size", "moe_shared_expert_intermediate_size"},
        "text config");

    TextConfig out;
    out.hidden_size             = dimension(config, "hidden_size");
    out.vocab_size              = dimension(config, "vocab_size");
    out.max_position_embeddings = dimension(config, "max_position_embeddings");
    out.tie_word_embeddings     = config.at("tie_word_embeddings").get<bool>();
    out.rms_norm_eps            = positive_float(config, "layer_norm_epsilon");

    const auto& blocks = config.at("block_configs");
    if (!blocks.is_array() || blocks.empty()) {
        throw ArtifactError("block_configs must be a non-empty array");
    }
    out.block_kinds.reserve(blocks.size());
    out.moe_banks.resize(blocks.size());

    bool saw_mamba     = false;
    bool saw_attention = false;
    for (std::size_t layer = 0; layer < blocks.size(); ++layer) {
        const auto& block = blocks[layer];
        if (!block.is_object()) { throw ArtifactError("block_configs entries must be objects"); }
        require_members(block, {"block_type"},
                        {"n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"},
                        "block_configs entry");
        const BlockKind kind = block_kind(require_id(block.at("block_type"), "block_type"));
        out.block_kinds.push_back(kind);
        switch (kind) {
        case BlockKind::Mamba:
            saw_mamba = true;
            break;
        case BlockKind::Attention:
            saw_attention = true;
            break;
        case BlockKind::Moe:
            out.moe_banks[layer] = moe_bank(block, config);
            break;
        }
    }

    // The checkpoint's own rule: a per-block field is either set on every block that defines it or
    // on none of them. Mixed presence would silently change a layer's shape, so it is refused here.
    for (std::size_t layer = 0; layer < blocks.size(); ++layer) {
        const bool declared = blocks[layer].contains("n_routed_experts");
        const bool is_moe   = out.block_kinds[layer] == BlockKind::Moe;
        if (declared != is_moe) {
            throw ArtifactError(
                "block_configs: MoE fields must be present on exactly the moe blocks (layer " +
                std::to_string(layer) + ")");
        }
        if (!is_moe && out.moe_banks[layer].has_value()) {
            throw ArtifactError("block_configs: a non-moe block carries an expert bank");
        }
    }

    // A shape is required exactly when a block of its kind exists: a model with mamba blocks and no
    // mamba config would otherwise load with zeroed dimensions and compute nothing useful.
    if (saw_mamba) {
        if (!config.contains("mamba")) {
            throw ArtifactError("a mamba block requires the mamba config");
        }
        out.mamba = mamba(config.at("mamba"));
    }
    if (saw_attention) {
        if (!config.contains("attention")) {
            throw ArtifactError("an attention block requires the attention config");
        }
        out.attention = attention(config.at("attention"));
    }
    if (out.moe_layer_count() != 0) {
        if (!config.contains("router")) {
            throw ArtifactError("a moe block requires the router config");
        }
        out.router = router(config.at("router"));
    }
    return out;
}

} // namespace

std::uint32_t TextConfig::moe_layer_count() const noexcept {
    return static_cast<std::uint32_t>(
        std::count(block_kinds.begin(), block_kinds.end(), BlockKind::Moe));
}

const MoeBankConfig& TextConfig::moe_bank(std::uint32_t layer) const {
    if (layer >= moe_banks.size() || !moe_banks[layer].has_value()) {
        throw ArtifactError("layer " + std::to_string(layer) + " has no routed-expert bank");
    }
    return *moe_banks[layer];
}

BlockKind TextConfig::block_kind(std::uint32_t layer) const {
    if (layer >= block_kinds.size()) {
        throw ArtifactError("layer " + std::to_string(layer) + " is outside the block list");
    }
    return block_kinds[layer];
}

Config parse_config(const artifact::Directory& directory) {
    Config out{parse_text(directory.component("text").config)};
    // A draft rides the artifact as its own component, so the tower's duty is to read and validate
    // it when it is there rather than to decide whether it exists.
    if (const auto found = directory.components.find("dflash2");
        found != directory.components.end()) {
        if (found->second.target != "text") { throw ArtifactError("dflash2: target must be text"); }
        out.draft = models::draft::parse_lilicorr(
            found->second.config, out.text.hidden_size, out.text.vocab_size,
            static_cast<std::uint32_t>(out.text.block_kinds.size()));
    } else if (const auto found = directory.components.find("dflash");
               found != directory.components.end()) {
        // A plain DFlash companion: the same tower-side validation, without the correlation head.
        if (found->second.target != "text") { throw ArtifactError("dflash: target must be text"); }
        out.draft = models::draft::parse_dflash(
            found->second.config, out.text.hidden_size, out.text.vocab_size,
            static_cast<std::uint32_t>(out.text.block_kinds.size()));
    }
    return out;
}

} // namespace ninfer::models::nemotron_h
