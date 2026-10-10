#pragma once

// Gemma 4 31B (text) configuration, read from the artifact's own config.
//
// The mathematics is in docs/maintainer/gemma4-model.md. This is the immutable shape the loader and
// the Program consume, and it is deliberately strict: every field the converter writes is named
// here, an unknown member is an error, and the invariants the mathematics depends on -- the layer
// pattern, the two attention geometries, proportional RoPE, a plain norm -- are checked rather than
// assumed.

#include "artifact/schema.h"
#include "models/registry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {

// One attention geometry. The sliding layers use `TextConfig::sliding`, the global layers
// `TextConfig::global`, and nothing else differs between them.
struct AttentionGeometry {
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    float rope_theta                  = 0.0F;
    // Rotating dimensions: the whole head for sliding layers, the leading block of the pair
    // rotation for global ones.
    std::uint32_t rotary_dim = 0;
};

struct GlobalAttentionGeometry {
    // The global layers store no value projection: K is V, normalized without a weight and scaled
    // by the key norm afterwards, so the value vector serves both q·kᵀ and the output.
    AttentionGeometry shared;
    // Proportional RoPE: `rope_angles` frequencies taken against the full head width, so
    // `rotary_dim == 2 * rope_angles` dims rotate and the rest carry no rotation at all.
    std::uint32_t rope_angles           = 0;
    std::uint32_t denominator_head_dim  = 0;
};

enum class MixerKind : std::uint8_t { SlidingAttention, FullAttention };

struct TextConfig {
    Architecture architecture             = Architecture::Gemma4;
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    // Query heads. The artifact's config names it and the loader's invariants used it, but nothing
    // downstream could read it until the weight bindings needed the query projection's row count.
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t sliding_window          = 0;
    std::uint32_t global_layers           = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0.0F;
    float embedding_scale                 = 0.0F;
    float attention_scale                 = 0.0F;
    float final_logit_softcapping         = 0.0F;
    std::string hidden_act;
    std::vector<MixerKind> layer_types;
    AttentionGeometry sliding;
    GlobalAttentionGeometry global;

    [[nodiscard]] bool sliding_attention(std::size_t layer) const {
        return layer_types.at(layer) == MixerKind::SlidingAttention;
    }

    // The geometry layer `layer` attends with.
    [[nodiscard]] const AttentionGeometry& geometry(std::size_t layer) const {
        return sliding_attention(layer) ? sliding : global.shared;
    }
};

// The assistant drafter (the artifact's `mtp` component). It has no key or value projections: each of
// its layers attends to the target's cache of the same kind, the target's last sliding layer and its
// last global layer, so its attention geometry is the target's and only its widths are its own.
struct DraftConfig {
    std::uint32_t hidden_size          = 0;
    std::uint32_t intermediate_size    = 0;
    std::uint32_t num_hidden_layers    = 0;
    std::uint32_t backbone_hidden_size = 0;
    float rms_norm_eps                 = 0.0F;
    std::vector<MixerKind> layer_types;
    // The target layers the drafter's sliding and global layers read.
    std::uint32_t target_sliding_layer = 0;
    std::uint32_t target_global_layer  = 0;
};

// The image encoder (the artifact's `vision` component): a ViT over 16-pixel patches whose soft
// tokens, 3x3-pooled and standardized, are projected into the text width. The converter stores
// every head's query and key rows with the middle two 18-dim blocks swapped, so the 2-D RoPE Op's
// layout applies.
struct VisionConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t pooling_kernel_size     = 0;
    std::uint32_t position_embedding_size = 0;
    std::uint32_t soft_tokens_per_image   = 0;
    std::uint32_t output_hidden_size      = 0;
    float rms_norm_eps                    = 0.0F;
    float rope_theta                      = 0.0F;
    std::int32_t image_token_id           = 0;
    std::int32_t boi_token_id             = 0;
    std::int32_t eoi_token_id             = 0;

    // The most patches one image has: its soft tokens times the pooling cell.
    [[nodiscard]] std::uint32_t max_patches() const noexcept {
        return soft_tokens_per_image * pooling_kernel_size * pooling_kernel_size;
    }
};

struct Config {
    TextConfig text;
};

// Parses one text component config. Throws ArtifactError on an unknown member or a violated
// invariant, so a checkpoint this build cannot execute is refused at load rather than misread.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);

// Parses the `mtp` component against the target it drafts for, with the same strictness.
[[nodiscard]] DraftConfig parse_draft_config(const artifact::Json& value, const TextConfig& target);

// Parses the `vision` component against the text width it projects into.
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value, const TextConfig& text);

} // namespace ninfer::models::gemma4
