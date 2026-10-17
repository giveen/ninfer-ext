#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "models/draft/draft_spec.h"

// Nemotron-H text configuration, for Nemotron-3.5-Bolt-74B-A9B.
//
// This family is a hybrid: an 88-entry block list interleaves Mamba-2 state-space blocks, LatentMoE
// blocks and GQA attention blocks, and -- unlike every other family here -- the MoE *shape* varies
// per layer as well: 40 MoE layers carry banks of 240, 304, 368, 432 or 496 experts at four
// different intermediate widths. So the FFN is a per-layer list rather than the single `ffn`
// variant the other families use, and the type is a per-layer vector in the same way their
// `layer_types` already is.
//
// The Mamba and attention blocks are each internally identical in this checkpoint, so those are
// single-valued. Facts that are easy to get wrong and are deliberate here:
//
//   * `n_groups` is the number of B/C groups inside the mixer and has nothing to do with the
//   router's
//     `n_group`; both names appear in the checkpoint config with different values (8 and 1).
//   * `conv_bias` is true while the checkpoint's `use_bias` is false, so the depthwise convolution
//     carries a bias even though the projections do not.
//   * only the *minimum* time step clamps: the reference's maximum clamp is commented out and its
//     `time_step_limit` is unused on the torch path.
//   * the experts are not gated -- `down(relu2(up(x)))` -- so a bank holds two matrices per expert.
//   * `rescale_prenorm_residual` is an initialisation rule, not a runtime one, so nothing here
//   models
//     it: the checkpoint already carries the rescaled weights.

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::nemotron_h {

enum class BlockKind { Mamba, Moe, Attention };

// Selective state-space (Mamba-2) block.
struct MambaConfig {
    std::uint32_t num_heads   = 0; // mamba_num_heads
    std::uint32_t head_dim    = 0; // mamba_head_dim
    std::uint32_t state_size  = 0; // ssm_state_size
    std::uint32_t n_groups    = 0; // B/C grouping within the mixer
    std::uint32_t conv_kernel = 0; // depthwise conv1d kernel
    std::uint32_t chunk_size  = 0; // chunked scan length
    float time_step_min       = 0; // softplus floor applied to dt
    bool conv_bias            = false;

    [[nodiscard]] std::uint64_t inner_size() const noexcept {
        return std::uint64_t(num_heads) * head_dim;
    }

    // The convolution sees x, B and C: the mixer's inner width plus one B and one C vector per
    // group.
    [[nodiscard]] std::uint64_t conv_channels() const noexcept {
        return inner_size() + 2ULL * n_groups * state_size;
    }

    // in_proj emits z, x/B/C and dt, in that order. The reference also tolerates a gated-MLP pair
    // ahead of them (it computes their width and discards it), which is zero for this checkpoint --
    // so the projection is exactly these three, and that is asserted at parse time rather than
    // assumed here.
    [[nodiscard]] std::uint64_t in_projection_size() const noexcept {
        return inner_size() + conv_channels() + num_heads;
    }
};

struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    float rope_theta                  = 0;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

// Router shape for this checkpoint. The group machinery exists in the reference but is degenerate
// here: `n_group` is 1 and `topk_group` is 1, so the group mask selects every expert and top-k runs
// over the layer's whole bank. The fields are kept because the reference's rule is the
// specification, and a future checkpoint may set them.
struct RouterConfig {
    float scaling_factor      = 0;     // routed_scaling_factor applied after normalisation
    bool normalize_top_k      = false; // norm_topk_prob
    bool correction_bias      = false; // e_score_correction_bias is present in the checkpoint
    std::uint32_t groups      = 1;     // n_group
    std::uint32_t groups_used = 1;     // topk_group
};

// One MoE layer's bank. Both numbers vary between layers of this checkpoint.
struct MoeBankConfig {
    std::uint32_t num_experts              = 0;
    std::uint32_t experts_per_token        = 0;
    std::uint32_t intermediate_size        = 0;
    std::uint32_t shared_intermediate_size = 0;
    // `fc1_latent_proj`/`fc2_latent_proj` around the experts. Zero means the experts run on the
    // block's own hidden width (the reference makes both projections identities in that case).
    std::uint32_t latent_size = 0;

    [[nodiscard]] bool latent() const noexcept { return latent_size != 0; }

    [[nodiscard]] std::uint32_t expert_input_width(std::uint32_t hidden_size) const noexcept {
        return latent() ? latent_size : hidden_size;
    }

    // Up and down only: the experts are not gated.
    [[nodiscard]] std::uint64_t parameters_per_expert(std::uint32_t hidden_size) const noexcept {
        const std::uint64_t in = expert_input_width(hidden_size);
        return 2ULL * in * intermediate_size;
    }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t max_position_embeddings = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0;

    // One entry per block, in order; this *is* the model's layer count.
    std::vector<BlockKind> block_kinds;
    // Indexed by layer: engaged exactly on the layers whose kind is Moe. The optional is what makes
    // a per-layer bank unrepresentable-by-accident: a non-MoE layer cannot silently carry one.
    std::vector<std::optional<MoeBankConfig>> moe_banks;

    std::optional<MambaConfig> mamba;         // engaged when any layer is Mamba
    std::optional<AttentionConfig> attention; // engaged when any layer is Attention
    RouterConfig router;

    [[nodiscard]] std::uint32_t num_hidden_layers() const noexcept {
        return static_cast<std::uint32_t>(block_kinds.size());
    }

    [[nodiscard]] std::uint32_t moe_layer_count() const noexcept;

    // The bank of an MoE layer. Throws ArtifactError if the layer is not MoE, so a caller that
    // guesses wrong hears about it instead of reading a default.
    [[nodiscard]] const MoeBankConfig& moe_bank(std::uint32_t layer) const;

    [[nodiscard]] BlockKind block_kind(std::uint32_t layer) const;
};

struct Config {
    TextConfig text;
    // The LiLiCorr draft the artifact carries alongside the tower, when it carries one: absent
    // means this artifact has no speculative draft, since nothing supplies weights it does not
    // store.
    std::optional<models::draft::Spec> draft;
};

// Parses the artifact's own text component. Every field this family needs is required or implied by
// what is present, and any input the family cannot represent exactly is refused rather than
// defaulted: a wrong-but-plausible shape would load and compute the wrong model.
[[nodiscard]] Config parse_config(const artifact::Directory& directory);

} // namespace ninfer::models::nemotron_h
