#pragma once

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::draft {

/**
 * The LiLiCorr speculative draft's stored configuration.
 *
 * The draft is a DFlash2-style block drafter -- five sliding-attention layers over a set of the
 * target tower's residual taps -- followed by the correlation head that reorders its per-position
 * candidates. This is the *artifact's* reading of that: the converter pins the release (five
 * layers, six taps, mask token 990, rotation 1e4) and this validates the stored representation
 * against the tower it drafts for, which is the loader's obligation. Duplicating the release's
 * constants here would make a second place own them.
 *
 * The member sets are exact, per level: the top level, `dflash_config`, and the head's own keys are
 * each validated as a closed set, so a field from another level is a refusal rather than a field
 * this parser silently ignores. The head's parameters are nested rather than flattened because they
 * belong to the projector, not to the body.
 */
struct Spec {
    // Which drafter this is. The two share a six-layer qwen3 body over the tower's residual taps
    // and differ in what surrounds it: a plain DFlash (what Qwen, Gemma 4 and the 30B Lightning
    // companions carry) has no convolution, no attention sink and no head of its own -- it proposes
    // through the tower's output head and its block length is the round's choice. LiLiCorr (the
    // 74B's `dflash2`) adds the grouped convolution, the sink and the correlation head, and stores
    // the block length it was trained with. Keeping the flavour explicit is what lets one execution
    // path serve both without either guessing from a zero field.
    enum class Flavor : std::uint8_t { Lilicorr, Dflash };

    Flavor flavor                         = Flavor::Lilicorr;
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t num_key_value_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t block_size              = 0;
    std::uint32_t sliding_window          = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id    = 0;
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t conv_group_size  = 0;

    struct Head {
        std::uint32_t candidate_top_k = 0;
        std::uint32_t hidden_size     = 0;
        std::uint32_t num_layers      = 0;
        std::uint32_t num_heads       = 0;
        float mlp_ratio               = 0;
        std::uint32_t factor_dim      = 0;
        float vector_eps              = 0;
        float logit_scale             = 0;
    } head;

    /** The block's slot count, which is also the draft length a round proposes. */
    [[nodiscard]] std::uint32_t draft_length() const noexcept { return block_size; }
};

/**
 * Parses the `dflash2` component's config for a LiLiCorr draft over one target tower.
 *
 * `target_hidden`, `target_vocab` and `target_blocks` are the tower's own values: the draft's input
 * width is the tower's residual width, its vocabulary is the tower's, and its taps are absolute
 * block indices into the tower. A stored config that disagrees with any of them, or that declares a
 * member set other than the one this flavor stores, throws `artifact::ArtifactError`.
 */
Spec parse_lilicorr(const artifact::Json& value, std::uint32_t target_hidden,
                    std::uint32_t target_vocab, std::uint32_t target_blocks);


/**
 * Parses a plain `dflash` component's config -- the block-diffusion drafter without the LiLiCorr
 * correlation head.
 *
 * The member set is the converter's own output (`draft_config` in `tools/convert/qwen3_5.py`): the
 * tower's width and vocabulary, the drafter's six layers, rope at its base theta and the tap list.
 * There are no conv keys and no head, because a plain DFlash has neither; `block_size` stays zero
 * because a plain draft's length is a per-round choice rather than a stored geometry.
 */
Spec parse_dflash(const artifact::Json& value, std::uint32_t target_hidden,
                  std::uint32_t target_vocab, std::uint32_t target_blocks);

} // namespace ninfer::models::draft
