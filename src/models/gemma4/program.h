#pragma once

// The decoder state and execution of up to `lanes` independent sequences, composed from the layer
// forward.
//
// The Program owns everything mutable: one KV cache and one sampler state per lane, and the scratch
// arena, token, hidden and logits buffers the lanes share, since a pass runs one lane at a time. It
// shares no device allocation with any other Program, and the execution view is supplied per call,
// so a Program is bound to streaming but not to a stream. Every allocation is made at construction:
// a request never discovers half way through that its cache does not fit.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/model.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::gemma4 {

class Program {
public:
    // `capacity` bounds the tokens one lane's global layers can hold; sliding layers always hold the
    // window. `draft_tokens` > 0 enables MTP rounds of up to that many drafts, which needs the drafter
    // loaded, and gives the sliding rings that much slack.
    Program(const Model& model, std::int32_t capacity, std::int32_t lanes, DeviceContext& device,
            std::int32_t draft_tokens = 0);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Device bytes the Program allocates for `lanes` lanes of `capacity` tokens, so a caller can size
    // the lanes before constructing one.
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config, std::int32_t capacity,
                                                  std::int32_t lanes, std::int32_t draft_tokens = 0);

    // The largest draft count a round may use, zero without speculation.
    [[nodiscard]] std::int32_t draft_tokens() const noexcept { return draft_tokens_; }
    static constexpr std::int32_t kMaximumDraftTokens = 7;

    // The outcome of one speculative round, before the caller decides how much of it to keep.
    struct Round {
        // The licensed tokens: the accepted drafts, then the target's correction or bonus token.
        std::vector<std::int32_t> tokens;
        std::int32_t accepted_drafts = 0;
    };

    /**
     * One MTP round for `lane`, whose last accepted token `anchor` the target has not consumed yet.
     * The drafter proposes `drafts` tokens greedily from the anchor and the target's hidden state that
     * produced it; the target then runs [anchor, drafts] in one pass, and the lane's sampler accepts
     * the longest prefix its distribution licenses (exactly the greedy chain for a greedy sampler,
     * speculative rejection sampling otherwise) plus one token of its own. Synchronizes. Nothing is
     * committed until `commit_round`.
     */
    [[nodiscard]] Round speculate(std::int32_t lane, std::int32_t anchor, std::int32_t drafts,
                                  DeviceExecutionView execution);

    // Keeps the first `kept` tokens of `lane`'s last round (1 to its token count): the target has then
    // consumed the anchor and the first kept-1 of them, and the last kept token is the next anchor.
    // The rows written for the rest are rolled back.
    void commit_round(std::int32_t lane, std::int32_t kept, DeviceExecutionView execution);

    [[nodiscard]] std::int32_t lanes() const noexcept {
        return static_cast<std::int32_t>(lanes_.size());
    }
    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept { return allocated_bytes_; }

    // Forgets every token of `lane` and installs its sampler, keeping the allocations. The sampler's
    // token counts start empty.
    void reset(std::int32_t lane, const ops::SamplingConfig& sampling = {});

    // Tokens `lane` has consumed, which is also the position its next token occupies.
    [[nodiscard]] std::int32_t position(std::int32_t lane) const { return lane_at(lane).position; }

    // The lane's cache, read-only, for checks of what it holds.
    [[nodiscard]] const KvCache& cache(std::int32_t lane) const { return lane_at(lane).cache; }

    // Consumes tokens for `lane` and leaves the logits of the last one. The tokens are run in passes of
    // at most kPass, so a long prompt costs one pass per kPass tokens rather than one per token.
    void prefill(std::int32_t lane, std::span<const std::int32_t> ids, DeviceExecutionView execution);

    // Consumes one token for `lane` and leaves its logits.
    void decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution);

    // Draws the token after the one `lane` consumed last from the logits the last call left, with the
    // lane's sampler. The draw is keyed by the position the token will occupy, so it does not depend on
    // how the lanes interleave, and the lane's token counts include it afterwards. Synchronizes.
    [[nodiscard]] std::int32_t sample(std::int32_t lane, DeviceExecutionView execution);

    // Contiguous BF16 [vocabulary,1] logits for the token the last call consumed, already soft-capped
    // by the head. A scoring pass's logits live side by side, so this is a view of the last column.
    [[nodiscard]] Tensor logits() const noexcept {
        const std::size_t column = static_cast<std::size_t>(last_batch_ - 1);
        return Tensor(static_cast<std::uint8_t*>(logits_.data) +
                          column * static_cast<std::size_t>(logits_.ne[0]) * sizeof(std::uint16_t),
                      DType::BF16, {logits_.ne[0], 1});
    }

    /**
     * Scores `ids` causally on lane 0, from an empty sequence, and returns the natural
     * log-probability of each target token: one entry for every position from `first_target` on, in
     * order. The probability is taken from the soft-capped logits the head produced for the preceding
     * position, which is the distribution the model itself would sample from, and it is normalized
     * over the whole vocabulary in double precision.
     *
     * `first_target` must be at least 1 and less than the token count. The lane is reset first, so a
     * Program can score many windows in a row.
     */
    [[nodiscard]] std::vector<float> causal_score(std::span<const std::int32_t> ids,
                                                  std::int32_t first_target, const LogitsSink& sink,
                                                  DeviceExecutionView execution);

    // The most tokens one pass covers, which fixes the layer scratch. Prefill applies the head to the
    // pass's last token only; scoring needs every column's logits and runs passes of kScorePass,
    // which fixes the logits buffer.
    static constexpr std::int32_t kPass      = 512;
    static constexpr std::int32_t kScorePass = 128;

private:
    struct Lane {
        KvCache cache;
        std::int32_t position = 0;
        // Device [vocabulary] I32 occurrence counts the sampler's penalties read.
        void* token_counts = nullptr;
        // With speculation: device BF16 [hidden], the target's post-final-norm state at the token
        // before the anchor, which the drafter's first step reads.
        void* hidden = nullptr;
        // The round speculate() left for commit_round(): its first position and draft count.
        std::int32_t round_position = -1;
        std::int32_t round_drafts   = 0;
    };

    [[nodiscard]] Lane& lane_at(std::int32_t lane);
    [[nodiscard]] const Lane& lane_at(std::int32_t lane) const;

    // Embeds `tokens` ids starting at `first_position`, runs every layer over them in one pass against
    // `cache`, and applies the head to every column, or to the last one only unless `every_column`.
    void run_batch(KvCache& cache, const std::int32_t* ids, std::int32_t tokens,
                   std::int32_t first_position, bool every_column, DeviceExecutionView execution);
    // The same pass over ids already in the device id buffer.
    void run_pass(KvCache& cache, std::int32_t tokens, std::int32_t first_position,
                  bool every_column, DeviceExecutionView execution);
    // Copies column `column` of the last pass's post-final-norm hidden state into the lane's.
    void keep_hidden(Lane& lane, std::int32_t column, cudaStream_t stream);

    const Model* model_ = nullptr;
    std::vector<Lane> lanes_;
    DeviceArena arena_;
    DeviceArena sampling_workspace_;
    Tensor logits_; // [vocabulary, kScorePass], one column per token of a scoring pass
    // [hidden, kScorePass]: the post-final-norm state of every column the head ran over.
    Tensor head_hidden_;
    std::int32_t draft_tokens_ = 0;
    // The drafter's buffers: its input [2 * hidden], output state [hidden], logits [vocabulary], and
    // the round's I32 scalars and vectors (see speculate()).
    Tensor draft_input_, draft_hidden_, draft_logits_;
    void* round_ints_ = nullptr;
    std::vector<std::int32_t> round_host_;
    std::vector<std::int32_t> id_host_;
    void* id_buffer_       = nullptr;
    void* state_[2]        = {nullptr, nullptr};
    void* sampling_[3]     = {nullptr, nullptr, nullptr}; // configs [lanes], token [1], position [1]
    // Host staging for one pass's logits, which scoring reads back per target token.
    std::vector<std::uint16_t> logits_host_;
    std::int32_t last_batch_      = 1;
    std::int32_t capacity_        = 0;
    std::size_t allocated_bytes_  = 0;
};

} // namespace ninfer::models::gemma4
