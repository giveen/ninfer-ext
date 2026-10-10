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
    // window.
    Program(const Model& model, std::int32_t capacity, std::int32_t lanes, DeviceContext& device);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Device bytes the Program allocates for `lanes` lanes of `capacity` tokens, so a caller can size
    // the lanes before constructing one.
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config, std::int32_t capacity,
                                                  std::int32_t lanes);

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

    // Consumes tokens for `lane` and leaves the logits of the last one. The tokens are run in passes of
    // at most kBatch, so a long prompt costs one pass per batch rather than one per token.
    void prefill(std::int32_t lane, std::span<const std::int32_t> ids, DeviceExecutionView execution);

    // Consumes one token for `lane` and leaves its logits.
    void decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution);

    // Draws the token after the one `lane` consumed last from the logits the last call left, with the
    // lane's sampler. The draw is keyed by the position the token will occupy, so it does not depend on
    // how the lanes interleave, and the lane's token counts include it afterwards. Synchronizes.
    [[nodiscard]] std::int32_t sample(std::int32_t lane, DeviceExecutionView execution);

    // Contiguous BF16 [vocabulary,1] logits for the token the last call consumed, already soft-capped
    // by the head. The batch's logits live side by side, so this is a view of the last column.
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

    // The number of tokens one pass covers, which fixes the scratch and logits sizes and the slack the
    // sliding rings carry.
    static constexpr std::int32_t kBatch = 128;

private:
    struct Lane {
        KvCache cache;
        std::int32_t position = 0;
        // Device [vocabulary] I32 occurrence counts the sampler's penalties read.
        void* token_counts = nullptr;
    };

    [[nodiscard]] Lane& lane_at(std::int32_t lane);
    [[nodiscard]] const Lane& lane_at(std::int32_t lane) const;

    // Embeds `tokens` ids starting at `first_position`, runs every layer over them in one pass against
    // `cache`, and applies the head to every column.
    void run_batch(KvCache& cache, const std::int32_t* ids, std::int32_t tokens,
                   std::int32_t first_position, DeviceExecutionView execution);

    const Model* model_ = nullptr;
    std::vector<Lane> lanes_;
    DeviceArena arena_;
    DeviceArena sampling_workspace_;
    Tensor logits_; // [vocabulary, kBatch], one column per token of a pass
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
