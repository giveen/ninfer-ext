#pragma once

// One sequence's decoder state and execution, composed from the layer forward.
//
// The Program owns everything mutable: the KV cache, the per-layer scratch arena, the token and
// hidden buffers, and the logits buffer. It shares no device allocation with any other Program, and
// the model and the execution view are supplied per call, so a Program is bound to streaming but not
// to a stream.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/model.h"

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::gemma4 {

class Program {
public:
    // `capacity` bounds the tokens the global layers can hold; sliding layers always hold the window.
    Program(const Model& model, std::int32_t capacity, DeviceContext& device);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Forgets every token, keeping the allocations.
    void reset();

    // Tokens consumed so far, which is also the position the next one occupies.
    [[nodiscard]] std::int32_t position() const noexcept { return position_; }

    // Consumes prompt tokens and leaves the logits of the last one. The tokens are run in batches, so
    // a long prompt costs one pass per batch rather than one per token.
    void prefill(std::span<const std::int32_t> ids, DeviceExecutionView execution);

    // Consumes one token and leaves its logits.
    void decode(std::int32_t id, DeviceExecutionView execution);

    // Contiguous BF16 [vocabulary,1] logits for the token the last call consumed, already soft-capped
    // by the head. Sampling is the caller's business. The batch's logits live side by side, so this is
    // a view of the last column.
    [[nodiscard]] Tensor logits() const noexcept {
        const std::size_t column = static_cast<std::size_t>(last_batch_ - 1);
        return Tensor(static_cast<std::uint8_t*>(logits_.data) +
                          column * static_cast<std::size_t>(logits_.ne[0]) * sizeof(std::uint16_t),
                      DType::BF16, {logits_.ne[0], 1});
    }

    /**
     * Scores `ids` causally, from an empty sequence, and returns the natural log-probability of each
     * target token: one entry for every position from `first_target` on, in order. The probability is
     * taken from the soft-capped logits the head produced for the preceding position, which is the
     * distribution the model itself would sample from, and it is normalized over the whole vocabulary
     * in double precision.
     *
     * `first_target` must be at least 1 and less than the token count. The sequence is reset first, so
     * a Program can score many windows in a row.
     */
    [[nodiscard]] std::vector<float> causal_score(std::span<const std::int32_t> ids,
                                                  std::int32_t first_target, const LogitsSink& sink,
                                                  DeviceExecutionView execution);

private:
    // Embeds `tokens` ids starting at `first_position`, runs every layer over them in one pass, and
    // applies the head to every column.
    void run_batch(const std::int32_t* ids, std::int32_t tokens, std::int32_t first_position,
                   DeviceExecutionView execution);

    // The number of tokens one pass covers, which fixes the scratch and logits sizes.
    static constexpr std::int32_t kBatch = 128;

    const Model* model_ = nullptr;
    KvCache cache_;
    DeviceArena arena_;
    Tensor logits_;      // [vocabulary, kBatch], one column per token of a pass
    std::vector<std::int32_t> id_host_;
    void* id_buffer_    = nullptr;
    void* state_[2]     = {nullptr, nullptr};
    // Host staging for one position's logits, which scoring reads back per target token.
    std::vector<std::uint16_t> logits_host_;
    std::int32_t position_   = 0;
    std::int32_t last_batch_ = 1;
    std::int32_t capacity_ = 0;
};

} // namespace ninfer::models::gemma4
