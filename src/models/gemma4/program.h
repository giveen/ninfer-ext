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

    // Consumes prompt tokens in order and leaves the logits of the last one.
    void prefill(std::span<const std::int32_t> ids, DeviceExecutionView execution);

    // Consumes one token and leaves its logits.
    void decode(std::int32_t id, DeviceExecutionView execution);

    // Contiguous BF16 [vocabulary,1] logits for the token the last call consumed, already soft-capped
    // by the head. Sampling is the caller's business.
    [[nodiscard]] const Tensor& logits() const noexcept { return logits_; }

private:
    // Embeds `id`, runs every layer at `position`, and applies the head.
    void run_token(std::int32_t id, std::int32_t position, DeviceExecutionView execution);

    const Model* model_ = nullptr;
    KvCache cache_;
    DeviceArena arena_;
    Tensor logits_;
    void* id_buffer_    = nullptr;
    void* state_[2]     = {nullptr, nullptr};
    std::int32_t position_ = 0;
    std::int32_t capacity_ = 0;
};

} // namespace ninfer::models::gemma4
