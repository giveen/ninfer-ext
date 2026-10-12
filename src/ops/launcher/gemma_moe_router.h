// Preconditions, all established by the wrapper: x is contiguous BF16 [H,T] with T>=1 and H a
// multiple of 8, the three weight pointers are 16-byte aligned BF16 matrices/vectors of the
// geometry's sizes, logits is FP32 [E,T] of caller workspace, expert_ids is I32 [K,T] and
// expert_weights FP32 [K,T] with 1<=K<=16, K<=E and E<=256.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void gemma_moe_router_launch(const void* x, const void* router, const void* router_scale,
                             const void* per_expert_scale, float eps, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, std::int32_t tokens,
                             float* logits, void* expert_ids, float* expert_weights,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
