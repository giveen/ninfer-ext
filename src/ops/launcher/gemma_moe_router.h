// Preconditions, all established by the wrapper: x is contiguous BF16 [H,T] with T>=1, the three
// weight pointers are BF16 matrices/vectors of the geometry's sizes, expert_ids is I32 [K,T] and
// expert_weights FP32 [K,T] with 1<=K<=16 and K<=E. Only the shared-memory footprint depends on the
// geometry, and the launcher sizes it.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void gemma_moe_router_launch(const void* x, const void* router, const void* router_scale,
                             const void* per_expert_scale, float eps, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, std::int32_t tokens,
                             void* expert_ids, float* expert_weights, cudaStream_t stream);

} // namespace ninfer::ops::detail
