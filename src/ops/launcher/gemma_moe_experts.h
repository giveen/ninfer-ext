// Preconditions, all established by the wrapper: x and out are contiguous BF16 [H,T], expert_ids is
// I32 [K,T] and expert_weights FP32 [K,T] with 1<=K<=E, and both banks are BF16 at the geometry's
// expert-major shapes. Only the shared footprint depends on the geometry, and the launcher sizes
// it.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const void* gate_up, const void* down, std::int32_t hidden,
                              std::int32_t intermediate, std::int32_t top_k, std::int32_t tokens,
                              void* out, cudaStream_t stream);

} // namespace ninfer::ops::detail
