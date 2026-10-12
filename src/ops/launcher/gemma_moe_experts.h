// Preconditions, all established by the wrapper: x and out are contiguous BF16 [H,T], expert_ids is
// I32 [K,T] and expert_weights FP32 [K,T] with 1<=K<=E, and both banks carry the geometry's shapes
// in the same format. Only the shared footprint depends on the geometry, and the launcher sizes it.
#pragma once

#include "core/weight.h" // QType

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One bank as the kernel reads it. A BF16 bank uses `payload`; an NVFP4 bank uses `payload`
// (codes), `scales` and the divisor of each source matrix in `divisors` over `divisor_rows` rows.
struct GemmaMoeBankLaunch {
    QType qtype               = QType::BF16;
    const void* payload       = nullptr;
    const void* scales        = nullptr;
    const float* divisors     = nullptr;
    std::int32_t divisor_rows = 0;
    std::int32_t columns      = 0;
};

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                              std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k,
                              std::int32_t tokens, void* out, cudaStream_t stream);

} // namespace ninfer::ops::detail
