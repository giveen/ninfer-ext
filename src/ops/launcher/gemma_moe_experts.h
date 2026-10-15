// Preconditions, all established by the wrapper: x and out are contiguous BF16 [H,T], expert_ids is
// I32 [K,T] and expert_weights FP32 [K,T] with 1<=K<=E, both banks carry the geometry's shapes in
// the same format, and `workspace` is 16-byte aligned and holds the bytes the query below names.
#pragma once

#include "core/weight.h" // QType

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One bank as the kernel reads it. A BF16 bank uses `payload`; an NVFP4 bank uses `payload`
// (codes), `scales` and the divisor of each source matrix in `divisors` over `divisor_rows` rows;
// an int4 group-32 row bank uses `payload` (codes) and `scales`.
struct GemmaMoeBankLaunch {
    QType qtype               = QType::BF16;
    const void* payload       = nullptr;
    const void* scales        = nullptr;
    const float* divisors     = nullptr;
    std::int32_t divisor_rows = 0;
    std::int32_t columns      = 0;
};

// The workspace a call over T tokens takes for this format and geometry: the streamed route's FP32
// products, the grouped route's route maps and BF16 intermediates, or nothing for the per-token
// route.
[[nodiscard]] std::size_t gemma_moe_experts_workspace(QType qtype, std::int32_t hidden,
                                                      std::int32_t experts,
                                                      std::int32_t intermediate,
                                                      std::int32_t top_k, std::int32_t tokens);

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                              std::int32_t hidden, std::int32_t experts, std::int32_t intermediate,
                              std::int32_t top_k, std::int32_t tokens, void* workspace, void* out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
