// Preconditions, all established by the wrapper: x and out are contiguous BF16 [H,T], expert_ids is
// I32 [K,T] and expert_weights FP32 [K,T] with 1<=K<=E, both banks carry the geometry's shapes in
// the same format, and `workspace` is 16-byte aligned and holds the bytes the query below names.
#pragma once

#include "core/weight.h"            // QType
#include "ninfer/ops/offload_moe.h" // ExpertBankSpan, ExpertWeights

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One bank as the kernel reads it. A BF16 bank uses `payload`; a low-bit bank adds its planes and,
// when its payload is a stack of separately quantised sources, the divisor of the source each row
// belongs to.
struct NemotronHMoeBankLaunch {
    QType qtype               = QType::BF16;
    const void* payload       = nullptr;
    const void* scales        = nullptr;
    const float* divisors     = nullptr;
    std::int32_t divisor_rows = 0;
    std::int32_t columns      = 0;
};

// The workspace a call over T tokens takes for this format and geometry: none for either landed
// route, because the per-token route keeps the activation row in shared memory and the
// plane-addressed route reads a bank that is already staged.
[[nodiscard]] std::size_t nemotron_h_moe_experts_workspace(QType qtype, std::int32_t latent,
                                                           std::int32_t experts,
                                                           std::int32_t intermediate,
                                                           std::int32_t top_k, std::int32_t tokens);

void nemotron_h_moe_experts_launch(const void* x, const void* expert_ids,
                                   const void* expert_weights, const NemotronHMoeBankLaunch& up,
                                   const NemotronHMoeBankLaunch& down, std::int32_t latent,
                                   std::int32_t experts, std::int32_t intermediate,
                                   std::int32_t top_k, std::int32_t tokens, void* workspace,
                                   void* out, cudaStream_t stream);

// The plane-addressed route: one whole layer's NVFP4 bank at `staged` (planes 0/1 gate-up and 2/3
// down, expert or slot i at `base[p] + i*stride[p]`), with `span.experts` the layer's own expert
// count. `staged` may address a device-resident upload, a staging buffer, the artifact's pinned
// host mapping read over UVA, or the expert cache's slot pool at the pool's own stride; this route
// reads whatever it is given, which is what makes the in-place, the staged and the gathered
// configurations the same code.
void nemotron_h_moe_experts_planar_launch(const void* x, const void* expert_ids,
                                          const void* expert_weights, const ExpertWeights& staged,
                                          ExpertBankSpan span, std::int32_t latent,
                                          std::int32_t intermediate, std::int32_t top_k,
                                          std::int32_t tokens, void* out, cudaStream_t stream);

} // namespace ninfer::ops::detail
