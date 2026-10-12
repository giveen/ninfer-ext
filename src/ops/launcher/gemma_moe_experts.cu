// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// One CTA per token, with the token's activation row, one selected expert's gate and up vectors,
// the product and the running output row in shared memory. A wider geometry only needs the dynamic
// shared opt-in below.
#include "ops/launcher/gemma_moe_experts.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/gemma_moe_experts.cuh"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads                   = 128;
constexpr std::size_t kStaticSharedBytes = 48 * 1024;

// The kernel's shared layout: the activation row, then the gate, up and product vectors, then the
// accumulator row.
std::size_t expert_shared_bytes(std::int32_t hidden, std::int32_t intermediate) {
    const std::size_t floats = static_cast<std::size_t>(hidden) + 3U * intermediate + hidden;
    return floats * sizeof(float);
}

} // namespace

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const void* gate_up, const void* down, std::int32_t hidden,
                              std::int32_t intermediate, std::int32_t top_k, std::int32_t tokens,
                              void* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    const std::size_t shared = expert_shared_bytes(hidden, intermediate);
    if (shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma_moe_experts_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    }
    gemma_moe_experts_kernel<<<tokens, kThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const std::int32_t*>(expert_ids),
        static_cast<const float*>(expert_weights), static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(down), hidden, intermediate, top_k, tokens,
        static_cast<__nv_bfloat16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
