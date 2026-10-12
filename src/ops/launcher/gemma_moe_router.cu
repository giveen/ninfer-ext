// Implements: include/ninfer/ops/gemma_moe_router.h
//
// One CTA per token: the admitted geometry puts a token's whole hidden row and expert logit vector
// in shared memory, and a wider row only needs the dynamic-shared opt-in below.
#include "ops/launcher/gemma_moe_router.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/gemma_moe_router.cuh"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

// A power of two, so the block reductions halve cleanly, and at least the kernel's 16-rank bound.
constexpr int kThreads                   = 128;
constexpr std::size_t kStaticSharedBytes = 48 * 1024;

// The kernel's shared layout: [hidden] scaled activations, [experts] probabilities, [experts] bytes
// of selection flags rounded to whole floats, one RouterBest per thread, one float per thread.
std::size_t router_shared_bytes(std::int32_t hidden, std::int32_t experts) {
    const std::size_t floats = static_cast<std::size_t>(hidden) +
                               static_cast<std::size_t>(experts) +
                               static_cast<std::size_t>(experts + 3) / 4U + 3U * kThreads;
    return floats * sizeof(float);
}

} // namespace

void gemma_moe_router_launch(const void* x, const void* router, const void* router_scale,
                             const void* per_expert_scale, float eps, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, std::int32_t tokens,
                             void* expert_ids, float* expert_weights, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    const std::size_t shared = router_shared_bytes(hidden, experts);
    if (shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma_moe_router_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    }
    gemma_moe_router_kernel<<<tokens, kThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(router),
        static_cast<const __nv_bfloat16*>(router_scale),
        static_cast<const __nv_bfloat16*>(per_expert_scale), eps, hidden, experts, top_k, tokens,
        static_cast<std::int32_t*>(expert_ids), expert_weights);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
