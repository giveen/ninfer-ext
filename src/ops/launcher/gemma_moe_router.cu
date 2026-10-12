// Implements: include/ninfer/ops/gemma_moe_router.h
//
// The projection over (expert, token) CTAs up to kSplitMaxTokens tokens and over (expert group,
// token tile) CTAs above, then the selection a warp per token. Only the wide projection's shared
// footprint depends on the geometry, and a wider row only needs the dynamic-shared opt-in below.
#include "ops/launcher/gemma_moe_router.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/gemma_moe_router.cuh"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kStaticSharedBytes = 48 * 1024;
constexpr int kSelectWarps               = 8;
// Up to this many tokens a CTA takes one expert of one token and splits its row.
constexpr std::int32_t kSplitMaxTokens = 8;

} // namespace

void gemma_moe_router_launch(const void* x, const void* router, const void* router_scale,
                             const void* per_expert_scale, float eps, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, std::int32_t tokens,
                             float* logits, void* expert_ids, float* expert_weights,
                             cudaStream_t stream) {
    if (tokens <= 0) { return; }
    const auto* input = static_cast<const __nv_bfloat16*>(x);
    const auto* rows  = static_cast<const __nv_bfloat16*>(router);
    const auto* scale = static_cast<const __nv_bfloat16*>(router_scale);
    if (tokens <= kSplitMaxTokens) {
        gemma_moe_router_logits_split_kernel<<<dim3(experts, tokens), kRouterExpertsPerBlock * 32,
                                               0, stream>>>(input, rows, scale, eps, hidden,
                                                            tokens, logits);
    } else {
        const std::size_t shared =
            static_cast<std::size_t>(kRouterTokensPerBlock) * hidden * sizeof(float);
        if (shared > kStaticSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(gemma_moe_router_logits_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(shared)));
        }
        const dim3 grid((experts + kRouterExpertsPerBlock - 1) / kRouterExpertsPerBlock,
                        (tokens + kRouterTokensPerBlock - 1) / kRouterTokensPerBlock);
        gemma_moe_router_logits_kernel<<<grid, kRouterExpertsPerBlock * 32, shared, stream>>>(
            input, rows, scale, eps, hidden, experts, tokens, logits);
    }
    CUDA_CHECK(cudaGetLastError());
    gemma_moe_router_select_kernel<<<(tokens + kSelectWarps - 1) / kSelectWarps, kSelectWarps * 32,
                                     0, stream>>>(
        logits, static_cast<const __nv_bfloat16*>(per_expert_scale), experts, top_k, tokens,
        static_cast<std::int32_t*>(expert_ids), expert_weights);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
