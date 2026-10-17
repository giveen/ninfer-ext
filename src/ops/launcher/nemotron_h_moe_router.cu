// Implements: include/ninfer/ops/nemotron_h_moe_router.h
//
// The projection over (expert, token tile) CTAs, then the selection a warp per token. Unlike the
// Gemma router there is no second, split-row projection kernel for a decode token: this Op's
// projection reads the whole weight matrix once per token tile and nothing else, so at T = 1 it is
// already at the bandwidth floor and a split-K variant would only redistribute the same reads.
#include "ops/launcher/nemotron_h_moe_router.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/nemotron_h_moe_router.cuh"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kStaticSharedBytes = 48 * 1024;
constexpr int kSelectWarps               = 8;

} // namespace

void nemotron_h_moe_router_launch(const void* x, const void* router, const void* correction_bias,
                                  std::int32_t hidden, std::int32_t experts, std::int32_t top_k,
                                  std::int32_t groups, std::int32_t groups_used, bool normalize,
                                  float scaling_factor, std::int32_t tokens, float* logits,
                                  void* expert_ids, float* expert_weights, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    const auto* input = static_cast<const __nv_bfloat16*>(x);
    const auto* rows  = static_cast<const __nv_bfloat16*>(router);

    // Only the staged token tile sizes the shared footprint, and a wider row only needs the
    // dynamic-shared opt-in.
    const std::size_t shared =
        static_cast<std::size_t>(kNemotronRouterTokensPerBlock) * hidden * sizeof(float);
    if (shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(nemotron_router_logits_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    }
    const dim3 grid((experts + kNemotronRouterExpertsPerBlock - 1) / kNemotronRouterExpertsPerBlock,
                    (tokens + kNemotronRouterTokensPerBlock - 1) / kNemotronRouterTokensPerBlock);
    nemotron_router_logits_kernel<<<grid, kNemotronRouterExpertsPerBlock * 32, shared, stream>>>(
        input, rows, hidden, experts, tokens, logits);
    CUDA_CHECK(cudaGetLastError());

    nemotron_router_select_kernel<<<(tokens + kSelectWarps - 1) / kSelectWarps, kSelectWarps * 32,
                                    0, stream>>>(
        logits, static_cast<const float*>(correction_bias), experts, top_k, groups, groups_used,
        normalize ? 1 : 0, scaling_factor, tokens, static_cast<std::int32_t*>(expert_ids),
        expert_weights);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
