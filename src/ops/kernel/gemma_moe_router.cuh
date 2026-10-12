// Implements: include/ninfer/ops/gemma_moe_router.h
//
// Two kernels. The projection spreads the router's rows over the machine: a CTA owns a few experts
// for a tile of tokens, recomputes those tokens' weightless RMS scale itself (a row is a few KB,
// far cheaper than a round trip), stages the scaled rows in shared memory and gives each expert a
// warp that streams its weight row in 16-byte vectors. The logits land in the caller's workspace.
// The selection is one warp per token over those logits: the softmax, the K-pass masked argmax and
// the renormalize-then-scale weights.
//
// Ties compare equal floats exactly and resolve to the lower expert id, which is what the
// (value, id) order in `router_better` gives whatever the reduction tree.
#pragma once

#include "ops/common/memory.cuh" // load_vec
#include "ops/common/warp.cuh"   // warp_reduce_sum

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kRouterExpertsPerBlock = 4;  // one warp each
inline constexpr int kRouterTokensPerBlock  = 4;  // the scaled rows staged per CTA
inline constexpr int kRouterMaxExperts      = 256; // eight logits a lane in the selection warp
inline constexpr int kRouterMaxRanks        = 16;

// The (value, id) pair a selection pass reduces to. The comparison is the whole tie-break contract.
struct RouterBest {
    float value;
    int id;
};

__device__ __forceinline__ RouterBest router_better(RouterBest a, RouterBest b) {
    if (b.value > a.value) { return b; }
    if (b.value < a.value) { return a; }
    return b.id < a.id ? b : a;
}

__device__ __forceinline__ RouterBest router_warp_argmax(RouterBest mine) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        RouterBest other;
        other.value = __shfl_xor_sync(0xffffffffu, mine.value, offset);
        other.id    = __shfl_xor_sync(0xffffffffu, mine.id, offset);
        mine        = router_better(mine, other);
    }
    return mine;
}

// Every lane gets the total, which `warp_reduce_sum` leaves in lane 0 only.
__device__ __forceinline__ float router_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float router_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

/**
 * Grid (ceil(E / kRouterExpertsPerBlock), ceil(T / kRouterTokensPerBlock)), kRouterExpertsPerBlock
 * warps. `shared` holds kRouterTokensPerBlock rows of `hidden` FP32 scaled activations. `hidden` is
 * a multiple of 8 and every row is 16-byte aligned. logits is FP32 [E,T], token fastest.
 */
__global__ void gemma_moe_router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                               const __nv_bfloat16* __restrict__ router,
                                               const __nv_bfloat16* __restrict__ router_scale,
                                               float eps, int hidden, int experts, int tokens,
                                               float* __restrict__ logits) {
    extern __shared__ __align__(16) float scaled[];
    __shared__ float inverse[kRouterTokensPerBlock];
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int token_base = static_cast<int>(blockIdx.y) * kRouterTokensPerBlock;
    const int live       = min(kRouterTokensPerBlock, tokens - token_base);

    // 1. The weightless RMS of each staged token, a warp per token: the unit-RMS row, then
    //    router_scale and H^-1/2. The norm carries no weight and no gain of its own.
    if (warp < live) {
        const __nv_bfloat16* row = x + static_cast<std::int64_t>(token_base + warp) * hidden;
        float squares            = 0.0F;
        for (int h = lane; h < hidden; h += 32) {
            const float value = __bfloat162float(row[h]);
            squares += value * value;
        }
        squares = warp_reduce_sum(squares);
        if (lane == 0) {
            inverse[warp] = rsqrtf(squares / static_cast<float>(hidden) + eps) *
                            rsqrtf(static_cast<float>(hidden));
        }
    }
    __syncthreads();
    for (int index = tid; index < live * hidden; index += blockDim.x) {
        const int t = index / hidden;
        const int h = index - t * hidden;
        scaled[index] =
            __bfloat162float(x[static_cast<std::int64_t>(token_base + t) * hidden + h]) *
            inverse[t] * __bfloat162float(router_scale[h]);
    }
    __syncthreads();

    // 2. The projection: a warp per expert, eight weights a lane a step, every staged token at
    //    once, accumulated in FP32.
    const int expert = static_cast<int>(blockIdx.x) * kRouterExpertsPerBlock + warp;
    if (expert >= experts) { return; }
    const __nv_bfloat16* weights_row = router + static_cast<std::int64_t>(expert) * hidden;
    float accumulator[kRouterTokensPerBlock] = {};
    for (int k = lane * 8; k < hidden; k += 32 * 8) {
        const uint4 packed = load_vec<uint4>(weights_row + k);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
        float w[8];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 value = __bfloat1622float2(pairs[pair]);
            w[2 * pair]        = value.x;
            w[2 * pair + 1]    = value.y;
        }
#pragma unroll
        for (int t = 0; t < kRouterTokensPerBlock; ++t) {
            if (t < live) {
                const float4 s0 = *reinterpret_cast<const float4*>(scaled + t * hidden + k);
                const float4 s1 = *reinterpret_cast<const float4*>(scaled + t * hidden + k + 4);
                float sum       = accumulator[t];
                sum             = fmaf(w[0], s0.x, sum);
                sum             = fmaf(w[1], s0.y, sum);
                sum             = fmaf(w[2], s0.z, sum);
                sum             = fmaf(w[3], s0.w, sum);
                sum             = fmaf(w[4], s1.x, sum);
                sum             = fmaf(w[5], s1.y, sum);
                sum             = fmaf(w[6], s1.z, sum);
                sum             = fmaf(w[7], s1.w, sum);
                accumulator[t]  = sum;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kRouterTokensPerBlock; ++t) {
        const float total = warp_reduce_sum(accumulator[t]);
        if (lane == 0 && t < live) {
            logits[static_cast<std::int64_t>(expert) * tokens + token_base + t] = total;
        }
    }
}

/**
 * The few-token projection: grid (E, T), kRouterExpertsPerBlock warps. A CTA is one expert of one
 * token, and its warps split the row's K, so a decode token spreads over E CTAs rather than
 * E / kRouterExpertsPerBlock. Each CTA takes its token's weightless RMS itself.
 */
__global__ void gemma_moe_router_logits_split_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ router,
                                                     const __nv_bfloat16* __restrict__ router_scale,
                                                     float eps, int hidden, int tokens,
                                                     float* __restrict__ logits) {
    __shared__ float partial[2][kRouterExpertsPerBlock];
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int expert = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const __nv_bfloat16* row         = x + static_cast<std::int64_t>(token) * hidden;
    const __nv_bfloat16* weights_row = router + static_cast<std::int64_t>(expert) * hidden;

    // One pass over the CTA's share of the row: the sum of squares for the RMS, and the dot of the
    // scaled activation, which the RMS factor multiplies once at the end.
    float squares = 0.0F;
    float dot     = 0.0F;
    for (int k = tid * 8; k < hidden; k += kRouterExpertsPerBlock * 32 * 8) {
        const uint4 xv    = load_vec<uint4>(row + k);
        const uint4 wv    = load_vec<uint4>(weights_row + k);
        const uint4 sv    = load_vec<uint4>(router_scale + k);
        const auto* xp    = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const auto* wp    = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const auto* sp    = reinterpret_cast<const __nv_bfloat162*>(&sv);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 xf = __bfloat1622float2(xp[p]);
            const float2 wf = __bfloat1622float2(wp[p]);
            const float2 sf = __bfloat1622float2(sp[p]);
            squares         = fmaf(xf.x, xf.x, squares);
            squares         = fmaf(xf.y, xf.y, squares);
            dot             = fmaf(wf.x, xf.x * sf.x, dot);
            dot             = fmaf(wf.y, xf.y * sf.y, dot);
        }
    }
    squares = router_warp_sum(squares);
    dot     = router_warp_sum(dot);
    if (lane == 0) {
        partial[0][warp] = squares;
        partial[1][warp] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float total_squares = 0.0F;
        float total_dot     = 0.0F;
#pragma unroll
        for (int w = 0; w < kRouterExpertsPerBlock; ++w) {
            total_squares += partial[0][w];
            total_dot += partial[1][w];
        }
        const float inverse = rsqrtf(total_squares / static_cast<float>(hidden) + eps) *
                              rsqrtf(static_cast<float>(hidden));
        logits[static_cast<std::int64_t>(expert) * tokens + token] = total_dot * inverse;
    }
}

/**
 * One warp per token over logits FP32 [E,T]; E <= kRouterMaxExperts and K <= kRouterMaxRanks.
 */
__global__ void gemma_moe_router_select_kernel(const float* __restrict__ logits,
                                               const __nv_bfloat16* __restrict__ per_expert_scale,
                                               int experts, int top_k, int tokens,
                                               std::int32_t* __restrict__ ids,
                                               float* __restrict__ weights) {
    constexpr int kPerLane = kRouterMaxExperts / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int token =
        static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) +
        (static_cast<int>(threadIdx.x) >> 5);
    if (token >= tokens) { return; }

    // Lane l holds experts l, l + 32, ...: ascending ids within a lane, so a slot's id is its own.
    float p[kPerLane];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int slot = 0; slot < kPerLane; ++slot) {
        const int e = lane + 32 * slot;
        p[slot]     = e < experts ? logits[static_cast<std::int64_t>(e) * tokens + token]
                                  : -CUDART_INF_F;
        maximum     = fmaxf(maximum, p[slot]);
    }
    maximum     = router_warp_max(maximum);
    float total = 0.0F;
#pragma unroll
    for (int slot = 0; slot < kPerLane; ++slot) {
        const int e = lane + 32 * slot;
        p[slot]     = e < experts ? __expf(p[slot] - maximum) : 0.0F;
        total += p[slot];
    }
    total = router_warp_sum(total);
#pragma unroll
    for (int slot = 0; slot < kPerLane; ++slot) { p[slot] /= total; }

    // The K ranks, each a masked argmax over the probabilities. A taken slot drops to -inf, below
    // every probability, so it cannot win again; an absent expert never could.
    RouterBest chosen[kRouterMaxRanks];
    for (int rank = 0; rank < top_k; ++rank) {
        RouterBest mine{-CUDART_INF_F, experts};
#pragma unroll
        for (int slot = 0; slot < kPerLane; ++slot) {
            const int e = lane + 32 * slot;
            if (e < experts) { mine = router_better(mine, RouterBest{p[slot], e}); }
        }
        chosen[rank] = router_warp_argmax(mine);
        if ((chosen[rank].id & 31) == lane) {
#pragma unroll
            for (int slot = 0; slot < kPerLane; ++slot) {
                if (lane + 32 * slot == chosen[rank].id) { p[slot] = -CUDART_INF_F; }
            }
        }
    }

    // Renormalize the selected probabilities to sum to one, then apply the per-expert scale. The
    // order is the contract: the scale multiplies an already-normalized weight.
    if (lane == 0) {
        float sum = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) { sum += chosen[rank].value; }
        const float inverse_sum = sum > 0.0F ? 1.0F / sum : 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert                                       = chosen[rank].id;
            ids[static_cast<std::int64_t>(rank) * tokens + token] = expert;
            weights[static_cast<std::int64_t>(rank) * tokens + token] =
                chosen[rank].value * inverse_sum * __bfloat162float(per_expert_scale[expert]);
        }
    }
}

} // namespace ninfer::ops::detail
