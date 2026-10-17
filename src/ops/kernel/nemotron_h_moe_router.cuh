// Implements: include/ninfer/ops/nemotron_h_moe_router.h
//
// Two kernels. The projection spreads the router's rows over the machine: a CTA owns a few experts
// for a tile of tokens, stages those tokens' rows in shared memory as FP32 and gives each expert a
// warp that streams its weight row in 16-byte vectors. The selection is one warp per token over
// those logits: the sigmoid, the correction bias, the group rule and the masked rank sweep.
//
// The staging is FP32 rather than BF16 because every staged element is read by each of the CTA's
// expert warps, so decoding once beats decoding E/warps times, and because the decode is exact:
// a BF16 value is an FP32 value, and the assembled sum is the FP32 one the reference forms.
//
// Ties compare equal floats exactly and resolve to the lower id -- of the group or of the expert --
// which is what the (value, id) order in `nemotron_router_better` gives whatever the reduction
// tree.
#pragma once

#include "ops/common/math.cuh"   // sigmoid
#include "ops/common/memory.cuh" // load_vec
#include "ops/common/warp.cuh"   // warp_sum

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kNemotronRouterExpertsPerBlock = 4;  // one warp each
inline constexpr int kNemotronRouterTokensPerBlock  = 4;  // the rows staged per CTA
inline constexpr int kNemotronRouterMaxSlots        = 16; // E <= 512, so 16 logits a lane
inline constexpr int kNemotronRouterMaxRanks        = 16;
inline constexpr int kNemotronRouterMaxGroups       = 32; // one group per lane in the ranking pass

// The (value, id) pair a sweep reduces to. The comparison is the whole tie-break contract, and it
// is why the group and entry sweeps below can be re-associated freely.
struct NemotronRouterBest {
    float value;
    int id;
};

__device__ __forceinline__ NemotronRouterBest nemotron_router_better(NemotronRouterBest a,
                                                                     NemotronRouterBest b) {
    if (b.value > a.value) { return b; }
    if (b.value < a.value) { return a; }
    return b.id < a.id ? b : a;
}

__device__ __forceinline__ NemotronRouterBest nemotron_router_argmax(NemotronRouterBest mine) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        NemotronRouterBest other;
        other.value = __shfl_xor_sync(0xffffffffu, mine.value, offset);
        other.id    = __shfl_xor_sync(0xffffffffu, mine.id, offset);
        mine        = nemotron_router_better(mine, other);
    }
    return mine;
}

// The largest entry in [begin, end) other than `exclude`, over the lanes' own slots. Every lane
// ends with the same answer, which is what lets the group ranking read its own score out of a lane.
template <int Slots>
__device__ __forceinline__ NemotronRouterBest nemotron_router_entry_argmax(
    const float (&score)[Slots], int lane, int begin, int end, int exclude) {
    NemotronRouterBest mine{-CUDART_INF_F, end};
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
        const int expert = lane + 32 * slot;
        if (expert >= begin && expert < end && expert != exclude) {
            mine = nemotron_router_better(mine, NemotronRouterBest{score[slot], expert});
        }
    }
    return nemotron_router_argmax(mine);
}

/**
 * Grid (ceil(E / kNemotronRouterExpertsPerBlock), ceil(T / kNemotronRouterTokensPerBlock)),
 * kNemotronRouterExpertsPerBlock warps. `staged` holds kNemotronRouterTokensPerBlock rows of
 * `hidden` FP32 activations. `hidden` is a multiple of 8, so every 8-element weight vector a lane
 * reads is in bounds and x's token rows are contiguous.
 */
__global__ void nemotron_router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ router, int hidden,
                                              int experts, int tokens, float* __restrict__ logits) {
    extern __shared__ __align__(16) float staged[];
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int token_base = static_cast<int>(blockIdx.y) * kNemotronRouterTokensPerBlock;
    const int live       = min(kNemotronRouterTokensPerBlock, tokens - token_base);

    // 1. The tile's activations, decoded once. x is [H,T] with the token axis the slower one, so a
    //    tile of live tokens is one contiguous run.
    for (int index = tid; index < live * hidden; index += kNemotronRouterExpertsPerBlock * 32) {
        staged[index] = __bfloat162float(x[static_cast<std::int64_t>(token_base) * hidden + index]);
    }
    __syncthreads();

    // 2. The projection: a warp per expert, eight weights a lane a step, every staged token at
    // once,
    //    accumulated in FP32.
    const int expert = static_cast<int>(blockIdx.x) * kNemotronRouterExpertsPerBlock + warp;
    if (expert >= experts) { return; }
    const __nv_bfloat16* weights_row = router + static_cast<std::int64_t>(expert) * hidden;
    float accumulator[kNemotronRouterTokensPerBlock] = {};
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
        for (int t = 0; t < kNemotronRouterTokensPerBlock; ++t) {
            if (t < live) {
                const float4 s0 = *reinterpret_cast<const float4*>(staged + t * hidden + k);
                const float4 s1 = *reinterpret_cast<const float4*>(staged + t * hidden + k + 4);
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
    for (int t = 0; t < kNemotronRouterTokensPerBlock; ++t) {
        const float total = warp_reduce_sum(accumulator[t]);
        if (lane == 0 && t < live) {
            logits[static_cast<std::int64_t>(expert) * tokens + token_base + t] = total;
        }
    }
}

/**
 * One warp per token over logits FP32 [E,T]; E <= 512 (kNemotronRouterMaxSlots a lane), K and the
 * group count within kNemotronRouterMaxRanks and kNemotronRouterMaxGroups.
 */
__global__ void nemotron_router_select_kernel(const float* __restrict__ logits,
                                              const float* __restrict__ correction_bias,
                                              int experts, int top_k, int groups, int groups_used,
                                              int normalize, float scaling_factor, int tokens,
                                              std::int32_t* __restrict__ ids,
                                              float* __restrict__ weights) {
    constexpr int kSlots = kNemotronRouterMaxSlots;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int token      = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) +
                           (static_cast<int>(threadIdx.x) >> 5);
    if (token >= tokens) { return; }

    // Lane l holds experts l, l + 32, ...: ascending ids within a lane, so a slot's id is its own.
    // `score` is the corrected score the selection ranks, `uncorrected` the sigmoid the emitted
    // weight is taken from; an absent expert is -inf in both so it can never win a sweep.
    float score[kSlots];
    float uncorrected[kSlots];
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
        const int expert = lane + 32 * slot;
        if (expert < experts) {
            const float logit = logits[static_cast<std::int64_t>(expert) * tokens + token];
            const float s     = sigmoid(logit);
            uncorrected[slot] = s;
            score[slot]       = s + correction_bias[expert];
        } else {
            uncorrected[slot] = -CUDART_INF_F;
            score[slot]       = -CUDART_INF_F;
        }
    }

    // The group rule. Groups tile the bank in expert-id order, so a group is a range of ids and its
    // score is the sum of the two largest corrected scores inside it. Lane g keeps group g's score,
    // which is what the ranking pass below then ranks.
    const int width   = experts / groups; // exact: the wrapper requires E % groups == 0
    float group_score = -CUDART_INF_F;
    for (int group = 0; group < groups; ++group) {
        const int begin = group * width;
        const int end   = begin + width;
        // -1 excludes nothing, so the first sweep is the group's maximum and the second is the
        // maximum without it: the two largest.
        const NemotronRouterBest first = nemotron_router_entry_argmax(score, lane, begin, end, -1);
        float sum                      = first.value;
        if (width > 1) {
            sum += nemotron_router_entry_argmax(score, lane, begin, end, first.id).value;
        }
        if (lane == group) { group_score = sum; }
    }

    // `groups_used` sweeps over the group scores, each dropping the group it took, so the selected
    // groups are the U largest with the lower id winning a tie; the wrapper's groups_used <= groups
    // is what guarantees an unselected group survives every sweep. The selected set is a bitmask,
    // which is what makes the group test below one shift.
    std::uint32_t selected_groups = 0U;
    for (int rank = 0; rank < groups_used; ++rank) {
        NemotronRouterBest mine{-CUDART_INF_F, groups};
        if (lane < groups && ((selected_groups >> lane) & 1U) == 0U) {
            mine.value = group_score;
            mine.id    = lane;
        }
        selected_groups |= 1U << nemotron_router_argmax(mine).id;
    }
    // One bit a slot: the group test is a shift of a lane-varying index, so it is done once here
    // rather than in each of the top_k sweeps below, and kSlots bits fit a register.
    std::uint32_t eligible = 0U;
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
        const int expert = lane + 32 * slot;
        if (expert < experts && ((selected_groups >> (expert / width)) & 1U) != 0U) {
            eligible |= 1U << slot;
        }
    }

    // The K ranks, each a masked argmax over the corrected scores. A taken slot drops to -inf,
    // below every candidate, so it cannot win again; one outside the selected groups never could.
    NemotronRouterBest chosen[kNemotronRouterMaxRanks];
    for (int rank = 0; rank < top_k; ++rank) {
        NemotronRouterBest mine{-CUDART_INF_F, experts};
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            if (((eligible >> slot) & 1U) != 0U) {
                mine =
                    nemotron_router_better(mine, NemotronRouterBest{score[slot], lane + 32 * slot});
            }
        }
        chosen[rank] = nemotron_router_argmax(mine);
        if ((chosen[rank].id & 31) == lane) {
#pragma unroll
            for (int slot = 0; slot < kSlots; ++slot) {
                if (lane + 32 * slot == chosen[rank].id) { score[slot] = -CUDART_INF_F; }
            }
        }
    }

    // The emitted weight is the *uncorrected* sigmoid at each chosen id, so it has to be gathered
    // from the slot that owns that expert. The wrapper bounds top_k by the experts the selected
    // groups can supply, so every chosen id here is a real expert.
    float emitted[kNemotronRouterMaxRanks];
    for (int rank = 0; rank < top_k; ++rank) {
        const int id = chosen[rank].id;
        float mine   = 0.0F;
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            if (lane + 32 * slot == id) { mine = uncorrected[slot]; }
        }
        emitted[rank] = warp_sum(mine);
    }

    // Renormalize and then scale, in the reference's order: the divisor is the sum of the emitted
    // weights plus 1e-20, and the scale multiplies the normalized weight.
    if (lane == 0) {
        float sum = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) { sum += emitted[rank]; }
        for (int rank = 0; rank < top_k; ++rank) {
            float weight = emitted[rank];
            if (normalize != 0) { weight /= sum + 1e-20F; }
            ids[static_cast<std::int64_t>(rank) * tokens + token]     = chosen[rank].id;
            weights[static_cast<std::int64_t>(rank) * tokens + token] = weight * scaling_factor;
        }
    }
}

} // namespace ninfer::ops::detail
