#pragma once

// Implements: include/ninfer/ops/residual_add_norm.h
// Match: aligned contiguous BF16 `[d, rows]` inputs with `d % 256 == 0` and `d <= 5376`.
//
// The body is `ops::rmsnorm`'s fixed-width CTA route with an accumulate epilogue, and it calls that
// route's own helpers: one 128-thread CTA owns a row, a thread sums the pairs `t, t + Block, ...`
// in ascending order with the same expression, `block_reduce_sum` folds the block, and
// `rmsnorm_epilogue<Plain>` forms the normalized value before its single BF16 rounding. The
// accumulator is then updated through `residual_add_pair` and `mul_scalar_pair` themselves -- the
// very expressions `ops::residual_add` and `ops::mul_scalar` store at, from their shared header --
// so the fused result is bit-identical to the unfused `rmsnorm` -> `residual_add` -> `mul_scalar`
// sequence at the widths the engine calls this Op with rather than merely close to it.

#include "ops/common/bf16_combine.cuh" // residual_add_pair, mul_scalar_pair
#include "ops/common/warp.cuh"         // block_reduce_sum
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// The CTA this route uses, and the unrolled trip counts the launcher instantiates. A row is staged
// in registers, so the count has to cover the row and the launcher picks the smallest that fits: a
// row that exactly fills it has no predication left, and 2816 is then instruction for instruction
// the route `ops::rmsnorm`'s fixed 2816 width takes.
inline constexpr int kResidualAddNormBlock = 128;
// 11 pairs a thread cover 2816 elements (Gemma 4 26B-A4B's hidden); 21 cover 5376 (the 31B's).
inline constexpr int kResidualAddNormPairs2816 = 11;
inline constexpr int kResidualAddNormPairs5376 = 21;

template <int MaxPairs>
__launch_bounds__(kResidualAddNormBlock) __global__
    void residual_add_norm_kernel(const __nv_bfloat162* __restrict__ x,
                                  const __nv_bfloat162* __restrict__ weight,
                                  __nv_bfloat162* __restrict__ accumulator, std::int32_t input_d,
                                  std::int64_t rows, float eps, float scale) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int d                = input_d;
    const int pairs            = d / 2;
    const int pairs_per_thread = pairs / kResidualAddNormBlock;
    const std::int64_t base    = row * static_cast<std::int64_t>(pairs);

    // All three operands are staged in the sum pass: neither the weight nor the accumulator depends
    // on the sum of squares, so issuing them together with x exposes one memory round trip instead
    // of a second one after the reduction. That is what `ops::rmsnorm`'s route does for the weight,
    // and it is worth more than the fusion saves: measured on this card, one row of 2816 costs
    // 4.7 us a call when the weight and accumulator loads wait for the reduction and 1.4 us when
    // they are staged alongside x, against 1.7-2.5 us for the unfused sequence.
    __nv_bfloat162 values[MaxPairs];
    __nv_bfloat162 gains[MaxPairs];
    __nv_bfloat162 accumulators[MaxPairs];
    float sum = 0.0F;
#pragma unroll
    for (int k = 0; k < MaxPairs; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * kResidualAddNormBlock;
            values[k]       = x[base + pair];
            gains[k]        = weight[pair];
            accumulators[k] = accumulator[base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[kResidualAddNormBlock / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<kResidualAddNormBlock>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairs; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * kResidualAddNormBlock;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(gains[k]);
            // The normalized value's one BF16 rounding, then the two `residual_add` and
            // `mul_scalar` store: the sequence the unfused pair performs.
            const __nv_bfloat162 normalized =
                __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Plain>(xf.x, inv, wf.x, 0.0F),
                                      rmsnorm_epilogue<RmsEpilogue::Plain>(xf.y, inv, wf.y, 0.0F));
            accumulator[base + pair] =
                mul_scalar_pair(residual_add_pair(normalized, accumulators[k]), scale);
        }
    }
}

} // namespace ninfer::ops
