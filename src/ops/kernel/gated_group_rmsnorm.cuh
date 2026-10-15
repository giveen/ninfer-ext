#pragma once

// ninfer::ops - gated grouped RMSNorm kernels over contiguous BF16 rows.
//
// The gate is part of the reduction, not an epilogue: the reference (Zamba2RMSNormGated, which
// NemotronHMamba2Mixer uses for its output norm) forms h = x * SiLU(gate) in FP32 and then
// normalizes h within groups, so the gate changes the mean square each group divides by. That is
// what separates this kernel from rmsnorm.cuh's Gated epilogue, which multiplies an
// already-normalized row by its gate and cannot be re-parameterized into this form.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGatedGroupRmsNormBlock = 256;

// One warp owns one (row, group) task, so a group's sum of squares never leaves the warp and the
// traversal does not depend on the group size: a different grouping only changes the task count,
// which is why group_size can stay a runtime argument. A group smaller than the warp leaves lanes
// inactive rather than needing a second reduction path.
//
// `Pair` reads the row as BF16x2 pairs, one pair per lane per step. The launcher selects the scalar
// instantiation for an odd group size or a buffer that is not BF16-pair aligned, where pairing
// would read across the group boundary.
template <bool Pair>
__launch_bounds__(kGatedGroupRmsNormBlock) __global__
    void gated_group_rmsnorm_warp_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                         const __nv_bfloat16* gate, __nv_bfloat16* out,
                                         std::int32_t row_width, std::int32_t group_size,
                                         std::int64_t tasks, float eps) {
    constexpr int kWarpsPerBlock = kGatedGroupRmsNormBlock / kWarpSize;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    // The early return is warp-uniform: the task index depends on the warp, not on the lane, so the
    // shuffle mask below is either full or the warp is gone.
    const std::int64_t task = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (task >= tasks) { return; }

    // Groups partition the row, so one division places the task and the rest is arithmetic. The
    // gain is per column of the full row width, so the weight address is the group's column offset
    // and does not carry the row stride.
    const std::int32_t groups_per_row = row_width / group_size;
    const std::int64_t row            = task / groups_per_row;
    const std::int32_t group          = static_cast<std::int32_t>(task - row * groups_per_row);
    const std::int64_t column         = static_cast<std::int64_t>(group) * group_size;
    const std::int64_t base           = row * static_cast<std::int64_t>(row_width) + column;
    const float inverse_group_size    = 1.0F / static_cast<float>(group_size);

    float sum = 0.0F;
    if constexpr (Pair) {
        const int pairs = group_size / 2;
        for (int pair = lane; pair < pairs; pair += kWarpSize) {
            const std::int64_t offset = base + 2 * static_cast<std::int64_t>(pair);
            const float2 xv           = __bfloat1622float2(load_vec<__nv_bfloat162>(x + offset));
            const float2 gv           = __bfloat1622float2(load_vec<__nv_bfloat162>(gate + offset));
            const float hx            = xv.x * silu(gv.x);
            const float hy            = xv.y * silu(gv.y);
            sum += hx * hx + hy * hy;
        }
    } else {
        for (int element = lane; element < group_size; element += kWarpSize) {
            const std::int64_t offset = base + element;
            const float h = __bfloat162float(x[offset]) * silu(__bfloat162float(gate[offset]));
            sum += h * h;
        }
    }

    const float total = warp_reduce_sum(sum);
    const float inv   = rsqrtf(__shfl_sync(kFullWarpMask, total, 0) * inverse_group_size + eps);

    // x and gate are re-read here rather than staged in registers: the pair count is a runtime
    // quantity, so a register array would have to be a fixed capacity with a runtime trip count.
    if constexpr (Pair) {
        const int pairs = group_size / 2;
        for (int pair = lane; pair < pairs; pair += kWarpSize) {
            // The gain is indexed by column, so this pass carries the group's column offset and the
            // row's element offset separately.
            const std::int64_t pair_offset = 2 * static_cast<std::int64_t>(pair);
            const std::int64_t offset      = base + pair_offset;
            const std::int64_t gain        = column + pair_offset;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + offset));
            const float2 gv = __bfloat1622float2(load_vec<__nv_bfloat162>(gate + offset));
            const float2 wv = __bfloat1622float2(load_vec<__nv_bfloat162>(weight + gain));
            const float hx  = xv.x * silu(gv.x);
            const float hy  = xv.y * silu(gv.y);
            store_vec(out + offset, __floats2bfloat162_rn(wv.x * hx * inv, wv.y * hy * inv));
        }
    } else {
        for (int element = lane; element < group_size; element += kWarpSize) {
            const std::int64_t offset = base + element;
            const float h = __bfloat162float(x[offset]) * silu(__bfloat162float(gate[offset]));
            out[offset] = __float2bfloat16_rn(__bfloat162float(weight[column + element]) * h * inv);
        }
    }
}

} // namespace ninfer::ops
