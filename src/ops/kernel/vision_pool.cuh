#pragma once

// Implements: include/ninfer/ops/vision_pool.h
//
// One block per pooled cell; threads stride over the features, each summing its k x k patches in
// FP32. A cell is at most a few hundred patches' worth of one feature, so the reads are short and
// coalesced across the threads of a patch column.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__global__ void vision_pool_standardize_kernel(const __nv_bfloat16* __restrict__ x,
                                               const float* __restrict__ bias,
                                               const float* __restrict__ scale,
                                               __nv_bfloat16* __restrict__ out, int features,
                                               int grid_width, int kernel, float multiplier) {
    const int cell          = static_cast<int>(blockIdx.x);
    const int cells_per_row = grid_width / kernel;
    const int first_column  = (cell % cells_per_row) * kernel;
    const int first_row     = (cell / cells_per_row) * kernel;
    const float mean_scale  = multiplier / static_cast<float>(kernel * kernel);
    for (int d = static_cast<int>(threadIdx.x); d < features; d += static_cast<int>(blockDim.x)) {
        float sum = 0.0F;
        for (int row = first_row; row < first_row + kernel; ++row) {
            for (int column = first_column; column < first_column + kernel; ++column) {
                const std::int64_t patch = static_cast<std::int64_t>(row) * grid_width + column;
                sum += __bfloat162float(x[patch * features + d]);
            }
        }
        out[static_cast<std::int64_t>(cell) * features + d] =
            __float2bfloat16_rn((sum * mean_scale - bias[d]) * scale[d]);
    }
}

} // namespace ninfer::ops::detail
