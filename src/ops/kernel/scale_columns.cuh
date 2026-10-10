#pragma once

// Implements: include/ninfer/ops/scale_columns.h
// Match: one block dimension owns the requested dim range and one row is visited per grid step.
// The registered aligned domains use one 16-byte pack per thread; the BF16x2 and scalar routes
// preserve correctness for ranges and storage that the pack route cannot address.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh" // load_vec, store_vec

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// One pair of the range: the FP32 product of the two represented values, rounded once.
__device__ __forceinline__ __nv_bfloat162 scale_columns_pair(__nv_bfloat162 x, __nv_bfloat162 s) {
    return __floats2bfloat162_rn(__low2float(x) * __low2float(s),
                                 __high2float(x) * __high2float(s));
}

__launch_bounds__(256) __global__
    void scale_columns_bf16x8_kernel(const Bf16x8Pack* scale, Bf16x8Pack* x, std::int32_t begin,
                                     std::int32_t words, std::int64_t row_stride_words,
                                     std::int64_t rows) {
    const std::int32_t word = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= words) { return; }
    for (std::int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::int64_t offset = row * row_stride_words + (begin / 8) + word;
        const Bf16x8Pack xv       = load_vec<Bf16x8Pack>(x + offset);
        const Bf16x8Pack sv       = load_vec<Bf16x8Pack>(scale + (begin / 8) + word);
        Bf16x8Pack out{};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            out.pair[pair] = scale_columns_pair(xv.pair[pair], sv.pair[pair]);
        }
        store_vec(x + offset, out);
    }
}

__launch_bounds__(256) __global__
    void scale_columns_bf16x2_kernel(const __nv_bfloat162* scale, __nv_bfloat162* x,
                                     std::int32_t begin, std::int32_t pairs,
                                     std::int64_t row_stride_pairs, std::int64_t rows) {
    const std::int32_t pair = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) { return; }
    for (std::int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::int64_t offset = row * row_stride_pairs + (begin / 2) + pair;
        x[offset] = scale_columns_pair(x[offset], scale[(begin / 2) + pair]);
    }
}

__launch_bounds__(256) __global__
    void scale_columns_scalar_kernel(const __nv_bfloat16* scale, __nv_bfloat16* x,
                                     std::int32_t begin, std::int32_t length,
                                     std::int64_t row_stride, std::int64_t rows) {
    const std::int32_t index = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= length) { return; }
    const std::int32_t dim = begin + index;
    for (std::int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::int64_t offset = row * row_stride + dim;
        x[offset] = __float2bfloat16_rn(__bfloat162float(x[offset]) *
                                        __bfloat162float(scale[dim]));
    }
}

} // namespace ninfer::ops
