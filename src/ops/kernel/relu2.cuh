// Implements: include/ninfer/ops/relu2.h
//
// Match: contiguous BF16 storage. The Op writes each element back where it read it, so the three
// routes below differ only in how many elements a thread handles at once -- a 16-byte pack while
// the tensor stays in the cache-sized regime, BF16 pairs above it, and one element at a time when
// the storage cannot be paired.
#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRelu2PairsPerThread = 4;

// The activation both this Op and the Nemotron-H routed experts apply, kept in one place so the
// two cannot drift: `max(z, 0)^2`, with the rectification before the square, in FP32.
//
// The comparison is written so that a non-positive `z` -- including a zero and either signed zero
// -- takes the zero branch and is squared there, rather than being squared first and clamped
// afterwards: those agree on the value but not on a NaN, which this form sends to zero. Nothing in
// the model feeds this activation a non-finite value, and the branch is the cheaper one.
__device__ __forceinline__ float relu2_one(float value) {
    const float rectified = value > 0.0F ? value : 0.0F;
    return rectified * rectified;
}

template <int Block>
__launch_bounds__(Block) __global__ void relu2_bf16x8_kernel(Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        Bf16x8Pack value = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = __floats2bfloat162_rn(relu2_one(__low2float(value.pair[pair])),
                                                     relu2_one(__high2float(value.pair[pair])));
        }
        store_vec(x + i, value);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void relu2_bf16x2_kernel(__nv_bfloat162* x, std::int64_t pairs, __nv_bfloat16* scalar,
                             bool has_scalar) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * Block + threadIdx.x) * kRelu2PairsPerThread;
#pragma unroll
    for (int item = 0; item < kRelu2PairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) {
            const __nv_bfloat162 value = x[pair];
            x[pair]                    = __floats2bfloat162_rn(relu2_one(__low2float(value)),
                                                               relu2_one(__high2float(value)));
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && has_scalar) {
        *scalar = __float2bfloat16_rn(relu2_one(__bfloat162float(*scalar)));
    }
}

// Templated for the same reason the two above are: this header is the single definition of the
// activation and is included by the experts' kernel as well, so a non-template __global__ here
// would be defined twice in the library.
template <int Block>
__launch_bounds__(Block) __global__ void relu2_scalar_kernel(__nv_bfloat16* x, std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(relu2_one(__bfloat162float(x[i])));
    }
}

} // namespace ninfer::ops
