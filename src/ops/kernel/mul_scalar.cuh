#pragma once

// Implements: include/ninfer/ops/mul_scalar.h
// Match: aligned eight-element domains use one 16-byte pack per thread; the BF16x2 route carries an
// odd tail through its first thread and the scalar route handles two-byte-aligned sliced storage.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMulScalarPairsPerThread = 4;

__device__ __forceinline__ float mul_scalar_one(float value, float factor) { return value * factor; }

__device__ __forceinline__ __nv_bfloat162 mul_scalar_pair(__nv_bfloat162 input, float factor) {
    return __floats2bfloat162_rn(mul_scalar_one(__low2float(input), factor),
                                 mul_scalar_one(__high2float(input), factor));
}

__launch_bounds__(256) __global__
    void mul_scalar_bf16x8_kernel(Bf16x8Pack* x, std::int64_t packs, float factor) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        Bf16x8Pack value = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = mul_scalar_pair(value.pair[pair], factor);
        }
        store_vec(x + i, value);
    }
}

__launch_bounds__(256) __global__
    void mul_scalar_bf16x2_kernel(__nv_bfloat162* x, std::int64_t pairs, __nv_bfloat16* scalar,
                                  bool has_scalar, float factor) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kMulScalarPairsPerThread;
#pragma unroll
    for (int item = 0; item < kMulScalarPairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) { x[pair] = mul_scalar_pair(x[pair], factor); }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && has_scalar) {
        *scalar = __float2bfloat16_rn(mul_scalar_one(__bfloat162float(*scalar), factor));
    }
}

__launch_bounds__(256) __global__ void mul_scalar_kernel(__nv_bfloat16* x, std::int64_t n,
                                                         float factor) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(mul_scalar_one(__bfloat162float(x[i]), factor));
    }
}

} // namespace ninfer::ops
