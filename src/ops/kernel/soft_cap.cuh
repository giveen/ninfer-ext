#pragma once

// Implements: include/ninfer/ops/soft_cap.h
// Match: contiguous BF16 storage. Aligned eight-element domains use one 16-byte pack per thread;
// the BF16x2 route carries an odd tail through its first thread and the scalar route handles
// two-byte-aligned sliced storage.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSoftCapPairsPerThread = 4;

__device__ __forceinline__ float soft_cap_one(float value, float cap) {
    return cap * tanhf(value / cap);
}

__device__ __forceinline__ __nv_bfloat162 soft_cap_pair(__nv_bfloat162 input, float cap) {
    return __floats2bfloat162_rn(soft_cap_one(__low2float(input), cap),
                                 soft_cap_one(__high2float(input), cap));
}

__launch_bounds__(256) __global__
    void soft_cap_bf16x8_kernel(Bf16x8Pack* x, std::int64_t packs, float cap) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        Bf16x8Pack value = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = soft_cap_pair(value.pair[pair], cap);
        }
        store_vec(x + i, value);
    }
}

__launch_bounds__(256) __global__
    void soft_cap_bf16x2_kernel(__nv_bfloat162* x, std::int64_t pairs, __nv_bfloat16* scalar,
                                bool has_scalar, float cap) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kSoftCapPairsPerThread;
#pragma unroll
    for (int item = 0; item < kSoftCapPairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) { x[pair] = soft_cap_pair(x[pair], cap); }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && has_scalar) {
        *scalar = __float2bfloat16_rn(soft_cap_one(__bfloat162float(*scalar), cap));
    }
}

__launch_bounds__(256) __global__ void soft_cap_scalar_kernel(__nv_bfloat16* x, std::int64_t n,
                                                              float cap) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(soft_cap_one(__bfloat162float(x[i]), cap));
    }
}

} // namespace ninfer::ops
