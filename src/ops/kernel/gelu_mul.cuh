#pragma once

// Implements: include/ninfer/ops/gelu_mul.h
// Match: paired contiguous storage takes one BF16x2 per thread; the general kernel walks the four
// logical axes so strided halves of a fused projection need no copy of their own.

#include "ops/common/math.cuh"
#include "ops/kernel/gelu.cuh" // gelu_one<true>: the same tanh-GELU the gelu Op uses

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float gelu_mul_one(float gate, float up) {
    return gelu_one<true>(gate) * up;
}

__device__ __forceinline__ __nv_bfloat162 gelu_mul_pair(__nv_bfloat162 gate, __nv_bfloat162 up) {
    return __floats2bfloat162_rn(gelu_mul_one(__low2float(gate), __low2float(up)),
                                 gelu_mul_one(__high2float(gate), __high2float(up)));
}

__launch_bounds__(256) __global__
    void gelu_mul_kernel(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
                         __nv_bfloat16* __restrict__ out, std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t pairs  = n / 2;
    const auto* gate2         = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2           = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2                = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t i = start; i < pairs; i += stride) { out2[i] = gelu_mul_pair(gate2[i], up2[i]); }
    if (start == 0 && (n & 1) != 0) {
        const std::int64_t last = n - 1;
        out[last] = __float2bfloat16_rn(
            gelu_mul_one(__bfloat162float(gate[last]), __bfloat162float(up[last])));
    }
}

__launch_bounds__(256) __global__ void gelu_mul_strided_kernel(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out, std::int64_t n, std::int32_t ne0, std::int32_t ne1,
    std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1, std::int64_t gnb2, std::int64_t gnb3,
    std::int64_t unb0, std::int64_t unb1, std::int64_t unb2, std::int64_t unb3) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t gate_offset =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t up_offset =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gate_value = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + gate_offset);
        const auto up_value   = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + up_offset);
        out[i] = __float2bfloat16_rn(
            gelu_mul_one(__bfloat162float(gate_value), __bfloat162float(up_value)));
    }
}

} // namespace ninfer::ops
