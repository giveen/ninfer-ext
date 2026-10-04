#pragma once

// Device helpers shared by the EXL3 routed-expert kernels (decode and prefill): the 128-point
// Hadamard, half packing and the m16n8k16 FP16 MMA.

#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {

inline constexpr float kExl3InvSqrt128 = 0.08838834764831845F;

// In-register unnormalized 128-point Hadamard: lane holds elements 4*lane..4*lane+3.
__device__ __forceinline__ void fwht128(float v[4], int lane) {
    const float a = v[0], b = v[1], c = v[2], d = v[3];
    const float ab = a + b, amb = a - b, cd = c + d, cmd = c - d;
    v[0] = ab + cd;
    v[1] = amb + cmd;
    v[2] = ab - cd;
    v[3] = amb - cmd;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xFFFFFFFFU, v[j], m);
            v[j]              = (lane & m) ? (other - v[j]) : (v[j] + other);
        }
    }
}

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
    const __half2 value = __floats2half2_rn(lo, hi);
    std::uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__device__ __forceinline__ std::uint32_t load_half2(const __half* pair) {
    return *reinterpret_cast<const std::uint32_t*>(pair);
}

__device__ __forceinline__ void mma_f16(float d[4], const std::uint32_t a[4],
                                        const std::uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

} // namespace ninfer::ops::detail
