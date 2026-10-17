#pragma once

// A dedicated NVFP4 A16 route for widths the chunked SIMT route's phase tiling cannot take.
//
// The shared route walks K in phases of `32 * ValuesPerLane` (256 at the narrowest), which is why a
// K of 2688 -- 168 groups of 16 -- has no phase tile it divides. This route walks K in single
// 16-value groups instead: one E4M3 block scale, one 8-byte code pack and sixteen BF16 activations
// per step, so its only constraint is `K % 16 == 0`. The Lightning 30B DFlash companion's gate/up
// projection is the shape it exists for.
//
// Arithmetic is the Open's A16 contract for a dense route: one FP32 accumulator per (row, token),
// the 16 values of a group in ascending K, one BF16 rounding at the store, and the bank's FP32
// divisor folded into the group's scale. The op test checks it against the FP64 oracle.
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/memory.cuh"
#include "ops/linear/bf16/bf16_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Tokens a lane carries: the weight pack is read once and reused across them, which is what keeps
// the route's traffic proportional to the matrix rather than to the matrix times the block width.
inline constexpr int kNvfp4GroupTokensPerThread = 2;

template <class Geometry, int TokensPerThread = kNvfp4GroupTokensPerThread,
          int ThreadsPerCta = 32>
__global__ __launch_bounds__(ThreadsPerCta) void nvfp4_group_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_divisor,
    __nv_bfloat16* __restrict__ out, int tokens) {
    static_assert((Geometry::kInputRows % 16) == 0);
    constexpr int kGroups     = Geometry::kInputRows / 16;
    constexpr int kScaleTiles = Geometry::kInputRows / 64;

    const int row = static_cast<int>(blockIdx.y) * ThreadsPerCta + threadIdx.x;
    if (row >= Geometry::kOutputRows) { return; }
    const int token0 = static_cast<int>(blockIdx.x) * TokensPerThread;
    const int last   = tokens - 1;

    const __nv_bfloat16* token_base[TokensPerThread];
#pragma unroll
    for (int slot = 0; slot < TokensPerThread; ++slot) {
        token_base[slot] =
            x + static_cast<std::int64_t>(min(token0 + slot, last)) * Geometry::kInputRows;
    }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * Geometry::kCodeBytesPerRow;

    float accumulator[TokensPerThread] = {};
    for (int group = 0; group < kGroups; ++group) {
        const uint2 packed = load_vec<uint2>(code_row + group * 8);
        const float scale =
            decode_nvfp4_e4m3(scales[nvfp4_scale_byte<kScaleTiles>(row, group)]) * inverse_divisor;
        Bf16GemvPack<16> activation[TokensPerThread];
#pragma unroll
        for (int slot = 0; slot < TokensPerThread; ++slot) {
            activation[slot] = load_bf16_pack<16>(token_base[slot] + group * 16);
        }
        const std::uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(words[byte / 4] >> (8 * (byte % 4))));
            const float2 weight = make_float2(code.x * scale, code.y * scale);
            // Byte `byte` carries values {2 * byte, 2 * byte + 1}, which are word `byte` of the
            // activation pack: two BF16 a word, low half first.
#pragma unroll
            for (int slot = 0; slot < TokensPerThread; ++slot) {
                const __nv_bfloat162 bits =
                    *reinterpret_cast<const __nv_bfloat162*>(&activation[slot].words[byte]);
                const float2 values = __bfloat1622float2(bits);
                accumulator[slot] =
                    fmaf(weight.x, values.x, fmaf(weight.y, values.y, accumulator[slot]));
            }
        }
    }

#pragma unroll
    for (int slot = 0; slot < TokensPerThread; ++slot) {
        const int token = token0 + slot;
        if (token < tokens) {
            out[static_cast<std::int64_t>(token) * Geometry::kOutputRows + row] =
                __float2bfloat16_rn(accumulator[slot]);
        }
    }
}

template <class Geometry, int TokensPerThread = kNvfp4GroupTokensPerThread>
void launch_nvfp4_group_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { return; }
    const dim3 grid(static_cast<unsigned>((tokens + TokensPerThread - 1) / TokensPerThread),
                    static_cast<unsigned>((Geometry::kOutputRows + 31) / 32));
    nvfp4_group_gemv_kernel<Geometry, TokensPerThread><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales),
        1.0F / weight.weight_scale_divisor, static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
