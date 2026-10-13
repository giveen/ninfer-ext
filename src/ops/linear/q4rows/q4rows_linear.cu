// The int4 group-32 row format (q4_g32_fp16_rows in row_group_v1) as a linear weight.
//
// Two routes, both over any N and K that the layout and the tile admit, with no shape registry:
// the format's rows are unpadded and its decode is one nibble and one binary16 scale a value, so
// nothing about a geometry needs hand tuning to be correct, and the converter's QAT grid recovery
// is what makes the format worth reading at all.
//
//   * up to kQ4RowsSimtMaxTokens tokens, a warp per output row streams the row's groups once and
//     applies each decoded group to the activations, accumulating in FP32;
//   * above that, the 64 x 64 BF16 tensor-core tile of q4_rows_mma.cuh, one CTA per (64 rows,
//     64 tokens), with the weights decoded to BF16 in shared memory.
#include "ops/linear/q4rows/q4rows_dispatch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/q4_rows_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Output rows a warp owns: one. A decode GEMV's grid is then one CTA per kSimtWarps rows, so the
// machine fills with warps instead of a few hundred CTAs: the 26B's projections have 64-512 CTAs
// at four rows a warp, which leaves one CTA on an SM and the kernel latency-bound, and one row a
// warp measures 4.8% of end-to-end decode on this card. A row's arithmetic does not depend on which
// block covers it, so the two forms are bit-identical. Each lane takes whole 32-value scale groups
// as one 16-byte load a row, and a lane's strided group range keeps several of the row's loads in
// flight; the warp's 32 lanes cover 32 of the row's groups at a time.
constexpr int kSimtRows       = 1;
constexpr int kSimtWarps      = 4;
constexpr int kSimtRowsPerCta = kSimtRows * kSimtWarps;

// Eight codes of one row group (one 32-bit word, element 2i in the low nibble of byte i, stored as
// code + 8) times their binary16 scale. 1024 + n in binary16 minus 1032 is exactly n - 8.
__device__ __forceinline__ void decode_q4_rows_eight(std::uint32_t packed, float scale,
                                                     float (&weights)[8]) {
    const __half2 bias = __half2half2(__ushort_as_half(0x6408));
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const float2 values =
            __half22float2(__hsub2(*reinterpret_cast<const __half2*>(&bits), bias));
        weights[pair]     = values.x * scale;
        weights[pair + 4] = values.y * scale;
    }
}

template <int Tokens>
__global__ __launch_bounds__(kSimtWarps * 32) void q4rows_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, int n, int k) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = (static_cast<int>(blockIdx.x) * kSimtWarps + warp) * kSimtRows;
    if (row0 >= n) { return; }
    const int groups = k / 32;

    float acc[kSimtRows][Tokens] = {};
    for (int group = lane; group < groups; group += 32) {
        uint4 packed[kSimtRows];
        float scale[kSimtRows];
#pragma unroll
        for (int r = 0; r < kSimtRows; ++r) {
            const std::int64_t row = row0 + r;
            packed[r]              = load_vec<uint4>(codes + row * (k / 2) + group * 16);
            scale[r]               = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(scales + (row * (k / 32) + group) * 2)));
        }
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            const int column = group * 32 + chunk * 8;
            float values[Tokens][8];
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                const uint4 input = load_vec<uint4>(x + static_cast<std::int64_t>(t) * k + column);
                const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&input);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 value   = __bfloat1622float2(pairs[p]);
                    values[t][2 * p]     = value.x;
                    values[t][2 * p + 1] = value.y;
                }
            }
#pragma unroll
            for (int r = 0; r < kSimtRows; ++r) {
                const std::uint32_t word = chunk == 0   ? packed[r].x
                                           : chunk == 1 ? packed[r].y
                                           : chunk == 2 ? packed[r].z
                                                        : packed[r].w;
                float weights[8];
                decode_q4_rows_eight(word, scale[r], weights);
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[r][t] = fmaf(weights[i], values[t][i], acc[r][t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kSimtRows; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float total = acc[r][t];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_xor_sync(0xffffffffu, total, offset);
            }
            if (lane == 0) {
                out[static_cast<std::int64_t>(t) * n + row0 + r] = __float2bfloat16_rn(total);
            }
        }
    }
}

__global__ __launch_bounds__(kQ4RowsTileThreads) void q4rows_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, int n, int k,
    int tokens) {
    __shared__ __align__(16) __nv_bfloat16 smem[kQ4RowsTileSmemElements];
    const int tid          = static_cast<int>(threadIdx.x);
    const int row0         = static_cast<int>(blockIdx.x) * kQ4RowsTileM;
    const int col0         = static_cast<int>(blockIdx.y) * kQ4RowsTileN;
    const std::int64_t row = row0 + (tid >> 1);

    const __nv_bfloat16* b_rows[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int column = min(col0 + (tid >> 3) + 16 * c, tokens - 1);
        b_rows[c]        = x + static_cast<std::int64_t>(column) * k;
    }
    float acc[2][4][4];
    q4_rows_tile_mainloop(codes + row * (k / 2), scales + row * (k / 32) * 2, b_rows,
                          k / kQ4RowsTileK, smem, acc);
    const float* const c_tile = q4_rows_tile_to_shared(acc, smem);

    // Thirty-two output rows a thread for its column.
    const int column = col0 + (tid >> 1);
    if (column >= tokens) { return; }
    const int r0 = (tid & 1) * 32;
    __align__(16) __nv_bfloat16 values[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        values[r] = __float2bfloat16_rn(c_tile[(r0 + r) * kQ4RowsTileCStride + (tid >> 1)]);
    }
    __nv_bfloat16* const target = out + static_cast<std::int64_t>(column) * n + row0 + r0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        store_vec(target + v * 8, *reinterpret_cast<const uint4*>(values + v * 8));
    }
}

template <int Tokens>
void launch_simt(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                 __nv_bfloat16* out, int n, int k, cudaStream_t stream) {
    q4rows_simt_kernel<Tokens>
        <<<(n + kSimtRowsPerCta - 1) / kSimtRowsPerCta, kSimtWarps * 32, 0, stream>>>(
            x, codes, scales, out, n, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4rows_launch(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                   __nv_bfloat16* out, std::int32_t n, std::int32_t k, std::int32_t tokens,
                   cudaStream_t stream) {
    switch (tokens) {
    case 1:
        launch_simt<1>(x, codes, scales, out, n, k, stream);
        return;
    case 2:
        launch_simt<2>(x, codes, scales, out, n, k, stream);
        return;
    case 3:
        launch_simt<3>(x, codes, scales, out, n, k, stream);
        return;
    case 4:
        launch_simt<4>(x, codes, scales, out, n, k, stream);
        return;
    case 5:
        launch_simt<5>(x, codes, scales, out, n, k, stream);
        return;
    case 6:
        launch_simt<6>(x, codes, scales, out, n, k, stream);
        return;
    case 7:
        launch_simt<7>(x, codes, scales, out, n, k, stream);
        return;
    case 8:
        launch_simt<8>(x, codes, scales, out, n, k, stream);
        return;
    default:
        break;
    }
    const dim3 grid(n / kQ4RowsTileM, (tokens + kQ4RowsTileN - 1) / kQ4RowsTileN);
    q4rows_mma_kernel<<<grid, kQ4RowsTileThreads, 0, stream>>>(x, codes, scales, out, n, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
