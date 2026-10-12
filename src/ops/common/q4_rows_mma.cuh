#pragma once

// The BF16 tensor-core tile over an int4 group-32 row matrix (q4_g32_fp16_rows in row_group_v1):
// a 64 x 64 output tile of 64 weight rows against 64 activation columns, accumulated in FP32 by
// four warps (2 x 2, each 32 x 32) with m16n8k16. The weights are decoded `code * binary16 scale`
// to BF16 into shared memory; the activations arrive as BF16 rows through cp.async. Each K tile is
// 64 wide, two scale groups, and the pipeline holds two stages.
//
// The caller decides which weight row each staging thread reads and which activation row each B
// chunk comes from, so the same tile serves a dense projection and a routed expert bank.

#include "ops/common/memory.cuh" // load_vec, store_vec
#include "ops/common/mma.cuh"    // ldmatrix_x4, mma_bf16

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kQ4RowsTileM       = 64;
inline constexpr int kQ4RowsTileN       = 64;
inline constexpr int kQ4RowsTileK       = 64;
inline constexpr int kQ4RowsTileThreads = 128;
inline constexpr int kQ4RowsTileStride  = kQ4RowsTileK + 8; // padded smem row, in BF16
// Two stages of A and of B.
inline constexpr int kQ4RowsTileSmemElements = 2 * 2 * kQ4RowsTileM * kQ4RowsTileStride;
// The FP32 epilogue tile [M][N + 4] reuses the same storage.
inline constexpr int kQ4RowsTileCStride = kQ4RowsTileN + 4;
static_assert(kQ4RowsTileM * kQ4RowsTileCStride * 4 <= kQ4RowsTileSmemElements * 2);

__device__ __forceinline__ unsigned q4_rows_shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void q4_rows_cp_async_16(void* shared, const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     q4_rows_shared_address(shared)),
                 "l"(global));
}

__device__ __forceinline__ void q4_rows_cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Thirty-two codes of one row group (16 bytes, element 2i in the low nibble of byte i, stored as
// code + 8) times their binary16 scale, as BF16 into shared memory.
__device__ __forceinline__ void decode_q4_rows_group(uint4 packed, std::uint16_t scale_bits,
                                                     __nv_bfloat16* out) {
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const std::uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        __align__(16) __nv_bfloat162 pairs[4];
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const std::uint32_t b = (w[word] >> (8 * byte)) & 0xffu;
            pairs[byte]           = __floats2bfloat162_rn(
                static_cast<float>(static_cast<int>(b & 0x0fu) - 8) * scale,
                static_cast<float>(static_cast<int>(b >> 4) - 8) * scale);
        }
        store_vec(out + word * 8, *reinterpret_cast<const uint4*>(pairs));
    }
}

/**
 * The K loop of one 64 x 64 tile. Thread t stages weight row t / 2, group half t % 2 of each K
 * tile, from `a_codes` / `a_scales` (that row's first code byte and first scale word); and the B
 * chunks of activation rows (t / 8) + 16c, c = 0..3, from `b_rows[c]` (that row's first element),
 * 16 bytes at (t % 8) * 8 along K. `k_tiles` is K / 64. On return `smem` is free and the caller's
 * FP32 accumulators hold the tile in mma C-fragment order.
 */
__device__ __forceinline__ void q4_rows_tile_mainloop(const std::uint8_t* __restrict__ a_codes,
                                                      const std::uint8_t* __restrict__ a_scales,
                                                      const __nv_bfloat16* const (&b_rows)[4],
                                                      int k_tiles, __nv_bfloat16* smem,
                                                      float (&acc)[2][4][4]) {
    __nv_bfloat16* const a_tiles = smem;
    __nv_bfloat16* const b_tiles = smem + 2 * kQ4RowsTileM * kQ4RowsTileStride;
    const int tid                = static_cast<int>(threadIdx.x);
    const int lane               = tid & 31;
    const int warp               = tid >> 5;
    const int warp_m             = warp >> 1;
    const int warp_n             = warp & 1;
    const int a_row              = tid >> 1;
    const int a_half             = tid & 1;
    const int b_k                = (tid & 7) * 8;

    auto load_a = [&](int kt, uint4& packed, std::uint16_t& scale) {
        const int group = kt * 2 + a_half;
        packed          = load_vec<uint4>(a_codes + group * 16);
        scale           = *reinterpret_cast<const std::uint16_t*>(a_scales + group * 2);
    };
    auto store_a = [&](int stage, uint4 packed, std::uint16_t scale) {
        decode_q4_rows_group(
            packed, scale,
            a_tiles + (stage * kQ4RowsTileM + a_row) * kQ4RowsTileStride + a_half * 32);
    };
    auto load_b = [&](int stage, int kt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = (tid >> 3) + 16 * c;
            q4_rows_cp_async_16(b_tiles + (stage * kQ4RowsTileN + n) * kQ4RowsTileStride + b_k,
                                b_rows[c] + kt * kQ4RowsTileK + b_k);
        }
    };

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mi][ni][e] = 0.0F; }
        }
    }
    {
        uint4 packed;
        std::uint16_t scale;
        load_a(0, packed, scale);
        load_b(0, 0);
        store_a(0, packed, scale);
        q4_rows_cp_async_wait_all();
        __syncthreads();
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt & 1;
        const bool more = kt + 1 < k_tiles;
        uint4 next_packed{};
        std::uint16_t next_scale = 0;
        if (more) {
            load_b(stage ^ 1, kt + 1);
            load_a(kt + 1, next_packed, next_scale);
        }

        const __nv_bfloat16* a_tile = a_tiles + stage * kQ4RowsTileM * kQ4RowsTileStride;
        const __nv_bfloat16* b_tile = b_tiles + stage * kQ4RowsTileN * kQ4RowsTileStride;
#pragma unroll
        for (int ks = 0; ks < kQ4RowsTileK / 16; ++ks) {
            unsigned af[2][4];
            unsigned bf[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row = warp_m * 32 + mi * 16 + (lane & 15);
                const int col = ks * 16 + (lane >> 4) * 8;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            q4_rows_shared_address(a_tile + row * kQ4RowsTileStride + col));
            }
#pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                const int n   = warp_n * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int col = ks * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(bf[nj * 2][0], bf[nj * 2][1], bf[nj * 2 + 1][0], bf[nj * 2 + 1][1],
                            q4_rows_shared_address(b_tile + n * kQ4RowsTileStride + col));
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                }
            }
        }

        if (more) { store_a(stage ^ 1, next_packed, next_scale); }
        q4_rows_cp_async_wait_all();
        __syncthreads();
    }
}

// The accumulators as an FP32 [M][N + 4] tile in `smem`, followed by a barrier.
__device__ __forceinline__ float* q4_rows_tile_to_shared(const float (&acc)[2][4][4],
                                                         __nv_bfloat16* smem) {
    float* const c_tile = reinterpret_cast<float*>(smem);
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m    = warp >> 1;
    const int warp_n    = warp & 1;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int row = warp_m * 32 + mi * 16 + (lane >> 2);
            const int col = warp_n * 32 + ni * 8 + (lane & 3) * 2;
            c_tile[row * kQ4RowsTileCStride + col]           = acc[mi][ni][0];
            c_tile[row * kQ4RowsTileCStride + col + 1]       = acc[mi][ni][1];
            c_tile[(row + 8) * kQ4RowsTileCStride + col]     = acc[mi][ni][2];
            c_tile[(row + 8) * kQ4RowsTileCStride + col + 1] = acc[mi][ni][3];
        }
    }
    __syncthreads();
    return c_tile;
}

} // namespace ninfer::ops::detail
