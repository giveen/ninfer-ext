#pragma once

// Q8G32 RowSplit x BF16 warp-per-row SIMT GEMM.
//
//   - One warp owns one output row. blockIdx.y selects a tile of kTt activation
//     columns; for T <= kTt the weights are streamed exactly once.
//   - K is processed in 1024-value slabs staged through shared memory with a
//     cp.async double buffer: all
//     weight-plane reads are coalesced 128-bit loads, so DRAM latency is hidden
//     without relying on occupancy alone. Deeper pipelines and launch-bounds
//     occupancy caps were both measured slower (spills / no gain, see report).
//   - Consume is phase-interleaved: in phase c (0..3), lane L owns the 8
//     consecutive K-values [256c + 8L, 256c + 8L + 8). The warp's 32 x loads per
//     (column, phase) are one 16-byte uint4 per lane covering 512 *consecutive*
//     bytes, i.e. minimal L1 wavefronts (a per-lane-contiguous layout would
//     stride lanes 64B apart and burn 4x the L1 throughput on the same bytes).
//     The weight planes index out conflict-free or broadcast under the same
//     ownership.
//   - Q8 bytes are converted to FP32 and scaled per 32-value group.
//     fp32 FMA into per-column accumulators; warp-shuffle reduction per column.
//   - kTt is 4 or 8 only. Larger column tiles blow past the register budget
//     (kTt=16 needs ~98 regs -> 2 blocks/SM -> latency-bound at ~20% of DRAM),
//     so T > 8 re-streams the weights once per 8-column tile instead; the mma
//     tensor-core GEMM takes over where that stops winning.
//
// Correctness for arbitrary shapes: vector_k is k when k % 8 == 0 and x is
// 16-byte aligned, else 0. Full 1024-value slabs cover [0, vector_k/1024*1024);
// the rest of [0, vector_k) is read straight from global memory in the same
// phase layout, 256 values per warp step, so a short or ragged k (640, 2560)
// costs a few independent steps rather than one dependent step per group. When
// vector_k is 0 every group uses the scalar per-pair path, masked at the k
// boundary. Weights in the padded region [k, padded_k) are never used.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q8/q8_rowsplit_output.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Q8 slab traits + dequant. A slab is 1024 K-values.
// (Scales are staged with 4-byte cp.async because a row's scale plane is only
// guaranteed 4-byte aligned for generic kg.)
//
// dequant_chunk(s_nib, s_hi, s_sc, c, lane, w): dequantize the lane's 8 values
// of phase c into w[0..7], scale applied. All shared reads are conflict-free or
// broadcast under the phase-interleaved ownership.

struct Q8RowSplitSimtSchedule {
    using Codec                             = Q8ScalarDecodeAtom;
    static constexpr int kNibU4             = 64;
    static constexpr int kHighU4            = 0;
    static constexpr int kScaleU32          = 16;
    static constexpr int kHighBytesPerGroup = 0;

    // Byte j is signed value j. The lane's 8 values sit inside one 32-value
    // group (offset 8L % 32), so a single scale applies.
    __device__ static __forceinline__ void dequant_chunk(const uint4* s_nib, const uint4* /*s_hi*/,
                                                         const std::uint32_t* s_sc, int c, int lane,
                                                         float (&w)[8]) {
        const uint2 words = reinterpret_cast<const uint2*>(s_nib)[c * 32 + lane];
        const float scale = __half2float(
            __ushort_as_half(reinterpret_cast<const std::uint16_t*>(s_sc)[c * 8 + (lane >> 2)]));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t word = (&words.x)[j];
            w[4 * j + 0] = static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
            w[4 * j + 1] =
                static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
            w[4 * j + 2] =
                static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
            w[4 * j + 3] =
                static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
        }
    }
};

template <class Schedule>
__device__ __forceinline__ void
q8_simt_issue_slab(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi,
                   std::uint32_t* __restrict__ s_sc, const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ high_row,
                   const std::uint8_t* __restrict__ scale_row, int slab, int lane) {
    static_assert(Schedule::kNibU4 % 32 == 0,
                  "Q8 code plane must be a whole number of warp copies");
#pragma unroll
    for (int j = 0; j < Schedule::kNibU4 / 32; ++j) {
        const int i = j * 32 + lane;
        pipe_copy<16>(&s_nib[i], code_row +
                                     static_cast<std::int64_t>(slab) * (Schedule::kNibU4 * 16) +
                                     i * 16);
    }
    if constexpr (Schedule::kHighU4 > 0) {
        if (lane < Schedule::kHighU4) {
            pipe_copy<16>(&s_hi[lane],
                          high_row + static_cast<std::int64_t>(slab) * (Schedule::kHighU4 * 16) +
                              lane * 16);
        }
    }
    if (lane < Schedule::kScaleU32) {
        pipe_copy<4>(&s_sc[lane], scale_row +
                                      static_cast<std::int64_t>(slab) * (Schedule::kScaleU32 * 4) +
                                      lane * 4);
    }
    pipe_commit();
}

// x0 points at the column tile base (x + col0*k); xslab is the slab's first
// K-value. Requires k % 8 == 0 and 16-byte aligned x.
template <class Schedule, int ColsPerTile>
__device__ __forceinline__ void
q8_simt_consume_slab(const __nv_bfloat16* __restrict__ x0, std::int64_t xslab, std::int32_t k,
                     int ncols, const uint4* __restrict__ s_nib, const uint4* __restrict__ s_hi,
                     const std::uint32_t* __restrict__ s_sc, int lane, float (&acc)[ColsPerTile]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float w[8];
        Schedule::dequant_chunk(s_nib, s_hi, s_sc, c, lane, w);
        const std::int64_t xoff = xslab + c * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < ColsPerTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv  = load_vec<uint4>(x0 + static_cast<std::int64_t>(tt) * k + xoff);
                const float2 f0 = bf16x2_bits_to_float2(xv.x);
                const float2 f1 = bf16x2_bits_to_float2(xv.y);
                const float2 f2 = bf16x2_bits_to_float2(xv.z);
                const float2 f3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
                acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
                acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
                acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
                acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
                acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
                acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
                acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
            }
        }
    }
}

// One lane's step outside the slab pipeline: its 8 codes at kk (kk % 8 == 0) against every live
// column, read straight from global memory. Requires k % 8 == 0 and 16-byte aligned x.
template <int ColsPerTile>
__device__ __forceinline__ void
q8_simt_vector_step(const std::uint8_t* __restrict__ code_row,
                    const std::uint8_t* __restrict__ scale_row, const __nv_bfloat16* __restrict__ x0,
                    std::int32_t k, int kk, int ncols, float (&acc)[ColsPerTile]) {
    const uint2 words = *reinterpret_cast<const uint2*>(code_row + kk);
    const float scale = __half2float(
        __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scale_row)[kk / 32]));
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const std::uint32_t word = (&words.x)[j / 4];
        w[j] = static_cast<float>(static_cast<std::int8_t>((word >> (8 * (j % 4))) & 0xffu)) * scale;
    }
#pragma unroll
    for (int tt = 0; tt < ColsPerTile; ++tt) {
        if (tt < ncols) {
            const uint4 xv  = load_vec<uint4>(x0 + static_cast<std::int64_t>(tt) * k + kk);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
            acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
            acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
            acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
            acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
            acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
            acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
            acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
        }
    }
}

// vector_k is computed on the host: q8_simt_vector_k (k when k % 8 == 0 and x is
// 16-byte aligned, else 0, when everything runs through the scalar tail).
__host__ __device__ constexpr std::int32_t q8_simt_vector_k(std::int32_t k, bool aligned_x) {
    return aligned_x && k % 8 == 0 ? k : 0;
}

template <class Schedule, int ColsPerTile, int RowsPerCta, int PipelineStages, bool Full,
          Q8Epilogue Epilogue = Q8Epilogue::Store, class Output = Q8ContiguousOutput>
__global__ void q8_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                             const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ scales, Output output,
                                             std::int32_t rows, std::int32_t k, std::int32_t cols,
                                             std::int32_t padded_k, std::int32_t vector_k) {
    const std::int32_t full_slabs = vector_k / 1024;
    using Codec                = typename Schedule::Codec;
    constexpr int kPrefetch    = PipelineStages - 1;
    constexpr int kHighU4Alloc = Schedule::kHighU4 > 0 ? Schedule::kHighU4 : 1;

    __shared__ __align__(16) uint4 s_nib[RowsPerCta][PipelineStages][Schedule::kNibU4];
    __shared__ __align__(16) uint4 s_hi[RowsPerCta][PipelineStages][kHighU4Alloc];
    __shared__ __align__(16) std::uint32_t s_sc[RowsPerCta][PipelineStages][Schedule::kScaleU32];

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0 = static_cast<int>(blockIdx.x) * RowsPerCta;
    const int row      = cta_row0 + warp;
    if constexpr (!Full) {
        if (row >= rows) { return; }
    }
    const Q8OutputTile output_tile = output.tile(cta_row0);
    const int col0                 = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols                = Full ? ColsPerTile : min(ColsPerTile, cols - col0);

    const int kg_padded           = padded_k / Codec::kGroupK;
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row  = nullptr;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;
    const __nv_bfloat16* x0       = x + static_cast<std::int64_t>(col0) * k;

    float acc[ColsPerTile];
#pragma unroll
    for (int i = 0; i < ColsPerTile; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < full_slabs) {
            q8_simt_issue_slab<Schedule>(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row,
                                         high_row, scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int s = 0; s < full_slabs; ++s) {
        const int fetch = s + kPrefetch;
        if (fetch < full_slabs) {
            const int buf = fetch % PipelineStages;
            q8_simt_issue_slab<Schedule>(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf],
                                         code_row, high_row, scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = s % PipelineStages;
        q8_simt_consume_slab<Schedule, ColsPerTile>(x0, static_cast<std::int64_t>(s) * 1024, k,
                                                    ncols, s_nib[warp][buf], s_hi[warp][buf],
                                                    s_sc[warp][buf], lane, acc);
        __syncwarp();
    }

    // Vector tail: the rest of [0, vector_k) in the slab's phase layout, 256 values per step.
#pragma unroll 4
    for (int kk = full_slabs * 1024 + lane * 8; kk < vector_k; kk += 256) {
        q8_simt_vector_step<ColsPerTile>(code_row, scale_row, x0, k, kk, ncols, acc);
    }

    // Scalar tail (vector_k == 0 only): groups read global memory directly, masked at k.
    const int g0      = vector_k > 0 ? div_up(k, Codec::kGroupK) : 0;
    const int kg_used = div_up(k, Codec::kGroupK);
    for (int g = g0; g < kg_used; ++g) {
        const int kk = g * Codec::kGroupK + lane * 2;
        if (kk >= k) { continue; }
        float w0 = 0.0f;
        float w1 = 0.0f;
        Codec::load_pair(codes, nullptr, scales, static_cast<std::int64_t>(row) * kg_padded + g,
                         lane, w0, w1);
#pragma unroll
        for (int tt = 0; tt < ColsPerTile; ++tt) {
            if (tt < ncols) {
                const std::int64_t xb = static_cast<std::int64_t>(tt) * k + kk;
                acc[tt]               = fmaf(w0, __bfloat162float(x0[xb]), acc[tt]);
                if (kk + 1 < k) { acc[tt] = fmaf(w1, __bfloat162float(x0[xb + 1]), acc[tt]); }
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < ColsPerTile; ++tt) {
        if (tt >= ncols) { continue; }
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) {
            __nv_bfloat16* destination = output_tile.at(row, col0 + tt);
            if constexpr (Epilogue == Q8Epilogue::Residual) {
                a = __bfloat162float(*destination) + a;
            }
            *destination = __float2bfloat16_rn(a);
        }
    }
}

// K-split variant for few rows over a long K (Qwen4Exp hyper-connection [324, 10240]): the
// warp-per-row kernel would run one warp per row across all of K on a handful of CTAs.
// WarpsPerRow warps share a row, warp p taking steps p, p + WarpsPerRow, ... of 256 values, and
// reduce through shared memory. Requires k % 8 == 0 and 16-byte aligned x (vector_k == k).
template <int ColsPerTile, int RowsPerCta, int WarpsPerRow>
__global__ void __launch_bounds__(RowsPerCta* WarpsPerRow * 32)
    q8_rowsplit_ksplit_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ scales,
                                   Q8ContiguousOutput output, std::int32_t rows, std::int32_t k,
                                   std::int32_t cols, std::int32_t padded_k) {
    __shared__ float partial[RowsPerCta][WarpsPerRow][ColsPerTile];
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int local = warp / WarpsPerRow;
    const int part  = warp % WarpsPerRow;
    const int row   = static_cast<int>(blockIdx.x) * RowsPerCta + local;
    const int col0  = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols = min(ColsPerTile, cols - col0);

    float acc[ColsPerTile];
#pragma unroll
    for (int i = 0; i < ColsPerTile; ++i) { acc[i] = 0.0f; }
    if (row < rows) {
        const int kg                  = padded_k / 32;
        const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kg * 32;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg * 2;
        const __nv_bfloat16* x0       = x + static_cast<std::int64_t>(col0) * k;
#pragma unroll 4
        for (int kk = part * 256 + lane * 8; kk < k; kk += 256 * WarpsPerRow) {
            q8_simt_vector_step<ColsPerTile>(code_row, scale_row, x0, k, kk, ncols, acc);
        }
    }
#pragma unroll
    for (int tt = 0; tt < ColsPerTile; ++tt) {
        const float a = warp_reduce_sum(acc[tt]);
        if (lane == 0) { partial[local][part][tt] = a; }
    }
    __syncthreads();
    if (part == 0 && lane < ncols && row < rows) {
        float a = 0.0f;
#pragma unroll
        for (int p = 0; p < WarpsPerRow; ++p) { a += partial[local][p][lane]; }
        *output.tile(0).at(row, col0 + lane) = __float2bfloat16_rn(a);
    }
}

} // namespace ninfer::ops::detail
