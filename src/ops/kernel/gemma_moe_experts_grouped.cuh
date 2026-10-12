// Implements: include/ninfer/ops/gemma_moe_experts.h (the grouped route)
//
// The prefill route for an int4 group-32 row bank pair at a registered geometry. A pass's K*T
// assignments are grouped by expert so that one expert's weights serve every token routed to it,
// then three tensor-core passes and a combine:
//
//   1. route: per-expert counts, then offsets and a job list (one job per 64 assignments of an
//      expert), then each assignment's packed column, written in assignment order by one CTA per
//      expert. No atomics decide an order, so the packed layout is a function of the ids alone.
//   2. gate/up: a CTA per (32 intermediate rows, job) computes the 32 gate and 32 up rows of its
//      expert against the job's 64 gathered tokens, and writes GeGLU(gate) * up as BF16.
//   3. down: a CTA per (64 hidden rows, job) computes the expert's down rows against the job's
//      BF16 products and writes BF16, one row per assignment.
//   4. combine: each token's K rows are summed in ascending rank order in FP32, times their
//      routing weights, and narrowed to BF16 once.
//
// The weights are decoded `code * binary16 scale` straight to BF16 in shared memory. For a bank
// converted from a QAT checkpoint that is the checkpoint's own BF16 value to within one ulp, so
// the GEMMs are plain BF16 m16n8k16 with FP32 accumulation. The two BF16 boundaries -- the GeGLU
// product and each expert's output -- are where the checkpoint's reference rounds as well.
#pragma once

#include "ops/common/memory.cuh" // load_vec, store_vec
#include "ops/common/q4_rows_mma.cuh" // q4_rows_tile_mainloop
#include "ops/kernel/gelu.cuh"   // gelu_one

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kGroupedBM      = kQ4RowsTileM; // weight rows a CTA computes
inline constexpr int kGroupedBN      = kQ4RowsTileN; // assignments a job covers
inline constexpr int kGroupedThreads = kQ4RowsTileThreads;
inline constexpr int kRouteThreads   = 256;

// --- route -------------------------------------------------------------------------------------

// A slice of a pass: `tokens` columns starting at `first` of the router's rank-major [K, stride]
// maps. Slice-local assignment a is rank a / tokens of token a % tokens.
struct GroupedSlice {
    int stride;
    int first;
    int tokens;
    int top_k;

    __device__ __forceinline__ int assignments() const { return top_k * tokens; }

    __device__ __forceinline__ int id(const std::int32_t* ids, int a) const {
        return ids[(a / tokens) * stride + first + a % tokens];
    }
};

// One CTA per expert counts the assignments that selected it.
__global__ void gemma_moe_grouped_count_kernel(const std::int32_t* __restrict__ ids,
                                               GroupedSlice slice, int* __restrict__ counts) {
    __shared__ int partial[kRouteThreads / 32];
    const int expert = static_cast<int>(blockIdx.x);
    int mine         = 0;
    for (int a = static_cast<int>(threadIdx.x); a < slice.assignments(); a += kRouteThreads) {
        mine += slice.id(ids, a) == expert;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        mine += __shfl_xor_sync(0xffffffffu, mine, offset);
    }
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = mine; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < kRouteThreads / 32; ++w) { total += partial[w]; }
        counts[expert] = total;
    }
}

// One thread lays the experts out in id order and cuts each into 64-assignment jobs.
__global__ void gemma_moe_grouped_plan_kernel(const int* __restrict__ counts, int experts,
                                              int* __restrict__ offsets,
                                              int* __restrict__ job_experts,
                                              int* __restrict__ job_columns,
                                              int* __restrict__ job_count) {
    if (threadIdx.x != 0) { return; }
    int base = 0;
    int jobs = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = base;
        for (int c = 0; c < counts[e]; c += kGroupedBN) {
            job_experts[jobs] = e;
            job_columns[jobs] = base + c;
            ++jobs;
        }
        base += counts[e];
    }
    offsets[experts] = base;
    job_count[0]     = jobs;
}

// One CTA per expert walks the assignments in order and gives each of its own the next packed
// column: the token it reads, and the inverse map the combine reads back.
__global__ void gemma_moe_grouped_place_kernel(const std::int32_t* __restrict__ ids,
                                               GroupedSlice slice,
                                               const int* __restrict__ offsets,
                                               int* __restrict__ packed_token,
                                               int* __restrict__ inverse) {
    __shared__ int warp_totals[kRouteThreads / 32];
    __shared__ int carry;
    const int expert = static_cast<int>(blockIdx.x);
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    if (threadIdx.x == 0) { carry = offsets[expert]; }
    __syncthreads();
    for (int begin = 0; begin < slice.assignments(); begin += kRouteThreads) {
        const int a        = begin + static_cast<int>(threadIdx.x);
        const bool mine    = a < slice.assignments() && slice.id(ids, a) == expert;
        const unsigned set = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) { warp_totals[warp] = __popc(set); }
        __syncthreads();
        int before = carry;
        for (int w = 0; w < warp; ++w) { before += warp_totals[w]; }
        if (mine) {
            const int column     = before + __popc(set & ((1u << lane) - 1u));
            packed_token[column] = a % slice.tokens;
            inverse[a]           = column;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < kRouteThreads / 32; ++w) { carry += warp_totals[w]; }
        }
        __syncthreads();
    }
}

// --- grouped GEMMs -----------------------------------------------------------------------------

struct GroupedJobs {
    const int* __restrict__ experts;
    const int* __restrict__ columns;
    const int* __restrict__ count;
    const int* __restrict__ offsets; // [E+1], an expert's columns end at offsets[e + 1]
};

// The two projections differ in which weight rows a CTA stages, where its B columns come from,
// and what its epilogue writes.
enum class GroupedProjection : std::uint8_t { GateUp, Down };

/**
 * Grid (row blocks, job bound), kGroupedThreads. GateUp: K = Hidden, a row block is 32
 * intermediate rows (their gate and up rows), B columns are tokens of `x` [Hidden,T] through
 * `packed_token`, and the epilogue writes BF16 GeGLU products to `out` [assignments, Intermediate].
 * Down: K = Intermediate, a row block is 64 hidden rows, B columns are rows of `b` [assignments,
 * Intermediate], and the epilogue writes BF16 to `out` [assignments, Hidden].
 */
template <GroupedProjection Projection, int Hidden, int Intermediate>
__global__ __launch_bounds__(kGroupedThreads) void gemma_moe_grouped_mma_kernel(
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ packed_token,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    GroupedJobs jobs, __nv_bfloat16* __restrict__ out) {
    constexpr bool kGateUp = Projection == GroupedProjection::GateUp;
    constexpr int K        = kGateUp ? Hidden : Intermediate;
    static_assert(K % kQ4RowsTileK == 0 && Intermediate % 32 == 0 && Hidden % kQ4RowsTileM == 0);
    __shared__ __align__(16) __nv_bfloat16 smem[kQ4RowsTileSmemElements];

    const int job = static_cast<int>(blockIdx.y);
    if (job >= jobs.count[0]) { return; }
    const int expert       = jobs.experts[job];
    const int column_begin = jobs.columns[job];
    const int column_end   = min(column_begin + kGroupedBN, jobs.offsets[expert + 1]);
    const int tid          = static_cast<int>(threadIdx.x);

    // This thread's staged weight row: gate/up stages 32 gate rows then the 32 matching up rows.
    const int a_row = tid >> 1;
    const int row0  = static_cast<int>(blockIdx.x) * (kGateUp ? kQ4RowsTileM / 2 : kQ4RowsTileM);
    const int bank_row =
        kGateUp ? expert * 2 * Intermediate + (a_row < 32 ? row0 + a_row
                                                         : Intermediate + row0 + a_row - 32)
                : expert * Hidden + row0 + a_row;

    const __nv_bfloat16* b_rows[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int column = min(column_begin + (tid >> 3) + 16 * c, column_end - 1);
        b_rows[c]        = kGateUp ? b + static_cast<std::int64_t>(packed_token[column]) * Hidden
                                   : b + static_cast<std::int64_t>(column) * Intermediate;
    }

    float acc[2][4][4];
    q4_rows_tile_mainloop(codes + static_cast<std::int64_t>(bank_row) * (K / 2),
                          scales + static_cast<std::int64_t>(bank_row) * (K / 32) * 2, b_rows,
                          K / kQ4RowsTileK, smem, acc);
    const float* const c_tile = q4_rows_tile_to_shared(acc, smem);

    const int n      = tid >> 1;
    const int column = column_begin + n;
    if (column >= column_end) { return; }
    if constexpr (kGateUp) {
        // Sixteen intermediate rows a thread: GeGLU(gate) * up for its column.
        const int j0 = (tid & 1) * 16;
        __align__(16) __nv_bfloat16 values[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            values[j] = __float2bfloat16_rn(
                gelu_one<true>(c_tile[(j0 + j) * kQ4RowsTileCStride + n]) *
                c_tile[(32 + j0 + j) * kQ4RowsTileCStride + n]);
        }
        __nv_bfloat16* const target =
            out + static_cast<std::int64_t>(column) * Intermediate + row0 + j0;
        store_vec(target, *reinterpret_cast<const uint4*>(values));
        store_vec(target + 8, *reinterpret_cast<const uint4*>(values + 8));
    } else {
        // Thirty-two hidden rows a thread for its column.
        const int r0 = (tid & 1) * 32;
        __align__(16) __nv_bfloat16 values[32];
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            values[r] = __float2bfloat16_rn(c_tile[(r0 + r) * kQ4RowsTileCStride + n]);
        }
        __nv_bfloat16* const target = out + static_cast<std::int64_t>(column) * Hidden + row0 + r0;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            store_vec(target + v * 8, *reinterpret_cast<const uint4*>(values + v * 8));
        }
    }
}

// --- combine -----------------------------------------------------------------------------------

/**
 * Grid T, Hidden / 8 threads: out[:,t] = sum over ranks in ascending order of
 * weight[r,t] * expert_out[inverse[r*T + t], :], accumulated in FP32 and narrowed once.
 */
template <int Hidden, int TopK>
__global__ void gemma_moe_grouped_combine_kernel(const __nv_bfloat16* __restrict__ expert_out,
                                                 const int* __restrict__ inverse,
                                                 const float* __restrict__ weights,
                                                 GroupedSlice slice,
                                                 __nv_bfloat16* __restrict__ out) {
    const int token = static_cast<int>(blockIdx.x);
    const int h     = static_cast<int>(threadIdx.x) * 8;
    float sum[8]    = {};
#pragma unroll
    for (int rank = 0; rank < TopK; ++rank) {
        const int slot     = rank * slice.tokens + token;
        const float weight = weights[rank * slice.stride + slice.first + token];
        const uint4 packed =
            load_vec<uint4>(expert_out + static_cast<std::int64_t>(inverse[slot]) * Hidden + h);
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 value = __bfloat1622float2(pairs[p]);
            sum[2 * p]         = fmaf(weight, value.x, sum[2 * p]);
            sum[2 * p + 1]     = fmaf(weight, value.y, sum[2 * p + 1]);
        }
    }
    __align__(16) __nv_bfloat162 result[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) { result[p] = __floats2bfloat162_rn(sum[2 * p], sum[2 * p + 1]); }
    store_vec(out + static_cast<std::int64_t>(token) * Hidden + h,
              *reinterpret_cast<const uint4*>(result));
}

} // namespace ninfer::ops::detail
