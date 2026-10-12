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
#include "ops/common/mma.cuh"    // ldmatrix_x4, mma_bf16
#include "ops/kernel/gelu.cuh"   // gelu_one

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kGroupedBM      = 64; // weight rows a CTA computes
inline constexpr int kGroupedBN      = 64; // assignments a job covers
inline constexpr int kGroupedBK      = 64;
inline constexpr int kGroupedThreads = 128; // four warps, 2 x 2, each 32 x 32
inline constexpr int kGroupedStride  = kGroupedBK + 8; // padded smem row, in BF16
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

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_16(void* shared, const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_address(shared)),
                 "l"(global));
}

__device__ __forceinline__ void cp_async_commit_wait_all() {
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
    constexpr int KT       = K / kGroupedBK;
    static_assert(K % kGroupedBK == 0 && Intermediate % 32 == 0 && Hidden % kGroupedBM == 0);

    __shared__ __align__(16) __nv_bfloat16 smem[2 * 2 * kGroupedBM * kGroupedStride];
    __nv_bfloat16* const a_tiles = smem;                                    // [2][BM][stride]
    __nv_bfloat16* const b_tiles = smem + 2 * kGroupedBM * kGroupedStride; // [2][BN][stride]

    const int job = static_cast<int>(blockIdx.y);
    if (job >= jobs.count[0]) { return; }
    const int expert       = jobs.experts[job];
    const int column_begin = jobs.columns[job];
    const int column_end   = min(column_begin + kGroupedBN, jobs.offsets[expert + 1]);
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid & 31;
    const int warp         = tid >> 5;
    const int warp_m       = warp >> 1;
    const int warp_n       = warp & 1;

    // This thread's staged weight row and 32-wide group within a K tile.
    const int a_row   = tid >> 1;
    const int a_half  = tid & 1;
    const int row0    = static_cast<int>(blockIdx.x) * (kGateUp ? kGroupedBM / 2 : kGroupedBM);
    const int a_bank_row =
        kGateUp ? expert * 2 * Intermediate + (a_row < 32 ? row0 + a_row
                                                           : Intermediate + row0 + a_row - 32)
                : expert * Hidden + row0 + a_row;
    const std::uint8_t* const a_codes = codes + static_cast<std::int64_t>(a_bank_row) * (K / 2);
    const std::uint8_t* const a_scales =
        scales + static_cast<std::int64_t>(a_bank_row) * (K / 32) * 2;

    // This thread's four B chunks: rows (tid / 8) + 16 c, 16 bytes at (tid % 8) * 8 along K.
    const __nv_bfloat16* b_rows[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int n      = (tid >> 3) + 16 * c;
        const int column = min(column_begin + n, column_end - 1);
        b_rows[c]        = kGateUp ? b + static_cast<std::int64_t>(packed_token[column]) * Hidden
                                   : b + static_cast<std::int64_t>(column) * Intermediate;
    }
    const int b_k = (tid & 7) * 8;

    auto load_a = [&](int kt, uint4& packed, std::uint16_t& scale) {
        const int group = kt * 2 + a_half;
        packed          = load_vec<uint4>(a_codes + group * 16);
        scale           = *reinterpret_cast<const std::uint16_t*>(a_scales + group * 2);
    };
    auto store_a = [&](int stage, uint4 packed, std::uint16_t scale) {
        decode_q4_rows_group(packed, scale,
                             a_tiles + (stage * kGroupedBM + a_row) * kGroupedStride + a_half * 32);
    };
    auto load_b = [&](int stage, int kt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = (tid >> 3) + 16 * c;
            cp_async_16(b_tiles + (stage * kGroupedBN + n) * kGroupedStride + b_k,
                        b_rows[c] + kt * kGroupedBK + b_k);
        }
    };

    float acc[2][4][4] = {};
    {
        uint4 packed;
        std::uint16_t scale;
        load_a(0, packed, scale);
        load_b(0, 0);
        store_a(0, packed, scale);
        cp_async_commit_wait_all();
        __syncthreads();
    }

    for (int kt = 0; kt < KT; ++kt) {
        const int stage = kt & 1;
        const bool more = kt + 1 < KT;
        uint4 next_packed{};
        std::uint16_t next_scale = 0;
        if (more) {
            load_b(stage ^ 1, kt + 1);
            load_a(kt + 1, next_packed, next_scale);
        }

        const __nv_bfloat16* a_tile = a_tiles + stage * kGroupedBM * kGroupedStride;
        const __nv_bfloat16* b_tile = b_tiles + stage * kGroupedBN * kGroupedStride;
#pragma unroll
        for (int ks = 0; ks < kGroupedBK / 16; ++ks) {
            unsigned af[2][4];
            unsigned bf[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row = warp_m * 32 + mi * 16 + (lane & 15);
                const int col = ks * 16 + (lane >> 4) * 8;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            shared_address(a_tile + row * kGroupedStride + col));
            }
#pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                const int n   = warp_n * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int col = ks * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(bf[nj * 2][0], bf[nj * 2][1], bf[nj * 2 + 1][0], bf[nj * 2 + 1][1],
                            shared_address(b_tile + n * kGroupedStride + col));
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
        cp_async_commit_wait_all();
        __syncthreads();
    }

    // Epilogue through shared memory: the tile as FP32 [BM][BN + 4].
    constexpr int kCStride = kGroupedBN + 4;
    float* const c_tile    = reinterpret_cast<float*>(smem);
    static_assert(kGroupedBM * kCStride * 4 <= sizeof(smem));
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int row = warp_m * 32 + mi * 16 + (lane >> 2);
            const int col = warp_n * 32 + ni * 8 + (lane & 3) * 2;
            c_tile[row * kCStride + col]           = acc[mi][ni][0];
            c_tile[row * kCStride + col + 1]       = acc[mi][ni][1];
            c_tile[(row + 8) * kCStride + col]     = acc[mi][ni][2];
            c_tile[(row + 8) * kCStride + col + 1] = acc[mi][ni][3];
        }
    }
    __syncthreads();

    const int n      = tid >> 1;
    const int column = column_begin + n;
    if (column >= column_end) { return; }
    if constexpr (kGateUp) {
        // Sixteen intermediate rows a thread: GeGLU(gate) * up for its column.
        const int j0 = (tid & 1) * 16;
        __align__(16) __nv_bfloat16 values[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            values[j] = __float2bfloat16_rn(gelu_one<true>(c_tile[(j0 + j) * kCStride + n]) *
                                            c_tile[(32 + j0 + j) * kCStride + n]);
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
            values[r] = __float2bfloat16_rn(c_tile[(r0 + r) * kCStride + n]);
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
