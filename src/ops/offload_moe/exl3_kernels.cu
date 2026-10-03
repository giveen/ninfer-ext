// Implements: include/ninfer/ops/offload_moe.h (EXL3 routed experts)
//
// One expert is W = diag(suh) H128 Z H128 diag(svh) per matrix (orthonormal H128, Z the mul1
// trellis integers). Per assignment this runs
//
//   xr   = H(suh_g . x), H(suh_u . x)               (rotated into the codebook basis, FP16)
//   g, u = svh . H(Z xr)                            (two tensor-core GEMMs, K split over 4 CTAs)
//   act  = H(suhd . silu(g) u)                      (the down input, rotated for the next GEMM)
//   out  = svhd . H(Zd act)                         (per-assignment partial for the merge)
//
// The activations are FP16 because FP16 holds every mul1 integer (+-510) exactly, which BF16 does
// not; the Hadamards are orthonormal so the rotated values stay within a small multiple of the
// input. Every sum has a fixed order (K split slices added in order, the merge sums the ten
// assignments in order), so cache placement and job order never change a result.
#include "ops/offload_moe/jobs.cuh"
#include "ops/offload_moe/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/exl3/exl3_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {
namespace {

constexpr int H = kOffloadMoeHidden;
constexpr int I = kOffloadMoeIntermediate;
constexpr int K = kOffloadMoeTopK;

constexpr int kThreads = 256; // 8 warps, one 16-wide output tile each
constexpr int kKSplit  = 4;   // gate/up K slices (one CTA each), summed in order
constexpr int kBlocks  = I / 128;      // 128-wide Hadamard blocks of the intermediate
constexpr int kDownBlocks = H / 128;   // ... and of the hidden output
constexpr int kSliceK  = H / kKSplit;  // 640 input features per gate/up CTA
constexpr int kSliceKTiles = kSliceK / 16;
constexpr int kDownKTiles  = I / 16;
constexpr int kRows    = 16;           // m16 rows per pass over a job
constexpr int kAStride = kSliceK + 8;  // padded halves per smem row: conflict-free fragment loads
constexpr float kInvSqrt128 = 0.08838834764831845F;

static_assert(kSliceK == I && kAStride == I + 8, "gate/up slice and down input share one row stride");

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

// A fragment of k-tile `kt` from a row-major smem tile with stride kAStride halves.
__device__ __forceinline__ void load_a(const __half* tile, int row, int col, int kt,
                                       std::uint32_t a[4]) {
    const __half* p = tile + row * kAStride + kt * 16 + col;
    a[0]            = load_half2(p);
    a[1]            = load_half2(p + 8 * kAStride);
    a[2]            = load_half2(p + 8);
    a[3]            = load_half2(p + 8 * kAStride + 8);
}

// One 16-row, 16-column tile over `ktiles` k-tiles: c[j] is the n8 half j of the m16n8 result.
__device__ __forceinline__ void tile_gemm(const __half* a_tile, const std::uint8_t* trellis,
                                          int ktiles, int tile_bytes, const Exl3TileDecoder& dec,
                                          int lane, float c[2][4]) {
    const int row = lane >> 2;
    const int col = (lane & 3) * 2;
    for (int kt = 0; kt < ktiles; ++kt) {
        float d[8];
        dec.decode(trellis + static_cast<std::size_t>(kt) * tile_bytes, lane, d);
        const std::uint32_t b0[2] = {pack_half2(d[0], d[1]), pack_half2(d[2], d[3])};
        const std::uint32_t b1[2] = {pack_half2(d[4], d[5]), pack_half2(d[6], d[7])};
        std::uint32_t a[4];
        load_a(a_tile, row, col, kt, a);
        mma_f16(c[0], a, b0);
        mma_f16(c[1], a, b1);
    }
}

// Gate/up. CTA (job, 128-column block, K slice): rotate the job's rows into smem, run both GEMMs
// over the slice and store the FP32 partials. The last of the four slices of a block applies the
// output Hadamard and scale, the SwiGLU, and the down input's scale and Hadamard.
__global__ void __launch_bounds__(kThreads)
    exl3_gate_up_kernel(const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ ids,
                        const std::int32_t* __restrict__ sorted_assign,
                        const std::int32_t* __restrict__ sorted_slot,
                        const std::int32_t* __restrict__ jobs,
                        const std::int32_t* __restrict__ job_count, std::int32_t assignments,
                        JobFilter filter, ExpertWeights w, float* __restrict__ partial,
                        __half* __restrict__ act, std::int32_t* __restrict__ counters) {
    __shared__ __half as[2][kRows * kAStride];
    __shared__ int last_s;
    const JobView job = load_job(jobs, job_count, sorted_slot, sorted_assign, ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int nb = static_cast<int>(blockIdx.y), ks = static_cast<int>(blockIdx.z);
    const std::int64_t slot = job.slot;
    const auto* suh = reinterpret_cast<const float*>(w.base[1] + slot * w.stride[1]); // [2][H]
    const auto* trellis = reinterpret_cast<const std::uint8_t*>(w.base[0] + slot * w.stride[0]);
    const int half_bits = w.layout.gate_up_half_bits;
    const int tile_bytes = 16 * half_bits;
    const Exl3TileDecoder dec(half_bits, lane);
    const int row = lane >> 2, col = (lane & 3) * 2;
    constexpr int kTilesK = H / 16;

    for (int g0 = 0; g0 < job.count; g0 += kRows) {
        const int rows = min(kRows, job.count - g0);
        for (int item = warp; item < 2 * kRows; item += kThreads / 32) {
            const int r = item >> 1, proj = item & 1;
            __half* dst = &as[proj][r * kAStride];
            if (r >= rows) {
                for (int i = lane; i < kSliceK; i += 32) { dst[i] = __float2half_rn(0.0F); }
                continue;
            }
            const int token = sorted_assign[job.begin + g0 + r] / K;
            const __nv_bfloat16* xr = x + static_cast<std::int64_t>(token) * H + ks * kSliceK;
            const float* sp         = suh + proj * H + ks * kSliceK;
            for (int blk = 0; blk < kSliceK / 128; ++blk) {
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int idx = blk * 128 + lane * 4 + j;
                    v[j]          = __bfloat162float(xr[idx]) * sp[idx];
                }
                fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dst[blk * 128 + lane * 4 + j] = __float2half_rn(v[j] * kInvSqrt128);
                }
            }
        }
        __syncthreads();
        float cg[2][4] = {}, cu[2][4] = {};
        const std::int64_t gate_tile = static_cast<std::int64_t>(nb * 8 + warp);
        const std::int64_t up_tile   = static_cast<std::int64_t>(I / 16 + nb * 8 + warp);
        tile_gemm(as[0], trellis + (gate_tile * kTilesK + ks * kSliceKTiles) * tile_bytes,
                  kSliceKTiles, tile_bytes, dec, lane, cg);
        tile_gemm(as[1], trellis + (up_tile * kTilesK + ks * kSliceKTiles) * tile_bytes,
                  kSliceKTiles, tile_bytes, dec, lane, cu);
        for (int proj = 0; proj < 2; ++proj) {
            const auto& c = proj == 0 ? cg : cu;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = nb * 128 + warp * 16 + j * 8 + col;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int r = row + 8 * half;
                    if (r < rows) {
                        float* out = partial + ((static_cast<std::int64_t>(ks) * assignments +
                                                 job.begin + g0 + r) * 2 + proj) * I + n;
                        out[0] = c[j][2 * half];
                        out[1] = c[j][2 * half + 1];
                    }
                }
            }
        }
        __syncthreads();
    }

    // The last slice to finish a block runs its epilogue.
    __threadfence();
    __syncthreads();
    std::int32_t* counter = counters + static_cast<std::int64_t>(blockIdx.x) * kBlocks + nb;
    if (tid == 0) { last_s = atomicAdd(counter, 1) == kKSplit - 1; }
    __syncthreads();
    if (!last_s) { return; }
    __threadfence();
    const auto* svh = reinterpret_cast<const float*>(w.base[2] + slot * w.stride[2]); // [2][I]
    const auto* sd  = reinterpret_cast<const float*>(w.base[4] + slot * w.stride[4]); // [I]
    for (int pos = warp; pos < job.count; pos += kThreads / 32) {
        float g[4], u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = nb * 128 + lane * 4 + j;
            float sg = 0.0F, su = 0.0F;
            for (int s = 0; s < kKSplit; ++s) {
                const std::int64_t row_base =
                    (static_cast<std::int64_t>(s) * assignments + job.begin + pos) * 2 * I;
                sg += __ldcg(partial + row_base + n);
                su += __ldcg(partial + row_base + I + n);
            }
            g[j] = sg;
            u[j] = su;
        }
        fwht128(g, lane);
        fwht128(u, lane);
        float a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = nb * 128 + lane * 4 + j;
            const float gv = g[j] * kInvSqrt128 * svh[n];
            const float uv = u[j] * kInvSqrt128 * svh[I + n];
            a[j]           = gv / (1.0F + expf(-gv)) * uv * sd[n];
        }
        fwht128(a, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            act[static_cast<std::int64_t>(job.begin + pos) * I + nb * 128 + lane * 4 + j] =
                __float2half_rn(a[j] * kInvSqrt128);
        }
    }
    if (tid == 0) { *counter = 0; }
}

// Down. CTA (job, 128-column block of the hidden output): the whole K = 640 contraction, then the
// output Hadamard and scale, written as the assignment's partial row for the merge.
__global__ void __launch_bounds__(kThreads)
    exl3_down_kernel(const __half* __restrict__ act, const std::int32_t* __restrict__ ids,
                     const std::int32_t* __restrict__ sorted_assign,
                     const std::int32_t* __restrict__ sorted_slot,
                     const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
                     std::int32_t assignments, JobFilter filter, ExpertWeights w,
                     float* __restrict__ partial) {
    __shared__ __half as[kRows * kAStride];
    __shared__ float res[kRows][129];
    const JobView job = load_job(jobs, job_count, sorted_slot, sorted_assign, ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int nb = static_cast<int>(blockIdx.y);
    const std::int64_t slot = job.slot;
    const auto* trellis = reinterpret_cast<const std::uint8_t*>(w.base[3] + slot * w.stride[3]);
    const auto* svh     = reinterpret_cast<const float*>(w.base[5] + slot * w.stride[5]); // [H]
    const int half_bits = w.layout.down_half_bits;
    const int tile_bytes = 16 * half_bits;
    const Exl3TileDecoder dec(half_bits, lane);
    const int row = lane >> 2, col = (lane & 3) * 2;

    for (int g0 = 0; g0 < job.count; g0 += kRows) {
        const int rows = min(kRows, job.count - g0);
        for (int i = tid; i < kRows * (I / 2); i += kThreads) {
            const int r = i / (I / 2), k2 = i % (I / 2);
            std::uint32_t pair = 0;
            if (r < rows) {
                pair = *reinterpret_cast<const std::uint32_t*>(
                    act + static_cast<std::int64_t>(job.begin + g0 + r) * I + 2 * k2);
            }
            *reinterpret_cast<std::uint32_t*>(&as[r * kAStride + 2 * k2]) = pair;
        }
        __syncthreads();
        float c[2][4] = {};
        const std::int64_t tile = static_cast<std::int64_t>(nb * 8 + warp);
        tile_gemm(as, trellis + tile * kDownKTiles * tile_bytes, kDownKTiles, tile_bytes, dec, lane, c);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int n = warp * 16 + j * 8 + col;
            res[row][n]         = c[j][0];
            res[row][n + 1]     = c[j][1];
            res[row + 8][n]     = c[j][2];
            res[row + 8][n + 1] = c[j][3];
        }
        __syncthreads();
        for (int pos = warp; pos < rows; pos += kThreads / 32) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { v[j] = res[pos][lane * 4 + j]; }
            fwht128(v, lane);
            const std::int64_t a = sorted_assign[job.begin + g0 + pos];
            float4 out;
            const int n = nb * 128 + lane * 4;
            out.x = v[0] * kInvSqrt128 * svh[n];
            out.y = v[1] * kInvSqrt128 * svh[n + 1];
            out.z = v[2] * kInvSqrt128 * svh[n + 2];
            out.w = v[3] * kInvSqrt128 * svh[n + 3];
            *reinterpret_cast<float4*>(partial + a * H + n) = out;
        }
        __syncthreads();
    }
}

} // namespace

void moe_project_exl3(const MoeChunk& chunk, const std::int32_t* misses, bool missed,
                      cudaStream_t stream) {
    const int assignments = chunk.columns * K;
    const JobFilter filter{misses, missed};
    exl3_gate_up_kernel<<<dim3(assignments, kBlocks, kKSplit), kThreads, 0, stream>>>(
        chunk.x, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
        chunk.job_count, assignments, filter, chunk.source, chunk.gu_partial,
        reinterpret_cast<__half*>(chunk.act), chunk.counters);
    CUDA_CHECK(cudaGetLastError());
    exl3_down_kernel<<<dim3(assignments, kDownBlocks), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(chunk.act), chunk.expert_ids, chunk.sorted_assign,
        chunk.sorted_slot, chunk.jobs, chunk.job_count, assignments, filter, chunk.source,
        chunk.partial);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
