// Implements: include/ninfer/ops/offload_moe.h (EXL3 routed experts, prefill route)
//
// The wide-chunk counterpart of exl3_kernels.cu. A chunk's assignments are grouped by cache slot
// into jobs of up to 64 assignments (moe_jobs_kernel), and a job keeps four m16 row tiles, so every
// trellis tile a warp decodes into a B fragment feeds four tensor-core row tiles instead of half of
// one. Three passes, all keyed by the job list so the resident/missed split of the fetch overlap
// still applies:
//
//   rotate   xr = H(suh . x) for the gate and the up input of each job row (FP16, grouped order)
//   gate/up  CTA (job, 128-column block): the whole K = 2560 contraction in 256-wide slices staged
//            through shared memory, then the output Hadamard and scale, SwiGLU, and the down
//            input's scale and Hadamard, written as the FP16 down input in grouped order
//   down     CTA (job, 128-column block of the hidden output): K = 640, output Hadamard and scale,
//            one FP32 partial row per assignment for the shared merge
//
// Every reduction has a fixed order (K ascending in registers, the merge sums the ten assignments in
// order), so cache placement and job order never change a result. The values are those of the
// decode route up to FP32 summation order.
#include "ops/offload_moe/exl3_common.cuh"
#include "ops/offload_moe/jobs.cuh"
#include "ops/offload_moe/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/linear/exl3/exl3_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int H = kOffloadMoeHidden;
constexpr int I = kOffloadMoeIntermediate;
constexpr int K = kOffloadMoeTopK;

constexpr int kThreads    = 256; // 8 warps, one 16-wide output tile of each matrix per warp
constexpr int kGroups    = kJobTokens / 8; // n8 token groups per job
constexpr int kBatch      = 8;   // tile loads in flight per warp before any is decoded
constexpr int kSliceK     = 128; // contraction columns staged in shared memory at a time (one Hadamard block)
constexpr int kSliceTiles = kSliceK / 16;
constexpr int kTilesK     = H / 16;
constexpr int kDownTiles  = I / 16;
constexpr int kBlocks     = I / 128;
constexpr int kDownBlocks = H / 128;
constexpr int kGuStride   = kSliceK + 8; // padded halves per smem row: conflict-free A fragments
constexpr int kHalfRows   = kJobTokens / 2; // rows per epilogue pass
constexpr int kRes        = 129; // FP32 epilogue row stride

static_assert(kJobTokens % 16 == 0);
static_assert(H % kSliceK == 0 && I % kSliceK == 0 && kSliceTiles == kBatch);

// The RTX 5090 offers 100 KB of shared memory per SM, so the staged A tiles stay small enough for two
// CTAs; the epilogue reuses them for half of the job's rows at a time.
constexpr int kGuSmemBytes   = 2 * kJobTokens * kGuStride * 2;
constexpr int kDownSmemBytes = kJobTokens * kGuStride * 2;
static_assert(2 * kHalfRows * kRes * 4 <= kGuSmemBytes, "the epilogue tile reuses the A tiles");
static_assert(kHalfRows * kRes * 4 <= kDownSmemBytes);

// FP16 rotated inputs, [position in grouped order][gate|up][H].
__global__ void __launch_bounds__(kThreads)
    exl3_rotate_kernel(const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ ids,
                       const std::int32_t* __restrict__ sorted_assign,
                       const std::int32_t* __restrict__ sorted_slot,
                       const std::int32_t* __restrict__ jobs,
                       const std::int32_t* __restrict__ job_count, std::int32_t assignments,
                       JobFilter filter, ExpertWeights w, __half* __restrict__ xr) {
    const JobView job = load_job(jobs, job_count, sorted_slot, sorted_assign, ids, assignments, filter);
    if (job.count == 0) { return; }
    const int first = static_cast<int>(blockIdx.y) * 16;
    if (first >= job.count) { return; }
    const int rows = min(16, job.count - first);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto* suh = reinterpret_cast<const float*>(w.base[1] + job.slot * w.stride[1]); // [2][H]
    constexpr int kBlk = H / 128;
    for (int item = warp; item < rows * 2 * kBlk; item += kThreads / 32) {
        const int blk = item % kBlk, proj = (item / kBlk) & 1, r = item / (2 * kBlk);
        const int token = sorted_assign[job.begin + first + r] / K;
        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(token) * H + blk * 128;
        const float* sp           = suh + proj * H + blk * 128;
        const uint2 raw = *reinterpret_cast<const uint2*>(xrow + lane * 4);
        const float4 sv = *reinterpret_cast<const float4*>(sp + lane * 4);
        const float2 x01 = bf16x2_bits_to_float2(raw.x);
        const float2 x23 = bf16x2_bits_to_float2(raw.y);
        float v[4]       = {x01.x * sv.x, x01.y * sv.y, x23.x * sv.z, x23.y * sv.w};
        fwht128(v, lane);
        __half* dst = xr + (static_cast<std::int64_t>(job.begin + first + r) * 2 + proj) * H +
                      blk * 128 + lane * 4;
        uint2 packed;
        packed.x = pack_half2(v[0] * kExl3InvSqrt128, v[1] * kExl3InvSqrt128);
        packed.y = pack_half2(v[2] * kExl3InvSqrt128, v[3] * kExl3InvSqrt128);
        *reinterpret_cast<uint2*>(dst) = packed;
    }
}

// Contraction of `kMats` matrices over `ktiles` k-tiles against the job's tokens in shared memory.
// The decoded weights are the m16 operand (16 output columns by 16 k) and the tokens the n8 operand,
// so rows are padded to 8, not 16: c[m][g] is the 16 output columns of matrix m's tile by the 8
// tokens of group g. `kG` is the number of token groups the job occupies and is a template
// parameter: a runtime test around each MMA is compiled to predication, and a predicated-off HMMA
// still occupies the tensor pipe.
template <int kHB, int kMats, int kG>
__device__ __forceinline__ void tile_gemm_groups(const __half* const (&a_base)[kMats], int a_stride,
                                                 const std::uint8_t* const (&trellis)[kMats],
                                                 int ktiles, int tile_bytes,
                                                 const Exl3TileDecoder& dec, int lane,
                                                 float (&c)[kMats][kGroups][4]) {
    const int token = lane >> 2;
    const int col   = (lane & 3) * 2;
    auto apply = [&](int m, const float (&d)[8], int kt) {
        // The decoder's B-fragment registers are also the A fragment of the transposed product.
        const std::uint32_t a[4] = {pack_half2(d[0], d[1]), pack_half2(d[4], d[5]),
                                    pack_half2(d[2], d[3]), pack_half2(d[6], d[7])};
        const __half* p          = a_base[m] + token * a_stride + kt * 16 + col;
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            const __half* q          = p + g * 8 * a_stride;
            const std::uint32_t b[2] = {load_half2(q), load_half2(q + 8)};
            mma_f16(c[m][g], a, b);
        }
    };
    if (kHB == 8 || dec.prefetchable) {
        for (int kb = 0; kb < ktiles; kb += kBatch) {
            Exl3TileWords words[kMats][kBatch];
#pragma unroll
            for (int m = 0; m < kMats; ++m) {
#pragma unroll
                for (int i = 0; i < kBatch; ++i) {
                    words[m][i] = dec.load_words<kHB>(
                        trellis[m] + static_cast<std::size_t>(kb + i) * tile_bytes, lane);
                }
            }
#pragma unroll
            for (int i = 0; i < kBatch; ++i) {
#pragma unroll
                for (int m = 0; m < kMats; ++m) {
                    float d[8];
                    dec.decode_words<kHB>(words[m][i], d);
                    apply(m, d, kb + i);
                }
            }
        }
    } else {
        for (int kt = 0; kt < ktiles; ++kt) {
#pragma unroll
            for (int m = 0; m < kMats; ++m) {
                float d[8];
                dec.decode(trellis[m] + static_cast<std::size_t>(kt) * tile_bytes, lane, d);
                apply(m, d, kt);
            }
        }
    }
}

template <int kHB, int kMats>
__device__ __forceinline__ void tile_gemm_rows(const __half* const (&a_base)[kMats], int a_stride,
                                               const std::uint8_t* const (&trellis)[kMats],
                                               int ktiles, int tile_bytes,
                                               const Exl3TileDecoder& dec, int lane, int count,
                                               float (&c)[kMats][kGroups][4]) {
    switch ((count + 7) / 8) {
#define NINFER_EXL3_GROUPS(G)                                                                    \
    case G:                                                                                      \
        tile_gemm_groups<kHB, kMats, G>(a_base, a_stride, trellis, ktiles, tile_bytes, dec, lane, c); \
        break;
        NINFER_EXL3_GROUPS(1)
        NINFER_EXL3_GROUPS(2)
        NINFER_EXL3_GROUPS(3)
        NINFER_EXL3_GROUPS(4)
        NINFER_EXL3_GROUPS(5)
        NINFER_EXL3_GROUPS(6)
        NINFER_EXL3_GROUPS(7)
#undef NINFER_EXL3_GROUPS
    default:
        tile_gemm_groups<kHB, kMats, kGroups>(a_base, a_stride, trellis, ktiles, tile_bytes, dec, lane, c);
        break;
    }
}

// Spill the accumulators of the token groups of one half of the job to res[token][n] (row stride
// kRes) for the Hadamard epilogues. Groups the job does not occupy hold zeros.
template <int kMats>
__device__ __forceinline__ void store_accumulators(float* res, int half, int warp, int lane,
                                                   const float (&c)[kMats][kGroups][4]) {
    const int g = lane >> 2;
    const int t = (lane & 3) * 2;
#pragma unroll
    for (int m = 0; m < kMats; ++m) {
#pragma unroll
        for (int h = 0; h < kGroups / 2; ++h) {
            const int group = half * (kGroups / 2) + h;
            float* base     = res + (m * kHalfRows + h * 8 + t) * kRes + warp * 16 + g;
            base[0]         = c[m][group][0]; // token t, column g
            base[kRes]      = c[m][group][1]; // token t + 1, column g
            base[8]         = c[m][group][2]; // token t, column g + 8
            base[kRes + 8]  = c[m][group][3]; // token t + 1, column g + 8
        }
    }
}

template <int kHB>
__global__ void __launch_bounds__(kThreads, 2) exl3_gate_up_prefill_kernel(
    const std::int32_t* __restrict__ ids, const std::int32_t* __restrict__ sorted_assign,
    const std::int32_t* __restrict__ sorted_slot, const std::int32_t* __restrict__ jobs,
    const std::int32_t* __restrict__ job_count, std::int32_t assignments, JobFilter filter,
    ExpertWeights w, const __half* __restrict__ xr, __half* __restrict__ act) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* as = reinterpret_cast<__half*>(smem_raw);
    float* res = reinterpret_cast<float*>(smem_raw);
    const JobView job = load_job(jobs, job_count, sorted_slot, sorted_assign, ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int nb = static_cast<int>(blockIdx.y);
    const std::int64_t slot = job.slot;
    const auto* trellis = reinterpret_cast<const std::uint8_t*>(w.base[0] + slot * w.stride[0]);
    const int half_bits  = w.layout.gate_up_half_bits;
    const int tile_bytes = 16 * half_bits;
    const Exl3TileDecoder dec(half_bits, lane);
    const std::int64_t gate_tile = static_cast<std::int64_t>(nb * 8 + warp);
    const std::int64_t up_tile   = static_cast<std::int64_t>(I / 16 + nb * 8 + warp);
    const std::uint8_t* const tiles[2] = {trellis + gate_tile * kTilesK * tile_bytes,
                                          trellis + up_tile * kTilesK * tile_bytes};
    const __half* const a_base[2] = {as, as + kJobTokens * kGuStride};
    float c[2][kGroups][4] = {};

    for (int slice = 0; slice < H / kSliceK; ++slice) {
        __syncthreads();
        for (int i = tid; i < 2 * kJobTokens * (kSliceK / 8); i += kThreads) {
            const int proj = i / (kJobTokens * (kSliceK / 8));
            const int rem  = i % (kJobTokens * (kSliceK / 8));
            const int r = rem / (kSliceK / 8), v = rem % (kSliceK / 8);
            uint4 value = make_uint4(0U, 0U, 0U, 0U);
            if (r < job.count) {
                value = *reinterpret_cast<const uint4*>(
                    xr + (static_cast<std::int64_t>(job.begin + r) * 2 + proj) * H + slice * kSliceK +
                    v * 8);
            }
            *reinterpret_cast<uint4*>(as + (proj * kJobTokens + r) * kGuStride + v * 8) = value;
        }
        __syncthreads();
        const std::uint8_t* const slice_tiles[2] = {tiles[0] + slice * kSliceTiles * tile_bytes,
                                                    tiles[1] + slice * kSliceTiles * tile_bytes};
        tile_gemm_rows<kHB, 2>(a_base, kGuStride, slice_tiles, kSliceTiles, tile_bytes, dec, lane,
                               job.count, c);
    }

    const auto* svh = reinterpret_cast<const float*>(w.base[2] + slot * w.stride[2]); // [2][I]
    const auto* sd  = reinterpret_cast<const float*>(w.base[4] + slot * w.stride[4]); // [I]
    for (int half = 0; half * kHalfRows < job.count; ++half) {
    __syncthreads();
    store_accumulators<2>(res, half, warp, lane, c);
    __syncthreads();
    const int rows = min(kHalfRows, job.count - half * kHalfRows);
    for (int local = warp; local < rows; local += kThreads / 32) {
        const int pos = half * kHalfRows + local;
        float g[4], u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g[j] = res[local * kRes + lane * 4 + j];
            u[j] = res[(kHalfRows + local) * kRes + lane * 4 + j];
        }
        fwht128(g, lane);
        fwht128(u, lane);
        float a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n    = nb * 128 + lane * 4 + j;
            const float gv = g[j] * kExl3InvSqrt128 * svh[n];
            const float uv = u[j] * kExl3InvSqrt128 * svh[I + n];
            a[j]           = gv / (1.0F + expf(-gv)) * uv * sd[n];
        }
        fwht128(a, lane);
        uint2 packed;
        packed.x = pack_half2(a[0] * kExl3InvSqrt128, a[1] * kExl3InvSqrt128);
        packed.y = pack_half2(a[2] * kExl3InvSqrt128, a[3] * kExl3InvSqrt128);
        *reinterpret_cast<uint2*>(act + static_cast<std::int64_t>(job.begin + pos) * I + nb * 128 +
                                  lane * 4) = packed;
    }
    }
}

template <int kHB>
__global__ void __launch_bounds__(kThreads, 2) exl3_down_prefill_kernel(
    const __half* __restrict__ act, const std::int32_t* __restrict__ ids,
    const std::int32_t* __restrict__ sorted_assign, const std::int32_t* __restrict__ sorted_slot,
    const std::int32_t* __restrict__ jobs, const std::int32_t* __restrict__ job_count,
    std::int32_t assignments, JobFilter filter, ExpertWeights w, float* __restrict__ partial) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* as = reinterpret_cast<__half*>(smem_raw);
    float* res = reinterpret_cast<float*>(smem_raw);
    const JobView job = load_job(jobs, job_count, sorted_slot, sorted_assign, ids, assignments, filter);
    if (job.count == 0) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int nb = static_cast<int>(blockIdx.y);
    const std::int64_t slot = job.slot;
    const auto* trellis = reinterpret_cast<const std::uint8_t*>(w.base[3] + slot * w.stride[3]);
    const auto* svh     = reinterpret_cast<const float*>(w.base[5] + slot * w.stride[5]); // [H]
    const int half_bits  = w.layout.down_half_bits;
    const int tile_bytes = 16 * half_bits;
    const Exl3TileDecoder dec(half_bits, lane);

    float c[1][kGroups][4] = {};
    const std::int64_t tile = static_cast<std::int64_t>(nb * 8 + warp);
    const __half* const a_base[1]      = {as};
    const std::uint8_t* const tiles[1] = {trellis + tile * kDownTiles * tile_bytes};
    for (int slice = 0; slice < I / kSliceK; ++slice) {
        __syncthreads();
        for (int i = tid; i < kJobTokens * (kSliceK / 8); i += kThreads) {
            const int r = i / (kSliceK / 8), v = i % (kSliceK / 8);
            uint4 value = make_uint4(0U, 0U, 0U, 0U);
            if (r < job.count) {
                value = *reinterpret_cast<const uint4*>(
                    act + static_cast<std::int64_t>(job.begin + r) * I + slice * kSliceK + v * 8);
            }
            *reinterpret_cast<uint4*>(as + r * kGuStride + v * 8) = value;
        }
        __syncthreads();
        const std::uint8_t* const slice_tiles[1] = {tiles[0] + slice * kSliceTiles * tile_bytes};
        tile_gemm_rows<kHB, 1>(a_base, kGuStride, slice_tiles, kSliceTiles, tile_bytes, dec, lane,
                               job.count, c);
    }
    for (int half = 0; half * kHalfRows < job.count; ++half) {
    __syncthreads();
    store_accumulators<1>(res, half, warp, lane, c);
    __syncthreads();
    const int rows = min(kHalfRows, job.count - half * kHalfRows);
    for (int local = warp; local < rows; local += kThreads / 32) {
        const int pos = half * kHalfRows + local;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { v[j] = res[local * kRes + lane * 4 + j]; }
        fwht128(v, lane);
        const std::int64_t a = sorted_assign[job.begin + pos];
        const int n          = nb * 128 + lane * 4;
        float4 out;
        out.x = v[0] * kExl3InvSqrt128 * svh[n];
        out.y = v[1] * kExl3InvSqrt128 * svh[n + 1];
        out.z = v[2] * kExl3InvSqrt128 * svh[n + 2];
        out.w = v[3] * kExl3InvSqrt128 * svh[n + 3];
        *reinterpret_cast<float4*>(partial + a * H + n) = out;
    }
    }
}

template <class Kernel>
void allow_dynamic_smem(Kernel kernel, int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    // Without the full carveout the driver leaves ~100 KB of shared memory per SM, i.e. one CTA.
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
}

} // namespace

void moe_project_exl3_prefill(const MoeChunk& chunk, const std::int32_t* misses, bool missed,
                              cudaStream_t stream) {
    const int assignments = chunk.columns * K;
    const JobFilter filter{misses, missed};
    const ExpertLayout& layout = chunk.source.layout;
    auto* xr                   = static_cast<__half*>(chunk.xr);
    const auto gate_up =
        layout.gate_up_half_bits == 8 ? exl3_gate_up_prefill_kernel<8> : exl3_gate_up_prefill_kernel<0>;
    const auto down =
        layout.down_half_bits == 8 ? exl3_down_prefill_kernel<8> : exl3_down_prefill_kernel<0>;
    static const bool configured = [&] {
        allow_dynamic_smem(exl3_gate_up_prefill_kernel<8>, kGuSmemBytes);
        allow_dynamic_smem(exl3_gate_up_prefill_kernel<0>, kGuSmemBytes);
        allow_dynamic_smem(exl3_down_prefill_kernel<8>, kDownSmemBytes);
        allow_dynamic_smem(exl3_down_prefill_kernel<0>, kDownSmemBytes);
        return true;
    }();
    (void)configured;

    exl3_rotate_kernel<<<dim3(assignments, kJobTokens / 16), kThreads, 0, stream>>>(
        chunk.x, chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs,
        chunk.job_count, assignments, filter, chunk.source, xr);
    CUDA_CHECK(cudaGetLastError());
    gate_up<<<dim3(assignments, kBlocks), kThreads, kGuSmemBytes, stream>>>(
        chunk.expert_ids, chunk.sorted_assign, chunk.sorted_slot, chunk.jobs, chunk.job_count,
        assignments, filter, chunk.source, xr, reinterpret_cast<__half*>(chunk.act));
    CUDA_CHECK(cudaGetLastError());
    down<<<dim3(assignments, kDownBlocks), kThreads, kDownSmemBytes, stream>>>(
        reinterpret_cast<const __half*>(chunk.act), chunk.expert_ids, chunk.sorted_assign,
        chunk.sorted_slot, chunk.jobs, chunk.job_count, assignments, filter, chunk.source,
        chunk.partial);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
