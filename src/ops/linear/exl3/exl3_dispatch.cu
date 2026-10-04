// EXL3 mul1 linear decode. Correctness-first SIMT implementation of the stored reconstruction:
//
//   W[n,k] = su[k] · sv[n] · (H128 · Z · H128)[k,n],   out[n,t] = Σ_k W[n,k] x[k,t]
//
// A first kernel scales the activation rows by su and applies the input Hadamard over K. A second
// kernel contracts the decoded trellis with that rotated activation, applies the output Hadamard
// over N, and scales by sv. The 1/sqrt(128) factor of each Hadamard is folded into one 1/128 at the
// end. This is the "simple correct" M3 route; M4 replaces it with MMA kernels.

#include "ops/linear/exl3/exl3_dispatch.h"
#include "ops/linear/exl3/exl3_decode.cuh"

#include "core/device.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {


// The contraction kernels decode a trellis window per weight, so they are latency-bound rather
// than DRAM-bound: NCU shows them register-limited at 2 blocks/SM (about 16% occupancy, SMs active
// 8% of elapsed cycles). launch_bounds makes nvcc spend registers on occupancy instead, which is
// the whole lever for decode and prefill alike. The two families peak differently -- the GEMV wants
// 5 blocks/SM and the MMA 6, where a tighter cap starts spilling its accumulators -- so they carry
// separate targets.
constexpr int kExl3Threads        = 256;
constexpr int kExl3GemvMinBlocks = 6;
// The MMA block keeps a [2][TILE_T/16][4] FP32 accumulator per thread; the 6-block register budget
// (42 registers) cannot hold it and ptxas spills to local memory, so the register target is 3
// blocks (85 registers) and the 16 KiB shared footprint leaves room for more if registers allow.
// Measured: at 4 blocks the accumulator no longer fits and prefill falls to 832 tok/s from 1,420.
constexpr int kExl3MmaTarget     = 3;

// In-place unnormalized 128-point Sylvester Hadamard over 128 shared floats; all 128 threads join.
__device__ __forceinline__ void butterfly128(float* s, int lane) {
    for (int stride = 1; stride < 128; stride <<= 1) {
        const float a = s[lane];
        const float b = s[lane ^ stride];
        __syncthreads();
        s[lane] = (lane & stride) ? (b - a) : (a + b);
        __syncthreads();
    }
}

// u[k,t] = BF16(H128(su[k] * x[k,t])) over 128-blocks of k. u is BF16 scratch [K,T] in the same
// column-major order as the caller's tensors (dim 0 is contiguous). The Hadamard runs in FP32 and
// rounds once on store: every contraction consumes a BF16 A operand (the MMA atoms are BF16, and
// the GEMV rounds on pack), so a FP32 stage would only round later while costing twice the scratch
// traffic and, in the MMA, a bank-conflicted strided A read.
__global__ void exl3_input_transform(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ su, __nv_bfloat16* __restrict__ u,
                                     int k_extent, int columns) {
    const int block = blockIdx.x;
    const int t     = blockIdx.y;
    const int lane  = threadIdx.x;
    const int k     = block * 128 + lane;
    if (k >= k_extent) { return; }
    __shared__ float s[128];
    s[lane] = su[k] * __bfloat162float(x[k + static_cast<std::size_t>(t) * k_extent]);
    __syncthreads();
    butterfly128(s, lane);
    u[static_cast<std::size_t>(t) * k_extent + k] = __float2bfloat16(s[lane]);
}

// Decode contraction: one column, so parallelism comes from splitting K across the grid. A warp owns
// one 16-row trellis tile (n = 16w..16w+15) and lane L decodes the eight windows t = 8L..8L+7, which
// are the m16n8k16 B-fragment windows: for 4-bit rates that is exactly one 32-bit word per lane, so
// the tile loads are coalesced instead of scattered. The lane's four h=0 windows sum into n = L/4 and
// the four h=1 windows into n = L/4+8; a four-lane shuffle folds the k-split and one lane per n
// atomically adds into a small FP32 buffer. A second kernel applies the 128-point output Hadamard.
constexpr int kExl3GemvSplits = 32;

// Seeded per-element Gaussian noise for the sensitivity probe. A 32-bit hash of (n, k, seed) drives
// Box-Muller, so a trellis decode is exact but the perturbation is repeatable and independent of the
// launch geometry.
__device__ __forceinline__ float probe_gaussian(std::uint32_t n, std::uint32_t k,
                                                std::uint32_t seed) {
    const auto hash = [](std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    };
    const std::uint32_t h0 = hash(n * 0x9e3779b9U ^ k ^ seed);
    const std::uint32_t h1 = hash(h0 ^ 0x85ebca6bU);
    const float u0         = (static_cast<float>(h0 >> 8) + 0.5F) * (1.0F / 16777216.0F);
    const float u1         = (static_cast<float>(h1 >> 8) + 0.5F) * (1.0F / 16777216.0F);
    return sqrtf(-2.0F * logf(u0)) * cosf(6.283185307179586F * u1);
}
constexpr int kExl3GemvMaxM   = 8; // largest m the per-column GEMV beats the tiled MMA at

// T_MAX columns share one decode of the trellis tile: the weight is the expensive operand, and the
// verify pass runs the same weight against every drafted token. Each lane keeps T_MAX x 8
// accumulators (its two n8 groups by four k positions), which the tensor-core GEMV of the reference
// gets from the C fragment; here it is 8 floats per column.
// `kProbe` is a compile-time switch. The sensitivity probe's per-window Gaussian is dead weight in
// the served path, where `probe_sigma` is zero: keeping it in the same instantiation costs a uniform
// branch per k-tile per column and, worse, the probe's temporaries sit in the kernel's register
// budget, which is what caps this GEMV at five blocks per SM.
// `kRate4Bit` pins the 4-bit decode, which is the bulk of the model's projections by call count. As
// a template parameter it removes the per-k-tile rate branch chain from the hot loop and keeps the
// wide rates' six window registers out of this instantiation's register budget.
template <int T_MAX, bool kProbe, bool kRate4Bit>
__global__ void __launch_bounds__(kExl3Threads, kExl3GemvMinBlocks) exl3_gemv_split(
    const __nv_bfloat16* __restrict__ u, const std::uint8_t* __restrict__ trellis,
    float* __restrict__ partial, int k_extent, int n_extent, int half_bits, int columns,
    float probe_sigma, std::uint32_t probe_seed) {
    if constexpr (!kProbe) {
        (void)probe_sigma;
        (void)probe_seed;
    }
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_tile  = blockIdx.x * 8 + warp;
    const int n_base  = n_tile * 16;
    if (n_base >= n_extent) { return; }
    const int kt_tiles = k_extent / 16;
    const int splits   = gridDim.y;
    const int per      = (kt_tiles + splits - 1) / splits;
    const int kt_begin = blockIdx.y * per;
    const int kt_end   = min(kt_tiles, kt_begin + per);
    const int bits  = half_bits >> 1;
    const int words = 4 * half_bits;
    const bool fast = (half_bits & 1) == 0;
    if constexpr (kRate4Bit) {
        (void)bits;
        (void)words;
        (void)fast;
    }

    // Even rates resolve the two four-window groups with two lane-constant funnel shifts (exllamav3's
    // dq4 applied twice); the word indices and shifts do not depend on the k-tile.
    int wlo0 = 0, wn0 = 0, sft0 = 0, wlo1 = 0, wn1 = 0, sft1 = 0;
    if constexpr (!kRate4Bit) {
        if (fast && bits != 4) {
            const int total = words * 32;
            const int s0    = ((8 * lane + 1) * bits - 16 + total) % total;
            const int s1    = ((8 * lane + 5) * bits - 16 + total) % total;
            wlo0            = s0 >> 5;
            sft0            = s0 & 31;
            wn0             = (wlo0 + 1 == words) ? 0 : wlo0 + 1;
            wlo1            = s1 >> 5;
            sft1            = s1 & 31;
            wn1             = (wlo1 + 1 == words) ? 0 : wlo1 + 1;
        }
    }

    float a0[T_MAX][4];
    float a1[T_MAX][4];
#pragma unroll
    for (int t = 0; t < T_MAX; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a0[t][i] = 0.0F;
            a1[t][i] = 0.0F;
        }

    const std::uint8_t* tile_base =
        trellis + static_cast<std::size_t>(n_tile) * kt_tiles * (16 * half_bits);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const std::uint8_t* tile8 =
            tile_base + static_cast<std::size_t>(kt) * (16 * half_bits);
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        float d[8];
        if constexpr (kRate4Bit) {
            exl3_windows_4bit(tile32, lane, d);
        } else if (half_bits == 8) {
            exl3_windows_4bit(tile32, lane, d);
        } else if (fast) {
            // A four-window group can span more than 32 bits at 5 and 6 bits per weight, so resolve it
            // from a 64-bit window over the lane's word and its successor.
            const unsigned long long w0 =
                (static_cast<unsigned long long>(tile32[wn0]) << 32) | tile32[wlo0];
            const unsigned long long w1 =
                (static_cast<unsigned long long>(tile32[wn1]) << 32) | tile32[wlo1];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[i] = mul1_value(
                    static_cast<std::uint16_t>((w0 >> (sft0 + bits * i)) & 0xFFFFU));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[4 + i] = mul1_value(
                    static_cast<std::uint16_t>((w1 >> (sft1 + bits * i)) & 0xFFFFU));
            }
        } else if (exl3_half_fast(half_bits)) {
            exl3_windows_half(tile32, lane, half_bits, d);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                d[r] = mul1_value(tile_state(tile8, half_bits, 8 * lane + r));
            }
        }
        if constexpr (kProbe) {
            if (probe_sigma > 0.0F) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int nn = n_base + (lane >> 2) + 8 * (r >> 2);
                    const int kk = kt * 16 + 2 * (lane & 3) + (r & 1) + 8 * ((r >> 1) & 1);
                    d[r] += probe_sigma * probe_gaussian(static_cast<std::uint32_t>(nn),
                                                         static_cast<std::uint32_t>(kk),
                                                         probe_seed);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T_MAX; ++t) {
            const bool active = t < columns;
            const __nv_bfloat16* uk =
                u + static_cast<std::size_t>(active ? t : 0) * k_extent + kt * 16;
            // The lane's four k positions are the adjacent pairs 2c, 2c+1 and 2c+8, 2c+9, so one
            // 32-bit BF16 pair load serves each half of the fragment instead of two 16-bit halves:
            // two latency-bearing loads per column instead of four, on a kernel NCU calls latency
            // bound rather than pipe bound.
            const int c = 2 * (lane & 3);
            const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(uk + c);
            const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(uk + c + 8);
            const float op[4] = {active ? __low2float(lo) : 0.0F, active ? __high2float(lo) : 0.0F,
                                 active ? __low2float(hi) : 0.0F,
                                 active ? __high2float(hi) : 0.0F};
            a0[t][0] = fmaf(d[0], op[0], a0[t][0]);
            a0[t][1] = fmaf(d[1], op[1], a0[t][1]);
            a0[t][2] = fmaf(d[2], op[2], a0[t][2]);
            a0[t][3] = fmaf(d[3], op[3], a0[t][3]);
            a1[t][0] = fmaf(d[4], op[0], a1[t][0]);
            a1[t][1] = fmaf(d[5], op[1], a1[t][1]);
            a1[t][2] = fmaf(d[6], op[2], a1[t][2]);
            a1[t][3] = fmaf(d[7], op[3], a1[t][3]);
        }
    }
#pragma unroll
    for (int t = 0; t < T_MAX; ++t) {
        if (t >= columns) { break; }
        float acc0 = (a0[t][0] + a0[t][1]) + (a0[t][2] + a0[t][3]);
        float acc1 = (a1[t][0] + a1[t][1]) + (a1[t][2] + a1[t][3]);
        acc0 += __shfl_down_sync(0xFFFFFFFFU, acc0, 2);
        acc0 += __shfl_down_sync(0xFFFFFFFFU, acc0, 1);
        acc1 += __shfl_down_sync(0xFFFFFFFFU, acc1, 2);
        acc1 += __shfl_down_sync(0xFFFFFFFFU, acc1, 1);
        if ((lane & 3) == 0) {
            const int g = lane >> 2;
            float* column = partial + static_cast<std::size_t>(t) * n_extent;
            atomicAdd(&column[n_base + g], acc0);
            atomicAdd(&column[n_base + g + 8], acc1);
        }
    }
}

__global__ void exl3_gemv_finish(const float* __restrict__ partial, const float* __restrict__ sv,
                                 __nv_bfloat16* __restrict__ out, int n_extent) {
    const int lane = threadIdx.x;
    const int n    = blockIdx.x * 128 + lane;
    if (n >= n_extent) { return; }
    __shared__ float s[128];
    s[lane] = partial[n];
    __syncthreads();
    butterfly128(s, lane);
    out[n] = __float2bfloat16(s[lane] * sv[n] * (1.0F / 128.0F));
}

// Two bf16 values in the 32-bit register the MMA atom consumes.
__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 value = __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
    std::uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// The same register for a pair that is already packed in storage: an aligned 32-bit load instead of
// two converts and a permute. Both operands are BF16 in memory, so no rounding happens here.
__device__ __forceinline__ std::uint32_t load_bf16x2(const __nv_bfloat16* pair) {
    const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(pair);
    std::uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__device__ __forceinline__ void mma_m16n8k16(float d[4], const std::uint32_t a[4],
                                             const std::uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Small-m tensor-core GEMV: the verify pass runs one weight against up to eight drafted tokens, so
// the m16n8k16 A fragment carries the tokens and one C fragment carries all of them. The reference's
// exl3_gemv_kernel does exactly this for 2 <= m <= 8. The FFMA alternative needs eight accumulators
// per column (128 registers at four columns, halving occupancy), which is why it loses.
__global__ void __launch_bounds__(kExl3Threads, kExl3GemvMinBlocks) exl3_gemv_mma(
    const __nv_bfloat16* __restrict__ u, const std::uint8_t* __restrict__ trellis,
    float* __restrict__ partial, int k_extent, int n_extent, int half_bits, int columns) {
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_tile  = blockIdx.x * 8 + warp;
    const int n_base  = n_tile * 16;
    if (n_base >= n_extent) { return; }
    const int kt_tiles = k_extent / 16;
    const int splits   = gridDim.y;
    const int per      = (kt_tiles + splits - 1) / splits;
    const int kt_begin = blockIdx.y * per;
    const int kt_end   = min(kt_tiles, kt_begin + per);
    const int bits     = half_bits >> 1;
    const int words    = 4 * half_bits;
    const bool fast    = (half_bits & 1) == 0;

    int wlo0 = 0, wn0 = 0, sft0 = 0, wlo1 = 0, wn1 = 0, sft1 = 0;
    if (fast && bits != 4) {
        const int total = words * 32;
        const int s0    = ((8 * lane + 1) * bits - 16 + total) % total;
        const int s1    = ((8 * lane + 5) * bits - 16 + total) % total;
        wlo0            = s0 >> 5;
        sft0            = s0 & 31;
        wn0             = (wlo0 + 1 == words) ? 0 : wlo0 + 1;
        wlo1            = s1 >> 5;
        sft1            = s1 & 31;
        wn1             = (wlo1 + 1 == words) ? 0 : wlo1 + 1;
    }

    // A fragment: rows are the drafted tokens (only rows < columns exist), the k pair is the lane's.
    const int row = lane >> 2;
    const int col = (lane & 3) * 2;
    float c[2][4] = {{0.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};

    const std::uint8_t* tile_base =
        trellis + static_cast<std::size_t>(n_tile) * kt_tiles * (16 * half_bits);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const std::uint8_t* tile8 =
            tile_base + static_cast<std::size_t>(kt) * (16 * half_bits);
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        float d[8];
        if (half_bits == 8) {
            exl3_windows_4bit(tile32, lane, d);
        } else if (fast) {
            const unsigned long long w0 =
                (static_cast<unsigned long long>(tile32[wn0]) << 32) | tile32[wlo0];
            const unsigned long long w1 =
                (static_cast<unsigned long long>(tile32[wn1]) << 32) | tile32[wlo1];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[i] = mul1_value(
                    static_cast<std::uint16_t>((w0 >> (sft0 + bits * i)) & 0xFFFFU));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[4 + i] = mul1_value(
                    static_cast<std::uint16_t>((w1 >> (sft1 + bits * i)) & 0xFFFFU));
            }
        } else if (exl3_half_fast(half_bits)) {
            exl3_windows_half(tile32, lane, half_bits, d);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                d[r] = mul1_value(tile_state(tile8, half_bits, 8 * lane + r));
            }
        }
        const std::uint32_t b[2][2] = {{pack_bf16x2(d[0], d[1]), pack_bf16x2(d[2], d[3])},
                                       {pack_bf16x2(d[4], d[5]), pack_bf16x2(d[6], d[7])}};
        std::uint32_t a[4] = {0U, 0U, 0U, 0U};
        if (row < columns) {
            const __nv_bfloat16* urow =
                u + static_cast<std::size_t>(row) * k_extent + kt * 16;
            a[0] = load_bf16x2(urow + col);
            a[2] = load_bf16x2(urow + col + 8);
        }
        mma_m16n8k16(c[0], a, b[0]);
        mma_m16n8k16(c[1], a, b[1]);
    }
    if (row < columns) {
        float* out = partial + static_cast<std::size_t>(row) * n_extent + n_base + col;
        atomicAdd(out, c[0][0]);
        atomicAdd(out + 1, c[0][1]);
        atomicAdd(out + 8, c[1][0]);
        atomicAdd(out + 9, c[1][1]);
    }
}

// Tensor-core contraction. A block covers one 128-row Hadamard block and up to TILE_T tokens.
// The stored tile order is the m16n8k16 B-fragment order, so each warp decodes the trellis tile for
// its own 16 output rows straight into its two B fragments with no shuffle: every lane reads and
// converts its eight windows exactly once per k-tile, instead of re-reading a full [TILE_T][16]
// A tile from shared memory for every (fragment, h) pair after a whole-block barrier. The A stage
// is a double-buffered __pipeline copy (the next k-tile overlaps the current mma chain). A first
// version of this kernel shared one decoded B across the whole block and measured 2 blocks/SM and
// 18.8 ms per [5120,34816]x64 call at 4.0 bpw; the per-warp form reaches 6 blocks/SM and 3.2 ms on
// the same shape.
constexpr int kExl3MmaN       = 128;
constexpr int kExl3MmaThreads = 256;
constexpr int kExl3MmaT       = 64; // columns per block at large T
constexpr int kExl3MmaSmallT  = 16; // columns per block at small T, so the grid still has blocks

// Double-buffered A staging via __pipeline: the next k-tile's 4 KiB copy overlaps the current
// mma chain, so the shared-memory read and the u load stop serializing on the same barrier pair.
constexpr int kExl3AStages = 2;

// TILE_T is a multiple of 16 (the m16n8k16 m). A smaller tile gives the grid more blocks when the
// column count is small, which is the difference between an occupancy-starved and a busy kernel.
//
// `kProbe` and `kRate4Bit` are compile-time switches for the two branches that otherwise sit in the
// hot K loop: the sensitivity probe's per-window Gaussian (never taken in service) and the
// half_bits-class chain, which also carries the wide even rates' eight-register window plan. The
// 4-bit rate is the bulk of the model, and nothing else needs `plan`, so the served 4-bit
// instantiation carries neither.
template <int TILE_T, bool kProbe = true, bool kRate4Bit = false>
__global__ void __launch_bounds__(kExl3Threads, kExl3MmaTarget)
exl3_mma(const __nv_bfloat16* __restrict__ u, const std::uint8_t* __restrict__ trellis,
         const float* __restrict__ sv, __nv_bfloat16* __restrict__ out,
         int k_extent, int n_extent, int columns, int half_bits,
         float probe_sigma, std::uint32_t probe_seed) {
    constexpr int MT = TILE_T / 16;
    if constexpr (!kProbe) {
        (void)probe_sigma;
        (void)probe_seed;
    }
    const int n_base   = blockIdx.x * kExl3MmaN;
    const int t0       = blockIdx.y * TILE_T;
    const int lane     = threadIdx.x & 31;
    const int warp     = threadIdx.x >> 5;
    const int t_count  = min(TILE_T, columns - t0);
    const int n_tile   = blockIdx.x * 8 + warp;
    const int kt_tiles = k_extent / 16;
    const int bits     = half_bits >> 1;
    const int words    = 4 * half_bits;
    const bool fast    = (half_bits & 1) == 0;

    // A is staged as BF16 split into two k-planes: plane p holds k = 8p..8p+7 of the k-tile, so a
    // plane row is 16 bytes (four shared banks) and the eight rows a fragment spans land on the 32
    // banks one word apart. Each of a lane's four A-fragment registers is then a single 32-bit load
    // with no conflict. Staging FP32 in one 16-wide row instead cost a measured 3.9-way conflict on
    // every A load -- 74% of this kernel's shared wavefronts.
    __shared__ __nv_bfloat16 us[kExl3AStages][2][TILE_T][8];
    // One 16-row sub-tile of pre-Hadamard outputs, reused across the MT rounds of the epilogue.
    // A full [TILE_T][128] buffer would double the block's footprint and cut occupancy in half.
    __shared__ float vs[16][kExl3MmaN];
    float acc[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[h][m][c] = 0.0F; }

    const std::size_t tile_stride = static_cast<std::size_t>(16 * half_bits);
    const std::uint8_t* warp_tile =
        trellis + static_cast<std::size_t>(n_tile) * kt_tiles * tile_stride;

    // The wide even rates resolve a window pair from a lane-constant word/shift plan; the 4-bit
    // instantiation decodes the tile with `exl3_windows_4bit` and does not carry its eight registers.
    std::uint16_t plan[8];
    if constexpr (kRate4Bit) {
        (void)plan;
        (void)bits;
        (void)words;
        (void)fast;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) { plan[i] = exl3_window(8 * lane + i, bits, words); }
    }

    // One [TILE_T][16] BF16 tile copies as TILE_T*2 16-byte copies (all 256 threads at
    // TILE_T = 64): copy 2c + p fetches token column c's plane p, i.e. its eight k values
    // 8p..8p+7, as one contiguous 16-byte row. Columns past t_count zero-fill, which the mma
    // consumes as padding.
    const auto stage_tile = [&](int tile) {
        const int copy_id = static_cast<int>(threadIdx.x);
        if (copy_id < TILE_T * 2) {
            const int src_col = copy_id >> 1;      // 0..TILE_T-1: token column
            const int plane   = copy_id & 1;       // k 0..7 or 8..15 of the k-tile
            const int bytes   = (src_col < t_count) ? 16 : 0;
            const __nv_bfloat16* source =
                u + static_cast<std::size_t>(t0 + src_col) * k_extent + tile * 16 + plane * 8;
            cp_async_zfill<16>(&us[tile & (kExl3AStages - 1)][plane][src_col][0], source, bytes);
        }
    };

    // Iteration kt consumes the group staged for kt and issues the stage for kt + 1, whose DRAM
    // latency hides behind kt's mma chain. The wait leaves only the groups issued after kt's
    // pending, so kt's stage is complete; the trailing barrier keeps the next stage from
    // overwriting a slot a warp is still reading. A first version staged and waited inside the
    // same iteration with wait_group 0, which serialized every k-tile on its own load -- and,
    // missing the wait at kt = 0, let the first mma read uninitialized shared memory.
    if (kt_tiles > 0) { stage_tile(0); }
    cp_commit();
    for (int kt = 0; kt < kt_tiles; ++kt) {
        // When a new stage joins the queue the wait can leave one group outstanding (the one just
        // staged); the last iteration stages nothing, so its own tile is the only group left and
        // the wait must drain it completely before the mma reads the slot.
        if (kt + 1 < kt_tiles) { stage_tile(kt + 1); }
        cp_commit();
        if (kt + 1 < kt_tiles) { cp_wait<kExl3AStages - 1>(); } else { cp_wait<0>(); }
        __syncthreads();

        const int slot = kt & (kExl3AStages - 1);

        const std::uint8_t* tile8 = warp_tile + static_cast<std::size_t>(kt) * tile_stride;
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        if (!kRate4Bit && fast) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float d[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    d[i] = exl3_window_value(tile32, plan[4 * h + i], words);
                }
                if constexpr (kProbe) {
                    if (probe_sigma > 0.0F) {
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const int nn = n_base + h * 8 + (lane >> 2);
                            const int kk = kt * 16 + 2 * (lane & 3) + (r & 1) + 8 * (r >> 1);
                            d[r] += probe_sigma *
                                    probe_gaussian(static_cast<std::uint32_t>(nn),
                                                   static_cast<std::uint32_t>(kk), probe_seed);
                        }
                    }
                }
                const std::uint32_t b0 = pack_bf16x2(d[0], d[1]);
                const std::uint32_t b1 = pack_bf16x2(d[2], d[3]);
                const std::uint32_t b[2] = {b0, b1};
                // The m16n8k16 A fragment: rows groupID and groupID + 8, the lane's k pair and
                // its +8 partner. Plane 0 carries k 0..7 and plane 1 k 8..15, so the +8 partner of
                // a[0]/a[1] is a[2]/a[3] at the same in-plane offset.
                const int col = (lane & 3) * 2; // bf16 index within a plane row
                const int grp = lane >> 2;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const int t = mt * 16 + grp;
                    const std::uint32_t a0 = load_bf16x2(&us[slot][0][t][col]);
                    const std::uint32_t a1 = load_bf16x2(&us[slot][0][t + 8][col]);
                    const std::uint32_t a2 = load_bf16x2(&us[slot][1][t][col]);
                    const std::uint32_t a3 = load_bf16x2(&us[slot][1][t + 8][col]);
                    const std::uint32_t a[4] = {a0, a1, a2, a3};
                    mma_m16n8k16(acc[h][mt], a, b);
                }
            }
        } else {
            float d[8];
            if constexpr (kRate4Bit) {
                exl3_windows_4bit(tile32, lane, d);
            } else if (exl3_half_fast(half_bits)) {
                exl3_windows_half(tile32, lane, half_bits, d);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    d[i] = mul1_value(tile_state(tile8, half_bits, 8 * lane + i));
                }
            }
            if constexpr (kProbe) {
                if (probe_sigma > 0.0F) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const int nn = n_base + h * 8 + (lane >> 2);
                            const int kk = kt * 16 + 2 * (lane & 3) + (r & 1) + 8 * (r >> 1);
                            d[4 * h + r] += probe_sigma * probe_gaussian(
                                                        static_cast<std::uint32_t>(nn),
                                                        static_cast<std::uint32_t>(kk), probe_seed);
                        }
                    }
                }
            }
            std::uint32_t b[2][2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                b[h][0] = pack_bf16x2(d[4 * h + 0], d[4 * h + 1]);
                b[h][1] = pack_bf16x2(d[4 * h + 2], d[4 * h + 3]);
            }
            const int col = (lane & 3) * 2; // bf16 index within a plane row
            const int grp = lane >> 2;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int t = mt * 16 + grp;
                const std::uint32_t a0 = load_bf16x2(&us[slot][0][t][col]);
                const std::uint32_t a1 = load_bf16x2(&us[slot][0][t + 8][col]);
                const std::uint32_t a2 = load_bf16x2(&us[slot][1][t][col]);
                const std::uint32_t a3 = load_bf16x2(&us[slot][1][t + 8][col]);
                const std::uint32_t a[4] = {a0, a1, a2, a3};
                mma_m16n8k16(acc[0][mt], a, b[0]);
                mma_m16n8k16(acc[1][mt], a, b[1]);
            }
        }

        __syncthreads();
    }
    __syncthreads();

    // Publish one 16-row sub-tile at a time, apply the output Hadamard over its 128 columns, and
    // store, then reuse the buffer for the next sub-tile. Threads 0..127 transform even rows and
    // 128..255 odd ones; every thread reaches every barrier.
    const int half_lane = threadIdx.x & 127;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n0     = (2 * warp + h) * 8 + (lane & 3) * 2;
            const int t_a    = lane >> 2;
            vs[t_a][n0]      = acc[h][mt][0];
            vs[t_a][n0 + 1]  = acc[h][mt][1];
            vs[t_a + 8][n0]  = acc[h][mt][2];
            vs[t_a + 8][n0 + 1] = acc[h][mt][3];
        }
        __syncthreads();
        for (int pair = 0; pair < 8; ++pair) {
            const int t_even = 2 * pair;
            const int t_odd  = 2 * pair + 1;
            float* target    = (threadIdx.x < 128) ? &vs[t_even][0] : &vs[t_odd][0];
            butterfly128(target, half_lane);
            const int t = (threadIdx.x < 128) ? t_even : t_odd;
            if (mt * 16 + t < t_count) {
                const int n = n_base + half_lane;
                out[n + static_cast<std::size_t>(t0 + mt * 16 + t) * n_extent] =
                    __float2bfloat16(target[half_lane] * sv[n] * (1.0F / 128.0F));
            }
        }
        __syncthreads();
    }
}

} // namespace

void exl3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("exl3 linear: only A16Only is supported");
    }
    if (w.qtype != QType::EXL3_MUL1 || w.layout != QuantLayout::TrellisT16 ||
        w.input_scales == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("exl3 linear: weight is not a complete EXL3 parent");
    }
    if (w.bitrate_half_bits < 2 || w.bitrate_half_bits > 16) {
        throw std::invalid_argument("exl3 linear: unsupported bitrate");
    }
    if (w.k % 128 != 0 || w.n % 128 != 0) {
        throw std::invalid_argument("exl3 linear: N and K must be 128-aligned");
    }
    const std::int32_t columns = x.ne[1];
    if (columns <= 0) { throw std::invalid_argument("exl3 linear: T must be positive"); }

    const std::size_t u_bytes =
        static_cast<std::size_t>(w.k) * static_cast<std::size_t>(columns) * sizeof(__nv_bfloat16);
    const std::size_t partial_bytes =
        columns <= kExl3GemvMaxM ? static_cast<std::size_t>(columns) * w.n * sizeof(float) : 0;
    auto scope               = workspace.scope();
    const DeviceSpan scratch = workspace.alloc_bytes(u_bytes + partial_bytes, 16);
    __nv_bfloat16* u         = static_cast<__nv_bfloat16*>(scratch.data);
    float* partial = reinterpret_cast<float*>(static_cast<std::uint8_t*>(scratch.data) + u_bytes);

    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output      = static_cast<__nv_bfloat16*>(out.data);
    // The sensitivity probe perturbs one weight's decoded values; it is off unless the registry names
    // this weight's trellis plane. `probe_sigma` is the per-element standard deviation, so
    // ||noise||_F = rfn * ||W||_F.
    const Exl3WeightProbe& probe = exl3_weight_probe();
    const float probe_sigma =
        (probe.target == w.qdata && probe.rfn > 0.0F) ? probe.rfn * probe.rms : 0.0F;
    const std::uint32_t probe_seed = probe.seed;
    exl3_input_transform<<<dim3(static_cast<unsigned>(w.k / 128), columns), 128, 0, stream>>>(
        input, static_cast<const float*>(w.input_scales), u, w.k, columns);
    if (columns > kExl3GemvMaxM) {
        // A small column count takes the smaller tile so the grid keeps enough blocks to fill the GPU.
        // The 4-bit rate is pinned for service -- it carries most of the model's projections -- and
        // the probe instantiation is only entered by `ninfer-sensitivity`.
        const int mma_half_bits = static_cast<int>(w.bitrate_half_bits);
        const auto* mma_trellis   = static_cast<const std::uint8_t*>(w.qdata);
        const auto* mma_scales    = static_cast<const float*>(w.scales);
        const bool rate4          = mma_half_bits == 8;
        const bool probe_on       = probe_sigma > 0.0F;
        if (columns <= 64) {
            const dim3 tile_grid(static_cast<unsigned>(w.n / 128),
                                 static_cast<unsigned>((columns + kExl3MmaSmallT - 1) /
                                                       kExl3MmaSmallT));
            if (probe_on) {
                exl3_mma<kExl3MmaSmallT, true, false><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            } else if (rate4) {
                exl3_mma<kExl3MmaSmallT, false, true><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            } else {
                exl3_mma<kExl3MmaSmallT, false, false><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            }
        } else {
            const dim3 tile_grid(static_cast<unsigned>(w.n / 128),
                                 static_cast<unsigned>((columns + kExl3MmaT - 1) / kExl3MmaT));
            if (probe_on) {
                exl3_mma<kExl3MmaT, true, false><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            } else if (rate4) {
                exl3_mma<kExl3MmaT, false, true><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            } else {
                exl3_mma<kExl3MmaT, false, false><<<tile_grid, kExl3MmaThreads, 0, stream>>>(
                    u, mma_trellis, mma_scales, output, w.k, w.n, columns, mma_half_bits,
                    probe_sigma, probe_seed);
            }
        }
    } else {
        // One decode of every trellis tile serves all columns: the verify pass runs each weight
        // against every drafted token, and the decode, not the multiply, is the cost. T_MAX = 1
        // keeps the single-column decode byte-for-byte the same. Splitting K only pays while the grid
        // is short of blocks; a vocabulary-sized head already has thousands of row blocks, so it takes
        // one slice and avoids a scheduling- and atomic-bound grid.
        const int n_blocks = static_cast<int>(w.n / 128);
        const int splits   = std::min<int>(kExl3GemvSplits, std::max<int>(1, 8192 / n_blocks));
        const int half_bits = static_cast<int>(w.bitrate_half_bits);
        const auto* trellis = static_cast<const std::uint8_t*>(w.qdata);
        CUDA_CHECK(cudaMemsetAsync(partial, 0, partial_bytes, stream));
        const dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(splits));
        if (columns == 1) {
            // The 4-bit rate covers most of the model's projections by call count, so pinning it
            // keeps the rate branch chain and the wide rates' window registers out of that
            // instantiation; the probe variant is only entered by `ninfer-sensitivity`.
            const bool rate4 = half_bits == 8;
            const bool probe = probe_sigma > 0.0F;
            if (probe) {
                if (rate4) {
                    exl3_gemv_split<1, true, true><<<grid, 256, 0, stream>>>(
                        u, trellis, partial, w.k, w.n, half_bits, 1, probe_sigma, probe_seed);
                } else {
                    exl3_gemv_split<1, true, false><<<grid, 256, 0, stream>>>(
                        u, trellis, partial, w.k, w.n, half_bits, 1, probe_sigma, probe_seed);
                }
            } else if (rate4) {
                exl3_gemv_split<1, false, true><<<grid, 256, 0, stream>>>(
                    u, trellis, partial, w.k, w.n, half_bits, 1, probe_sigma, probe_seed);
            } else {
                exl3_gemv_split<1, false, false><<<grid, 256, 0, stream>>>(
                    u, trellis, partial, w.k, w.n, half_bits, 1, probe_sigma, probe_seed);
            }
        } else {
            exl3_gemv_mma<<<grid, 256, 0, stream>>>(u, trellis, partial, w.k, w.n, half_bits,
                                                    columns);
        }
        for (int t = 0; t < columns; ++t) {
            exl3_gemv_finish<<<dim3(static_cast<unsigned>(w.n / 128)), 128, 0, stream>>>(
                partial + static_cast<std::size_t>(t) * w.n, static_cast<const float*>(w.scales),
                output + static_cast<std::size_t>(t) * w.n, w.n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

Exl3WeightProbe& exl3_weight_probe() {
    static Exl3WeightProbe probe;
    return probe;
}

} // namespace ninfer::ops::detail
