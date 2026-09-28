// EXL3 mul1 linear decode. Correctness-first SIMT implementation of the stored reconstruction:
//
//   W[n,k] = su[k] · sv[n] · (H128 · Z · H128)[k,n],   out[n,t] = Σ_k W[n,k] x[k,t]
//
// A first kernel scales the activation rows by su and applies the input Hadamard over K. A second
// kernel contracts the decoded trellis with that rotated activation, applies the output Hadamard
// over N, and scales by sv. The 1/sqrt(128) factor of each Hadamard is folded into one 1/128 at the
// end. This is the "simple correct" M3 route; M4 replaces it with MMA kernels.

#include "ops/linear/exl3/exl3_dispatch.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::uint32_t kMul1Multiplier = 0x83DCD12DU;

__device__ __forceinline__ float mul1_value(std::uint16_t state) {
    // The byte sum of the multiplied state is the mul1 codebook; one dp4a replaces six shift/mask/add
    // ops, which matters because every weight of the model passes through here.
    const std::uint32_t product = static_cast<std::uint32_t>(state) * kMul1Multiplier;
    const std::uint32_t sum     = __dp4a(product, 0x01010101U, 0U);
    return static_cast<float>(static_cast<std::int32_t>(sum) - 510);
}

// State index whose 16x16 tile element is (k_local, n_local); the inverse of the trellis order in
// storage-layouts.md 9.2, i.e. of k = 2 b3 + 4 b4 + b0 + 8 b1, n = b5 + 2 b6 + 4 b7 + 8 b2.
__device__ __forceinline__ int tile_state_index(int k_local, int n_local) {
    return (k_local & 1) | (((k_local >> 3) & 1) << 1) | (((n_local >> 3) & 1) << 2) |
           (((k_local >> 1) & 1) << 3) | (((k_local >> 2) & 1) << 4) | ((n_local & 7) << 5);
}

// 16-bit circular window of the tile's tail-biting bitstream that decodes weight `t`.
__device__ __forceinline__ std::uint16_t tile_state(const std::uint8_t* tile, int half_bits,
                                                    int t) {
    const int low   = half_bits >> 1;
    const int count = t + 1;
    const int end   = count * low + ((half_bits & 1) ? count / 2 : 0);
    const int total = 128 * half_bits;
    const int begin = (end + total - 16) % total;
    std::uint32_t state = 0;
    for (int b = 0; b < 16; ++b) {
        const int position = (begin + b) % total;
        state |= (static_cast<std::uint32_t>(tile[position >> 3] >> (position & 7)) & 1U) << b;
    }
    return static_cast<std::uint16_t>(state);
}

// A window's position in the tile, precomputed once per thread and packed into 16 bits (word in the
// low byte, shift in the high byte) so a 16-window plan costs eight registers instead of forty-eight.
// The state index, bitrate and tile width are loop-invariant, so the two integer modulos that located
// the window move out of the K loop entirely; only the two loads and the funnel shift remain.
__device__ __forceinline__ std::uint16_t exl3_window(int state, int bits, int words) {
    const int total = words * 32;
    int start       = (state + 1) * bits - 16;
    if (start < 0) { start += total; }
    return static_cast<std::uint16_t>((start >> 5) | ((start & 31) << 8));
}

__device__ __forceinline__ float exl3_window_value(const std::uint32_t* tile,
                                                   std::uint16_t window, int words) {
    const int word  = window & 0xFF;
    const int shift = (window >> 8) & 0x1F;
    const int next  = (word + 1 == words) ? 0 : word + 1;
    return mul1_value(static_cast<std::uint16_t>(
        __funnelshift_r(tile[word], tile[next], shift) & 0xFFFFU));
}

// The eight B-fragment windows of one 16x16 tile at a 4-bit rate, resolved from the lane's word and
// its circular predecessor with one funnel shift and five bit-field extracts (exllamav3's
// dq8_regs_4bits). d[r] is the window for state 8*lane + r, which is the state the contraction
// indexes. The tile is 32 words, so `lane - 1` wraps inside it.
__device__ __forceinline__ void exl3_windows_4bit(const std::uint32_t* tile, int lane, float d[8]) {
    const std::uint32_t b = tile[lane];
    const std::uint32_t a = tile[(lane + 31) & 31];
    const std::uint32_t s = __funnelshift_r(a, b, 20);
    d[0] = mul1_value(static_cast<std::uint16_t>(s & 0xFFFFU));
    d[1] = mul1_value(static_cast<std::uint16_t>((s >> 4) & 0xFFFFU));
    d[2] = mul1_value(static_cast<std::uint16_t>((s >> 8) & 0xFFFFU));
    d[3] = mul1_value(static_cast<std::uint16_t>(b & 0xFFFFU));
    d[4] = mul1_value(static_cast<std::uint16_t>((b >> 4) & 0xFFFFU));
    d[5] = mul1_value(static_cast<std::uint16_t>((b >> 8) & 0xFFFFU));
    d[6] = mul1_value(static_cast<std::uint16_t>((b >> 12) & 0xFFFFU));
    d[7] = mul1_value(static_cast<std::uint16_t>((b >> 16) & 0xFFFFU));
}

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

// u[k,t] = H128(su[k] * x[k,t]) over 128-blocks of k. u is FP32 scratch [K,T] in the same
// column-major order as the caller's tensors (dim 0 is contiguous).
__global__ void exl3_input_transform(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ su, float* __restrict__ u,
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
    u[static_cast<std::size_t>(t) * k_extent + k] = s[lane];
}

// Decode contraction: one column, so parallelism comes from splitting K across the grid. A warp owns
// one 16-row trellis tile (n = 16w..16w+15) and lane L decodes the eight windows t = 8L..8L+7, which
// are the m16n8k16 B-fragment windows: for 4-bit rates that is exactly one 32-bit word per lane, so
// the tile loads are coalesced instead of scattered. The lane's four h=0 windows sum into n = L/4 and
// the four h=1 windows into n = L/4+8; a four-lane shuffle folds the k-split and one lane per n
// atomically adds into a small FP32 buffer. A second kernel applies the 128-point output Hadamard.
constexpr int kExl3GemvSplits = 32;
constexpr int kExl3GemvMaxM   = 8; // largest m the per-column GEMV beats the tiled MMA at

__global__ void exl3_gemv_split(const float* __restrict__ u,
                                const std::uint8_t* __restrict__ trellis,
                                float* __restrict__ partial, int k_extent, int n_extent,
                                int half_bits) {
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

    // Even rates resolve the two four-window groups with two lane-constant funnel shifts (exllamav3's
    // dq4 applied twice); the word indices and shifts do not depend on the k-tile.
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

    float a0[4] = {0.0F, 0.0F, 0.0F, 0.0F}; // n = L/4
    float a1[4] = {0.0F, 0.0F, 0.0F, 0.0F}; // n = L/4 + 8
    const std::uint8_t* tile_base =
        trellis + static_cast<std::size_t>(n_tile) * kt_tiles * (16 * half_bits);
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const std::uint8_t* tile8 =
            tile_base + static_cast<std::size_t>(kt) * (16 * half_bits);
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        const float* uk             = u + kt * 16;
        float d[8];
        if (bits == 4) {
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
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                d[r] = mul1_value(tile_state(tile8, half_bits, 8 * lane + r));
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int local = r & 3;
            const int k     = 2 * (lane & 3) + (local & 1) + 8 * (local >> 1);
            if (r < 4) {
                a0[local] = fmaf(d[r], uk[k], a0[local]);
            } else {
                a1[local] = fmaf(d[r], uk[k], a1[local]);
            }
        }
    }
    float acc0 = (a0[0] + a0[1]) + (a0[2] + a0[3]);
    float acc1 = (a1[0] + a1[1]) + (a1[2] + a1[3]);
    acc0 += __shfl_down_sync(0xFFFFFFFFU, acc0, 2);
    acc0 += __shfl_down_sync(0xFFFFFFFFU, acc0, 1);
    acc1 += __shfl_down_sync(0xFFFFFFFFU, acc1, 2);
    acc1 += __shfl_down_sync(0xFFFFFFFFU, acc1, 1);
    if ((lane & 3) == 0) {
        const int g = lane >> 2;
        atomicAdd(&partial[n_base + g], acc0);
        atomicAdd(&partial[n_base + g + 8], acc1);
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

__device__ __forceinline__ void mma_m16n8k16(float d[4], const std::uint32_t a[4],
                                             const std::uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Tensor-core contraction. A block covers one 128-row Hadamard block and up to 64 tokens. Warp w
// decodes the trellis tile for n = 16w..16w+15 (both n8 fragments) and runs it against the four m16
// token fragments. The stored tile order is the m16n8k16 B-fragment order, so a lane's eight
// windows pack straight into two B fragments with no shuffle.
constexpr int kExl3MmaN       = 128;
constexpr int kExl3MmaT       = 64;
constexpr int kExl3MmaThreads = 256;

__global__ void exl3_mma(const float* __restrict__ u, const std::uint8_t* __restrict__ trellis,
                         const float* __restrict__ sv, __nv_bfloat16* __restrict__ out,
                         int k_extent, int n_extent, int columns, int half_bits) {
    const int n_base  = blockIdx.x * kExl3MmaN;
    const int t0      = blockIdx.y * kExl3MmaT;
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int t_count = min(kExl3MmaT, columns - t0);
    const int kt_tiles = k_extent / 16;
    const int bits     = half_bits >> 1;
    const int words    = 4 * half_bits;
    const bool fast    = (half_bits & 1) == 0;

    __shared__ __nv_bfloat16 us[kExl3MmaT][16];
    __shared__ float vs[kExl3MmaT][kExl3MmaN];

    float acc[2][4][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[h][m][c] = 0.0F; }

    const std::size_t tile_stride = static_cast<std::size_t>(16 * half_bits);
    const std::uint8_t* warp_tile =
        trellis + static_cast<std::size_t>(blockIdx.x * 8 + warp) * kt_tiles * tile_stride;

    std::uint16_t plan[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { plan[i] = exl3_window(8 * lane + i, bits, words); }

    for (int kt = 0; kt < kt_tiles; ++kt) {
        for (int i = threadIdx.x; i < kExl3MmaT * 16; i += kExl3MmaThreads) {
            const int t  = i >> 4;
            const int kk = i & 15;
            us[t][kk]    = (t0 + t < columns)
                               ? __float2bfloat16(
                                     u[static_cast<std::size_t>(t0 + t) * k_extent + kt * 16 + kk])
                               : __float2bfloat16(0.0F);
        }
        __syncthreads();

        const std::uint8_t* tile8 = warp_tile + static_cast<std::size_t>(kt) * tile_stride;
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        std::uint32_t b[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float d[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                d[r] = fast ? exl3_window_value(tile32, plan[4 * h + r], words)
                            : mul1_value(tile_state(tile8, half_bits, 8 * lane + 4 * h + r));
            }
            b[h][0] = pack_bf16x2(d[0], d[1]);
            b[h][1] = pack_bf16x2(d[2], d[3]);
        }

        std::uint32_t a[4][4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int row = mt * 16 + (lane >> 2);
            const int col = (lane & 3) * 2;
            a[mt][0]      = pack_bf16x2(us[row][col], us[row][col + 1]);
            a[mt][1]      = pack_bf16x2(us[row + 8][col], us[row + 8][col + 1]);
            a[mt][2]      = pack_bf16x2(us[row][col + 8], us[row][col + 9]);
            a[mt][3]      = pack_bf16x2(us[row + 8][col + 8], us[row + 8][col + 9]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) { mma_m16n8k16(acc[h][mt], a[mt], b[h]); }

        __syncthreads();
    }

    // Publish the accumulators, then apply the output Hadamard over the 128 rows. Threads 0..127
    // transform even columns and 128..255 odd ones; every thread still reaches the block barriers.
    const int half_lane = threadIdx.x & 127;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int n0        = (2 * warp + h) * 8 + (lane & 3) * 2;
            const int t_a       = mt * 16 + (lane >> 2);
            vs[t_a][n0]         = acc[h][mt][0];
            vs[t_a][n0 + 1]     = acc[h][mt][1];
            vs[t_a + 8][n0]     = acc[h][mt][2];
            vs[t_a + 8][n0 + 1] = acc[h][mt][3];
        }
    }
    __syncthreads();
    for (int pair = 0; pair < kExl3MmaT / 2; ++pair) {
        const int t_even = 2 * pair;
        const int t_odd  = 2 * pair + 1;
        float* target    = (threadIdx.x < 128) ? &vs[t_even][0] : &vs[t_odd][0];
        butterfly128(target, half_lane);
        const int t = (threadIdx.x < 128) ? t_even : t_odd;
        if (t < t_count) {
            const int n = n_base + half_lane;
            out[n + static_cast<std::size_t>(t0 + t) * n_extent] =
                __float2bfloat16(target[half_lane] * sv[n] * (1.0F / 128.0F));
        }
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
        static_cast<std::size_t>(w.k) * static_cast<std::size_t>(columns) * sizeof(float);
    const std::size_t partial_bytes =
        columns <= kExl3GemvMaxM ? static_cast<std::size_t>(w.n) * sizeof(float) : 0;
    auto scope               = workspace.scope();
    const DeviceSpan scratch = workspace.alloc_bytes(u_bytes + partial_bytes, 16);
    float* u                 = static_cast<float*>(scratch.data);
    float* partial = reinterpret_cast<float*>(static_cast<std::uint8_t*>(scratch.data) + u_bytes);

    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output      = static_cast<__nv_bfloat16*>(out.data);
    exl3_input_transform<<<dim3(static_cast<unsigned>(w.k / 128), columns), 128, 0, stream>>>(
        input, static_cast<const float*>(w.input_scales), u, w.k, columns);
    if (columns > kExl3GemvMaxM) {
        exl3_mma<<<dim3(static_cast<unsigned>(w.n / 128),
                        static_cast<unsigned>((columns + kExl3MmaT - 1) / kExl3MmaT)),
                  kExl3MmaThreads, 0, stream>>>(
            u, static_cast<const std::uint8_t*>(w.qdata), static_cast<const float*>(w.scales),
            output, w.k, w.n, columns, static_cast<int>(w.bitrate_half_bits));
    } else {
        // Up to kExl3GemvMaxM columns run the GEMV once per column, as exllamav3 does for m <= 8: the
        // tiled MMA has too few blocks at small m, so re-decoding per column is faster than the pad.
        // Splitting K only pays while the grid is short of blocks; a vocabulary-sized head already has
        // thousands of row blocks, so it takes one slice and avoids a scheduling- and atomic-bound grid.
        const int n_blocks = static_cast<int>(w.n / 128);
        const int splits   = std::min<int>(kExl3GemvSplits, std::max<int>(1, 8192 / n_blocks));
        for (int t = 0; t < columns; ++t) {
            CUDA_CHECK(cudaMemsetAsync(partial, 0, partial_bytes, stream));
            exl3_gemv_split<<<dim3(static_cast<unsigned>(n_blocks), static_cast<unsigned>(splits)),
                              256, 0, stream>>>(
                u + static_cast<std::size_t>(t) * w.k, static_cast<const std::uint8_t*>(w.qdata),
                partial, w.k, w.n, static_cast<int>(w.bitrate_half_bits));
            exl3_gemv_finish<<<dim3(static_cast<unsigned>(w.n / 128)), 128, 0, stream>>>(
                partial, static_cast<const float*>(w.scales),
                output + static_cast<std::size_t>(t) * w.n, w.n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
