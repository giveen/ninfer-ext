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

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::uint32_t kMul1Multiplier = 0x83DCD12DU;

__device__ __forceinline__ float mul1_value(std::uint16_t state) {
    const std::uint32_t product = static_cast<std::uint32_t>(state) * kMul1Multiplier;
    const std::uint32_t sum     = (product & 0xFFU) + ((product >> 8) & 0xFFU) +
                                  ((product >> 16) & 0xFFU) + ((product >> 24) & 0xFFU);
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

// Same window for an even (integer) bitrate, read as two 32-bit words and a funnel shift instead of
// 16 per-bit extractions. `bits` is the integer rate, `words` the uint32 per 256-weight tile. The
// window ends at bit (t+1)*bits of the circular stream.
__device__ __forceinline__ std::uint16_t tile_state_fast(const std::uint32_t* tile, int words,
                                                         int bits, int t) {
    const int total = words * 32;
    const int start = ((t + 1) * bits + total - 16) % total;
    const int word  = start >> 5;
    const int shift = start & 31;
    const std::uint32_t window =
        __funnelshift_r(tile[word], tile[(word + 1) % words], shift);
    return static_cast<std::uint16_t>(window & 0xFFFFU);
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

// out[n,t] = sv[n] · H128( Σ_i Z[i,n] u[i,t] ) with the 1/128 from both Hadamards. One block covers
// 128 output rows and up to kExl3ContractTile columns; each trellis window is decoded once and
// reused across those columns, so prefill is no longer decode-bound. u is t-major [T,K], so the 16
// operands of one column are contiguous.
constexpr int kExl3ContractTile = 16;
__global__ void exl3_contract(const float* __restrict__ u, const std::uint8_t* __restrict__ trellis,
                              const float* __restrict__ sv, __nv_bfloat16* __restrict__ out,
                              int k_extent, int n_extent, int columns, int half_bits) {
    const int lane    = threadIdx.x; // n_local
    const int n_block = blockIdx.x;  // 128 output rows
    const int t0      = blockIdx.y * kExl3ContractTile;
    const int n       = n_block * 128 + lane;
    if (n >= n_extent || t0 >= columns) { return; }
    const int t_count  = min(kExl3ContractTile, columns - t0);
    const int kt_tiles = k_extent / 16;
    const int nt       = n >> 4;
    const int n_local  = n & 15;
    const int bits     = half_bits >> 1;
    const int words    = 4 * half_bits;
    const bool fast    = (half_bits & 1) == 0;

    float acc[kExl3ContractTile];
#pragma unroll
    for (int j = 0; j < kExl3ContractTile; ++j) { acc[j] = 0.0F; }

    for (int kt = 0; kt < kt_tiles; ++kt) {
        const std::uint8_t* tile8 =
            trellis + (static_cast<std::size_t>(nt) * kt_tiles + kt) * (16 * half_bits);
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
        float d[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int t_idx = tile_state_index(i, n_local);
            d[i]            = fast ? mul1_value(tile_state_fast(tile32, words, bits, t_idx))
                                   : mul1_value(tile_state(tile8, half_bits, t_idx));
        }
#pragma unroll
        for (int j = 0; j < kExl3ContractTile; ++j) {
            if (j >= t_count) { break; }
            const float* uj = u + (static_cast<std::size_t>(t0 + j) * k_extent + kt * 16);
            float sum       = acc[j];
#pragma unroll
            for (int i = 0; i < 16; ++i) { sum = fmaf(d[i], uj[i], sum); }
            acc[j] = sum;
        }
    }

    __shared__ float s[128];
    for (int j = 0; j < t_count; ++j) {
        s[lane] = acc[j];
        __syncthreads();
        butterfly128(s, lane);
        out[n + static_cast<std::size_t>(t0 + j) * n_extent] =
            __float2bfloat16(s[lane] * sv[n] * (1.0F / 128.0F));
    }
}

// Decode-shaped contraction: one column, so parallelism comes from splitting K. blockDim is
// (128 n-rows, kExl3DecodeSplit k-slices); the slices are reduced in shared memory and every slice
// runs its own copy of the 128-point output Hadamard, which keeps the block-wide barriers valid for
// all threads and costs almost nothing. This lifts decode occupancy from one 128-thread block per
// 128 output rows to kExl3DecodeSplit of them.
constexpr int kExl3DecodeSplit = 8;
__global__ void exl3_contract_decode(const float* __restrict__ u,
                                     const std::uint8_t* __restrict__ trellis,
                                     const float* __restrict__ sv, __nv_bfloat16* __restrict__ out,
                                     int k_extent, int n_extent, int half_bits) {
    const int lane = threadIdx.x; // n_local
    const int s    = threadIdx.y; // k-slice
    const int n    = blockIdx.x * 128 + lane;
    if (n >= n_extent) { return; }
    const int kt_tiles = k_extent / 16;
    const int per      = (kt_tiles + kExl3DecodeSplit - 1) / kExl3DecodeSplit;
    const int kt_begin = s * per;
    const int kt_end   = min(kt_tiles, kt_begin + per);
    const int nt       = n >> 4;
    const int n_local  = n & 15;
    const int bits     = half_bits >> 1;
    const int words    = 4 * half_bits;
    const bool fast    = (half_bits & 1) == 0;

    float acc = 0.0F;
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const std::uint8_t* tile8 =
            trellis + (static_cast<std::size_t>(nt) * kt_tiles + kt) * (16 * half_bits);
        const std::uint32_t* tile32 = reinterpret_cast<const std::uint32_t*>(tile8);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int t_idx = tile_state_index(i, n_local);
            const float d   = fast ? mul1_value(tile_state_fast(tile32, words, bits, t_idx))
                                   : mul1_value(tile_state(tile8, half_bits, t_idx));
            acc             = fmaf(d, u[kt * 16 + i], acc);
        }
    }

    __shared__ float sh[kExl3DecodeSplit][128];
    __shared__ float row[kExl3DecodeSplit][128];
    sh[s][lane] = acc;
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kExl3DecodeSplit; ++w) { total += sh[w][lane]; }
    row[s][lane] = total;
    __syncthreads();
    butterfly128(&row[s][0], lane);
    if (s == 0) { out[n] = __float2bfloat16(row[0][lane] * sv[n] * (1.0F / 128.0F)); }
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
constexpr int kExl3MmaMinT    = 32;

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
            std::uint16_t st[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int t_idx = 8 * lane + 4 * h + r;
                st[r]           = fast ? tile_state_fast(tile32, words, bits, t_idx)
                                       : tile_state(tile8, half_bits, t_idx);
            }
            b[h][0] = pack_bf16x2(mul1_value(st[0]), mul1_value(st[1]));
            b[h][1] = pack_bf16x2(mul1_value(st[2]), mul1_value(st[3]));
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

    const std::size_t scratch_bytes =
        static_cast<std::size_t>(w.k) * static_cast<std::size_t>(columns) * sizeof(float);
    auto scope         = workspace.scope();
    const DeviceSpan scratch = workspace.alloc_bytes(scratch_bytes, 16);
    float* u = static_cast<float*>(scratch.data);

    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output      = static_cast<__nv_bfloat16*>(out.data);
    exl3_input_transform<<<dim3(static_cast<unsigned>(w.k / 128), columns), 128, 0, stream>>>(
        input, static_cast<const float*>(w.input_scales), u, w.k, columns);
    if (columns >= kExl3MmaMinT) {
        exl3_mma<<<dim3(static_cast<unsigned>(w.n / 128),
                        static_cast<unsigned>((columns + kExl3MmaT - 1) / kExl3MmaT)),
                  kExl3MmaThreads, 0, stream>>>(
            u, static_cast<const std::uint8_t*>(w.qdata), static_cast<const float*>(w.scales),
            output, w.k, w.n, columns, static_cast<int>(w.bitrate_half_bits));
    } else if (columns == 1) {
        exl3_contract_decode<<<dim3(static_cast<unsigned>(w.n / 128)),
                               dim3(128, kExl3DecodeSplit), 0, stream>>>(
            u, static_cast<const std::uint8_t*>(w.qdata), static_cast<const float*>(w.scales),
            output, w.k, w.n, static_cast<int>(w.bitrate_half_bits));
    } else {
        exl3_contract<<<dim3(static_cast<unsigned>(w.n / 128),
                             static_cast<unsigned>((columns + kExl3ContractTile - 1) /
                                                   kExl3ContractTile)),
                       128, 0, stream>>>(
            u, static_cast<const std::uint8_t*>(w.qdata), static_cast<const float*>(w.scales),
            output, w.k, w.n, columns, static_cast<int>(w.bitrate_half_bits));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
