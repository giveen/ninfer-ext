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

// u[k,t] = H128(su[k] * x[k,t]) over 128-blocks of k. u is FP32 scratch [K,T].
__global__ void exl3_input_transform(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ su, float* __restrict__ u,
                                     int k_extent, int columns) {
    const int block = blockIdx.x;
    const int t     = blockIdx.y;
    const int lane  = threadIdx.x;
    const int k     = block * 128 + lane;
    if (k >= k_extent) { return; }
    __shared__ float s[128];
    s[lane] = su[k] * __bfloat162float(x[static_cast<std::size_t>(k) * columns + t]);
    __syncthreads();
    butterfly128(s, lane);
    u[static_cast<std::size_t>(k) * columns + t] = s[lane];
}

// out[n,t] = sv[n] · H128( Σ_i Z[i,n] u[i,t] ) with the 1/128 from both Hadamards.
__global__ void exl3_contract(const float* __restrict__ u, const std::uint8_t* __restrict__ trellis,
                              const float* __restrict__ sv, __nv_bfloat16* __restrict__ out,
                              int k_extent, int n_extent, int columns, int half_bits) {
    const int lane      = threadIdx.x; // n_local
    const int n_block   = blockIdx.x;  // 128 output rows
    const int t         = blockIdx.y;
    const int n         = n_block * 128 + lane;
    const int kt_tiles  = k_extent / 16;
    if (n >= n_extent) { return; }
    const int nt      = n >> 4;
    const int n_local = n & 15;

    float v = 0.0F;
    for (int kt = 0; kt < kt_tiles; ++kt) {
        const std::uint8_t* tile =
            trellis + (static_cast<std::size_t>(nt) * kt_tiles + kt) * (16 * half_bits);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int t_idx = tile_state_index(i, n_local);
            v += mul1_value(tile_state(tile, half_bits, t_idx)) *
                 u[static_cast<std::size_t>(kt * 16 + i) * columns + t];
        }
    }
    __shared__ float s[128];
    s[lane] = v;
    __syncthreads();
    butterfly128(s, lane);
    out[static_cast<std::size_t>(n) * columns + t] =
        __float2bfloat16(s[lane] * sv[n] * (1.0F / 128.0F));
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
    exl3_contract<<<dim3(static_cast<unsigned>(w.n / 128), columns), 128, 0, stream>>>(
        u, static_cast<const std::uint8_t*>(w.qdata), static_cast<const float*>(w.scales), output,
        w.k, w.n, columns, static_cast<int>(w.bitrate_half_bits));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
