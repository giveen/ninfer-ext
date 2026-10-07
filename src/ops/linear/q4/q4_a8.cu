#include "core/weight.h"
#include "ops/linear/q4/q4_a8_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Per-token absmax int8 quantization: bf16 [K, T] (token-major) -> int8 codes [K, T] and one
// float scale per token.  Symmetric range, so the maximum element maps to +-127.
template <int InputRows, int Threads = 256>
__global__ __launch_bounds__(Threads, 2) void q4_a8_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::int8_t* __restrict__ codes,
    float* __restrict__ scales) {
    static_assert((InputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_token  = InputRows / 2;
    constexpr int pairs_per_thread = pairs_per_token / Threads;
    constexpr int warps            = Threads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float token_scale;

    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(
        input + static_cast<std::int64_t>(token) * InputRows);
    auto* output_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(token) * InputRows);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        values[item]   = bf16x2_bits_to_float2(input_pairs[pair]);
        maximum        = fmaxf(maximum, fabsf(values[item].x));
        maximum        = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = warp_max(maximum);
        if (lane == 0) { token_scale = maximum > 0.0F ? maximum / 127.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = token_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        const int lo   = max(-127, min(127, static_cast<int>(__float2int_rn(values[item].x * inverse))));
        const int hi   = max(-127, min(127, static_cast<int>(__float2int_rn(values[item].y * inverse))));
        output_pairs[pair] = static_cast<std::uint16_t>((lo & 0xFF) | ((hi & 0xFF) << 8));
    }
    if (tid == 0) { scales[token] = scale; }
}

template <int InputRows>
void launch_quantize_exact(const Tensor& x, Q4A8Workspace workspace, cudaStream_t stream) {
    constexpr int kThreads = 256;
    q4_a8_quantize_kernel<InputRows, kThreads><<<x.ne[1], kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_a8_quantize(const Tensor& x, Q4A8Workspace workspace, cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("q4 A8 requires caller workspace");
    }
    switch (x.ne[0]) {
    case 5120:
        launch_quantize_exact<5120>(x, workspace, stream);
        return;
    default:
        throw std::invalid_argument("q4 A8 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail
