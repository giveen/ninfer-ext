#include "quantize/exl3/hadamard.h"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr float kInvSqrt128 = 0.08838834764831845f;

// Transforms one 128-vector held as four consecutive elements per lane (index 4 * lane + j).
// Index bits 0-1 are within a lane; bits 2-6 are lane bits, exchanged through shuffles.
__device__ __forceinline__ void fwht128_warp(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float partner = __shfl_xor_sync(0xffffffffU, v[j], mask);
            v[j]                = (lane & mask) ? partner - v[j] : v[j] + partner;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] *= kInvSqrt128; }
}

__global__ void hadamard_rows_kernel(float* __restrict__ data, std::int64_t chunks) {
    const std::int64_t chunk =
        static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (chunk >= chunks) { return; }
    const int lane = static_cast<int>(threadIdx.x & 31);
    auto* p        = reinterpret_cast<float4*>(data + chunk * 128) + lane;
    const float4 x = *p;
    float v[4]{x.x, x.y, x.z, x.w};
    fwht128_warp(v, lane);
    *p = make_float4(v[0], v[1], v[2], v[3]);
}

// One CTA owns a 128-row by 32-column panel; each of its 8 warps transforms 4 columns.
__global__ void __launch_bounds__(256)
    hadamard_cols_kernel(float* __restrict__ data, std::int64_t cols) {
    __shared__ float panel[128][33];
    const std::int64_t row0 = static_cast<std::int64_t>(blockIdx.y) * 128;
    const std::int64_t col0 = static_cast<std::int64_t>(blockIdx.x) * 32;
    const int tx            = static_cast<int>(threadIdx.x & 31);
    const int ty            = static_cast<int>(threadIdx.x >> 5);
    const bool live         = col0 + tx < cols;
    for (int r = ty; r < 128; r += 8) {
        panel[r][tx] = live ? data[(row0 + r) * cols + col0 + tx] : 0.0f;
    }
    __syncthreads();
    for (int c = ty; c < 32; c += 8) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { v[j] = panel[4 * tx + j][c]; }
        fwht128_warp(v, tx);
#pragma unroll
        for (int j = 0; j < 4; ++j) { panel[4 * tx + j][c] = v[j]; }
    }
    __syncthreads();
    if (live) {
        for (int r = ty; r < 128; r += 8) { data[(row0 + r) * cols + col0 + tx] = panel[r][tx]; }
    }
}

} // namespace

void hadamard128_rows(float* data, std::int64_t rows, std::int64_t cols, cudaStream_t stream) {
    if (rows < 0 || cols <= 0 || cols % 128) {
        throw std::invalid_argument("hadamard128_rows needs cols % 128 == 0");
    }
    const std::int64_t chunks = rows * (cols / 128);
    if (chunks == 0) { return; }
    constexpr int kWarps = 8;
    const auto blocks    = static_cast<unsigned>((chunks + kWarps - 1) / kWarps);
    hadamard_rows_kernel<<<blocks, kWarps * 32, 0, stream>>>(data, chunks);
    CUDA_CHECK(cudaGetLastError());
}

void hadamard128_cols(float* data, std::int64_t rows, std::int64_t cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || rows % 128 || rows / 128 > 65535) {
        throw std::invalid_argument("hadamard128_cols needs rows % 128 == 0");
    }
    const dim3 grid(static_cast<unsigned>((cols + 31) / 32), static_cast<unsigned>(rows / 128));
    hadamard_cols_kernel<<<grid, 256, 0, stream>>>(data, cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
