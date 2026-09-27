#include "quantize/exl3/hessian.h"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kTile    = 64;  // output tile edge
constexpr int kDepth   = 16;  // tokens staged per step
constexpr int kThreads = 256; // 16 x 16 threads, 4 x 4 outputs each

// One CTA owns the 64x64 block (bi, bj) with bi <= bj of H and writes it and its mirror.
__global__ void __launch_bounds__(kThreads)
    hessian_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t tokens, std::int64_t k,
                   float* __restrict__ h) {
    const int bi = static_cast<int>(blockIdx.y);
    const int bj = static_cast<int>(blockIdx.x);
    if (bi > bj) { return; }
    __shared__ float a[kDepth][kTile];
    __shared__ float b[kDepth][kTile];
    const int tx            = static_cast<int>(threadIdx.x & 15);
    const int ty            = static_cast<int>(threadIdx.x >> 4);
    const std::int64_t row0 = static_cast<std::int64_t>(bi) * kTile;
    const std::int64_t col0 = static_cast<std::int64_t>(bj) * kTile;

    float acc[4][4] = {};
    for (std::int64_t t0 = 0; t0 < tokens; t0 += kDepth) {
        for (int i = static_cast<int>(threadIdx.x); i < kDepth * kTile; i += kThreads) {
            const int t     = i / kTile;
            const int c     = i % kTile;
            const bool live = t0 + t < tokens;
            a[t][c]         = live ? __bfloat162float(x[(t0 + t) * k + row0 + c]) : 0.0f;
            b[t][c]         = live ? __bfloat162float(x[(t0 + t) * k + col0 + c]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kDepth; ++t) {
            float av[4], bv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) { av[r] = a[t][ty + 16 * r]; }
#pragma unroll
            for (int c = 0; c < 4; ++c) { bv[c] = b[t][tx + 16 * c]; }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < 4; ++c) { acc[r][c] = fmaf(av[r], bv[c], acc[r][c]); }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const std::int64_t i = row0 + ty + 16 * r;
            const std::int64_t j = col0 + tx + 16 * c;
            h[i * k + j] += acc[r][c];
            if (bi != bj) { h[j * k + i] += acc[r][c]; }
        }
    }
}

} // namespace

void accumulate_hessian(const __nv_bfloat16* x, std::int64_t tokens, std::int64_t k, float* h,
                        cudaStream_t stream) {
    if (tokens < 0 || k <= 0 || k % 128 || k / kTile > 65535) {
        throw std::invalid_argument("accumulate_hessian needs k % 128 == 0");
    }
    if (tokens == 0) { return; }
    const auto blocks = static_cast<unsigned>(k / kTile);
    hessian_kernel<<<dim3(blocks, blocks), kThreads, 0, stream>>>(x, tokens, k, h);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// Same reduction as hessian_kernel, but x is the K x tokens activation the model's linear Ops
// take (row stride tokens), so the tile reads are transposed. Used only by offline calibration.
__global__ void __launch_bounds__(kThreads)
    hessian_kt_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t tokens, std::int64_t k,
                      float* __restrict__ h) {
    const int bi = static_cast<int>(blockIdx.y);
    const int bj = static_cast<int>(blockIdx.x);
    if (bi > bj) { return; }
    __shared__ float a[kDepth][kTile];
    __shared__ float b[kDepth][kTile];
    const int tx            = static_cast<int>(threadIdx.x & 15);
    const int ty            = static_cast<int>(threadIdx.x >> 4);
    const std::int64_t row0 = static_cast<std::int64_t>(bi) * kTile;
    const std::int64_t col0 = static_cast<std::int64_t>(bj) * kTile;

    float acc[4][4] = {};
    for (std::int64_t t0 = 0; t0 < tokens; t0 += kDepth) {
        for (int i = static_cast<int>(threadIdx.x); i < kDepth * kTile; i += kThreads) {
            const int t     = i / kTile;
            const int c     = i % kTile;
            const bool live = t0 + t < tokens;
            a[t][c] = live ? __bfloat162float(x[(row0 + c) * tokens + t0 + t]) : 0.0f;
            b[t][c] = live ? __bfloat162float(x[(col0 + c) * tokens + t0 + t]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kDepth; ++t) {
            float av[4], bv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) { av[r] = a[t][ty + 16 * r]; }
#pragma unroll
            for (int c = 0; c < 4; ++c) { bv[c] = b[t][tx + 16 * c]; }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < 4; ++c) { acc[r][c] = fmaf(av[r], bv[c], acc[r][c]); }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const std::int64_t i = row0 + ty + 16 * r;
            const std::int64_t j = col0 + tx + 16 * c;
            h[i * k + j] += acc[r][c];
            if (bi != bj) { h[j * k + i] += acc[r][c]; }
        }
    }
}

} // namespace

void accumulate_hessian_kt(const __nv_bfloat16* x_kt, std::int64_t tokens, std::int64_t k,
                           float* h, cudaStream_t stream) {
    if (tokens < 0 || k <= 0 || k % 128 || k / kTile > 65535) {
        throw std::invalid_argument("accumulate_hessian_kt needs k % 128 == 0");
    }
    if (tokens == 0) { return; }
    const auto blocks = static_cast<unsigned>(k / kTile);
    hessian_kt_kernel<<<dim3(blocks, blocks), kThreads, 0, stream>>>(x_kt, tokens, k, h);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
