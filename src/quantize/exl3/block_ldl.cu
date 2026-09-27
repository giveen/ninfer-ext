#include "quantize/exl3/block_ldl.h"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kPanel = 64;

// Factors the diagonal block of panel p in FP64 (right-looking, unblocked) and writes its lower
// Cholesky factor back. A non-positive pivot sets *info (1-based column) and is replaced by 1 so
// the remaining steps stay finite.
__global__ void __launch_bounds__(256)
    factor_diagonal_kernel(float* __restrict__ a, std::int64_t n, int p, int* __restrict__ info) {
    __shared__ double block[kPanel][kPanel + 1];
    const std::int64_t base = static_cast<std::int64_t>(p) * kPanel;
    for (int i = static_cast<int>(threadIdx.x); i < kPanel * kPanel; i += blockDim.x) {
        const int r = i / kPanel, c = i % kPanel;
        block[r][c] = r >= c ? a[(base + r) * n + base + c] : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < kPanel; ++k) {
        if (threadIdx.x == 0) {
            double pivot = block[k][k];
            if (!(pivot > 0.0)) {
                if (*info == 0) { *info = static_cast<int>(base) + k + 1; }
                pivot = 1.0;
            }
            block[k][k] = sqrt(pivot);
        }
        __syncthreads();
        for (int r = k + 1 + static_cast<int>(threadIdx.x); r < kPanel; r += blockDim.x) {
            block[r][k] /= block[k][k];
        }
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < kPanel * kPanel; i += blockDim.x) {
            const int r = i / kPanel, c = i % kPanel;
            if (c > k && r >= c) { block[r][c] -= block[r][k] * block[c][k]; }
        }
        __syncthreads();
    }
    for (int i = static_cast<int>(threadIdx.x); i < kPanel * kPanel; i += blockDim.x) {
        const int r = i / kPanel, c = i % kPanel;
        a[(base + r) * n + base + c] = r >= c ? static_cast<float>(block[r][c]) : 0.0f;
    }
}

// Rows below the diagonal block solve x L_ppᵀ = a[row, panel] by forward substitution.
__global__ void __launch_bounds__(128)
    solve_panel_kernel(float* __restrict__ a, std::int64_t n, int p) {
    __shared__ float l[kPanel][kPanel + 1];
    const std::int64_t base = static_cast<std::int64_t>(p) * kPanel;
    for (int i = static_cast<int>(threadIdx.x); i < kPanel * kPanel; i += blockDim.x) {
        l[i / kPanel][i % kPanel] = a[(base + i / kPanel) * n + base + i % kPanel];
    }
    __syncthreads();
    const std::int64_t row =
        base + kPanel + static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) { return; }
    float* r = a + row * n + base;
    float x[kPanel];
#pragma unroll
    for (int j = 0; j < kPanel; ++j) {
        float s = r[j];
#pragma unroll
        for (int m = 0; m < j; ++m) { s = fmaf(-x[m], l[j][m], s); }
        x[j] = s / l[j][j];
    }
#pragma unroll
    for (int j = 0; j < kPanel; ++j) { r[j] = x[j]; }
}

// Trailing update of the lower triangle: a[I, J] -= L[I, p] L[J, p]ᵀ for 64-blocks I >= J > p.
__global__ void __launch_bounds__(256)
    trailing_update_kernel(float* __restrict__ a, std::int64_t n, int p) {
    const int bi = p + 1 + static_cast<int>(blockIdx.y);
    const int bj = p + 1 + static_cast<int>(blockIdx.x);
    if (bj > bi) { return; }
    __shared__ float li[kPanel][kPanel + 1];
    __shared__ float lj[kPanel][kPanel + 1];
    const std::int64_t col   = static_cast<std::int64_t>(p) * kPanel;
    const std::int64_t row_i = static_cast<std::int64_t>(bi) * kPanel;
    const std::int64_t row_j = static_cast<std::int64_t>(bj) * kPanel;
    for (int i = static_cast<int>(threadIdx.x); i < kPanel * kPanel; i += blockDim.x) {
        const int r = i / kPanel, c = i % kPanel;
        li[r][c] = a[(row_i + r) * n + col + c];
        lj[r][c] = a[(row_j + r) * n + col + c];
    }
    __syncthreads();
    const int tx    = static_cast<int>(threadIdx.x & 15);
    const int ty    = static_cast<int>(threadIdx.x >> 4);
    float acc[4][4] = {};
    for (int m = 0; m < kPanel; ++m) {
        float av[4], bv[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { av[r] = li[ty + 16 * r][m]; }
#pragma unroll
        for (int c = 0; c < 4; ++c) { bv[c] = lj[tx + 16 * c][m]; }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[r][c] = fmaf(av[r], bv[c], acc[r][c]); }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            a[(row_i + ty + 16 * r) * n + row_j + tx + 16 * c] -= acc[r][c];
        }
    }
}

// Zeroes the strict upper triangle outside the diagonal 64-blocks (those are written exactly by
// factor_diagonal_kernel).
__global__ void zero_upper_kernel(float* __restrict__ a, std::int64_t n) {
    const std::int64_t row = blockIdx.y;
    for (std::int64_t c = (row / kPanel + 1) * kPanel + blockIdx.x * blockDim.x + threadIdx.x;
         c < n; c += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        a[row * n + c] = 0.0f;
    }
}

// L <- L blockdiag(D16)^-1: every 16-wide block column is multiplied by the inverse of its own
// lower-triangular diagonal block, which leaves identity diagonal blocks. One CTA per block column
// and 128-row chunk; the 16x16 inverse is formed in FP64.
__global__ void __launch_bounds__(128)
    normalize_block_columns_kernel(float* __restrict__ a, std::int64_t n) {
    constexpr int kB = kLdlBlock;
    __shared__ double inv[kB][kB];
    __shared__ double d[kB][kB];
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * kB;
    for (int i = static_cast<int>(threadIdx.x); i < kB * kB; i += blockDim.x) {
        d[i / kB][i % kB]   = a[(col + i / kB) * n + col + i % kB];
        inv[i / kB][i % kB] = 0.0;
    }
    __syncthreads();
    // Column c of the inverse of lower-triangular d by forward substitution (thread c).
    if (threadIdx.x < kB) {
        const int c = static_cast<int>(threadIdx.x);
        for (int r = c; r < kB; ++r) {
            double s = r == c ? 1.0 : 0.0;
            for (int m = c; m < r; ++m) { s -= d[r][m] * inv[m][c]; }
            inv[r][c] = s / d[r][r];
        }
    }
    __syncthreads();
    const std::int64_t row = col + static_cast<std::int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (row >= n) { return; }
    // The diagonal block itself is read by every CTA of this column; set_identity_blocks_kernel
    // replaces it afterwards.
    if (row < col + kB) { return; }
    float* r = a + row * n + col;
    double x[kB];
#pragma unroll
    for (int j = 0; j < kB; ++j) { x[j] = r[j]; }
#pragma unroll
    for (int j = 0; j < kB; ++j) {
        double s = 0.0;
#pragma unroll
        for (int m = j; m < kB; ++m) { s += x[m] * inv[m][j]; }
        r[j] = static_cast<float>(s);
    }
}

__global__ void set_identity_blocks_kernel(float* __restrict__ a, std::int64_t n) {
    const std::int64_t col     = static_cast<std::int64_t>(blockIdx.x) * kLdlBlock;
    const int r                = static_cast<int>(threadIdx.x / kLdlBlock);
    const int c                = static_cast<int>(threadIdx.x % kLdlBlock);
    a[(col + r) * n + col + c] = r == c ? 1.0f : 0.0f;
}

} // namespace

bool block_ldl(float* h, std::int64_t n, cudaStream_t stream) {
    if (n <= 0 || n % kPanel || n / kPanel > 65535) {
        throw std::invalid_argument("block_ldl needs n % 64 == 0");
    }
    int* info = nullptr;
    CUDA_CHECK(cudaMallocAsync(&info, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(info, 0, sizeof(int), stream));
    const int panels = static_cast<int>(n / kPanel);
    for (int p = 0; p < panels; ++p) {
        factor_diagonal_kernel<<<1, 256, 0, stream>>>(h, n, p, info);
        const std::int64_t below = n - static_cast<std::int64_t>(p + 1) * kPanel;
        if (below > 0) {
            solve_panel_kernel<<<static_cast<unsigned>((below + 127) / 128), 128, 0, stream>>>(h, n,
                                                                                               p);
            const auto trailing = static_cast<unsigned>(below / kPanel);
            trailing_update_kernel<<<dim3(trailing, trailing), 256, 0, stream>>>(h, n, p);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    zero_upper_kernel<<<dim3(8, static_cast<unsigned>(n)), 256, 0, stream>>>(h, n);
    normalize_block_columns_kernel<<<dim3(static_cast<unsigned>(n / kLdlBlock),
                                          static_cast<unsigned>((n + 127) / 128)),
                                     128, 0, stream>>>(h, n);
    set_identity_blocks_kernel<<<static_cast<unsigned>(n / kLdlBlock), kLdlBlock * kLdlBlock, 0,
                                 stream>>>(h, n);
    CUDA_CHECK(cudaGetLastError());
    int host_info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(info, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host_info == 0;
}

} // namespace ninfer::quantize::exl3
