#include "quantize/exl3/linalg.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kTile  = 64;
constexpr int kDepth = 16;

// 64x64 output tile per CTA, 16 x 16 threads with 4 x 4 outputs each; bounds-checked.
template <bool TransposeB>
__global__ void __launch_bounds__(256)
    gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                std::int64_t m, std::int64_t n, std::int64_t k) {
    __shared__ float as[kDepth][kTile + 1];
    __shared__ float bs[kDepth][kTile + 1];
    const std::int64_t row0 = static_cast<std::int64_t>(blockIdx.y) * kTile;
    const std::int64_t col0 = static_cast<std::int64_t>(blockIdx.x) * kTile;
    const int tx            = static_cast<int>(threadIdx.x & 15);
    const int ty            = static_cast<int>(threadIdx.x >> 4);
    float acc[4][4]         = {};
    for (std::int64_t k0 = 0; k0 < k; k0 += kDepth) {
        for (int i = static_cast<int>(threadIdx.x); i < kDepth * kTile; i += 256) {
            // A tile: rows row0 + r, depth k0 + d; loaded depth-fastest for coalescing.
            const int r = i / kDepth, d = i % kDepth;
            const std::int64_t ar = row0 + r, ad = k0 + d;
            as[d][r] = ar < m && ad < k ? a[ar * k + ad] : 0.0f;
            if constexpr (TransposeB) {
                const std::int64_t br = col0 + r;
                bs[d][r]              = br < n && ad < k ? b[br * k + ad] : 0.0f;
            }
        }
        if constexpr (!TransposeB) {
            for (int i = static_cast<int>(threadIdx.x); i < kDepth * kTile; i += 256) {
                const int d = i / kTile, cc = i % kTile;
                const std::int64_t bd = k0 + d, bc = col0 + cc;
                bs[d][cc] = bd < k && bc < n ? b[bd * n + bc] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            float av[4], bv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) { av[r] = as[d][ty + 16 * r]; }
#pragma unroll
            for (int q = 0; q < 4; ++q) { bv[q] = bs[d][tx + 16 * q]; }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int q = 0; q < 4; ++q) { acc[r][q] = fmaf(av[r], bv[q], acc[r][q]); }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::int64_t i = row0 + ty + 16 * r, j = col0 + tx + 16 * q;
            if (i < m && j < n) { c[i * n + j] = acc[r][q]; }
        }
    }
}

__global__ void scale_kernel(float* __restrict__ x, std::int64_t rows, std::int64_t cols,
                             const float* __restrict__ row, const float* __restrict__ col) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * cols) { return; }
    float v = x[i];
    if (row) { v *= row[i / cols]; }
    if (col) { v *= col[i % cols]; }
    x[i] = v;
}

// One CTA per output element: reduction of a[r][*]*b[r][*] (row mode) or a[*][c]*b[*][c].
template <bool Rows>
__global__ void __launch_bounds__(256)
    pair_reduce_kernel(const float* __restrict__ a, const float* __restrict__ b, std::int64_t rows,
                       std::int64_t cols, float* __restrict__ out, bool rms) {
    __shared__ double partial[256];
    const std::int64_t index = blockIdx.x;
    const std::int64_t count = Rows ? cols : rows;
    double sum               = 0.0;
    for (std::int64_t j = threadIdx.x; j < count; j += 256) {
        const std::int64_t at = Rows ? index * cols + j : j * cols + index;
        sum += static_cast<double>(a[at]) * static_cast<double>(b[at]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[index] = rms ? static_cast<float>(sqrt(partial[0] / static_cast<double>(count)))
                         : static_cast<float>(partial[0]);
    }
}

__global__ void dot_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           std::int64_t count, double* __restrict__ out) {
    __shared__ double partial[256];
    double sum = 0.0;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * 256) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(out, partial[0]); }
}

__global__ void add_diagonal_kernel(float* __restrict__ x, std::int64_t n, float value) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { x[i * n + i] += value; }
}

__global__ void multiply_kernel(float* __restrict__ a, const float* __restrict__ b,
                                std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { a[i] *= b[i]; }
}

__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x, float alpha,
                            std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { y[i] = fmaf(alpha, x[i], y[i]); }
}

__global__ void xpby_kernel(float* __restrict__ p, const float* __restrict__ r, float beta,
                            std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { p[i] = fmaf(beta, p[i], r[i]); }
}

unsigned blocks_for(std::int64_t count) { return static_cast<unsigned>((count + 255) / 256); }

} // namespace

void gemm(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n, std::int64_t k,
          bool transpose_b, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) { throw std::invalid_argument("gemm needs positive sizes"); }
    const dim3 grid(static_cast<unsigned>((n + kTile - 1) / kTile),
                    static_cast<unsigned>((m + kTile - 1) / kTile));
    if (transpose_b) {
        gemm_kernel<true><<<grid, 256, 0, stream>>>(a, b, c, m, n, k);
    } else {
        gemm_kernel<false><<<grid, 256, 0, stream>>>(a, b, c, m, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

void scale_rows_cols(float* x, std::int64_t rows, std::int64_t cols, const float* row,
                     const float* col, cudaStream_t stream) {
    scale_kernel<<<blocks_for(rows * cols), 256, 0, stream>>>(x, rows, cols, row, col);
    CUDA_CHECK(cudaGetLastError());
}

void row_rms(const float* x, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream) {
    pair_reduce_kernel<true>
        <<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, x, rows, cols, out, true);
    CUDA_CHECK(cudaGetLastError());
}

void col_rms(const float* x, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream) {
    pair_reduce_kernel<false>
        <<<static_cast<unsigned>(cols), 256, 0, stream>>>(x, x, rows, cols, out, true);
    CUDA_CHECK(cudaGetLastError());
}

void col_dot(const float* a, const float* b, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream) {
    pair_reduce_kernel<false>
        <<<static_cast<unsigned>(cols), 256, 0, stream>>>(a, b, rows, cols, out, false);
    CUDA_CHECK(cudaGetLastError());
}

void row_dot(const float* a, const float* b, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream) {
    pair_reduce_kernel<true>
        <<<static_cast<unsigned>(rows), 256, 0, stream>>>(a, b, rows, cols, out, false);
    CUDA_CHECK(cudaGetLastError());
}

double dot(const float* a, const float* b, std::int64_t count, cudaStream_t stream) {
    double* d_out = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_out, sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(double), stream));
    const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((count + 255) / 256, 1024));
    if (blocks > 0) { dot_kernel<<<blocks, 256, 0, stream>>>(a, b, count, d_out); }
    CUDA_CHECK(cudaGetLastError());
    double out = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

void add_diagonal(float* x, std::int64_t n, float value, cudaStream_t stream) {
    add_diagonal_kernel<<<blocks_for(n), 256, 0, stream>>>(x, n, value);
    CUDA_CHECK(cudaGetLastError());
}

void multiply(float* a, const float* b, std::int64_t count, cudaStream_t stream) {
    multiply_kernel<<<blocks_for(count), 256, 0, stream>>>(a, b, count);
    CUDA_CHECK(cudaGetLastError());
}

void axpy(float* y, const float* x, float alpha, std::int64_t count, cudaStream_t stream) {
    axpy_kernel<<<blocks_for(count), 256, 0, stream>>>(y, x, alpha, count);
    CUDA_CHECK(cudaGetLastError());
}

void xpby(float* p, const float* r, float beta, std::int64_t count, cudaStream_t stream) {
    xpby_kernel<<<blocks_for(count), 256, 0, stream>>>(p, r, beta, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
