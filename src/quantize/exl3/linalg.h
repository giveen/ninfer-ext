#pragma once

// Small FP32 device linear-algebra helpers for the offline EXL3 quantizer. Row-major throughout.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

// C[m][n] = A[m][k] * B[k][n]            (transpose_b = false)
// C[m][n] = A[m][k] * B[n][k]ᵀ           (transpose_b = true)
void gemm(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n, std::int64_t k,
          bool transpose_b, cudaStream_t stream);

// x[r][c] *= row[r] (row may be null) and *= col[c] (col may be null).
void scale_rows_cols(float* x, std::int64_t rows, std::int64_t cols, const float* row,
                     const float* col, cudaStream_t stream);

// out[r] = sqrt(mean_c x[r][c]^2)
void row_rms(const float* x, std::int64_t rows, std::int64_t cols, float* out, cudaStream_t stream);
// out[c] = sqrt(mean_r x[r][c]^2)
void col_rms(const float* x, std::int64_t rows, std::int64_t cols, float* out, cudaStream_t stream);

// Sum over all elements of a[i] * b[i] in FP64 (host result; synchronizes the stream).
[[nodiscard]] double dot(const float* a, const float* b, std::int64_t count, cudaStream_t stream);

// out[c] = sum_r a[r][c] * b[r][c]   (column-wise dot products)
void col_dot(const float* a, const float* b, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream);
// out[r] = sum_c a[r][c] * b[r][c]   (row-wise dot products)
void row_dot(const float* a, const float* b, std::int64_t rows, std::int64_t cols, float* out,
             cudaStream_t stream);

// x[i][i] += value for i < n (x is n x n)
void add_diagonal(float* x, std::int64_t n, float value, cudaStream_t stream);

// a[i] *= b[i]
void multiply(float* a, const float* b, std::int64_t count, cudaStream_t stream);
// y[i] += alpha * x[i]
void axpy(float* y, const float* x, float alpha, std::int64_t count, cudaStream_t stream);
// p[i] = r[i] + beta * p[i]
void xpby(float* p, const float* r, float beta, std::int64_t count, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
