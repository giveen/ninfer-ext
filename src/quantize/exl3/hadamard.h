#pragma once

// Normalized 128-point Walsh-Hadamard transforms (Sylvester order, scaled by 1/sqrt(128)) for the
// EXL3 quantizer, in place on row-major FP32 matrices. H128 is symmetric and orthogonal, so each
// transform is its own inverse.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

// x[r, 128b:128b+128] <- x[r, 128b:128b+128] * H128 for every row r and block b (cols % 128 == 0).
void hadamard128_rows(float* data, std::int64_t rows, std::int64_t cols, cudaStream_t stream);

// x[128b:128b+128, c] <- H128 * x[128b:128b+128, c] for every column c and block b
// (rows % 128 == 0).
void hadamard128_cols(float* data, std::int64_t rows, std::int64_t cols, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
