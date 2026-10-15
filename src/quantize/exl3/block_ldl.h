#pragma once

// Block LDL decomposition for LDLQ: H = L D Lᵀ with L unit block-lower-triangular (16x16 identity
// diagonal blocks) and D block diagonal. Only L is produced; LDLQ feeds quantization error forward
// through its off-diagonal blocks.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

inline constexpr int kLdlBlock = 16;

// In place on a row-major FP32 n x n matrix (n % 64 == 0, at most 65535 64-wide panels): reads the
// lower triangle of the symmetric positive-definite H and leaves L (strict upper triangle zeroed).
// Uses a blocked Cholesky with 64-wide FP32 panels and FP64 diagonal-block factorization. Throws
// std::invalid_argument when n <= 0, n % 64 != 0 or n / 64 > 65535. Returns false when a pivot is
// not positive, i.e. H is not numerically positive definite; the caller then increases the diagonal
// damping and retries from the original H. Synchronizes the stream.
[[nodiscard]] bool block_ldl(float* h, std::int64_t n, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
