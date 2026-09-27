#pragma once

// Block LDL decomposition for LDLQ: H = L D Lᵀ with L unit block-lower-triangular (16x16 identity
// diagonal blocks) and D block diagonal. Only L is produced; LDLQ feeds quantization error forward
// through its off-diagonal blocks.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

inline constexpr int kLdlBlock = 16;

// In place on a row-major FP32 n x n matrix (n % 64 == 0): reads the lower triangle of the
// symmetric positive-definite H and leaves L (strict upper triangle zeroed). Uses a blocked
// Cholesky with 64-wide FP32 panels and FP64 diagonal-block factorization. Returns false when a
// pivot is not positive, i.e. H is not numerically positive definite; the caller then increases
// the diagonal damping and retries from the original H. Synchronizes the stream.
[[nodiscard]] bool block_ldl(float* h, std::int64_t n, cudaStream_t stream);

} // namespace ninfer::quantize::exl3
