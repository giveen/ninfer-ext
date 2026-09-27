#pragma once

// Calibration Hessian accumulation for the EXL3 quantizer: H += Xᵀ X in FP32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::quantize::exl3 {

// x: [tokens][k] BF16 activations, row-major with row stride k. h: [k][k] FP32, updated in place
// (both triangles). k % 128 == 0.
void accumulate_hessian(const __nv_bfloat16* x, std::int64_t tokens, std::int64_t k, float* h,
                        cudaStream_t stream);

} // namespace ninfer::quantize::exl3
