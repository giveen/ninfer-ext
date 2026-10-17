#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                    cudaStream_t stream, bool sigmoid_gate = false, bool weightless = false);

// Plain epilogue only: x/out FP32, gain BF16 or absent (weight == nullptr means unit gain).
void rmsnorm_fp32_launch(const Tensor& x, const Tensor* weight, float eps, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
