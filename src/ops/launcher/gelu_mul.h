#pragma once

// ninfer::ops::detail - private launch prototype for gelu_mul.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gelu_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
