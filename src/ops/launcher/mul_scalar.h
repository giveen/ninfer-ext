#pragma once

// ninfer::ops::detail - private launch prototype for mul_scalar.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mul_scalar_launch(Tensor& x, float factor, cudaStream_t stream);

} // namespace ninfer::ops::detail
