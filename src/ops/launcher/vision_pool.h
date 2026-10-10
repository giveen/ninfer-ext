#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void vision_pool_standardize_launch(const Tensor& x, int grid_width, int kernel, float multiplier,
                                    const Tensor& bias, const Tensor& scale, Tensor& out,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
