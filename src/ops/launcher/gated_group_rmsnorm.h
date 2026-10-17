#pragma once

// ninfer::ops::detail - private launch prototype for gated_group_rmsnorm.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gated_group_rmsnorm_launch(const Tensor& x, const Tensor& weight, const Tensor& gate,
                                std::int32_t group_size, float eps, Tensor& out,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
