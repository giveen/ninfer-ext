#pragma once

// ninfer::ops::detail - private launch prototype for scale_columns.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void scale_columns_launch(const Tensor& scale, Tensor& x, std::int32_t dims_begin,
                          std::int32_t dims_end, cudaStream_t stream);

} // namespace ninfer::ops::detail
