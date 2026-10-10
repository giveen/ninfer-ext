#pragma once

// ninfer::ops::detail - private launch prototype for soft_cap.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void soft_cap_launch(Tensor& x, float cap, cudaStream_t stream);

} // namespace ninfer::ops::detail
