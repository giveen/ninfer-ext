#pragma once

// ninfer::ops::detail - private launch prototype for residual_add. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entries; assume inputs already validated by the wrapper. The dtype selects the route, so
// the wrapper dispatches on the tensor dtypes and never mixes them.
void residual_add_launch(const Tensor& y, Tensor& x, cudaStream_t stream);
void residual_add_fp32_launch(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops::detail
