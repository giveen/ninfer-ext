#pragma once

// ninfer::ops::detail - private launch prototype for residual_add_norm. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void residual_add_norm_launch(const Tensor& x, const Tensor& weight, float eps, Tensor& accumulator,
                              float scale, cudaStream_t stream);

} // namespace ninfer::ops::detail
