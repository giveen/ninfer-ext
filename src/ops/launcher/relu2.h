#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Preconditions, all established by the wrapper: x is a contiguous BF16 tensor with non-null data
// and 16-byte-aligned storage is a route condition rather than a requirement.
void relu2_launch(Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops::detail
