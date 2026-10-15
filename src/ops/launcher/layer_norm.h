#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Private launch prototype (host/cu only; not public api). Assumes the wrapper has already
// validated: x/weight/bias/out are BF16 and contiguous, weight and bias are 1-D [x.ne[0]], and
// x.ne[0] divides x.numel() (the .cu derives rows = x.numel() / x.ne[0] with no zero guard).

void layer_norm_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
