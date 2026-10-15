#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Private launch prototype (host/cu only; not public api). Assumes the wrapper has already
// validated shapes/contiguity: source and destination have equal numel, are contiguous, and do not
// alias.

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
