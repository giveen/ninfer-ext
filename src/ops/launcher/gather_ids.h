#pragma once

// ninfer::ops::detail - private launch prototype for gather_ids.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void gather_ids_launch(const Tensor& index, const Tensor& table, Tensor& out, std::int32_t count,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail
