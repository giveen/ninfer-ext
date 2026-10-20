#pragma once

// ninfer::ops::detail - private launch prototype for compact_kv_rows_quantize.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void compact_kv_rows_quantize_launch(const Tensor& rows, const Tensor& positions, Tensor& codes,
                                     Tensor& scales, std::int32_t kv_heads, std::int32_t capacity,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
