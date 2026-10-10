#pragma once

// ninfer::ops::detail - private launch prototype for compact_kv_rows.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void compact_kv_rows_launch(const Tensor& v, const Tensor& k, Tensor& out, std::int32_t rotary_dim,
                            std::int32_t rotary_pairs, cudaStream_t stream);

} // namespace ninfer::ops::detail
