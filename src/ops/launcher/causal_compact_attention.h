#pragma once

// ninfer::ops::detail - private launch prototype for causal_compact_attention.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void causal_compact_attention_launch(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                                     const Tensor& position_k, std::int32_t rotary_dim,
                                     std::int32_t rope_angles, std::int32_t query_heads,
                                     std::int32_t kv_heads, float scale, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
