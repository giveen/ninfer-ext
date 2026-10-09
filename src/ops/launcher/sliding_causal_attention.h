#pragma once

// ninfer::ops::detail - private launch prototype for sliding_causal_attention.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sliding_causal_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& position_q, const Tensor& position_k,
                                     std::int32_t head_dim, std::int32_t query_heads,
                                     std::int32_t kv_heads, std::uint32_t window, float scale,
                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
