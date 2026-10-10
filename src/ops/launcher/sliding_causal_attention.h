#pragma once

// ninfer::ops::detail - private launch prototype for sliding_causal_attention.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sliding_causal_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& position_q, const Tensor& position_q_high,
                                     const Tensor& position_k,
                                     const Tensor& k_extra, const Tensor& v_extra,
                                     const Tensor& position_extra, std::int32_t head_dim, std::int32_t query_heads,
                                     std::int32_t kv_heads, std::uint32_t window, float scale,
                                     Tensor& out, const Tensor& workspace, cudaStream_t stream);

std::size_t sliding_causal_attention_workspace_bytes(std::int32_t head_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t query_heads,
                                                     std::int32_t kv_heads, std::int32_t key_tokens,
                                                     std::int32_t batch);

} // namespace ninfer::ops::detail
