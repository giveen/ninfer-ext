#pragma once

// ninfer::ops::detail - private launch prototype for causal_compact_attention.

#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// `key_codes`/`key_scales` are null for BF16 rows; when set, `kv` is ignored and the compact rows
// are decoded from the planes as code * scale.

void causal_compact_attention_launch(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                                     const Tensor& position_k, const Tensor* key_codes,
                                     const Tensor* key_scales, std::int32_t rotary_dim,
                                     std::int32_t rope_angles, std::int32_t query_heads,
                                     std::int32_t kv_heads, float scale, bool indexed_keys,
                                     Tensor& out, const Tensor& workspace,
                                     DeviceExecutionView execution);

std::size_t causal_compact_attention_workspace_bytes(std::int32_t rotary_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t query_heads,
                                                     std::int32_t kv_heads, std::int32_t key_tokens,
                                                     std::int32_t batch);

} // namespace ninfer::ops::detail
