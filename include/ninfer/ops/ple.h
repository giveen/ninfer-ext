#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * PLE stream gate. `key` and `query` are grouped-normalized contiguous BF16 `[S*H, T]`, `value`
 * contiguous BF16 `[H, T]`, `gated` contiguous BF16 `[S*H, T]`:
 *
 *   g[s,t]         = sum_h key[s*H+h,t] * query[s*H+h,t] / sqrt(H)
 *   g'[s,t]        = sign(g) * sqrt(max(|g|, 1e-6))
 *   gated[s*H+h,t] = sigmoid(g'[s,t]) * value[h,t].
 */
void ple_gate(const Tensor& key, const Tensor& query, const Tensor& value, std::int32_t streams,
              Tensor& gated, cudaStream_t stream);

/**
 * Causal dilated depthwise convolution with SiLU, added to the residual with the gated values.
 *
 * `normed` and `gated` are contiguous BF16 `[C, W, B]`; `weight` is contiguous BF16 `[K, C]` (tap
 * major, tap K-1 multiplies the current column); `states` is contiguous BF16 `[(K-1)*dilation*C,
 * S]`, slot s holding the (K-1)*dilation previous `normed` columns oldest first. Lane b reads slot
 * `source_slots[b]`. For every valid column w:
 *
 *   conv[c,w,b]      = sum_j weight[j,c] * x[w - (K-1-j)*dilation]   over [state ; normed]
 *   residual[c,w,b] += gated[c,w,b] + silu(conv[c,w,b]).
 *
 * The state is not modified; ple_conv_advance commits it.
 */
void ple_dilated_conv(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                      std::int32_t dilation, const Tensor& states, const Tensor& source_slots,
                      Tensor& residual, cudaStream_t stream);

/**
 * Advance convolution history: slot `destination_slots[b]` receives the
 * last (K-1)*dilation columns of `[states(source_slots[b]) ; normed[:, 0:count_b, b]]`.
 */
void ple_conv_advance(const Tensor& normed, const Tensor* valid_columns, std::int32_t history,
                      Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                      cudaStream_t stream);

} // namespace ninfer::ops
