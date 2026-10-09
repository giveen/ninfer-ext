#pragma once

// One Gemma 4 decoder layer, composed from the Ops that already exist.
//
// The sequence follows the reference engines: pre-norm, the attention input projections, the per-head
// q/k norms and the weightless v norm, RoPE, attention, the output projection, then the branch's
// post-norm added to the residual; then the MLP's pre-norm, the gate and up projections, GeGLU, the
// down projection, and again the branch's post-norm added to the residual, with the layer scalar
// applied to the residual once. The layer's output is that residual itself, because the next layer
// applies its own input norm.
//
// The two layer kinds differ only in their attention. Sliding layers rotate the whole 256-wide head,
// attend over a linear window, and store a value projection. Global layers rotate `rope_angles` pairs
// against a 512-wide head, store no value projection because K is V, and attend over the compact KV
// representation `compact_kv_rows` writes.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/model.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::models::gemma4 {

// Transient capacity one layer needs, at any position and of either kind, so a caller can reuse one
// arena for every layer of a pass.
[[nodiscard]] std::size_t layer_workspace_bytes(const TextConfig& config);

/**
 * Runs one decoder layer for one token, dispatching on the layer's attention kind.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 [hidden_size,1] and must not overlap; the layer
 * reads the residual stream and writes its updated self to `hidden_out`. `position` is the token's
 * absolute position: the layer writes the token's key and value rows into `cache` and then attends
 * over everything the cache still makes visible to that position.
 */
void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::int32_t position, KvCache& cache, DeviceArena& arena, Tensor& hidden_out,
                   DeviceExecutionView execution);

/**
 * Applies the output head to a layer stack's final hidden state.
 *
 * `hidden_in` is contiguous BF16 [hidden_size,1] and `logits` contiguous BF16 [vocabulary,1]. The head
 * is the final norm, then the projection through the embedding matrix, which is tied to the token
 * embedding, and then the logit soft cap. The cap is applied here because it is part of the model's
 * logits; every consumer that must see uncapped values is the caller's business.
 */
void forward_head(const Model& model, const Tensor& hidden_in, DeviceArena& arena, Tensor& logits,
                  DeviceExecutionView execution);

} // namespace ninfer::models::gemma4
