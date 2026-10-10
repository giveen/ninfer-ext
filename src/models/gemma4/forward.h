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

// Transient capacity one layer needs to run `tokens` tokens at once, of either kind, so a caller can
// reuse one arena for every layer of a pass.
[[nodiscard]] std::size_t layer_workspace_bytes(const TextConfig& config, std::int32_t tokens);

/**
 * Runs one decoder layer for `tokens` tokens at once, dispatching on the layer's attention kind.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 [hidden_size,tokens] and must not overlap; the
 * layer reads the residual stream and writes its updated self to `hidden_out`. The tokens occupy the
 * consecutive absolute positions starting at `first_position`, so a prefill is one call and a decode
 * is the same call with one token. Every token's key and value rows are written into `cache` before
 * the attention runs, and each query then sees exactly what the cache makes visible to its own
 * position, which is what makes a batched prefill produce the same result as token-at-a-time.
 */
void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::int32_t first_position, std::int32_t tokens, KvCache& cache,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution);

/**
 * Applies the output head to a layer stack's final hidden state.
 *
 * `hidden_in` is contiguous BF16 [hidden_size,tokens] and `logits` contiguous BF16
 * [vocabulary,tokens], one column per token. The head
 * is the final norm, then the projection through the embedding matrix, which is tied to the token
 * embedding, and then the logit soft cap. The cap is applied here because it is part of the model's
 * logits; every consumer that must see uncapped values is the caller's business.
 */
void forward_head(const Model& model, const Tensor& hidden_in, std::int32_t tokens, DeviceArena& arena,
                  Tensor& logits, DeviceExecutionView execution, Tensor* normed_out = nullptr);

/**
 * One step of the assistant drafter, for one token.
 *
 * `input` is contiguous BF16 [2 * target hidden, 1]: the target's scaled embedding of the token the
 * step reads, then a target-width hidden state (the target's post-final-norm state for the step that
 * follows a target pass, the drafter's own projected state after that). `position` is the absolute
 * position of that token, which the target has not consumed: every step of a round uses the same
 * position. The drafter attends, at that RoPE position, to the keys the target's last sliding and last
 * global layers hold strictly before it, read from `cache`, which it does not change.
 *
 * Writes `logits` (BF16 [vocabulary, 1], the drafter's head with no soft cap) and `hidden_out` (BF16
 * [target hidden, 1], the post-projected state the next step reads).
 */
void forward_draft(const Model& model, const Tensor& input, std::int32_t position,
                   const KvCache& cache, DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution);

} // namespace ninfer::models::gemma4
