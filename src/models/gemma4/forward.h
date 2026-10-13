#pragma once

// One Gemma 4 decoder layer, composed from the Ops that already exist.
//
// The sequence follows the reference engines: pre-norm, the attention input projections, the
// per-head q/k norms and the weightless v norm, RoPE, attention, the output projection, then the
// branch's post-norm added to the residual; then the MLP's pre-norm, the gate and up projections,
// GeGLU, the down projection, and again the branch's post-norm added to the residual, with the
// layer scalar applied to the residual once. The layer's output is that residual itself, because
// the next layer applies its own input norm.
//
// The two layer kinds differ only in their attention. Sliding layers rotate the whole 256-wide
// head, attend over a linear window, and store a value projection. Global layers rotate
// `rope_angles` pairs against a 512-wide head, store no value projection because K is V, and attend
// over the compact KV representation `compact_kv_rows` writes.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/model.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include <cuda_runtime.h>

namespace ninfer::models::gemma4 {

// Transient capacity one pass needs: the pass's indices (`write_pass_indices`) plus the scratch of
// one layer running `tokens` tokens at once, of either kind, so a caller can reuse one arena for
// every layer of a pass. A quantized sliding storage adds the BF16 ring each layer decodes its
// quantized ring into, which is why the bound takes the storage and the ring's slack.
[[nodiscard]] std::size_t layer_workspace_bytes(const TextConfig& config, std::int32_t tokens,
                                                KvCacheStorage storage  = KvCacheStorage::BFloat16,
                                                std::int32_t ring_slack = 0);

// One lane's share of a pass: `tokens` consecutive columns from `column`, the lane's tokens at the
// positions from `first_position`, attending to and written into `cache`. A prefill or a verify
// pass is one segment; a decode round over several lanes is one single-token segment per lane, so
// the lanes share every weight read while each attends to its own cache.
struct PassSegment {
    KvCache* cache              = nullptr;
    std::int32_t first_position = 0;
    std::int32_t column         = 0;
    std::int32_t tokens         = 0;
};

// The device I32 [columns] vectors every layer of one pass reads: each column's absolute position,
// and the sliding-ring slot that position occupies in its lane's cache. The layers take positions
// and cache slots from here rather than from the host, so a pass's launches do not depend on where
// in the sequence it runs: that is what lets one captured decode graph serve every position.
struct PassIndices {
    Tensor positions;
    Tensor ring_slots;
};

// Allocates the indices for `segments`, which tile [0, columns) in order, from `arena` (the
// caller's scope owns them) and writes them on `stream`. The pageable source is staged before the
// call returns.
[[nodiscard]] PassIndices write_pass_indices(std::span<const PassSegment> segments,
                                             DeviceArena& arena, cudaStream_t stream);
// The same for one lane's tokens at `first_position`...
[[nodiscard]] PassIndices write_pass_indices(const KvCache& cache, std::int32_t first_position,
                                             std::int32_t tokens, DeviceArena& arena,
                                             cudaStream_t stream);

/**
 * Runs one decoder layer for `tokens` tokens at once, dispatching on the layer's attention kind.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 [hidden_size,tokens] and must not overlap; the
 * layer reads the residual stream and writes its updated self to `hidden_out`. The tokens occupy
 * the consecutive absolute positions starting at `first_position`, which `indices` holds on the
 * device, so a prefill is one call and a decode is the same call with one token. Every token's key
 * and value rows are written into `cache` before the attention runs, and each query then sees
 * exactly what the cache makes visible to its own position, which is what makes a batched prefill
 * produce the same result as token-at-a-time.
 *
 * A pass of at most `cache.ring_slack() + 1` tokens issues the same launches at every position: its
 * cache writes are indexed on the device and its attention scans the whole ring and the whole
 * global capacity, whose unwritten and future slots are invisible.
 */
//
// `position_high`, when given, is device I32 [tokens,1]: the latest position each token's sliding
// attention may see. An image's tokens see their whole image (its last position); every other
// token, and every global layer, stays causal. Empty means causal throughout.
void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::int32_t first_position, std::int32_t tokens, const PassIndices& indices,
                   KvCache& cache, DeviceArena& arena, Tensor& hidden_out,
                   DeviceExecutionView execution, const Tensor& position_high = {});

// The same layer over several lanes' `segments` at once, `hidden_in`/`hidden_out` [hidden_size,
// columns] with the segments tiling the columns in order. Every column-wise step (norms,
// projections, RoPE, the MLP) runs once over all columns; each segment's cache writes and attention
// run against its own cache, exactly as the one-lane call would run them. `position_high` needs a
// single segment.
void forward_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   std::span<const PassSegment> segments, const PassIndices& indices,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   const Tensor& position_high = {});

/**
 * Applies the output head to a layer stack's final hidden state.
 *
 * `hidden_in` is contiguous BF16 [hidden_size,tokens] and `logits` contiguous BF16
 * [vocabulary,tokens], one column per token. The head
 * is the final norm, then the projection through the embedding matrix, which is tied to the token
 * embedding, and then the logit soft cap. The cap is applied here because it is part of the model's
 * logits; every consumer that must see uncapped values is the caller's business.
 */
void forward_head(const Model& model, const Tensor& hidden_in, std::int32_t tokens,
                  DeviceArena& arena, Tensor& logits, DeviceExecutionView execution,
                  Tensor* normed_out = nullptr);

/**
 * One step of the assistant drafter, for one token.
 *
 * `input` is contiguous BF16 [2 * target hidden, 1]: the target's scaled embedding of the token the
 * step reads, then a target-width hidden state (the target's post-final-norm state for the step
 * that follows a target pass, the drafter's own projected state after that). `position` is the
 * absolute position of that token, which the target has not consumed: every step of a round uses
 * the same position. The drafter attends, at that RoPE position, to the keys the target's last
 * sliding and last global layers hold strictly before it, read from `cache`, which it does not
 * change.
 *
 * Writes `logits` (BF16 [vocabulary, 1], the drafter's head with no soft cap) and `hidden_out`
 * (BF16 [target hidden, 1], the post-projected state the next step reads).
 */
void forward_draft(const Model& model, const Tensor& input, std::int32_t position,
                   const KvCache& cache, DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution);

// One lane of a batched drafter step: the cache it reads and the position of its token.
struct DraftLane {
    const KvCache* cache  = nullptr;
    std::int32_t position = 0;
};

// The same step for several lanes at once: `input` [2 * target hidden, lanes], `logits`
// [vocabulary, lanes] and `hidden_out` [target hidden, lanes] hold one column per lane, in order.
// The projections and MLP run once over every column; each lane attends to its own cache.
void forward_draft(const Model& model, const Tensor& input, std::span<const DraftLane> lanes,
                   DeviceArena& arena, Tensor& logits, Tensor& hidden_out,
                   DeviceExecutionView execution);

} // namespace ninfer::models::gemma4
