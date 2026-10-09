#pragma once

// One Gemma 4 sliding decoder layer, composed from the Ops that already exist.
//
// The sequence follows the reference engines: pre-norm, the three attention input projections, the
// per-head q/k norms and the weightless v norm, RoPE, attention, the output projection, then the
// branch's post-norm added to the residual; then the MLP's pre-norm, the fused gate/up projection,
// GeGLU, the down projection, and again the branch's post-norm added to the residual, with the layer
// scalar applied to the residual once. The layer's output is that residual itself, because the next
// layer applies its own input norm.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/model.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::models::gemma4 {

// Transient capacity one layer needs, at any position.
[[nodiscard]] std::size_t sliding_layer_workspace_bytes(const TextConfig& config);

/**
 * Runs one sliding layer for one token.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 [hidden_size,1] and must not overlap; the layer
 * reads the residual stream and writes its updated self to `hidden_out`. `position` is the token's
 * absolute position, and the attention's key and value sets are the token itself, which is the
 * single-token case of the ring a Program maintains for longer contexts.
 */
void forward_sliding_layer(const Model& model, std::size_t layer, const Tensor& hidden_in,
                           std::int32_t position, DeviceArena& arena, Tensor& hidden_out,
                           DeviceExecutionView execution);

} // namespace ninfer::models::gemma4
