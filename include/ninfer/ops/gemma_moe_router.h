#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: gemma_moe_router
 *
 * Math / indexing:
 *   For each token t in [0,T) over contiguous BF16 x [H,T]:
 *
 *     n[t,h]  = x[t,h] * ( mean_h(x[t,:]^2) + eps )^(-1/2)   // unit RMS, no weight and no gain
 *     s[t,h]  = n[t,h] * router_scale[h] * H^(-1/2)
 *     logit[t,e] = sum_h W_router[e,h] * s[t,h]                // [E] per token, no bias
 *     p[t,e]  = softmax(logit[t,:])[e]
 *     (w, id)[t,0..K-1] = the K largest p[t,:] with ties broken on the lower expert id
 *     w[t,j]  = w[t,j] / sum_j w[t,j] * per_expert_scale[id[t,j]]
 *
 *   The renormalization happens before the per-expert scale, so the applied weights do not sum to
 *   one when the scales differ. The ranking is by `p`, i.e. by the pre-scale probability: the
 *   per-expert scale multiplies a selected weight and never changes the selection.
 *
 *   H^(-1/2) is a host-computed constant, not a learned value; it is folded into the scale above,
 *   so no separate scalar Op is needed. The norm is weightless: the unit-RMS vector is used
 *   directly, which is the reason the router's own norm is not `rmsnorm`'s learned-weight form.
 *
 * Logical shapes:
 *   x              contiguous BF16 [H,T], semantic axes [hidden, token], no padding.
 *   router_scale   BF16 [H], one per hidden index, as a tensor.
 *   per_expert_scale BF16 [E], as a tensor.
 *   W_router       Weight [E,H] in any registered matrix format; the decode is the format's.
 *   expert_ids     contiguous I32 [K,T]  (rank fastest axis), fully overwritten.
 *   expert_weights contiguous FP32 [K,T], fully overwritten.
 *
 * Supported domain:
 *   T >= 1, H >= 1, E >= 1, 1 <= K <= E. Any registered weight format the linear Op also accepts.
 *   The artifact geometry this Op was admitted for is H=2816, E=128, K=8.
 *
 * Numeric:
 *   The mean of squares, the reciprocal square root, the projection, the softmax and the weights
 *   are computed in FP32 from the decoded weights and the represented BF16 activations, with the
 *   sum over h accumulated in FP32. `eps` is the config's rms_norm_eps. An exact tie in `p` at the
 *   K-th boundary selects the lower expert id; ties among floats are exact equality, not a
 *   tolerance. The oracle evaluates the same formula naively in FP64.
 *
 * Effects:
 *   expert_ids and expert_weights are completely overwritten. x, router_scale, per_expert_scale,
 *   W_router and `eps` are read only. No state is kept between calls.
 *
 * Workspace:
 *   None. The Op allocates nothing and needs no caller scratch.
 *
 * Execution:
 *   The caller's stream; graph-capturable (no host synchronization, no allocation).
 */
struct GemmaMoeRouterWeights {
    Weight router; // [E, H]
    // The router's two scales are plain vectors with no packed representation, so the model hands
    // them over as tensors, exactly as it hands over a norm weight.
    Tensor router_scale;     // BF16 [H]
    Tensor per_expert_scale; // BF16 [E]
};

struct GemmaMoeRouterGeometry {
    std::int32_t hidden_size = 0; // H
    std::int32_t num_experts = 0; // E
    std::int32_t top_k       = 0; // K
    float rms_norm_eps       = 0.0F;
};

void gemma_moe_router(const Tensor& x, const GemmaMoeRouterWeights& weights,
                      const GemmaMoeRouterGeometry& geometry, Tensor& expert_ids,
                      Tensor& expert_weights, cudaStream_t stream);

} // namespace ninfer::ops
