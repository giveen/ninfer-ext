#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: nemotron_h_moe_router
 *
 * Math / indexing:
 *   For each token t in [0,T) over contiguous BF16 x [H,T], a BF16 router weight [E,H] and an FP32
 *   correction bias [E], with G = `groups`, U = `groups_used` and W = E/G:
 *
 *     d[t,e] = sum_h W_router[e,h] * x[t,h]                       // FP32, no bias
 *     s[t,e] = sigmoid(d[t,e])                                    // the *uncorrected* score
 *     c[t,e] = s[t,e] + correction_bias[e]                        // the score the selection ranks
 *     g[t,j] = the sum of the two largest c[t, jW .. (j+1)W)      // one score per group
 *     S      = the U groups with the largest g[t,:]               // lower group id wins a tie
 *     (id, w)[t,0..K-1] = the K largest c[t,:] over S's experts   // lower expert id wins a tie
 *     w[t,r] = s[t, id[t,r]]                                      // the UNCORRECTED score
 *     if normalize_top_k: w[t,r] = w[t,r] / (sum_r w[t,r] + 1e-20)
 *     w[t,r] *= scaling_factor
 *
 *   The correction bias selects, and the emitted weight does not carry it. Adding it moves an
 *   expert up or down the ranking *and* feeds the group scores, while `w` stays the plain sigmoid
 *   of that expert's own logit: on this checkpoint the bias is a large near-constant offset (4.47
 *   to 4.55 over layer 1's 240-expert bank, 17.56 to 17.64 over layer 8's 304-expert bank), so an
 *   implementation that emitted the corrected score would be off by that offset rather than by a
 *   rounding. Two consequences are visible in the outputs: a rank's weight is not a function of its
 *   rank (only of the expert), and the ranks are therefore *not* ordered by descending weight --
 *   they are ordered by descending `c`, which is this Op's convention for the unsorted top-k.
 *
 *   The group rule is implemented as written because it is the specification and because the fields
 *   are per-layer: this checkpoint sets G = U = 1, which selects every expert, so the step is a
 *   no-op here, but a checkpoint that sets U < G would be silently unroutable without it. The group
 *   score degenerates to that single score when W == 1 (a group of one expert has no second
 *   element; the reference's own `topk(2)` raises there, so W == 1 is this Op's extension of the
 *   rule rather than a behaviour the reference defines).
 *
 * Shapes:
 *   x                contiguous BF16 [H,T], semantic axes [hidden, token], no padding.
 *   W_router         BF16 Weight [E,H]. The checkpoint stores BF16, and the Op decodes it exactly
 *                    into FP32 -- which is the reference's own `weight.type(torch.float32)`, not an
 *                    approximation of it.
 *   correction_bias  contiguous FP32 [E], one per expert, as a tensor. The checkpoint stores this
 *                    buffer in FP32, which is the precision the reference adds it in, so a BF16
 *                    copy would lose what the selection is made of.
 *   expert_ids       contiguous I32 [K,T] (row-major, token fastest axis), fully overwritten.
 *   expert_weights   contiguous FP32 [K,T], fully overwritten.
 *
 * Supported domain:
 *   T >= 1, H >= 1 and a multiple of 8, 1 <= E <= 512 (the family config's own bound; this
 *   checkpoint's banks are 240 to 496), 1 <= K <= min(E,16), 1 <= G <= 32, 1 <= U <= G with E a
 *   multiple of G, and K <= U*(E/G) -- the selected groups must be able to supply K experts, which
 *   a smaller U*(E/G) cannot. `scaling_factor` is finite and `normalize_top_k` is a flag. x, the
 *   weight payload and the workspace are 16-byte aligned.
 *
 * Ties:
 *   An exact tie (equal FP32 values, compared exactly) resolves to the lower id, of the group or of
 *   the expert, which is also what `gemma_moe_router` does; the ranks themselves are emitted in
 *   descending `c` order, which is this Op's own convention. The reference ranks with
 *   `torch.topk(sorted=False)`, which defines no order among equal values and gives the ranks no
 *   order at all, so a tie is exactly where the two are free to differ. In the reference dump below
 *   no token has two equal corrected scores and the reference's own rows happen to be in descending
 *   order, so that comparison matches the ranks as a set and reports the one token of each layer
 *   whose order differs after the operand rounding rather than failing it.
 *
 * Numeric:
 *   The projection, the sigmoid, the correction, the group scores and the weights are computed in
 *   FP32 from the decoded BF16 weight and the represented BF16 activations, with each sum over h
 *   accumulated in FP32. The normalization divides by `sum + 1e-20` and only then multiplies by the
 *   scale, in the reference's order. The reference's own arithmetic for this stage is FP32
 *   sequential -- `F.linear` on the FP32 cast, then FP32 tensor ops -- so an oracle that evaluates
 *   this formula sequentially is exact enough by construction: the Op's only nonlinearity is the
 *   sigmoid, and one ulp per step is far inside the criteria. The oracle used here evaluates the
 *   same sequence in FP64 from the same represented operands, which bounds the reference's own
 *   rounding rather than adding to it.
 *
 * Effects:
 *   expert_ids and expert_weights are completely overwritten. x, W_router and correction_bias are
 *   read only. No state is kept between calls.
 *
 * Workspace:
 *   Caller-owned, graph-stable transient storage of at least
 * `nemotron_h_moe_router_workspace_bytes`, 16-byte aligned, carrying no state beyond the call: it
 * holds the FP32 [E,T] logits between the projection and the selection, so the projection alone is
 * observable to a verification harness (the reference dump exposes the same tensor). The Op
 * allocates nothing.
 *
 * Execution:
 *   The caller's stream; graph-capturable (no host synchronization, no allocation).
 */
struct NemotronHMoeRouterWeights {
    Weight router;          // [E, H]
    Tensor correction_bias; // FP32 [E]
};

struct NemotronHMoeRouterGeometry {
    std::int32_t hidden_size = 0;     // H
    std::int32_t num_experts = 0;     // E
    std::int32_t top_k       = 0;     // K
    std::int32_t groups      = 1;     // G  (n_group)
    std::int32_t groups_used = 1;     // U  (topk_group)
    bool normalize_top_k     = false; // norm_topk_prob
    float scaling_factor     = 1.0F;  // routed_scaling_factor
};

// The workspace a call over T tokens needs: the FP32 [E,T] logits. Throws on a non-positive E or T.
[[nodiscard]] std::size_t
nemotron_h_moe_router_workspace_bytes(const NemotronHMoeRouterGeometry& geometry,
                                      std::int32_t tokens);

// Applies the router to the T = x.ne[1] tokens (T is not a parameter). Throws std::invalid_argument
// for a non-positive geometry, a hidden_size that is not a multiple of 8, num_experts outside
// [1,512], top_k outside [1,min(E,16)], groups outside [1,32], an E that groups does not divide,
// groups_used outside [1,groups], a top_k that groups_used*(E/groups) cannot supply, a non-finite
// scaling_factor, mis-typed, mis-shaped, unbound or unaligned x/router/correction_bias tensors,
// mis-shaped outputs, and an undersized or unaligned workspace.
void nemotron_h_moe_router(const Tensor& x, const NemotronHMoeRouterWeights& weights,
                           const NemotronHMoeRouterGeometry& geometry, Tensor& expert_ids,
                           Tensor& expert_weights, Tensor& workspace, cudaStream_t stream);

} // namespace ninfer::ops
