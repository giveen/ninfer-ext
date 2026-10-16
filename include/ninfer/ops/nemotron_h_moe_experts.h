#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/offload_moe.h" // ExpertBankSpan, ExpertWeights (the plane-addressed route)

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: nemotron_h_moe_experts
 *
 * Math / indexing:
 *   For each token t in [0,T), over contiguous BF16 x [H,T] and the router's selection:
 *
 *     out[:,t] = sum over the K ranks j of  w[j,t] * down_e · relu2( up_e · x[:,t] )
 *
 *     with e = expert_ids[j,t], w = expert_weights[j,t], and relu2(z) = max(z, 0)^2.
 *
 *   The experts are **not gated**: an expert is one matrix pair, up into the intermediate width and
 *   down back out, with no gate projection and no elementwise product of two projections. The
 *   activation is applied to the up projection's own result, so an expert's contribution is
 *   `down · relu2(up · x)` and not `down · (relu2(up · x) ⊙ something)`; `ops::gemma_moe_experts`
 *   is the gated counterpart and is not a re-parameterization of this Op.
 *
 *   The experts are stored expert-major in two banks and are read in place, with no gather and no
 *   repack: expert e's up rows are `[e*I, (e+1)*I)` of `up` and its down rows are
 *   `[e*H, (e+1)*H)` of `down`. `H` is the width the bank is bound at, which for a layer with a
 *   latent projection is that projection's latent width (this checkpoint's 1024) and otherwise the
 *   block's hidden width; nothing in the Op depends on which, and a bank bound at the hidden width
 *   is the same Op over a different H. `E`, `I`, `K` and `H` are all runtime: this family's MoE
 *   layers carry banks of 240 to 496 experts at four intermediate widths (1536, 2048, 2304, 2560)
 *   around one latent width, so none of them is a property of the Op.
 *
 * Logical shapes:
 *   x               contiguous BF16 [H,T].
 *   expert_ids      contiguous I32 [K,T], rank fastest axis.
 *   expert_weights  contiguous FP32 [K,T], rank fastest axis.
 *   up              Weight [E*I, H].
 *   down            Weight [E*H, I].
 *   out             contiguous BF16 [H,T], completely overwritten.
 *
 * Supported domain:
 *   T >= 1, H >= 1, E >= 1, 1 <= K <= E, I >= 1. Both banks are BF16 and share one format. Every
 *   bank shape is checked against the geometry rather than trusted: `up` must have exactly E*I rows
 *   of H columns and `down` exactly E*H rows of I columns, so a bank that is one expert short, or
 *   bound at a latent width the geometry does not name, is refused instead of read out of range.
 *
 * Numeric:
 *   The activations are the represented BF16 values and the weights the exact decode of their
 *   stored form. Each expert's up projection, its `relu2`, and its down projection are computed in
 *   FP32, and the K selected contributions are accumulated per token in FP32 in ascending rank
 *   order, with one BF16 rounding at the output. The checkpoint's own reference accumulates the
 *   same sum in the routing weights' dtype -- FP32 -- and rounds to the activation dtype only when
 *   it returns, which is why that ordering and not a BF16 accumulator is the Op's contract. The
 *   oracle evaluates the formula above naively in FP64 from the same represented inputs; the
 *   reduction association, any private accumulator and the placement of the activation are
 *   execution choices rather than semantic boundaries. Only the selected experts' rows are read,
 *   so an unselected expert's bank content cannot affect the result. Every expert id in [0,E) is
 *   assumed, which the producing router guarantees.
 *
 * Effects:
 *   `out` is completely overwritten, including every token whose ranks are all zero-weighted.
 *   Every input and both banks are read only. No state is kept between calls.
 *
 * Workspace:
 *   Caller-owned, graph-stable, 16-byte aligned transient storage of at least
 *   `nemotron_h_moe_experts_workspace_bytes` for the call's geometry, bank format and T, carrying
 *   no state beyond the call. The Op allocates nothing. A route that needs no scratch reports zero
 *   and then accepts an unbound workspace; the query exists because the registered-format route
 *   this Op's dispatch is built to take stages the selected experts, and a caller that sizes its
 *   allocation from the query needs no change when that route lands.
 *
 * Execution:
 *   The caller's stream; graph-capturable (no host synchronization, no allocation). The output is
 *   bit-identical from run to run: each token's ranks are summed in ascending rank order.
 */
struct NemotronHMoeExpertWeights {
    Weight up;   // [num_experts * intermediate, latent]
    Weight down; // [num_experts * latent, intermediate]
};

struct NemotronHMoeExpertGeometry {
    std::int32_t latent_size       = 0; // H, the experts' own width (this checkpoint's 1024)
    std::int32_t num_experts       = 0; // E
    std::int32_t top_k             = 0; // K
    std::int32_t intermediate_size = 0; // I
};

// The workspace a call over T tokens needs for banks stored as `qtype`. Throws on a non-positive T.
[[nodiscard]] std::size_t
nemotron_h_moe_experts_workspace_bytes(const NemotronHMoeExpertGeometry& geometry, QType qtype,
                                       std::int32_t tokens);

void nemotron_h_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                            const NemotronHMoeExpertWeights& weights,
                            const NemotronHMoeExpertGeometry& geometry, Tensor& out,
                            Tensor& workspace, cudaStream_t stream);

/**
 * The same Op over a bank addressed as planes rather than as two whole matrices: the registered
 * (block-scaled NVFP4) form of one layer's routed experts.
 *
 * `bank` is the layer's bank as `ExpertWeights` addresses it -- plane 0 the gate/up codes, plane 1
 * the gate/up scales, plane 2 the down codes, plane 3 the down scales, expert `e` of plane `p` at
 * `base[p] + e * stride[p]`, and the weight divisors in `gate_up_divisors`/`down_divisors` -- and
 * `source` is that bank's expert count and the offset of its first expert in the pool it came from.
 * Where the bytes live is the caller's business and the Op's contract does not distinguish it: a
 * device-resident upload, a staging buffer a whole layer was copied into, and the artifact's own
 * pinned host mapping read over UVA are the same bank at three addresses. What the Op does require
 * is that the count is *the layer's*, so the two mistakes the offload path makes possible are
 * refused rather than read past:
 *
 *  - `source.experts` must equal `geometry.num_experts` exactly. A bank of 240 experts read with
 *    the pool's widest 496 bound (or with a global 512) would index `base[p] + e*stride[p]` past
 *    the row slice the layer owns, into the next layer's experts or the divisor tail.
 *  - `bank.layout` must be the layer's own layout: `nvfp4_expert_bank(source.experts, H, I, 1)`
 *    with `H = geometry.latent_size` and `I = geometry.intermediate_size`. A stride taken from
 *    another layer's bank mis-addresses every expert but the one it was taken from.
 *
 * The activation divisors must be zero: this checkpoint is W4A16, and a bank carrying calibrated
 * activation divisors is the A4 route (`moe_experts_a4`), which quantizes an activation once per
 * chunk and is not this Op.
 *
 * Numeric: identical to the whole-matrix entry point. Each expert's contribution is computed in
 * FP32 and a token's ranks are accumulated in ascending rank order with one BF16 rounding at the
 * output, so where the bank's bytes were read from cannot move a result.
 *
 * Workspace: none. The bank is already staged when this route is taken, which is the property that
 * makes it worth taking, so the Op allocates and takes nothing.
 */
void nemotron_h_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                            const ExpertWeights& bank, ExpertBankSpan source,
                            const NemotronHMoeExpertGeometry& geometry, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops
