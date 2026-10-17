#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * @brief The Mamba-2 selective scan: the discretization, the recurrence and the per-head D skip.
 *
 * @details This is the post-convolution core of the mixer that Nemotron-3.5-Bolt's mamba blocks
 * use, and it starts from the operands the depthwise convolution has already produced (its SiLU
 * included): the value stream `x`, the grouped input `b`, the grouped output `c` and the raw time
 * step `dt`. With `H` value heads, `G` groups, key (state) extent `S = 128`, value extent
 * `value_dim`, `g(h) = h / (H / G)`, and `h` and the group contraction in FP32:
 *
 * @f[
 *   u_t        = \mathrm{softplus}(dt_{t,h} + dtbias_h), \qquad
 *   dt_{t,h}   = \max(u_t, time\_step\_min), \qquad
 *   A_h        = -\exp(Alog_h),
 * @f]
 * @f[
 *   S_h        \leftarrow \exp(dt_{t,h} A_h) \cdot S_h + dt_{t,h} \cdot \mathrm{outer}(x_{t,h},
 * B_{t,g(h)}),
 *   \qquad
 *   ideal_{t,h,v} = \left( S_h \cdot C_{t,g(h)} \right)_v + D_h \cdot x_{t,h,v}.
 * @f]
 *
 * The scan is a recurrence in `t`: the published state is the state after the last token, and the
 * state handed in is the state before the first one. Every value head owns one `S x value_dim`
 * plane; B and C are shared by the `H / G` heads of a group and are *not* repeated in memory.
 *
 * The readout is **unscaled**. The gated-linear-attention Ops in this tree lock their readout to
 * `1/sqrt(state_dim)`; Mamba-2's is the raw state contraction, so this Op takes no scale argument
 * at all and applies none — a factor of `1/sqrt(128)` here would be a silent correctness bug, not a
 * tolerance question.
 *
 * `D` is the per-head skip, and it is applied *here* rather than left to the gated grouped norm
 * that follows: it multiplies the raw (pre-`dt`) `x`, and it is added before that norm, which
 * cannot absorb it (the norm gates first and then normalizes within groups, so it is not linear in
 * `x`).
 *
 * Contiguous operands:
 *
 * | operand | dtype | shape | index of the element `(t, h, v)` or `(t, g, k)` |
 * |---|---|---|---|
 * | `x` | BF16 | `[value_dim, H, T, 1]` | `(t H + h) value_dim + v` |
 * | `b`, `c` | BF16 | `[128, G, T, 1]` | `(t G + g) 128 + k` |
 * | `dt` | BF16 | `[H, T, 1, 1]` | `t H + h` |
 * | `dt_bias`, `A_log`, `D` | FP32 | `[H]` | `h` |
 * | `state` | FP32 | `[128, value_dim, H, 1]` | `(h value_dim + v) 128 + k` |
 * | `out` | BF16 | `[value_dim, H, T, 1]` | `(t H + h) value_dim + v` |
 *
 * The two state extents are **not** equal (`128` keys against `value_dim` values), so the state's
 * axis order is part of the contract rather than a convention: the key extent is the fastest axis,
 * the value extent is the row, and the state plane per head is `128 * value_dim` elements. That is
 * the reference's own `[head, head_dim, ssm_state_size]` plane, and it is what
 * `LinearAttentionStatePool` plans for the spec `{key_head_dim = 128, value_head_dim = value_dim,
 * value_heads = H}` — the pool needs no change, but a swapped pair of extents keeps the plane size
 * at `128 * value_dim` and would otherwise corrupt the scan silently, which is why the Op pins the
 * order in its test instead of a comment.
 *
 * `value_dim` is a runtime extent (any positive value; the kernel covers it with predicated
 * 16-value tiles), the state extent is fixed at 128, and `H` must be positive and divisible by `G`.
 * `time_step_min` is finite and non-negative; only the *minimum* clamps, because the reference's
 * maximum clamp is commented out and its `time_step_limit` is unused on this path.
 *
 * The oracle evaluates the recurrence above naively in FP64 from the represented BF16 operands and
 * the exact FP32 parameters and state, with no private staging and no output rounding: the single
 * BF16 output rounding belongs to the Op's numerical criterion, not the oracle. The reference forms
 * `dt + dt_bias` and the `D * x` product in its parameters' own dtype -- its decode path casts
 * `dt_bias` to `dt`'s dtype explicitly -- whereas this Op takes the three per-head parameters in
 * FP32 and forms both in FP32, which removes one rounding each and moves the result *towards* the
 * FP64 oracle. The state, the decay, the discretized inputs and the readout stay FP32; `softplus`,
 * `exp` and the `128`-term readout reduction are FP32 with implementation-chosen association and
 * transcendental accuracy. There is no workspace and no persistent state side effect.
 *
 * Inputs, output and both states must not overlap, except that the state input and state output may
 * be exactly the same storage (the single-state overload always is).
 */

/** Distinct-state form: `state_in` is read and `state_out` receives the state after `T` tokens. */
void mamba2_scan(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                 const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                 float time_step_min, const Tensor& state_in, Tensor& state_out, Tensor& out,
                 cudaStream_t stream);

/** Single-state form: `state` is the initial state on entry and the final state on return. */
void mamba2_scan(const Tensor& x, const Tensor& b, const Tensor& c, const Tensor& dt,
                 const Tensor& dt_bias, const Tensor& a_log, const Tensor& d_skip,
                 float time_step_min, Tensor& state, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
