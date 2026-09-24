#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Grouped zero-centred RMSNorm over a wide residual.
 *
 * `x` and `out` are contiguous BF16 `[G*H, T]`, `weight` is contiguous BF16 `[G*H]`. Every column t
 * and group g normalizes its own H-wide slice:
 *
 *   inv[g,t]    = 1 / sqrt((1/H) * sum_h x[g*H+h,t]^2 + eps)
 *   ideal[i,t]  = x[i,t] * inv[floor(i/H),t] * (1 + weight[i]).
 *
 * The oracle evaluates `ideal` in FP64 from represented inputs; BF16 output rounding belongs to the
 * numerical criterion. `x` and `out` must not overlap. There is no workspace.
 */
void grouped_offset_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream);

/**
 * Hyper-connection gate activations from the fused low-rank and inject projection.
 *
 * `projection` is contiguous BF16 `[R + I, T]` with I equal to `streams` when `inject` is given and
 * zero otherwise. Rows `[0,R)` produce `lowrank [R,T]` BF16 and rows `[R,R+I)` produce `inject
 * [streams,T]` FP32:
 *
 *   lowrank[r,t] = silu(projection[r,t] / streams)
 *   inject[s,t]  = 2 * sigmoid(projection[R+s,t] / streams).
 *
 * The oracle is FP64 from represented inputs.
 */
void hyper_connection_gates(const Tensor& projection, std::int32_t streams, Tensor& lowrank,
                            Tensor* inject, cudaStream_t stream);

/**
 * Collapse the normalized wide residual into one block input.
 *
 * `up` and `normalized` are contiguous BF16 `[S*H,T]`; `mixed` is contiguous BF16 `[H,T]`:
 *
 *   mixed[h,t] = (1/S) * sum_s sigmoid(up[s*H+h,t]) * normalized[s*H+h,t].
 */
void hyper_connection_collapse(const Tensor& up, const Tensor& normalized, std::int32_t streams,
                               Tensor& mixed, cudaStream_t stream);

/**
 * Inject one block output into every residual stream in place.
 *
 * `y` is contiguous BF16 `[H,T]`, `inject` contiguous FP32 `[S,T]`, `residual` contiguous BF16
 * `[S*H,T]`:
 *
 *   residual[s*H+h,t] += inject[s,t] * y[h,t].
 *
 * The oracle adds in FP64; the BF16 update rounding belongs to the numerical criterion.
 */
void hyper_connection_combine(const Tensor& y, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream);

/** Repeat `x [H,T]` into every stream of `residual [S*H,T]`. */
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

} // namespace ninfer::ops
