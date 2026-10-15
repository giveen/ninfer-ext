#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * @brief Accumulates an RMS-normalized row into the residual stream, then scales the result.
 *
 * @details `x` and `accumulator` are contiguous BF16 `[d,rows]` with the same shape and `weight` is
 * contiguous BF16 `[d]`, for a positive even `d` that is a multiple of 256 and at most 5376.
 * Dimension zero is stored fastest, so a row is one contiguous feature vector. `accumulator` is
 * updated in place; `x` and `weight` are preserved. For every row `r`, with
 * `inv(r) = rsqrt(FP32(sum_j x[j,r]^2) / d + eps)` and the gain applied to the already-normalized
 * activation -- the order `ops::rmsnorm` uses, `(x * inv) * weight` -- so
 * `norm(r,i) = FP32(x[i,r]) * inv(r) * FP32(weight[i])`:
 *
 * @f[
 *   \hat{n}_{i,r} = \mathrm{BF16}(norm(r,i)), \qquad
 *   \hat{s}_{i,r} = \mathrm{BF16}(\mathrm{FP32}(accumulator_{i,r}) + \mathrm{FP32}(\hat{n}_{i,r})),
 *   \qquad
 *   accumulator_{i,r} \leftarrow \mathrm{BF16}(\mathrm{FP32}(\hat{s}_{i,r}) \cdot scale)
 * @f]
 *
 * Those are the roundings the unfused `ops::rmsnorm` -> `ops::residual_add` pair stores, and the
 * final one is `ops::mul_scalar`'s. A row's sum of squares takes one fixed order: a 128-thread CTA
 * owns the row, each thread sums the pairs `t, t + 128, ...` in ascending order, and one block
 * reduction folds them. That is the order `ops::rmsnorm`'s fixed-width CTA route takes, so at the
 * widths the engine calls this Op with -- the Gemma 4 26B's hidden 2816 and the 31B's hidden 5376
 * -- the fused result is bit-identical to that pair, and to that pair followed by `ops::mul_scalar`
 * with the same `scale`; a `scale` of one is exact, so the two-term form needs no second entry
 * point. At any other admitted width `ops::rmsnorm` would have reached a different route with its
 * own reduction order, and the fused result is then within the criterion below rather than
 * identical.
 *
 * The oracle evaluates the formula in FP64 from the represented BF16 inputs; the FP32 reduction,
 * `rsqrtf`, and the Op's own BF16 output rounding are implementation effects covered by the Op's
 * numerical criterion.
 *
 * All three tensors must be non-null, contiguous, BF16 pair-aligned, and pairwise
 * non-overlapping. The Op updates all of `accumulator`, owns no workspace, and has no persistent
 * state effect.
 */
void residual_add_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& accumulator,
                       float scale, cudaStream_t stream);

} // namespace ninfer::ops
