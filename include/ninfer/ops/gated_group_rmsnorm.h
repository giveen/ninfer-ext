#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Applies a SiLU gate and then RMS-normalizes within groups of the fastest dimension.
 *
 * @details `x`, `gate`, and `out` are same-shaped contiguous BF16 tensors with fastest dimension
 * `D = ne[0]`, and `weight` is a contiguous BF16 `[D]` gain over the full row width. With
 * `G = group_size` a positive divisor of `D`, each row is `D / G` consecutive groups, and for a
 * row `r`, a group `g`, and `d` in group `g` (`h` and the group statistic are FP32):
 *
 * @f[
 *   h_d = \mathrm{FP32}(x_d) \cdot \mathrm{SiLU}(\mathrm{FP32}(gate_d)), \qquad
 *   inv_{g,r} = \frac{1}{\sqrt{\frac{1}{G}\sum_{j \in g} h_j^2 + \epsilon}}, \qquad
 *   ideal_d = \mathrm{FP32}(weight_d) \cdot h_d \cdot inv_{g(d),r}.
 * @f]
 *
 * The gate is applied *before* the reduction, so it changes the mean square each group divides by;
 * this is not `ops::gated_rmsnorm`, which normalizes the whole row first and multiplies the result
 * by its gate afterwards, and neither is a re-parameterization of the other (the gate scales the
 * sum of squares here). The divisor is the mean square of the gated value: there is no mean
 * subtraction, so this is not a variance-normalized form.
 *
 * `group_size` is a runtime parameter: one Op serves every grouping, and `group_size == D` is the
 * single-group case (a whole-row norm with the gate applied first). `eps` is positive and finite.
 * A `group_size` that does not divide `D`, or that is not positive, throws `std::invalid_argument`.
 *
 * Inputs and output must not overlap. The oracle evaluates `ideal` naively in FP64 from the
 * represented BF16 inputs; the FP32 gate, group reduction, `rsqrtf`, and the single BF16 output
 * rounding belong to the Op's numerical criterion, not the oracle. Kernel reduction order, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect.
 */
void gated_group_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate,
                         std::int32_t group_size, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
