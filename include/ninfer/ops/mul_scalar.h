#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Multiplies a tensor by one scalar, in place:
 *
 *   ideal[i] = x[i] * factor.
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor and `factor` is finite. The product is the FP32
 * product of the represented value and the factor, rounded once to BF16, which is what multiplying
 * a BF16 tensor by an FP32 scalar computes. The factor is deliberately a scalar here rather than a
 * one-element tensor: Gemma 4's layer scalar is stored FP32 and multiplies the whole hidden state in
 * one step, which is what the sandwich norms' `(h + post_norm) * s` needs between two other Ops.
 *
 * The oracle evaluates `ideal` in FP64 from the represented input; the updated BF16 x is promoted
 * and compared directly with that result, and output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. The Op mutates only x and uses no workspace or persistent state.
 */
void mul_scalar(Tensor& x, float factor, cudaStream_t stream);

} // namespace ninfer::ops
