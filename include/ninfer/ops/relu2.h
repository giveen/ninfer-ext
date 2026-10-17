#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: relu2
 *
 * Math / indexing:
 *   The squared ReLU, elementwise and in place. For every element i:
 *
 *     x[i] = BF16( relu2(FP32(x[i])) ),        relu2(z) = max(z, 0)^2.
 *
 *   `relu2` is the square of the rectified value, not a gate, a scale or a shifted form: it is
 *   exactly zero for every non-positive `z` and never negative, and it is the square of the value
 *   itself -- `relu2(2) = 4` where GELU returns 1.95 and SiLU 1.76, and `relu2(-1) = 0` where
 *   both of those are negative. The activation is not symmetric in any softening sense: only the
 *   positive half contributes, and it contributes quadratically. Nemotron-H names this as both its
 *   `hidden_act` and its `mlp_hidden_act`, so it is the activation of the routed experts and of
 *   the shared expert alike.
 *
 * Logical shapes:
 *   x is the tensor as its caller views it: any rank, contiguous, with `numel()` elements. The
 *   elements are independent, so the Op does not see the rank and needs no shape parameter.
 *
 * Supported domain:
 *   BF16, contiguous, non-null storage. A zero-element tensor is a no-op. The Op is defined on
 *   every finite input; a non-positive input, including a zero, yields exactly zero.
 *
 * Numeric:
 *   The input is the represented BF16 value promoted to FP32, the rectification and the square are
 *   FP32, and the result is rounded to BF16 once. The oracle evaluates `relu2` in FP64 from the
 *   represented input, so the FP32 arithmetic and the single output rounding belong to the Op's
 *   numerical criterion rather than to the oracle. Nothing else rounds: there is no accumulation
 *   and no reduction.
 *
 * Effects:
 *   x is completely overwritten in place; there is no other output and no aliasing question,
 *   because the Op reads each element exactly once. No state is kept between calls.
 *
 * Execution:
 *   The caller's stream; graph-capturable (no host synchronization, no allocation).
 */
void relu2(Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
