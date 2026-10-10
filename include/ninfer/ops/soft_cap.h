#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies the logit soft-cap elementwise in place. For z=x[i]:
 *
 *   ideal[i] = cap * tanh(z / cap).
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor and `cap` is positive and finite. The
 * transformation is monotone and saturates at ±cap, so a logit far outside the cap comes back
 * bounded rather than clipped. The oracle evaluates `ideal` in FP64 from the represented input; the
 * updated BF16 x is promoted and compared directly with that result, and output storage rounding
 * belongs to the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op mutates only x and uses no workspace or persistent state.
 *
 * The Op caps whatever tensor it is given; applying it to every consumer of a model's logits (the
 * sampler, grammar masks, scoring, and speculative acceptance) is the caller's responsibility.
 */
void soft_cap(Tensor& x, float cap, cudaStream_t stream);

} // namespace ninfer::ops
