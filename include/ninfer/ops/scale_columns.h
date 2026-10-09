#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * Scales a dim range in place by a per-dimension vector:
 *
 *   ideal[d, ...] = x[d, ...] * scale[d]     for dims_begin <= d < dims_end
 *
 * `x` is contiguous BF16 [D, ...] and `scale` is contiguous BF16 [D]; every row of x is scaled by
 * the same vector. `0 <= dims_begin <= dims_end <= D`, and dimensions outside the range are
 * bit-exact unchanged, so a caller can scale only the dimensions a partial rotation leaves alone.
 * Each product is the FP32 product of the two represented BF16 values rounded once to BF16, which
 * is what an elementwise multiply of BF16 tensors computes. `x` and `scale` must not overlap. The
 * oracle evaluates the ideal naively in FP64 from the represented inputs; the updated BF16 x is
 * promoted and compared directly with that result, and output storage rounding belongs to the Op's
 * numerical criterion, not the oracle. The Op uses no workspace or persistent state.
 */
void scale_columns(const Tensor& scale, Tensor& x, std::int32_t dims_begin, std::int32_t dims_end,
                   cudaStream_t stream);

} // namespace ninfer::ops
