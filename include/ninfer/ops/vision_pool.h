#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gemma 4's vision pooling: averages each k x k cell of an image's patch grid, scales the mean, and
 * standardizes it per feature.
 *
 *   patch p      = row * W + column, for a W x H patch grid with W and H multiples of k
 *   cell c       = column / k + (W / k) * (row / k), N = (W / k) * (H / k) cells
 *   ideal[d,c]   = (multiplier * mean_{p in c} x[d,p] - bias[d]) * scale[d]
 *
 * x is contiguous BF16 [D,W*H], bias and scale are contiguous FP32 [D], and out is contiguous BF16
 * [D,N]. The oracle evaluates `ideal` naively in FP64 from the represented inputs; the BF16 output
 * is promoted for comparison and its storage rounding belongs to the Op criterion. The kernel keeps
 * the mean, the scaling and the standardization in FP32 and rounds once (the reference
 * implementation also rounds the mean to BF16 before scaling; that intermediate is not an
 * observable boundary). Inputs are unchanged, out is completely overwritten and must not overlap
 * them, and the Op uses no workspace.
 */
void vision_pool_standardize(const Tensor& x, int grid_width, int grid_height, int kernel,
                             float multiplier, const Tensor& bias, const Tensor& scale, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops
