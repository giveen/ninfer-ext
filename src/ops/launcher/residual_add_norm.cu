// Implements: include/ninfer/ops/residual_add_norm.h
// One CTA a row, the row's own width: the admitted geometry follows from the kernel's block and
// pair count rather than a registry entry, so a wider model needs no new route.
#include "ops/launcher/residual_add_norm.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/residual_add_norm.cuh"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

void residual_add_norm_launch(const Tensor& x, const Tensor& weight, float eps, Tensor& accumulator,
                              float scale, cudaStream_t stream) {
    constexpr std::int32_t kMultiple = 2 * kResidualAddNormBlock;
    constexpr std::int32_t kMaxWidth = kResidualAddNormPairs5376 * kMultiple;
    const std::int32_t d             = x.ne[0];
    if (d <= 0 || d % kMultiple != 0 || d > kMaxWidth) {
        throw std::invalid_argument(
            "residual_add_norm: d must be a positive multiple of 256 and at most 5376");
    }
    const std::int64_t rows = accumulator.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("residual_add_norm: row count exceeds CUDA grid limit");
    }
    if (rows == 0) { return; }

    const auto* x_pairs     = reinterpret_cast<const __nv_bfloat162*>(x.data);
    const auto* w_pairs     = reinterpret_cast<const __nv_bfloat162*>(weight.data);
    auto* accumulator_pairs = reinterpret_cast<__nv_bfloat162*>(accumulator.data);
    const unsigned grid     = static_cast<unsigned>(rows);
    // The smallest staged trip count that covers the row, so an exact fit carries no predication.
    if (d <= kResidualAddNormPairs2816 * kMultiple) {
        residual_add_norm_kernel<kResidualAddNormPairs2816>
            <<<grid, kResidualAddNormBlock, 0, stream>>>(x_pairs, w_pairs, accumulator_pairs, d,
                                                         rows, eps, scale);
    } else {
        residual_add_norm_kernel<kResidualAddNormPairs5376>
            <<<grid, kResidualAddNormBlock, 0, stream>>>(x_pairs, w_pairs, accumulator_pairs, d,
                                                         rows, eps, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
