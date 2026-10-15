#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

// Coefficient rows the prepare projection's partial pass writes per token and split. The reduce
// pass re-reads the same layout; bf16_dynamic_grouped_conv_prepare_partial.cu declares its own
// copy of this count, so the two must stay equal.
inline constexpr std::size_t kPrepareCoefficientRows = 1280;
struct DynamicConvPrepareRoute {
    int rows;
    int columns;
    int split_k;
};

// `partial` is caller-owned scratch of at least split_k * tokens * kPrepareCoefficientRows floats,
// where `tokens` is the route's rows * columns. It is addressed as
// partial[(split * tokens + column) * kPrepareCoefficientRows + row] and the reduce launcher below
// re-reads exactly that layout, so a shorter allocation writes out of bounds with no diagnostic.

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream);
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream);
} // namespace ninfer::ops::detail
