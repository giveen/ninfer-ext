#pragma once

// ninfer::ops::detail - private launch prototypes for the Qwen4Exp PLE family.

#include "core/tensor.h"
#include "ninfer/ops/ple.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void ple_gate_launch(const Tensor& key, const Tensor& query, const Tensor& value,
                     std::int32_t streams, Tensor& gated, cudaStream_t stream);
void ple_dilated_conv_launch(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                             std::int32_t dilation, const Tensor& states,
                             const Tensor& source_slots, Tensor& residual, cudaStream_t stream);
void ple_conv_advance_launch(const Tensor& normed, const Tensor* valid_columns,
                             std::int32_t history, Tensor& states, const Tensor& source_slots,
                             const Tensor& destination_slots, cudaStream_t stream);

} // namespace ninfer::ops::detail
