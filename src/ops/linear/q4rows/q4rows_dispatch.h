#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Up to this many tokens a warp streams each pair of rows once for every token; above it the
// tensor-core tile takes over.
inline constexpr std::int32_t kQ4RowsSimtMaxTokens = 8;

// Validates a q4_g32_fp16_rows weight for the linear Op: row_group_v1, N and K multiples of 64,
// A16 only (the activations are the represented BF16 values), and enqueues the route.
void q4rows_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     cudaStream_t stream);

void q4rows_launch(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                   __nv_bfloat16* out, std::int32_t n, std::int32_t k, std::int32_t tokens,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
