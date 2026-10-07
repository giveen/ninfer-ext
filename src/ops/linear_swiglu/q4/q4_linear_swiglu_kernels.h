#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "core/arena.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
// Wide folded tiles for the leading column blocks plus a narrow route for a small remainder.
void q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(const Tensor& x, const Weight& w,
                                                               Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);
// W4A8: Q4 weights with per-token int8 activations (materialized in caller workspace).
void q4_linear_swiglu_a8_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
