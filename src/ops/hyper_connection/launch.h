#pragma once

// ninfer::ops::detail - private launch prototypes for the hyper-connection family.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_offset_rmsnorm_launch(const Tensor& x, const Tensor& weight, std::int32_t groups,
                                   float eps, Tensor& out, cudaStream_t stream);
void hyper_connection_gates_launch(const Tensor& projection, std::int32_t streams, Tensor& lowrank,
                                   Tensor* inject, cudaStream_t stream);
void hyper_connection_collapse_launch(const Tensor& up, const Tensor& normalized,
                                      std::int32_t streams, Tensor& mixed, cudaStream_t stream);
void hyper_connection_combine_launch(const Tensor& y, const Tensor& inject, Tensor& residual,
                                     cudaStream_t stream);
void hyper_connection_expand_launch(const Tensor& x, std::int32_t streams, Tensor& residual,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
