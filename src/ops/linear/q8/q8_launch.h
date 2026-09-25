#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_gemv_n2048_k16384(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
// Few rows over a long K at up to 4 / 8 columns: several warps share each row.
void launch_q8_simt_ksplit_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_simt_ksplit_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64x32_c64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
