// Implements: include/ninfer/ops/vision_pool.h
#include "ops/launcher/vision_pool.h"

#include "core/device.h"
#include "ops/kernel/vision_pool.cuh"

namespace ninfer::ops::detail {

void vision_pool_standardize_launch(const Tensor& x, int grid_width, int kernel, float multiplier,
                                    const Tensor& bias, const Tensor& scale, Tensor& out,
                                    cudaStream_t stream) {
    constexpr int kThreads = 128;
    vision_pool_standardize_kernel<<<static_cast<unsigned>(out.ne[1]), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(bias.data),
        static_cast<const float*>(scale.data), static_cast<__nv_bfloat16*>(out.data),
        static_cast<int>(x.ne[0]), grid_width, kernel, multiplier);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
