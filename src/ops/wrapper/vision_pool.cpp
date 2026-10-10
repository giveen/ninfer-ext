#include "ninfer/ops/vision_pool.h"

#include "ops/launcher/vision_pool.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

void vision_pool_standardize(const Tensor& x, int grid_width, int grid_height, int kernel,
                             float multiplier, const Tensor& bias, const Tensor& scale, Tensor& out,
                             cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || bias.dtype != DType::FP32 ||
        scale.dtype != DType::FP32) {
        throw std::invalid_argument(
            "vision_pool_standardize: x/out must be BF16 and bias/scale FP32");
    }
    if (kernel < 1 || grid_width < kernel || grid_height < kernel || grid_width % kernel != 0 ||
        grid_height % kernel != 0) {
        throw std::invalid_argument(
            "vision_pool_standardize: the grid must be a positive multiple of the kernel");
    }
    if (!std::isfinite(multiplier)) {
        throw std::invalid_argument("vision_pool_standardize: multiplier must be finite");
    }
    const std::int64_t d     = x.ne[0];
    const std::int64_t cells = static_cast<std::int64_t>(grid_width / kernel) * (grid_height / kernel);
    if (d <= 0 || x.ne[1] != static_cast<std::int64_t>(grid_width) * grid_height || x.ne[2] != 1 ||
        x.ne[3] != 1 || out.ne[0] != d || out.ne[1] != cells || out.ne[2] != 1 || out.ne[3] != 1 ||
        bias.numel() != d || scale.numel() != d) {
        throw std::invalid_argument(
            "vision_pool_standardize: expected x [D,W*H], bias/scale [D], out [D,cells]");
    }
    if (!x.is_contiguous() || !out.is_contiguous() || !bias.is_contiguous() ||
        !scale.is_contiguous()) {
        throw std::invalid_argument("vision_pool_standardize: tensors must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr || bias.data == nullptr || scale.data == nullptr) {
        throw std::invalid_argument("vision_pool_standardize: tensor data must be non-null");
    }
    detail::vision_pool_standardize_launch(x, grid_width, kernel, multiplier, bias, scale, out,
                                           stream);
}

} // namespace ninfer::ops
