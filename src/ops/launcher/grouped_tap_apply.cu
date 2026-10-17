#include "ops/launcher/grouped_tap_apply.h"

#include "core/device.h"
#include "ops/kernel/grouped_tap_apply.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_tap_apply_launch(const Tensor& x, const Tensor& base, const Tensor& delta, Tensor& out,
                              std::int32_t side, std::int32_t group_width, std::int32_t block_size,
                              DeviceExecutionView execution) {
    GroupedTapParams params{};
    params.x           = static_cast<const __nv_bfloat16*>(x.data);
    params.base        = static_cast<const __nv_bfloat16*>(base.data);
    params.delta       = static_cast<const __nv_bfloat16*>(delta.data);
    params.out         = static_cast<__nv_bfloat16*>(out.data);
    params.channels    = x.ne[0];
    params.window      = x.ne[1];
    params.batch       = x.ne[2];
    params.group_width = group_width;
    params.groups      = params.channels / group_width;
    params.taps        = base.ne[1] / params.channels;
    params.block_size  = block_size;
    params.side        = side;
    params.rows        = delta.ne[0];

    const std::int64_t owners =
        static_cast<std::int64_t>(params.channels) * params.window * params.batch;
    constexpr int kThreads    = 256;
    const std::int64_t blocks = (owners + kThreads - 1) / kThreads;
    grouped_tap_apply_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, execution.stream>>>(
        params, owners);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
