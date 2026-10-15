// ninfer::ops - gated grouped RMSNorm launcher: finite dispatch over the pairing routes.
#include "ops/launcher/gated_group_rmsnorm.h"

#include "ops/kernel/gated_group_rmsnorm.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <bool Pair>
void launch_gated_group_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate,
                                std::int32_t row_width, std::int32_t group_size, std::int64_t tasks,
                                float eps, Tensor& out, cudaStream_t stream) {
    const auto* x_bf16 = static_cast<const __nv_bfloat16*>(x.data);
    const auto* w_bf16 = static_cast<const __nv_bfloat16*>(weight.data);
    const auto* g_bf16 = static_cast<const __nv_bfloat16*>(gate.data);
    auto* out_bf16     = static_cast<__nv_bfloat16*>(out.data);

    // Eight (row, group) tasks per CTA makes the production geometry -- eight groups of 1024 over a
    // 8192-wide row -- exactly one CTA per row.
    constexpr int kBlock         = kGatedGroupRmsNormBlock;
    constexpr int kWarpsPerBlock = kBlock / kWarpSize;
    const auto blocks = static_cast<unsigned>((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gated_group_rmsnorm_warp_kernel<Pair><<<blocks, kBlock, 0, stream>>>(
        x_bf16, w_bf16, g_bf16, out_bf16, row_width, group_size, tasks, eps);
}

} // namespace

void gated_group_rmsnorm_launch(const Tensor& x, const Tensor& weight, const Tensor& gate,
                                std::int32_t group_size, float eps, Tensor& out,
                                cudaStream_t stream) {
    const std::int32_t row_width = x.ne[0];
    if (row_width <= 0) {
        throw std::invalid_argument("gated_group_rmsnorm: ne[0] must be positive");
    }
    if (group_size <= 0 || row_width % group_size != 0) {
        throw std::invalid_argument("gated_group_rmsnorm: group_size must divide ne[0]");
    }
    const std::int64_t rows  = out.numel() / row_width;
    const std::int64_t tasks = rows * (row_width / group_size);
    if (tasks > std::numeric_limits<int>::max()) {
        throw std::overflow_error("gated_group_rmsnorm: task count exceeds CUDA grid limit");
    }

    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto g_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto o_addr = reinterpret_cast<std::uintptr_t>(out.data);
    // A pair covers whole elements of one group only if the group starts on an even element, which
    // an even group size over a width it divides guarantees, and only if every buffer is BF16-pair
    // aligned. Odd group sizes therefore take the scalar route by construction.
    const bool pair_route = group_size % 2 == 0 && ((x_addr | w_addr | g_addr | o_addr) & 3U) == 0;

    if (pair_route) {
        launch_gated_group_rmsnorm<true>(x, weight, gate, row_width, group_size, tasks, eps, out,
                                         stream);
    } else {
        launch_gated_group_rmsnorm<false>(x, weight, gate, row_width, group_size, tasks, eps, out,
                                          stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
