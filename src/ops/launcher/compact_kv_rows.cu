// Implements: include/ninfer/ops/compact_kv_rows.h
// One route: the gather is a computed source index per output entry, so alignment and range
// regularity do not change what the kernel has to do.
#include "ops/launcher/compact_kv_rows.h"

#include "ops/common/math.h"
#include "ops/kernel/compact_kv_rows.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
// A grid dimension addresses at most this many rows; the kernel steps over rows beyond it.
constexpr std::int64_t kMaxGridDimY = 65535;

} // namespace

void compact_kv_rows_launch(const Tensor& v, const Tensor& k, Tensor& out, std::int32_t rotary_dim,
                            std::int32_t rotary_pairs, cudaStream_t stream) {
    const std::int32_t compact_width = rotary_dim + 2 * rotary_pairs;
    const std::int64_t rows = out.numel() / compact_width;
    if (rows <= 0) { return; }

    const int grid_x = static_cast<int>(
        std::max<std::int64_t>(1, div_up(static_cast<std::int64_t>(compact_width),
                                         static_cast<std::int64_t>(kBlock))));
    const auto grid_y = static_cast<unsigned>(std::min<std::int64_t>(rows, kMaxGridDimY));
    compact_kv_rows_kernel<<<dim3(grid_x, grid_y), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(out.data), rotary_dim, rotary_pairs, compact_width, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
