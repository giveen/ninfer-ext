// Implements: include/ninfer/ops/scale_columns.h
// Finite dispatch: the 16-byte pack route when the range and storage admit it, then the BF16x2
// route, then the scalar route for odd ranges and two-byte-aligned sliced storage.
#include "ops/launcher/scale_columns.h"

#include "ops/common/math.h"
#include "ops/kernel/scale_columns.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
// A grid dimension addresses at most this many rows; the kernels step over rows beyond it.
constexpr std::int64_t kMaxGridDimY = 65535;

std::int64_t grid_rows(std::int64_t rows) {
    return std::min<std::int64_t>(rows, kMaxGridDimY);
}

int grid_dimension(std::int32_t units) {
    return static_cast<int>(
        std::max<std::int64_t>(1, div_up(static_cast<std::int64_t>(units),
                                         static_cast<std::int64_t>(kBlock))));
}

} // namespace

void scale_columns_launch(const Tensor& scale, Tensor& x, std::int32_t dims_begin,
                          std::int32_t dims_end, cudaStream_t stream) {
    const std::int32_t d      = x.ne[0];
    const std::int32_t length = dims_end - dims_begin;
    const std::int64_t rows   = x.numel() / d;
    if (length <= 0 || rows <= 0) { return; }

    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto s_addr = reinterpret_cast<std::uintptr_t>(scale.data);
    const unsigned both = static_cast<unsigned>(x_addr | s_addr);

    if ((both & (alignof(Bf16x8Pack) - 1)) == 0 && (d % 8) == 0 && (dims_begin % 8) == 0 &&
        (length % 8) == 0) {
        scale_columns_bf16x8_kernel<<<dim3(grid_dimension(length / 8),
                                           static_cast<unsigned>(grid_rows(rows))),
                                      kBlock, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(scale.data), static_cast<Bf16x8Pack*>(x.data),
            dims_begin, length / 8, d / 8, rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if ((both & (alignof(__nv_bfloat162) - 1)) == 0 && (d % 2) == 0 && (dims_begin % 2) == 0 &&
        (length % 2) == 0) {
        scale_columns_bf16x2_kernel<<<dim3(grid_dimension(length / 2),
                                           static_cast<unsigned>(grid_rows(rows))),
                                      kBlock, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(scale.data),
            reinterpret_cast<__nv_bfloat162*>(x.data), dims_begin, length / 2, d / 2, rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    scale_columns_scalar_kernel<<<dim3(grid_dimension(length),
                                       static_cast<unsigned>(grid_rows(rows))),
                                  kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scale.data), static_cast<__nv_bfloat16*>(x.data),
        dims_begin, length, d, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
