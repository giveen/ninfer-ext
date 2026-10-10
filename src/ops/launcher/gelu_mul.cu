// Implements: include/ninfer/ops/gelu_mul.h
// Finite dispatch: the paired kernel when all three operands are contiguous and pair-aligned, then
// the axis-walking kernel, which is also what carries an odd element count.
#include "ops/launcher/gelu_mul.h"

#include "ops/common/math.h"
#include "ops/kernel/gelu_mul.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

int grid_for(std::int64_t units) {
    return static_cast<int>(std::min<std::int64_t>(
        std::max<std::int64_t>(1, div_up(units, static_cast<std::int64_t>(kBlock))),
        static_cast<std::int64_t>(std::numeric_limits<int>::max())));
}

} // namespace

void gelu_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    if (n <= 0) { return; }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    const bool paired =
        gate.is_contiguous() && up.is_contiguous() &&
        ((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) == 0;

    if (paired) {
        gelu_mul_kernel<<<grid_for(n / 2), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    gelu_mul_strided_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(out.data), n, gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0],
        gate.nb[1], gate.nb[2], gate.nb[3], up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
