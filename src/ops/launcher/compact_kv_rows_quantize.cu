// Implements: include/ninfer/ops/compact_kv_rows_quantize.h
#include "ops/launcher/compact_kv_rows_quantize.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/compact_kv_rows_quantize.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void compact_kv_rows_quantize_launch(const Tensor& rows, const Tensor& positions, Tensor& codes,
                                     Tensor& scales, std::int32_t kv_heads, std::int32_t capacity,
                                     cudaStream_t stream) {
    const auto tokens = positions.ne[0];
    const auto units  = static_cast<std::int64_t>(kv_heads) * tokens;
    if (units <= 0) { return; }

    const auto blocks =
        static_cast<unsigned>(div_up(units, static_cast<std::int64_t>(kCompactRowQuantizeWarps)));
    compact_kv_rows_quantize_kernel<<<blocks, kCompactRowQuantizeThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(rows.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<std::uint8_t*>(codes.data),
        static_cast<__half*>(scales.data), kv_heads, capacity, static_cast<std::int32_t>(units));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
