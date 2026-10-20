#pragma once

// Implements: include/ninfer/ops/compact_kv_rows_quantize.h
//
// One warp owns one (KV head, token) row: lane l holds dimensions l + 32 r of the fixed 640-wide
// compact row, so the row's 20 values per lane reduce to its absmax in one warp reduction and the
// codes land coalesced. The scale and codes are the D256 FP8 row codec's own arithmetic
// (`ops/kv_cache/fp8_e4m3_row_codec.cuh`), applied to a 640-value row instead of a 256-value one.

#include "ops/common/warp.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kCompactRowQuantizeWidth   = 640;
inline constexpr int kCompactRowQuantizeWarps   = 8;
inline constexpr int kCompactRowQuantizeThreads = kCompactRowQuantizeWarps * kWarpSize;
inline constexpr int kCompactRowQuantizePerLane = kCompactRowQuantizeWidth / kWarpSize;

__launch_bounds__(kCompactRowQuantizeThreads) __global__
    void compact_kv_rows_quantize_kernel(const __nv_bfloat16* __restrict__ rows,
                                         const std::int32_t* __restrict__ positions,
                                         std::uint8_t* __restrict__ codes,
                                         __half* __restrict__ scales, std::int32_t kv_heads,
                                         std::int32_t capacity, std::int32_t units) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kCompactRowQuantizeWarps + warp;
    if (unit >= units) { return; }

    const int head          = unit % kv_heads;
    const int token         = unit / kv_heads;
    const std::int32_t slot = positions[token];
    if (slot < 0 || slot >= capacity) { return; }

    const std::int64_t source =
        static_cast<std::int64_t>(head + kv_heads * token) * kCompactRowQuantizeWidth;
    float values[kCompactRowQuantizePerLane];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < kCompactRowQuantizePerLane; ++r) {
        values[r]    = __bfloat162float(rows[source + lane + 32 * r]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto quant = kv_cache_fp8_quant_params(warp_max(local_absmax));

    const std::int64_t destination =
        static_cast<std::int64_t>(head + kv_heads * slot) * kCompactRowQuantizeWidth;
#pragma unroll
    for (int r = 0; r < kCompactRowQuantizePerLane; ++r) {
        codes[destination + lane + 32 * r] =
            kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
    }
    if (lane == 0) { scales[head + kv_heads * slot] = quant.scale; }
}

} // namespace ninfer::ops::detail
