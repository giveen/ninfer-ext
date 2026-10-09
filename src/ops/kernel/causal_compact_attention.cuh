#pragma once

// Implements: include/ninfer/ops/causal_compact_attention.h
// One warp owns one (batch, query head, query token) row. Each lane owns a contiguous slice of the
// head dimension and keeps one accumulator per dimension of that slice. The score is a warp-wide dot
// product, so every lane sees the same score and computes the same running maximum and sum; the
// maximum is therefore taken in a first pass over the keys and the weighted value sum in a second, so
// the accumulator set stays in registers instead of a window-sized buffer.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalCompactThreads = 32;
// A lane's slice is ceil(rotary_dim / 32) dimensions wide; the registered domain bounds rotary_dim at
// 512, which is 16, and the wrapper rejects anything larger.
inline constexpr int kCausalCompactMaxDimsPerLane = 16;

__device__ __forceinline__ int compact_lane_begin(int head_dim, int lane) {
    return (head_dim * lane) / kCausalCompactThreads;
}

__device__ __forceinline__ int compact_lane_end(int head_dim, int lane) {
    return (head_dim * (lane + 1)) / kCausalCompactThreads;
}

__device__ __forceinline__ float compact_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// A rotated dimension: the low run and the high run of the pair rotation.
__device__ __forceinline__ bool compact_rotated(int d, int rotary_dim, int rope_angles) {
    return d < rope_angles || (d >= rotary_dim / 2 && d < rotary_dim / 2 + rope_angles);
}

// Where a rotated dim sits in the stored run: the low run first, then the high run.
__device__ __forceinline__ int compact_stored_rotated(int d, int rotary_dim, int rope_angles) {
    return d < rope_angles ? d : d - rotary_dim / 2 + rope_angles;
}

__launch_bounds__(kCausalCompactThreads) __global__ void causal_compact_attention_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
    const std::int32_t* __restrict__ position_q, const std::int32_t* __restrict__ position_k,
    std::int32_t rotary_dim, std::int32_t rope_angles, std::int32_t query_heads,
    std::int32_t kv_heads, std::int32_t query_tokens, std::int32_t key_tokens, std::int32_t batch,
    float scale, __nv_bfloat16* __restrict__ out) {
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int slice_begin = compact_lane_begin(rotary_dim, lane);
    const int slice_end   = compact_lane_end(rotary_dim, lane);
    const int slice       = slice_end - slice_begin;
    const std::int32_t compact_width = rotary_dim + 2 * rope_angles;

    const std::int32_t row    = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t column = row % (query_tokens * batch);
    const std::int32_t head   = row / (query_tokens * batch);
    const std::int32_t within = column / query_tokens;
    const std::int32_t query_position = position_q[column];
    const std::int32_t kv_head = head / (query_heads / kv_heads);

    const std::int64_t q_column = static_cast<std::int64_t>(head) +
                                  static_cast<std::int64_t>(query_heads) * column;
    const __nv_bfloat16* q_row = q + q_column * rotary_dim;
    // One stored row per (kv head, key): [compact_width, Hkv, S, B] puts (d, kvh, key, b) at
    // d + compact_width * (kvh + Hkv * (key + S * b)).
    const std::int64_t kv_batch = static_cast<std::int64_t>(kv_head) +
                                  static_cast<std::int64_t>(kv_heads) *
                                      static_cast<std::int64_t>(key_tokens) * within;
    __nv_bfloat16* out_row = out + q_column * rotary_dim;

    // How many keys are visible, and their order, is the same for every lane: walk causally.
    // Pass 1: the running maximum of the scores.
    float row_max = -CUDART_INF_F;
    for (std::int32_t key = 0; key < key_tokens; ++key) {
        const std::int32_t key_position =
            position_k[key + static_cast<std::int32_t>(key_tokens) * within];
        if (key_position > query_position) { continue; }
        const auto* row_data = kv + (kv_batch + static_cast<std::int64_t>(key) * kv_heads) *
                                   compact_width;
        float dot            = 0.0F;
        for (int i = 0; i < slice; ++i) {
            const int d      = slice_begin + i;
            const int stored = compact_rotated(d, rotary_dim, rope_angles)
                                   ? rotary_dim + compact_stored_rotated(d, rotary_dim, rope_angles)
                                   : d;
            dot += __bfloat162float(q_row[d]) * __bfloat162float(row_data[stored]);
        }
        row_max = fmaxf(row_max, compact_warp_sum(dot) * scale);
    }

    // Pass 2: the softmax denominator and the value accumulation, one register per owned dimension.
    float accumulator[kCausalCompactMaxDimsPerLane];
    for (int i = 0; i < slice; ++i) { accumulator[i] = 0.0F; }
    float row_sum = 0.0F;
    for (std::int32_t key = 0; key < key_tokens; ++key) {
        const std::int32_t key_position =
            position_k[key + static_cast<std::int32_t>(key_tokens) * within];
        if (key_position > query_position) { continue; }
        const auto* row_data = kv + (kv_batch + static_cast<std::int64_t>(key) * kv_heads) *
                                   compact_width;
        float dot            = 0.0F;
        for (int i = 0; i < slice; ++i) {
            const int d      = slice_begin + i;
            const int stored = compact_rotated(d, rotary_dim, rope_angles)
                                   ? rotary_dim + compact_stored_rotated(d, rotary_dim, rope_angles)
                                   : d;
            dot += __bfloat162float(q_row[d]) * __bfloat162float(row_data[stored]);
        }
        const float weight = __expf(compact_warp_sum(dot) * scale - row_max);
        row_sum += weight;
        for (int i = 0; i < slice; ++i) {
            accumulator[i] += weight * __bfloat162float(row_data[slice_begin + i]);
        }
    }

    for (int i = 0; i < slice; ++i) {
        const float value = row_sum > 0.0F ? accumulator[i] / row_sum : 0.0F;
        out_row[slice_begin + i] = __float2bfloat16_rn(value);
    }
}

} // namespace ninfer::ops
