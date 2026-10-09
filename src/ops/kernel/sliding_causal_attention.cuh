#pragma once

// Implements: include/ninfer/ops/sliding_causal_attention.h
// One warp owns one (batch, query head, query token) row: it compacts the visible keys into shared
// memory, softmaxes them in FP32, and reduces V. No tiling and no online softmax, because the
// visible set is bounded by the window.

#include <cuda_bf16.h>
#include <math_constants.h> // CUDART_INF_F

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSlidingCausalThreads = 32;

// The row is one warp: each lane covers a contiguous slice of the head dimension.
__device__ __forceinline__ int lane_slice_begin(int head_dim, int lane) {
    return (head_dim * lane) / 32;
}

__device__ __forceinline__ int lane_slice_end(int head_dim, int lane) {
    return (head_dim * (lane + 1)) / 32;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, offset); }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__launch_bounds__(kSlidingCausalThreads) __global__ void sliding_causal_attention_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ position_q,
    const std::int32_t* __restrict__ position_k, std::int32_t head_dim, std::int32_t query_heads,
    std::int32_t kv_heads, std::int32_t query_tokens, std::int32_t key_tokens, std::int32_t batch,
    std::int32_t window, float scale, __nv_bfloat16* __restrict__ out) {
    // Shared: one window of scores, then the key index of each compacted slot.
    extern __shared__ float shared[];
    float* scores     = shared;
    std::int32_t* keys = reinterpret_cast<std::int32_t*>(shared + window);

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t column  = row % (query_tokens * batch);
    const std::int32_t head    = row / (query_tokens * batch);
    const std::int32_t token   = column % query_tokens;
    const std::int32_t within  = column / query_tokens;

    const std::int32_t group   = query_heads / kv_heads;
    const std::int32_t kv_head = head / group;
    const std::int32_t query_position = position_q[column];

    // [D, Hq, T, B] contiguous: (d, head, token, batch) sits at d + D * (head + Hq * column),
    // where column already carries the token and batch, so the same expression serves both.
    const std::int64_t q_column = static_cast<std::int64_t>(head) +
                                  static_cast<std::int64_t>(query_heads) * column;
    const __nv_bfloat16* q_row = q + q_column * head_dim;
    __nv_bfloat16* out_row     = out + q_column * head_dim;

    // Compact the visible keys. The number of visible keys is bounded by the window, so a slot per
    // window position is enough.
    std::int32_t visible = 0;
    for (std::int32_t key = 0; key < key_tokens; ++key) {
        // positions are [tokens,batch] with tokens fastest: key + S * batch.
        const std::int32_t key_position =
            position_k[key + static_cast<std::int32_t>(key_tokens) * within];
        const std::int32_t distance     = query_position - key_position;
        if (distance < 0 || distance >= window) { continue; }
        keys[visible] = key;
        ++visible;
    }

    const int begin = lane_slice_begin(head_dim, lane);
    const int end   = lane_slice_end(head_dim, lane);

    // Scores, one lane-0 write per key after a warp-wide dot-product reduction.
    for (std::int32_t slot = 0; slot < visible; ++slot) {
        const std::int32_t key = keys[slot];
        const std::int64_t k_column = static_cast<std::int64_t>(kv_head) +
                                      static_cast<std::int64_t>(kv_heads) *
                                          (key + static_cast<std::int64_t>(key_tokens) * within);
        const __nv_bfloat16* k_row = k + k_column * head_dim;
        float dot = 0.0F;
        for (int d = begin; d < end; ++d) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
        }
        dot = warp_sum(dot);
        if (lane == 0) { scores[slot] = dot * scale; }
    }
    __syncwarp();

    // Stable Softmax over the compacted scores.
    float row_max = -CUDART_INF_F;
    for (std::int32_t slot = lane; slot < visible; slot += 32) {
        row_max = fmaxf(row_max, scores[slot]);
    }
    row_max = warp_max(row_max);

    float row_sum = 0.0F;
    for (std::int32_t slot = lane; slot < visible; slot += 32) {
        const float weight = __expf(scores[slot] - row_max);
        scores[slot]       = weight;
        row_sum += weight;
    }
    row_sum = warp_sum(row_sum);
    __syncwarp();

    // Weighted value reduction over the lane's own slice of the head.
    for (int d = begin; d < end; ++d) {
        float total = 0.0F;
        for (std::int32_t slot = 0; slot < visible; ++slot) {
            const std::int32_t key = keys[slot];
            const std::int64_t v_column = static_cast<std::int64_t>(kv_head) +
                                          static_cast<std::int64_t>(kv_heads) *
                                              (key + static_cast<std::int64_t>(key_tokens) * within);
            const __nv_bfloat16* v_row = v + v_column * head_dim;
            total += scores[slot] * __bfloat162float(v_row[d]);
        }
        out_row[d] = __float2bfloat16_rn(row_sum > 0.0F ? total / row_sum : 0.0F);
    }
}

} // namespace ninfer::ops
