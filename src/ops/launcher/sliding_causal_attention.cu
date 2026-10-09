// Implements: include/ninfer/ops/sliding_causal_attention.h
// One route: a warp per (batch, query head, query token) row, with the visible window in shared
// memory. The launcher only sizes that window and the grid.
#include "ops/launcher/sliding_causal_attention.h"

#include "ops/kernel/sliding_causal_attention.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void sliding_causal_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& position_q, const Tensor& position_k,
                                     std::int32_t head_dim, std::int32_t query_heads,
                                     std::int32_t kv_heads, std::uint32_t window, float scale,
                                     Tensor& out, cudaStream_t stream) {
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = k.ne[2];
    const std::int32_t batch        = q.ne[3];
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) { return; }

    // The visible keys are compacted into one window of scores plus their key indices.
    const std::size_t shared_bytes =
        static_cast<std::size_t>(window) * (sizeof(float) + sizeof(std::int32_t));
    const std::int64_t rows =
        static_cast<std::int64_t>(query_heads) * query_tokens * batch;

    sliding_causal_attention_kernel<<<static_cast<unsigned>(rows), kSlidingCausalThreads,
                                      shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(position_q.data),
        static_cast<const std::int32_t*>(position_k.data), head_dim, query_heads, kv_heads,
        query_tokens, key_tokens, batch, static_cast<std::int32_t>(window), scale,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
