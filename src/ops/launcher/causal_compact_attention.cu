// Implements: include/ninfer/ops/causal_compact_attention.h
// One route: a warp per (batch, query head, query token) row, walking the keys twice with the score
// computed cooperatively each time. No shared memory and no workspace.
#include "ops/launcher/causal_compact_attention.h"

#include "ops/kernel/causal_compact_attention.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

void causal_compact_attention_launch(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                                     const Tensor& position_k, std::int32_t rotary_dim,
                                     std::int32_t rope_angles, std::int32_t query_heads,
                                     std::int32_t kv_heads, float scale, Tensor& out,
                                     cudaStream_t stream) {
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = kv.ne[2];
    const std::int32_t batch        = q.ne[3];
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) { return; }

    const std::int64_t rows = static_cast<std::int64_t>(query_heads) * query_tokens * batch;
    causal_compact_attention_kernel<<<static_cast<unsigned>(rows), kCausalCompactThreads, 0,
                                      stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(kv.data),
        static_cast<const std::int32_t*>(position_q.data),
        static_cast<const std::int32_t*>(position_k.data), rotary_dim, rope_angles, query_heads,
        kv_heads, query_tokens, key_tokens, batch, scale,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
