// Implements: include/ninfer/ops/sliding_causal_attention.h
// The tensor-core route of ops/kernel/gemma_flash_attention.cuh, one instance per head dimension.
#include "ops/launcher/sliding_causal_attention.h"

#include "ops/launcher/gemma_flash_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void sliding_causal_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& position_q, const Tensor& position_q_high,
                                     const Tensor& position_k, const Tensor& k_extra,
                                     const Tensor& v_extra, const Tensor& position_extra,
                                     std::int32_t head_dim, std::int32_t query_heads,
                                     std::int32_t kv_heads, std::uint32_t window, float scale,
                                     Tensor& out, const Tensor& workspace, cudaStream_t stream) {
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = k.ne[2];
    const std::int32_t batch        = q.ne[3];
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) { return; }
    require_flash_alignment(q.data, head_dim, "sliding_causal_attention");
    require_flash_alignment(k.data, head_dim, "sliding_causal_attention");
    require_flash_alignment(v.data, head_dim, "sliding_causal_attention");

    FlashAttentionParams params{};
    params.q            = static_cast<const __nv_bfloat16*>(q.data);
    params.k            = static_cast<const __nv_bfloat16*>(k.data);
    params.v            = static_cast<const __nv_bfloat16*>(v.data);
    params.position_q   = static_cast<const std::int32_t*>(position_q.data);
    params.position_q_high = static_cast<const std::int32_t*>(position_q_high.data);
    params.position_k   = static_cast<const std::int32_t*>(position_k.data);
    params.out          = static_cast<__nv_bfloat16*>(out.data);
    params.query_heads  = query_heads;
    params.kv_heads     = kv_heads;
    params.query_tokens = query_tokens;
    params.key_tokens   = key_tokens;
    params.window       = static_cast<std::int32_t>(window);
    params.q_width      = head_dim;
    params.k_stride     = head_dim;
    params.v_stride     = head_dim;
    params.scale        = scale;
    if (position_extra.data != nullptr && position_extra.ne[0] > 0) {
        require_flash_alignment(k_extra.data, head_dim, "sliding_causal_attention");
        require_flash_alignment(v_extra.data, head_dim, "sliding_causal_attention");
        params.extra_k        = static_cast<const __nv_bfloat16*>(k_extra.data);
        params.extra_v        = static_cast<const __nv_bfloat16*>(v_extra.data);
        params.extra_position = static_cast<const std::int32_t*>(position_extra.data);
        params.extra_tokens   = position_extra.ne[0];
    }
    switch (head_dim) {
    case 32: launch_gemma_flash_attention<32, 32, false>(params, batch, workspace, stream); break;
    case 64: launch_gemma_flash_attention<64, 64, false>(params, batch, workspace, stream); break;
    case 128: launch_gemma_flash_attention<128, 128, false>(params, batch, workspace, stream); break;
    case 256: launch_gemma_flash_attention<256, 256, false>(params, batch, workspace, stream); break;
    default:
        throw std::invalid_argument(
            "sliding_causal_attention: head_dim must be 32, 64, 128 or 256");
    }
}

std::size_t sliding_causal_attention_workspace_bytes(std::int32_t head_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t query_heads,
                                                     std::int32_t kv_heads, std::int32_t key_tokens,
                                                     std::int32_t batch) {
    return flash_workspace_bytes(head_dim, query_tokens, query_heads, kv_heads, key_tokens, batch);
}

} // namespace ninfer::ops::detail
