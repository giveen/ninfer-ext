// Implements: include/ninfer/ops/causal_compact_attention.h
// The tensor-core route of ops/kernel/gemma_flash_attention.cuh over compact rows, one instance per
// (rotary_dim, rope_angles) the Op registers.
#include "ops/launcher/causal_compact_attention.h"

#include "ops/launcher/gemma_flash_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void causal_compact_attention_launch(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                                     const Tensor& position_k, std::int32_t rotary_dim,
                                     std::int32_t rope_angles, std::int32_t query_heads,
                                     std::int32_t kv_heads, float scale, bool indexed_keys,
                                     Tensor& out, const Tensor& workspace, cudaStream_t stream) {
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = kv.ne[2];
    const std::int32_t batch        = q.ne[3];
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) { return; }
    const std::int32_t width = rotary_dim + 2 * rope_angles;
    require_flash_alignment(kv.data, width, "causal_compact_attention");

    FlashAttentionParams params{};
    params.q            = static_cast<const __nv_bfloat16*>(q.data);
    params.k            = static_cast<const __nv_bfloat16*>(kv.data);
    params.v            = static_cast<const __nv_bfloat16*>(kv.data);
    params.position_q   = static_cast<const std::int32_t*>(position_q.data);
    params.position_k   = static_cast<const std::int32_t*>(position_k.data);
    params.out          = static_cast<__nv_bfloat16*>(out.data);
    params.query_heads  = query_heads;
    params.kv_heads     = kv_heads;
    params.query_tokens = query_tokens;
    params.key_tokens   = key_tokens;
    params.window       = 0;
    params.q_width      = rotary_dim;
    params.rope_angles  = rope_angles;
    params.k_stride     = width;
    params.v_stride     = width;
    params.scale        = scale;
    params.indexed_keys = indexed_keys;
    if (rotary_dim == 512 && rope_angles == 64) {
        launch_gemma_flash_attention<640, 512, true>(params, batch, workspace, stream);
    } else if (rotary_dim == 128 && rope_angles == 16) {
        launch_gemma_flash_attention<160, 128, true>(params, batch, workspace, stream);
    } else if (rotary_dim == 64 && rope_angles == 8) {
        launch_gemma_flash_attention<80, 64, true>(params, batch, workspace, stream);
    } else {
        throw std::invalid_argument(
            "causal_compact_attention: (rotary_dim, rope_angles) must be (512,64), (128,16) or "
            "(64,8)");
    }
}

std::size_t causal_compact_attention_workspace_bytes(std::int32_t rotary_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t query_heads,
                                                     std::int32_t kv_heads, std::int32_t key_tokens,
                                                     std::int32_t batch) {
    return flash_workspace_bytes(rotary_dim, query_tokens, query_heads, kv_heads, key_tokens, batch);
}

} // namespace ninfer::ops::detail
