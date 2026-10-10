// ninfer::ops - causal_compact_attention wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/causal_compact_attention.h"

#include "ops/launcher/causal_compact_attention.h" // detail::causal_compact_attention_launch

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// One accumulator per owned dimension, so the head width is bounded by the lane count times that.
constexpr std::int32_t kMaximumRotaryDim = 32 * 16;

void require_bf16(const Tensor& tensor, const char* role) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("causal_compact_attention: ") + role +
                                    " must be BF16");
    }
}

void require_positions(const Tensor& tensor, const char* role, std::int32_t tokens,
                       std::int32_t batch) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != tokens || tensor.ne[1] != batch ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("causal_compact_attention: ") + role +
                                    " must be I32 [tokens,batch]");
    }
}

} // namespace

void causal_compact_attention(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                              const Tensor& position_k, AttentionHeadGeometry geometry,
                              std::int32_t rotary_dim, std::int32_t rope_angles, float scale,
                              bool indexed_keys, Tensor& out, const Tensor& workspace,
                              cudaStream_t stream) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("causal_compact_attention: invalid head geometry");
    }
    if (rotary_dim <= 0 || rotary_dim > kMaximumRotaryDim || rotary_dim != geometry.head_dim) {
        throw std::invalid_argument(
            "causal_compact_attention: rotary_dim must equal head_dim and be in [1,512]");
    }
    if (rope_angles <= 0 || rope_angles > rotary_dim / 2) {
        throw std::invalid_argument("causal_compact_attention: rope_angles must be in [1,rd/2]");
    }
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("causal_compact_attention: scale must be positive and finite");
    }
    require_bf16(q, "q");
    require_bf16(kv, "kv");
    require_bf16(out, "out");

    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = kv.ne[2];
    const std::int32_t batch        = q.ne[3];
    const std::int32_t compact_width = rotary_dim + 2 * rope_angles;
    if (q.ne[0] != rotary_dim || out.ne[0] != rotary_dim || q.ne[1] != geometry.query_heads ||
        out.ne[1] != geometry.query_heads || out.ne[2] != query_tokens || out.ne[3] != batch ||
        kv.ne[0] != compact_width || kv.ne[1] != geometry.kv_heads || kv.ne[3] != batch) {
        throw std::invalid_argument(
            "causal_compact_attention: q/out must be [rd,Hq,T,B] and kv [rd+2*angles,Hkv,S,B]");
    }
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) {
        throw std::invalid_argument("causal_compact_attention: every extent must be positive");
    }
    require_positions(position_q, "position_q", query_tokens, batch);
    require_positions(position_k, "position_k", key_tokens, batch);

    if (q.numel() == 0) { return; }
    const Tensor* tensors[] = {&q, &kv, &position_q, &position_k, &out};
    for (const Tensor* tensor : tensors) {
        if (!tensor->is_contiguous()) {
            throw std::invalid_argument(
                "causal_compact_attention: every tensor must be contiguous");
        }
        if (tensor->data == nullptr) {
            throw std::invalid_argument(
                "causal_compact_attention: every tensor data must be non-null");
        }
    }

    detail::causal_compact_attention_launch(q, kv, position_q, position_k, rotary_dim, rope_angles,
                                            geometry.query_heads, geometry.kv_heads, scale,
                                            indexed_keys, out, workspace, stream);
}

std::size_t causal_compact_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t rotary_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("causal_compact_attention: invalid head geometry");
    }
    return detail::causal_compact_attention_workspace_bytes(rotary_dim, query_tokens,
                                                            geometry.query_heads,
                                                            geometry.kv_heads, key_tokens, batch);
}

} // namespace ninfer::ops
