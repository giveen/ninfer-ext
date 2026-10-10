// ninfer::ops - sliding_causal_attention wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/sliding_causal_attention.h"

#include "ops/launcher/sliding_causal_attention.h" // detail::sliding_causal_attention_launch

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::uint32_t kMaximumWindow = 4096;

void require_bf16(const Tensor& tensor, const char* role) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("sliding_causal_attention: ") + role +
                                    " must be BF16");
    }
}

void require_positions(const Tensor& tensor, const char* role, std::int32_t extents, std::int32_t batch) {
    if (tensor.dtype != DType::I32) {
        throw std::invalid_argument(std::string("sliding_causal_attention: ") + role +
                                    " must be I32");
    }
    if (tensor.ne[0] != extents || tensor.ne[1] != batch || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("sliding_causal_attention: ") + role +
                                    " must be [tokens,batch]");
    }
}

} // namespace

void sliding_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& position_q, const Tensor& position_k,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              Tensor& out, const Tensor& workspace, cudaStream_t stream) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("sliding_causal_attention: invalid head geometry");
    }
    if (window == 0 || window > kMaximumWindow) {
        throw std::invalid_argument("sliding_causal_attention: window must be in [1,4096]");
    }
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("sliding_causal_attention: scale must be positive and finite");
    }
    require_bf16(q, "q");
    require_bf16(k, "k");
    require_bf16(v, "v");
    require_bf16(out, "out");

    const std::int32_t head_dim     = geometry.head_dim;
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t key_tokens   = k.ne[2];
    const std::int32_t batch        = q.ne[3];
    if (q.ne[0] != head_dim || out.ne[0] != head_dim || k.ne[0] != head_dim ||
        v.ne[0] != head_dim || q.ne[1] != geometry.query_heads ||
        out.ne[1] != geometry.query_heads || k.ne[1] != geometry.kv_heads ||
        v.ne[1] != geometry.kv_heads) {
        throw std::invalid_argument(
            "sliding_causal_attention: q/k/v/out must be [head_dim,heads,tokens,batch] per the "
            "geometry");
    }
    if (key_tokens <= 0 || query_tokens <= 0 || batch <= 0 || q.ne[3] != k.ne[3] ||
        v.ne[3] != k.ne[3] || v.ne[2] != key_tokens) {
        throw std::invalid_argument("sliding_causal_attention: q/k/v/out extents must agree");
    }
    require_positions(position_q, "position_q", query_tokens, batch);
    require_positions(position_k, "position_k", key_tokens, batch);

    const bool empty = q.numel() == 0;
    if (empty) { return; }
    const Tensor* tensors[] = {&q, &k, &v, &position_q, &position_k, &out};
    for (const Tensor* tensor : tensors) {
        if (!tensor->is_contiguous()) {
            throw std::invalid_argument("sliding_causal_attention: every tensor must be contiguous");
        }
        if (tensor->data == nullptr) {
            throw std::invalid_argument("sliding_causal_attention: every tensor data must be non-null");
        }
    }

    detail::sliding_causal_attention_launch(q, k, v, position_q, position_k, head_dim,
                                            geometry.query_heads, geometry.kv_heads, window, scale,
                                            out, workspace, stream);
}

std::size_t sliding_causal_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("sliding_causal_attention: invalid head geometry");
    }
    return detail::sliding_causal_attention_workspace_bytes(geometry.head_dim, query_tokens,
                                                            geometry.query_heads,
                                                            geometry.kv_heads, key_tokens, batch);
}

} // namespace ninfer::ops
