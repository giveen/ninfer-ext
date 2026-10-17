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

void require_positions(const Tensor& tensor, const char* role, std::int32_t extents,
                       std::int32_t batch) {
    if (tensor.dtype != DType::I32) {
        throw std::invalid_argument(std::string("sliding_causal_attention: ") + role +
                                    " must be I32");
    }
    if (tensor.ne[0] != extents || tensor.ne[1] != batch || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("sliding_causal_attention: ") + role +
                                    " must be [tokens,batch]");
    }
}

} // namespace

void sliding_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& position_q, const Tensor& position_q_high,
                              const Tensor& position_k, const Tensor& k_extra,
                              const Tensor& v_extra, const Tensor& position_extra,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              Tensor& out, const Tensor& workspace, DeviceExecutionView execution,
                              const Tensor& sink) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("sliding_causal_attention: invalid head geometry");
    }
    // `window == 0` is the kernel's unbounded causal sentinel: the window test disappears and only
    // the upper bound decides, so the visible set is exactly the keys at or before `high` however
    // many keys the context holds. Every value in [1,4096] keeps the finite-window meaning.
    if (window > kMaximumWindow) {
        throw std::invalid_argument(
            "sliding_causal_attention: window must be 0 (unbounded causal) or in [1,4096]");
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
        v.ne[1] != geometry.kv_heads || out.ne[2] != query_tokens || out.ne[3] != batch) {
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
    const bool has_high = position_q_high.data != nullptr;
    if (has_high) {
        require_positions(position_q_high, "position_q_high", query_tokens, batch);
        if (!position_q_high.is_contiguous()) {
            throw std::invalid_argument(
                "sliding_causal_attention: position_q_high must be contiguous");
        }
    }
    const bool has_extra = position_extra.data != nullptr && position_extra.numel() > 0;
    if (has_extra) {
        const std::int32_t extra = position_extra.ne[0];
        require_bf16(k_extra, "k_extra");
        require_bf16(v_extra, "v_extra");
        require_positions(position_extra, "position_extra", extra, batch);
        if (k_extra.ne[0] != head_dim || v_extra.ne[0] != head_dim ||
            k_extra.ne[1] != geometry.kv_heads || v_extra.ne[1] != geometry.kv_heads ||
            k_extra.ne[2] != extra || v_extra.ne[2] != extra || k_extra.ne[3] != batch ||
            v_extra.ne[3] != batch || k_extra.data == nullptr || v_extra.data == nullptr ||
            !k_extra.is_contiguous() || !v_extra.is_contiguous() ||
            !position_extra.is_contiguous()) {
            throw std::invalid_argument(
                "sliding_causal_attention: the extra keys must be contiguous [D,Hkv,E,B] with "
                "[E,B] positions");
        }
    }

    const Tensor* tensors[] = {&q, &k, &v, &position_q, &position_k, &out};
    for (const Tensor* tensor : tensors) {
        if (!tensor->is_contiguous()) {
            throw std::invalid_argument(
                "sliding_causal_attention: every tensor must be contiguous");
        }
        if (tensor->data == nullptr) {
            throw std::invalid_argument(
                "sliding_causal_attention: every tensor data must be non-null");
        }
    }

    const Tensor none;
    if (sink.data != nullptr) {
        require_bf16(sink, "sink");
        if (sink.ne[0] != geometry.query_heads || sink.ne[1] != 1 || sink.ne[2] != 1 ||
            sink.ne[3] != 1 || !sink.is_contiguous()) {
            throw std::invalid_argument(
                "sliding_causal_attention: sink must be contiguous BF16 [Hq,1,1,1]");
        }
    }
    detail::sliding_causal_attention_launch(
        q, k, v, position_q, has_high ? position_q_high : none, position_k,
        has_extra ? k_extra : none, has_extra ? v_extra : none, has_extra ? position_extra : none,
        head_dim, geometry.query_heads, geometry.kv_heads, window, scale, out, workspace, execution,
        sink);
}

std::size_t sliding_causal_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("sliding_causal_attention: invalid head geometry");
    }
    return detail::sliding_causal_attention_workspace_bytes(geometry.head_dim, query_tokens,
                                                            geometry.query_heads, geometry.kv_heads,
                                                            key_tokens, batch);
}

} // namespace ninfer::ops
