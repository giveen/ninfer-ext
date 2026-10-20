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

void validate_geometry(
                              std::int32_t rotary_dim, std::int32_t rope_angles,
                       AttentionHeadGeometry geometry, float scale) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("causal_compact_attention: invalid head geometry");
    }
    if (rotary_dim <= 0 || rotary_dim > kMaximumRotaryDim || rotary_dim != geometry.head_dim) {
        throw std::invalid_argument(
            "causal_compact_attention: rotary_dim must equal head_dim and be in [1,512]");
    }
    // The route is implemented for three (rotary_dim, rope_angles) pairs; reject anything else here
    // rather than letting it reach the launcher, so an undocumented pair fails as a contract error.
    const bool implemented_pair = (rotary_dim == 512 && rope_angles == 64) ||
                                  (rotary_dim == 128 && rope_angles == 16) ||
                                  (rotary_dim == 64 && rope_angles == 8);
    if (!implemented_pair) {
        throw std::invalid_argument(
            "causal_compact_attention: (rotary_dim, rope_angles) must be (512,64), (128,16) or "
            "(64,8)");
    }
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("causal_compact_attention: scale must be positive and finite");
    }
}

void launch(const Tensor& q, const Tensor* kv, const Tensor* key_codes, const Tensor* key_scales,
            const Tensor& position_q, const Tensor& position_k, AttentionHeadGeometry geometry,
            std::int32_t rotary_dim, std::int32_t rope_angles, float scale, bool indexed_keys,
            Tensor& out, const Tensor& workspace, DeviceExecutionView execution) {
    validate_geometry(rotary_dim, rope_angles, geometry, scale);
    require_bf16(q, "q");
    require_bf16(out, "out");

    const bool coded                 = key_codes != nullptr;
    const std::int32_t query_tokens = q.ne[2];
    const std::int32_t batch        = q.ne[3];
    const std::int32_t compact_width = rotary_dim + 2 * rope_angles;
    std::int32_t key_tokens          = 0;
    if (coded) {
        if (key_codes->dtype != DType::FP8_E4M3FN || key_codes->ne[0] != compact_width ||
            key_codes->ne[1] != geometry.kv_heads || key_codes->ne[3] != batch ||
            key_codes->ne[2] <= 0) {
            throw std::invalid_argument(
                "causal_compact_attention_codes: codes must be FP8_E4M3FN [width,Hkv,S,B]");
        }
        if (key_scales->dtype != DType::FP16 || key_scales->ne[0] != geometry.kv_heads ||
            key_scales->ne[1] != key_codes->ne[2] || key_scales->ne[2] != batch ||
            key_scales->ne[3] != 1) {
            throw std::invalid_argument(
                "causal_compact_attention_codes: scales must be FP16 [Hkv,S,B]");
        }
        if (key_codes->data == nullptr || key_scales->data == nullptr ||
            !key_codes->is_contiguous() || !key_scales->is_contiguous()) {
            throw std::invalid_argument(
                "causal_compact_attention_codes: codes and scales must be non-null and contiguous");
        }
        key_tokens = key_codes->ne[2];
    } else {
        require_bf16(*kv, "kv");
        key_tokens = kv->ne[2];
        if (kv->ne[0] != compact_width || kv->ne[1] != geometry.kv_heads || kv->ne[3] != batch) {
            throw std::invalid_argument(
                "causal_compact_attention: q/out must be [rd,Hq,T,B] and kv [rd+2*angles,Hkv,S,B]");
        }
        if (kv->data == nullptr || !kv->is_contiguous()) {
            throw std::invalid_argument(
                "causal_compact_attention: kv data must be non-null and contiguous");
        }
    }
    if (q.ne[0] != rotary_dim || out.ne[0] != rotary_dim || q.ne[1] != geometry.query_heads ||
        out.ne[1] != geometry.query_heads || out.ne[2] != query_tokens || out.ne[3] != batch) {
        throw std::invalid_argument(
            "causal_compact_attention: q/out must be [rd,Hq,T,B] and kv [rd+2*angles,Hkv,S,B]");
    }
    if (query_tokens <= 0 || key_tokens <= 0 || batch <= 0) {
        throw std::invalid_argument("causal_compact_attention: every extent must be positive");
    }
    require_positions(position_q, "position_q", query_tokens, batch);
    require_positions(position_k, "position_k", key_tokens, batch);
    const Tensor* tensors[] = {&q, &position_q, &position_k, &out};
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

    detail::causal_compact_attention_launch(q, kv == nullptr ? Tensor{} : *kv, position_q, position_k, key_codes, key_scales, rotary_dim, rope_angles,
                                            geometry.query_heads, geometry.kv_heads,
                                            scale, indexed_keys, out, workspace, execution);
}

} // namespace

void causal_compact_attention(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                              const Tensor& position_k, AttentionHeadGeometry geometry,
                              std::int32_t rotary_dim, std::int32_t rope_angles, float scale,
                              bool indexed_keys, Tensor& out, const Tensor& workspace,
                              DeviceExecutionView execution) {
    launch(q, &kv, nullptr, nullptr, position_q, position_k, geometry, rotary_dim, rope_angles,
           scale, indexed_keys, out, workspace, execution);
}

void causal_compact_attention_codes(const Tensor& q, const Tensor& key_codes,
                                    const Tensor& key_scales, const Tensor& position_q,
                                    const Tensor& position_k, AttentionHeadGeometry geometry,
                                    std::int32_t rotary_dim, std::int32_t rope_angles, float scale,
                                    bool indexed_keys, Tensor& out, const Tensor& workspace,
                                    DeviceExecutionView execution) {
    launch(q, nullptr, &key_codes, &key_scales, position_q, position_k, geometry, rotary_dim,
           rope_angles, scale,
                                            indexed_keys, out, workspace, execution);
}

std::size_t causal_compact_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t rotary_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument("causal_compact_attention: invalid head geometry");
    }
    if (rotary_dim <= 0 || rotary_dim > kMaximumRotaryDim || rotary_dim != geometry.head_dim) {
        throw std::invalid_argument(
            "causal_compact_attention: rotary_dim must equal head_dim and be in [1,512]");
    }
    return detail::causal_compact_attention_workspace_bytes(rotary_dim, query_tokens,
                                                            geometry.query_heads,
                                                            geometry.kv_heads, key_tokens, batch);
}

} // namespace ninfer::ops
