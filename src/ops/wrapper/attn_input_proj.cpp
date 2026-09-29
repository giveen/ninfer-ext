#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear/exl3/exl3_dispatch.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4_G64_FP16 || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                         const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::EXL3_MUL1) {
        // One parent holds query, key, gate and value in that row order.
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = kQRows + kKvRows + kQRows + kKvRows;
        const std::int32_t columns     = x.ne[1];
        if (columns <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (weight.n != kRows || weight.k != kHidden || workspace == nullptr) {
            throw std::invalid_argument("exl3 attn_input_proj: unsupported weight shape");
        }
        require_matrix(x, kHidden, columns, "x");
        require_matrix(q, kQRows, columns, "q");
        require_matrix(gate, kQRows, columns, "gate");
        require_matrix(k, kKvRows, columns, "k");
        require_matrix(v, kKvRows, columns, "v");
        auto scope = workspace->scope();
        const DeviceSpan whole_storage =
            workspace->alloc_bytes(static_cast<std::size_t>(kRows) * columns * 2, 16);
        Tensor whole(whole_storage.data, DType::BF16, {kRows, columns});
        detail::exl3_dispatch(x, weight, whole, policy, *workspace, stream);
        detail::exl3_copy_rows(whole, 0, q, kQRows, columns, stream);
        detail::exl3_copy_rows(whole, kQRows, k, kKvRows, columns, stream);
        detail::exl3_copy_rows(whole, kQRows + kKvRows, gate, kQRows, columns, stream);
        detail::exl3_copy_rows(whole, kQRows + kKvRows + kQRows, v, kKvRows, columns, stream);
        return;
    }
    if (weight.qtype == QType::BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(weight, kRows, kHidden, "query/key/gate/value weight");
    detail::q8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4N14336K5120::kOutputRows ||
            input_rows != detail::Nvfp4N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        if (parent_rows != detail::Fp8N14336K5120::kOutputRows ||
            input_rows != detail::Fp8N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::Q8_G32_FP16:
        if (parent_rows != 9216 || input_rows != 2048) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported Q8 profile");
        }
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        return 0;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    case QType::EXL3_MUL1:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported EXL3 profile");
        }
        // The decoded parent [14336,T] plus the rotated activation [5120,T] BF16 and, for T <= 8, the
        // multi-column GEMV's [14336,T] partial accumulator.
        return static_cast<std::size_t>(max_tokens) * (14336ULL * 2 + 5120ULL * 2) +
               14336ULL * std::min<std::size_t>(static_cast<std::size_t>(max_tokens), 8) * 4;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

bool attn_input_proj_fused_rmsnorm_nvfp4_eligible(const Weight& weight, LinearPolicy policy,
                                                  std::int32_t tokens) {
    return policy == LinearPolicy::AllowA4 && detail::nvfp4_attn_input_tma_route(tokens) &&
           weight.qtype == QType::NVFP4 && weight.layout == QuantLayout::BlockScaleK16M128x4 &&
           weight.n == 14336 && weight.k == 5120;
}

void attn_input_proj_fused_rmsnorm_nvfp4(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& projection_weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (!attn_input_proj_fused_rmsnorm_nvfp4_eligible(projection_weight, policy, tokens)) {
        throw std::invalid_argument("fused rmsnorm attn_input_proj: unsupported route");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "fused rmsnorm attn_input_proj: eps must be positive and finite");
    }
    require_matrix(residual, 5120, tokens, "residual");
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != 5120 || norm_weight.ne[1] != 1 ||
        norm_weight.ne[2] != 1 || norm_weight.ne[3] != 1 || !norm_weight.is_contiguous() ||
        !aligned_to(norm_weight.data, 16)) {
        throw std::invalid_argument("fused rmsnorm attn_input_proj: invalid norm weight");
    }
    require_matrix(q, 6144, tokens, "q");
    require_matrix(gate, 6144, tokens, "gate");
    require_matrix(k, 1024, tokens, "k");
    require_matrix(v, 1024, tokens, "v");
    detail::validate_nvfp4_weight(projection_weight, "fused rmsnorm attn_input_proj");
    detail::nvfp4_attn_input_fused_rmsnorm_launch(residual, norm_weight, eps, projection_weight, q,
                                                  gate, k, v, workspace, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t hidden      = x.ne[0];
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (hidden != 2048 && hidden != 5120) {
        throw std::invalid_argument("attn_input_proj: unsupported Q8 Q/K/V profile");
    }
    require_matrix(x, hidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(query_key_value_weight, kRows, hidden, "query/key/value weight");

    detail::q8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops
