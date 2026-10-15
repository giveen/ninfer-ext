// ninfer::ops - nemotron_h_moe_experts wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/nemotron_h_moe_experts.h"

#include "ops/launcher/nemotron_h_moe_experts.h" // detail::nemotron_h_moe_experts_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("nemotron_h_moe_experts: " + message);
}

// A contiguous 2-D tensor: the leading two extents carry the shape and the trailing two are one,
// and the layout really is row-major rather than inferred from the extents. A tensor whose shape
// matches but whose strides do not -- a transposed view, say -- would otherwise be read as if it
// were contiguous, which is a wrong answer rather than a refusal.
void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous matrix of the geometry's shape");
    }
    if (!tensor.is_contiguous()) { fail(std::string(label) + " must be contiguous"); }
}

// One bank at its logical shape. The row count is the geometry's expert count times the width of
// that bank's own matrices, so a bank that holds fewer experts than the routing can name -- or one
// bound at a different latent width -- is refused here rather than read out of range by a kernel.
detail::NemotronHMoeBankLaunch prepare_bank(const Weight& weight, std::int64_t rows,
                                            std::int64_t columns, const char* label) {
    if (weight.payload == nullptr) { fail(std::string(label) + " must be bound"); }
    if (weight.ndim != 2 || static_cast<std::int64_t>(weight.shape[0]) != rows ||
        static_cast<std::int64_t>(weight.shape[1]) != columns) {
        fail(std::string(label) + " must be [" + std::to_string(rows) + ", " +
             std::to_string(columns) + "]");
    }
    if (weight.qtype != QType::BF16) { fail(std::string(label) + " must be BF16"); }

    detail::NemotronHMoeBankLaunch bank;
    bank.qtype   = weight.qtype;
    bank.payload = weight.payload;
    bank.columns = static_cast<std::int32_t>(columns);
    return bank;
}

} // namespace

std::size_t nemotron_h_moe_experts_workspace_bytes(const NemotronHMoeExpertGeometry& geometry,
                                                   QType qtype, std::int32_t tokens) {
    if (geometry.latent_size <= 0 || geometry.num_experts <= 0 || geometry.intermediate_size <= 0 ||
        geometry.top_k <= 0) {
        fail("the geometry must be positive");
    }
    if (tokens <= 0) { fail("T must be positive"); }
    return detail::nemotron_h_moe_experts_workspace(
        qtype, geometry.latent_size, geometry.num_experts, geometry.intermediate_size,
        geometry.top_k, tokens);
}

void nemotron_h_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                            const NemotronHMoeExpertWeights& weights,
                            const NemotronHMoeExpertGeometry& geometry, Tensor& out,
                            Tensor& workspace, cudaStream_t stream) {
    const std::int32_t latent       = geometry.latent_size;
    const std::int32_t experts      = geometry.num_experts;
    const std::int32_t top_k        = geometry.top_k;
    const std::int32_t intermediate = geometry.intermediate_size;
    if (latent <= 0 || experts <= 0 || intermediate <= 0) { fail("the geometry must be positive"); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, latent, tokens, "x");
    if (out.dtype != DType::BF16) { fail("out must be BF16"); }
    require_matrix(out, latent, tokens, "out");
    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    const std::int64_t up_rows   = static_cast<std::int64_t>(intermediate) * experts;
    const std::int64_t down_rows = static_cast<std::int64_t>(latent) * experts;
    const detail::NemotronHMoeBankLaunch up =
        prepare_bank(weights.up, up_rows, latent, "nemotron_h_moe_experts up");
    const detail::NemotronHMoeBankLaunch down =
        prepare_bank(weights.down, down_rows, intermediate, "nemotron_h_moe_experts down");
    if (up.qtype != down.qtype) { fail("the two banks must share one format"); }

    const std::size_t required = nemotron_h_moe_experts_workspace_bytes(geometry, up.qtype, tokens);
    if (required > 0 &&
        (workspace.data == nullptr || !workspace.is_contiguous() || workspace.bytes() < required ||
         reinterpret_cast<std::uintptr_t>(workspace.data) % 16 != 0)) {
        fail("workspace must be a contiguous, 16-byte aligned buffer of at least " +
             std::to_string(required) + " bytes");
    }

    detail::nemotron_h_moe_experts_launch(x.data, expert_ids.data, expert_weights.data, up, down,
                                          latent, experts, intermediate, top_k, tokens,
                                          workspace.data, out.data, stream);
}

} // namespace ninfer::ops
