// ninfer::ops - gemma_moe_experts wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/gemma_moe_experts.h"

#include "ops/launcher/gemma_moe_experts.h" // detail::gemma_moe_experts_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("gemma_moe_experts: " + message);
}

// A contiguous 2-D tensor: the leading two extents carry the shape and the trailing two are one.
void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous matrix of the geometry's shape");
    }
}

void require_bank(const Weight& weight, std::int64_t rows, std::int64_t columns,
                  const char* label) {
    if (weight.qtype != QType::BF16) { fail(std::string(label) + " must be BF16"); }
    if (weight.payload == nullptr) { fail(std::string(label) + " must be bound"); }
    if (weight.ndim != 2 || static_cast<std::int64_t>(weight.shape[0]) != rows ||
        static_cast<std::int64_t>(weight.shape[1]) != columns) {
        fail(std::string(label) + " must be [" + std::to_string(rows) + ", " +
             std::to_string(columns) + "]");
    }
}

} // namespace

void gemma_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                       const GemmaMoeExpertWeights& weights, const GemmaMoeExpertGeometry& geometry,
                       Tensor& out, cudaStream_t stream) {
    const std::int32_t hidden       = geometry.hidden_size;
    const std::int32_t experts      = geometry.num_experts;
    const std::int32_t top_k        = geometry.top_k;
    const std::int32_t intermediate = geometry.intermediate_size;
    if (hidden <= 0 || experts <= 0 || intermediate <= 0) { fail("the geometry must be positive"); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, hidden, tokens, "x");
    if (out.dtype != DType::BF16) { fail("out must be BF16"); }
    require_matrix(out, hidden, tokens, "out");
    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    const std::int64_t gate_up_rows = 2LL * intermediate * experts;
    const std::int64_t down_rows    = static_cast<std::int64_t>(hidden) * experts;
    require_bank(weights.gate_up, gate_up_rows, hidden, "gate_up");
    require_bank(weights.down, down_rows, intermediate, "down");

    detail::gemma_moe_experts_launch(x.data, expert_ids.data, expert_weights.data,
                                     weights.gate_up.payload, weights.down.payload, hidden,
                                     intermediate, top_k, tokens, out.data, stream);
}

} // namespace ninfer::ops
