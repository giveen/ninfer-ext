// ninfer::ops - gemma_moe_router wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/gemma_moe_router.h"

#include "ops/launcher/gemma_moe_router.h" // detail::gemma_moe_router_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kMaxRanks = 16; // the kernel holds the chosen ranks in registers

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("gemma_moe_router: " + message);
}

// A contiguous 2-D tensor: the leading two extents carry the shape and the trailing two are one.
void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous matrix of the geometry's shape");
    }
}

// The model stores a scale or a norm weight as a plain contiguous BF16 vector.
void require_vector(const Tensor& tensor, std::int32_t extent, const char* label) {
    if (tensor.dtype != DType::BF16) { fail(std::string(label) + " must be BF16"); }
    if (tensor.data == nullptr) { fail(std::string(label) + " must be bound"); }
    if (tensor.ne[0] != extent || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous vector of the geometry's size");
    }
}

} // namespace

void gemma_moe_router(const Tensor& x, const GemmaMoeRouterWeights& weights,
                      const GemmaMoeRouterGeometry& geometry, Tensor& expert_ids,
                      Tensor& expert_weights, cudaStream_t stream) {
    const std::int32_t hidden  = geometry.hidden_size;
    const std::int32_t experts = geometry.num_experts;
    const std::int32_t top_k   = geometry.top_k;
    if (hidden <= 0 || experts <= 0) { fail("the geometry must be positive"); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }
    if (top_k > kMaxRanks) { fail("top_k must be at most " + std::to_string(kMaxRanks)); }
    if (!(geometry.rms_norm_eps >= 0.0F)) { fail("rms_norm_eps must be nonnegative"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, hidden, tokens, "x");

    // The router and its two scales are the artifact's BF16 roles; the projection is a matrix, the
    // scales are vectors. The geometry is what says how big they are, so a mismatch is an error
    // rather than a read past either end.
    if (weights.router.qtype != QType::BF16) { fail("router must be BF16"); }
    if (weights.router.payload == nullptr) { fail("the router must be bound"); }
    if (weights.router.ndim != 2 || weights.router.shape[0] != experts ||
        weights.router.shape[1] != hidden) {
        fail("router must be [num_experts, hidden]");
    }
    require_vector(weights.router_scale, hidden, "router_scale");
    require_vector(weights.per_expert_scale, experts, "per_expert_scale");

    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    detail::gemma_moe_router_launch(x.data, weights.router.payload, weights.router_scale.data,
                                    weights.per_expert_scale.data, geometry.rms_norm_eps, hidden,
                                    experts, top_k, tokens, expert_ids.data,
                                    static_cast<float*>(expert_weights.data), stream);
}

} // namespace ninfer::ops
