// ninfer::ops - gemma_moe_router wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/gemma_moe_router.h"

#include "ops/launcher/gemma_moe_router.h" // detail::gemma_moe_router_launch

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kMaxRanks   = 16;  // the selection holds the chosen ranks per lane
constexpr std::int32_t kMaxExperts = 256; // eight logits a lane in the selection warp

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

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

} // namespace

std::size_t gemma_moe_router_workspace_bytes(const GemmaMoeRouterGeometry& geometry,
                                             std::int32_t tokens) {
    if (geometry.num_experts <= 0 || tokens <= 0) {
        fail("the workspace query needs a positive expert and token count");
    }
    return static_cast<std::size_t>(geometry.num_experts) * tokens * sizeof(float);
}

void gemma_moe_router(const Tensor& x, const GemmaMoeRouterWeights& weights,
                      const GemmaMoeRouterGeometry& geometry, Tensor& expert_ids,
                      Tensor& expert_weights, Tensor& workspace, cudaStream_t stream) {
    const std::int32_t hidden  = geometry.hidden_size;
    const std::int32_t experts = geometry.num_experts;
    const std::int32_t top_k   = geometry.top_k;
    if (hidden <= 0 || experts <= 0) { fail("the geometry must be positive"); }
    if (hidden % 8 != 0) { fail("hidden must be a multiple of 8"); }
    if (experts > kMaxExperts) { fail("num_experts must be at most " + std::to_string(kMaxExperts)); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }
    if (top_k > kMaxRanks) { fail("top_k must be at most " + std::to_string(kMaxRanks)); }
    if (!(geometry.rms_norm_eps >= 0.0F)) { fail("rms_norm_eps must be nonnegative"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, hidden, tokens, "x");
    if (!aligned16(x.data)) { fail("x must be 16-byte aligned"); }

    // The router and its two scales are the artifact's BF16 roles; the projection is a matrix, the
    // scales are vectors. The geometry is what says how big they are, so a mismatch is an error
    // rather than a read past either end.
    if (weights.router.qtype != QType::BF16) { fail("router must be BF16"); }
    if (weights.router.payload == nullptr) { fail("the router must be bound"); }
    if (weights.router.ndim != 2 || weights.router.shape[0] != experts ||
        weights.router.shape[1] != hidden) {
        fail("router must be [num_experts, hidden]");
    }
    if (!aligned16(weights.router.payload)) { fail("router must be 16-byte aligned"); }
    require_vector(weights.router_scale, hidden, "router_scale");
    require_vector(weights.per_expert_scale, experts, "per_expert_scale");

    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    const std::size_t required = gemma_moe_router_workspace_bytes(geometry, tokens);
    if (workspace.data == nullptr || !workspace.is_contiguous() || workspace.bytes() < required ||
        !aligned16(workspace.data)) {
        fail("the workspace must be a contiguous, 16-byte aligned span of at least " +
             std::to_string(required) + " bytes");
    }

    detail::gemma_moe_router_launch(x.data, weights.router.payload, weights.router_scale.data,
                                    weights.per_expert_scale.data, geometry.rms_norm_eps, hidden,
                                    experts, top_k, tokens, static_cast<float*>(workspace.data),
                                    expert_ids.data, static_cast<float*>(expert_weights.data),
                                    stream);
}

} // namespace ninfer::ops
