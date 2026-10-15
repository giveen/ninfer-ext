// ninfer::ops - nemotron_h_moe_router wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/nemotron_h_moe_router.h"

#include "ops/launcher/nemotron_h_moe_router.h" // detail::nemotron_h_moe_router_launch

#include <cmath>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// The family config carries num_local_experts = 512 as a bound on a bank; the released checkpoint
// banks 240 to 496 of them. The selection holds one slot a lane per 32 experts, so this bound is
// also what sizes its register arrays.
constexpr std::int32_t kMaxExperts = 512;
constexpr std::int32_t kMaxRanks   = 16;
constexpr std::int32_t kMaxGroups  = 32;

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("nemotron_h_moe_router: " + message);
}

// A contiguous 2-D tensor: the leading two extents carry the shape and the trailing two are one.
// `layout` states the payload's element order at the point of refusal, because the declared shape
// of this Op's outputs is the reference's own and reads as the transpose of what the Op writes: a
// caller told the extents alone would still not know which flat index to use.
void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label, const char* layout) {
    if (tensor.data == nullptr) { fail(std::string(label) + " must be bound"); }
    if (!tensor.is_contiguous()) { fail(std::string(label) + " must be contiguous"); }
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous [" + std::to_string(rows) + "," +
             std::to_string(columns) + "] matrix (" + layout + ")");
    }
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

} // namespace

std::size_t nemotron_h_moe_router_workspace_bytes(const NemotronHMoeRouterGeometry& geometry,
                                                  std::int32_t tokens) {
    if (geometry.num_experts <= 0 || tokens <= 0) {
        fail("the workspace query needs a positive expert and token count");
    }
    return static_cast<std::size_t>(geometry.num_experts) * tokens * sizeof(float);
}

void nemotron_h_moe_router(const Tensor& x, const NemotronHMoeRouterWeights& weights,
                           const NemotronHMoeRouterGeometry& geometry, Tensor& expert_ids,
                           Tensor& expert_weights, Tensor& workspace, cudaStream_t stream) {
    const std::int32_t hidden      = geometry.hidden_size;
    const std::int32_t experts     = geometry.num_experts;
    const std::int32_t top_k       = geometry.top_k;
    const std::int32_t groups      = geometry.groups;
    const std::int32_t groups_used = geometry.groups_used;
    if (hidden <= 0 || experts <= 0) { fail("the geometry must be positive"); }
    if (hidden % 8 != 0) { fail("hidden must be a multiple of 8"); }
    if (experts > kMaxExperts) {
        fail("num_experts must be at most " + std::to_string(kMaxExperts));
    }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }
    if (top_k > kMaxRanks) { fail("top_k must be at most " + std::to_string(kMaxRanks)); }
    if (groups <= 0 || groups > kMaxGroups || experts % groups != 0) {
        fail("groups must divide num_experts and lie in [1, " + std::to_string(kMaxGroups) + "]");
    }
    if (groups_used <= 0 || groups_used > groups) { fail("groups_used must lie in [1, groups]"); }
    // The selected groups are the only experts the masked top-k can reach, so a top_k they cannot
    // supply has no defined answer -- the reference would return -inf-ranked entries with real
    // gathered weights, which is a silently wrong model rather than a smaller one.
    if (top_k > groups_used * (experts / groups)) {
        fail("top_k must be at most groups_used * (num_experts / groups)");
    }
    if (!std::isfinite(geometry.scaling_factor)) { fail("scaling_factor must be finite"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, hidden, tokens, "x", "the hidden axis innermost, x[t*hidden + h]");
    if (!aligned16(x.data)) { fail("x must be 16-byte aligned"); }

    // The router weight is the artifact's BF16 role, exactly as the checkpoint stores it; the
    // correction bias is FP32 because that is what the checkpoint stores and what the reference
    // adds in. The geometry is what says how big they are, so a mismatch is an error rather than a
    // read past either end.
    if (weights.router.qtype != QType::BF16) { fail("router must be BF16"); }
    if (weights.router.payload == nullptr) { fail("the router must be bound"); }
    if (weights.router.ndim != 2 || weights.router.shape[0] != experts ||
        weights.router.shape[1] != hidden) {
        fail("router must be [num_experts, hidden]");
    }
    if (!aligned16(weights.router.payload)) { fail("router must be 16-byte aligned"); }
    if (weights.correction_bias.dtype != DType::FP32) { fail("correction_bias must be FP32"); }
    if (weights.correction_bias.data == nullptr) { fail("correction_bias must be bound"); }
    if (!weights.correction_bias.is_contiguous() || weights.correction_bias.ne[0] != experts ||
        weights.correction_bias.ne[1] != 1 || weights.correction_bias.ne[2] != 1 ||
        weights.correction_bias.ne[3] != 1) {
        fail("correction_bias must be a contiguous vector of num_experts");
    }

    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids",
                   "the payload is rank*tokens + token, the token axis innermost");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights",
                   "the payload is rank*tokens + token, the token axis innermost");

    const std::size_t required = nemotron_h_moe_router_workspace_bytes(geometry, tokens);
    if (workspace.data == nullptr || !workspace.is_contiguous() || workspace.bytes() < required ||
        !aligned16(workspace.data)) {
        fail("the workspace must be contiguous, 16-byte aligned and at least " +
             std::to_string(required) + " bytes");
    }

    detail::nemotron_h_moe_router_launch(
        x.data, weights.router.payload, weights.correction_bias.data, hidden, experts, top_k,
        groups, groups_used, geometry.normalize_top_k, geometry.scaling_factor, tokens,
        static_cast<float*>(workspace.data), expert_ids.data,
        static_cast<float*>(expert_weights.data), stream);
}

} // namespace ninfer::ops
