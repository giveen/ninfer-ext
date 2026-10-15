#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120,  &kFp8N16384K5120,  &kFp8N34816K5120, &kFp8N5120K6144,
                         &kFp8N5120K17408,  &kFp8N248320K5120, &kFp8N8192K5376,  &kFp8N4096K5376,
                         &kFp8N16384K5376,  &kFp8N2048K5376,   &kFp8N5376K8192,  &kFp8N5376K16384,
                         &kFp8N262144K5376, &kFp8N21504K5376,  &kFp8N5376K21504};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}

// The two-term route runs one fused kernel over both terms' split-K partial planes, which the
// shape's per-T schedule only sizes for a single term. The fused route is taken only above
// kFp8A8PairMinTokens (see launch_fp8_a8_tma), so a token interval that never reaches it pays for
// neither the pair plane nor the second plane's staging.
std::size_t fp8_a8_partial_bytes(const Fp8LinearShape& shape, LinearPolicy policy,
                                 std::int32_t max_tokens) {
    const std::size_t declared =
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(max_tokens) : 0;
    if (!allows_a8_pair(policy) || max_tokens <= kFp8A8PairMinTokens) return declared;
    return declared > kFp8A8PairPartialBytes ? declared : kFp8A8PairPartialBytes;
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    const bool routed =
        (allows_a8(policy) || allows_a8_pair(policy)) && shape.uses_a8(min_tokens, max_tokens);
    if (!routed) return 0;
    return fp8_a8_workspace_capacity_bytes(max_tokens, k,
                                           fp8_a8_partial_bytes(shape, policy, max_tokens),
                                           allows_a8_pair(policy) ? 2 : 1);
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    const bool pair   = allows_a8_pair(policy);
    if (!((allows_a8(policy) || pair) && shape.uses_a8(x.ne[1], x.ne[1])))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k, fp8_a8_partial_bytes(shape, policy, x.ne[1]), pair ? 2 : 1);
    if (shape.a8 == nullptr)
        throw std::logic_error("fp8 linear: shape registered for A8 without an A8 route");
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
