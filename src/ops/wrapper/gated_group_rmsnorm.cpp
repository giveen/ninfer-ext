// ninfer::ops - gated_group_rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/gated_group_rmsnorm.h"

#include "ops/launcher/gated_group_rmsnorm.h" // detail::gated_group_rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& t, const char* label) {
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) {
            throw std::invalid_argument(std::string("gated_group_rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[i] == 0) { return 0; }
        if (n > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
            throw std::overflow_error("gated_group_rmsnorm: tensor size overflows int64");
        }
        n *= t.ne[i];
    }
    return n;
}

bool vector_of(const Tensor& t, std::int32_t d) {
    return t.ne[0] == d && t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

// The contract excludes shared storage; the kernel reads x/gate/weight while writing out.
bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void gated_group_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate,
                         std::int32_t group_size, float eps, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || gate.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("gated_group_rmsnorm: x/weight/gate/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("gated_group_rmsnorm: eps must be positive and finite");
    }
    if (group_size <= 0) {
        throw std::invalid_argument("gated_group_rmsnorm: group_size must be positive");
    }
    const std::int64_t n = checked_numel(x, "x");
    (void)checked_numel(weight, "weight");
    (void)checked_numel(gate, "gate");
    (void)checked_numel(out, "out");
    for (int i = 0; i < 4; ++i) {
        if (out.ne[i] != x.ne[i] || gate.ne[i] != x.ne[i]) {
            throw std::invalid_argument("gated_group_rmsnorm: x/gate/out shapes must match");
        }
    }
    if (!vector_of(weight, x.ne[0])) {
        throw std::invalid_argument("gated_group_rmsnorm: weight must have shape [x.ne[0]]");
    }
    if (x.ne[0] % group_size != 0) {
        throw std::invalid_argument("gated_group_rmsnorm: group_size must divide x.ne[0]");
    }
    if (n == 0) { return; }
    if (!x.is_contiguous() || !weight.is_contiguous() || !gate.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("gated_group_rmsnorm: tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || gate.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("gated_group_rmsnorm: tensor data must be non-null");
    }
    if (overlaps(x, out) || overlaps(gate, out) || overlaps(weight, out) || overlaps(x, gate) ||
        overlaps(weight, x) || overlaps(weight, gate)) {
        throw std::invalid_argument("gated_group_rmsnorm: tensors must not overlap");
    }
    detail::gated_group_rmsnorm_launch(x, weight, gate, group_size, eps, out, stream);
}

} // namespace ninfer::ops
