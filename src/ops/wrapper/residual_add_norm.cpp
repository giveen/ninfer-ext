// ninfer::ops - residual_add_norm wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/residual_add_norm.h"

#include "ops/launcher/residual_add_norm.h" // detail::residual_add_norm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("residual_add_norm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("residual_add_norm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

// The contract excludes pairwise shared storage: accumulator is written where x/weight are read.
bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void residual_add_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& accumulator,
                       float scale, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || accumulator.dtype != DType::BF16) {
        throw std::invalid_argument("residual_add_norm: x/weight/accumulator must be BF16");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("residual_add_norm: eps must be positive and finite");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("residual_add_norm: scale must be finite");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("residual_add_norm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] != accumulator.ne[d]) {
            throw std::invalid_argument("residual_add_norm: x/accumulator shapes must match");
        }
    }
    if (numel_allow_zero(x, "x") == 0) { return; }
    if (!x.is_contiguous() || !weight.is_contiguous() || !accumulator.is_contiguous()) {
        throw std::invalid_argument("residual_add_norm: tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || accumulator.data == nullptr) {
        throw std::invalid_argument("residual_add_norm: tensors must be non-null");
    }
    if (overlaps(x, weight) || overlaps(x, accumulator) || overlaps(weight, accumulator)) {
        throw std::invalid_argument(
            "residual_add_norm: x, weight and accumulator must not overlap");
    }
    // A row is read and written as whole BF16 pairs, so the base addresses must be pair-aligned.
    // A pair is four bytes, which is what std::uint32_t's alignment states without a CUDA header.
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto a_addr = reinterpret_cast<std::uintptr_t>(accumulator.data);
    if (((x_addr | w_addr | a_addr) & (alignof(std::uint32_t) - 1)) != 0) {
        throw std::invalid_argument("residual_add_norm: tensors must be BF16 pair-aligned");
    }

    detail::residual_add_norm_launch(x, weight, eps, accumulator, scale, stream);
}

} // namespace ninfer::ops
