#include "ninfer/ops/add_bias.h"

#include "ops/launcher/add_bias.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& t) {
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) { throw std::invalid_argument("add_bias: negative dimension"); }
        if (t.ne[i] == 0) { return 0; }
        if (n > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
            throw std::overflow_error("add_bias: tensor size overflows int64");
        }
        n *= t.ne[i];
    }
    return n;
}

// The contract excludes any shared storage, not just identical base pointers.
bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    const bool fp32 = bias.dtype == DType::FP32 && x.dtype == DType::FP32;
    if (!fp32 && (bias.dtype != DType::BF16 || x.dtype != DType::BF16)) {
        throw std::invalid_argument("add_bias: bias/x must both be BF16 or both be F32");
    }
    const std::int64_t n = checked_numel(x);
    if (bias.ne[0] != x.ne[0] || bias.ne[1] != 1 || bias.ne[2] != 1 || bias.ne[3] != 1) {
        throw std::invalid_argument("add_bias: bias must have shape [x.ne[0]]");
    }
    if (n == 0) { return; }
    if (!bias.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("add_bias: bias/x must be contiguous");
    }
    if (bias.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("add_bias: bias/x data must be non-null");
    }
    if (overlaps(bias, x)) { throw std::invalid_argument("add_bias: bias and x must not overlap"); }
    if (fp32) {
        detail::add_bias_fp32_launch(bias, x, stream);
        return;
    }
    detail::add_bias_launch(bias, x, stream);
}

} // namespace ninfer::ops
