// ninfer::ops - scale_columns wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/scale_columns.h"

#include "ops/launcher/scale_columns.h" // detail::scale_columns_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("scale_columns: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("scale_columns: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void scale_columns(const Tensor& scale, Tensor& x, std::int32_t dims_begin, std::int32_t dims_end,
                   cudaStream_t stream) {
    if (scale.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("scale_columns: scale/x must be BF16");
    }
    if (scale.ne[0] != x.ne[0] || scale.ne[1] != 1 || scale.ne[2] != 1 || scale.ne[3] != 1) {
        throw std::invalid_argument("scale_columns: scale must be 1-D with ne[0] == x.ne[0]");
    }
    if (dims_begin < 0 || dims_end > x.ne[0] || dims_begin > dims_end) {
        throw std::invalid_argument(
            "scale_columns: the dim range must satisfy 0 <= begin <= end <= D");
    }
    if (numel_allow_zero(x, "x") == 0) { return; }
    (void)numel_allow_zero(scale, "scale");
    if (!scale.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("scale_columns: scale/x must be contiguous");
    }
    if (scale.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("scale_columns: scale/x data must be non-null");
    }
    // The contract requires disjoint storage: the Op reads scale while updating x in place.
    if (overlaps(scale, x)) {
        throw std::invalid_argument("scale_columns: scale and x must not overlap");
    }

    detail::scale_columns_launch(scale, x, dims_begin, dims_end, stream);
}

} // namespace ninfer::ops
