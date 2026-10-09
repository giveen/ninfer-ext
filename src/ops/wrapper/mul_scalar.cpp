// ninfer::ops - mul_scalar wrapper: implements the public api, validates parameters, and dispatches
// to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/mul_scalar.h"

#include "ops/launcher/mul_scalar.h" // detail::mul_scalar_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("mul_scalar: dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("mul_scalar: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void mul_scalar(Tensor& x, float factor, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("mul_scalar: x must be BF16"); }
    if (!std::isfinite(factor)) {
        throw std::invalid_argument("mul_scalar: factor must be finite");
    }
    if (numel(x) == 0) { return; }
    if (!x.is_contiguous()) { throw std::invalid_argument("mul_scalar: x must be contiguous"); }
    if (x.data == nullptr) { throw std::invalid_argument("mul_scalar: x data must be non-null"); }
    if (factor == 1.0F) { return; } // exact identity: multiplying every element by one changes none

    detail::mul_scalar_launch(x, factor, stream);
}

} // namespace ninfer::ops
