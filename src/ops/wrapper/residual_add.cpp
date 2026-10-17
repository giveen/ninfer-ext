// ninfer::ops - residual_add wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/residual_add.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("residual_add: y/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("residual_add: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    const bool fp32 = y.dtype == DType::FP32 && x.dtype == DType::FP32;
    if (!fp32 && (y.dtype != DType::BF16 || x.dtype != DType::BF16)) {
        throw std::invalid_argument("residual_add: y/x must both be BF16 or both be F32");
    }
    for (int d = 0; d < 4; ++d) {
        if (y.ne[d] != x.ne[d]) {
            throw std::invalid_argument("residual_add: y/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (!y.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("residual_add: y/x must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("residual_add: y/x data must be non-null");
    }

    if (fp32) {
        detail::residual_add_fp32_launch(y, x, stream);
        return;
    }
    detail::residual_add_launch(y, x, stream); // single variant -> direct dispatch
}

} // namespace ninfer::ops
