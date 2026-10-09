// ninfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

// `weight` is null for the weightless form, whose gain is exactly one.
void rmsnorm_impl(const Tensor& x, const Tensor* weight, float eps, bool unit_offset,
                  const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                  cudaStream_t stream, bool sigmoid_gate = false) {
    if (x.dtype != DType::BF16 || (weight != nullptr && weight->dtype != DType::BF16) ||
        out.dtype != DType::BF16 || (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    require_same_shape(x, out, "out");
    if (weight != nullptr) {
        (void)numel_allow_zero(*weight, "weight");
        if (weight->ne[0] != x.ne[0] || weight->ne[1] != 1 || weight->ne[2] != 1 ||
            weight->ne[3] != 1) {
            throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
        }
    } else if (unit_offset) {
        throw std::invalid_argument("rmsnorm: the weightless form has no offset to apply");
    }
    if (weight == nullptr && z != nullptr) {
        throw std::invalid_argument("rmsnorm: the weightless form has no gate");
    }
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "z");
        require_same_shape(x, *z, "z");
    }
    if (n == 0) { return; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || (weight != nullptr && !weight->is_contiguous()) ||
        !out.is_contiguous() || (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr || (weight != nullptr && weight->data == nullptr) ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    // The weightless epilogue never reads the weight, so the launch passes an unread placeholder
    // instead of a caller-visible one.
    const Tensor placeholder(nullptr, DType::BF16, {x.ne[0]});
    detail::rmsnorm_launch(x, weight == nullptr ? placeholder : *weight, eps, unit_offset, z, out,
                           multiprocessor_count, stream, sigmoid_gate, weight == nullptr);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, &weight, eps, unit_offset, nullptr, out, 0, stream);
}

void rmsnorm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    rmsnorm_impl(x, nullptr, eps, false, nullptr, out, 0, stream);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_rmsnorm: positive multiprocessor count required");
    }
    rmsnorm_impl(x, &weight, eps, false, &z, out, execution.multiprocessor_count,
                 execution.stream);
}

void gated_rmsnorm_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                           Tensor& out, DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_rmsnorm_sigmoid: positive multiprocessor count required");
    }
    rmsnorm_impl(x, &weight, eps, false, &z, out, execution.multiprocessor_count,
                 execution.stream, true);
}

} // namespace ninfer::ops
