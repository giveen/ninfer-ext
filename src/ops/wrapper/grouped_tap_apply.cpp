#include "ninfer/ops/grouped_tap_apply.h"

#include "ops/launcher/grouped_tap_apply.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("grouped_tap_apply: ") + label +
                                    " must be contiguous BF16 with data");
    }
}

} // namespace

void grouped_tap_apply(const Tensor& x, const Tensor& base, const Tensor& delta, Tensor& out,
                       std::int32_t side, std::int32_t group_width, std::int32_t block_size,
                       DeviceExecutionView execution) {
    if (side != 0 && side != 1) {
        throw std::invalid_argument("grouped_tap_apply: side must be 0 (prepare) or 1 (finish)");
    }
    require_bf16(x, "x");
    require_bf16(base, "base");
    require_bf16(delta, "delta");
    require_bf16(out, "out");
    const std::int32_t channels = x.ne[0];
    const std::int32_t window   = x.ne[1];
    const std::int32_t batch    = x.ne[2];
    if (channels <= 0 || window <= 0 || batch <= 0) {
        throw std::invalid_argument("grouped_tap_apply: x must have positive extents");
    }
    if (group_width <= 0 || channels % group_width != 0) {
        throw std::invalid_argument("grouped_tap_apply: group width must divide the channels");
    }
    if (block_size <= 0) {
        throw std::invalid_argument("grouped_tap_apply: block size must be positive");
    }
    if (out.ne[0] != channels || out.ne[1] != window || out.ne[2] != batch) {
        throw std::invalid_argument("grouped_tap_apply: out shape differs from x");
    }
    // `base` is [2, taps*channels] and `delta` is [2*taps*groups, window, batch], so the tap count
    // is derived from `base` and both sides must agree on it.
    if (base.ne[0] != 2 || base.ne[1] % channels != 0) {
        throw std::invalid_argument("grouped_tap_apply: base must be [2, taps*channels]");
    }
    const std::int32_t taps   = base.ne[1] / channels;
    const std::int32_t groups = channels / group_width;
    if (delta.ne[0] != 2 * taps * groups || delta.ne[1] != window || delta.ne[2] != batch) {
        throw std::invalid_argument("grouped_tap_apply: delta must be [2*taps*groups, W, B]");
    }
    detail::grouped_tap_apply_launch(x, base, delta, out, side, group_width, block_size, execution);
}

} // namespace ninfer::ops
