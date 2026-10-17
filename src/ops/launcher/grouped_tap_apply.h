#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Validated by the wrapper, which is the only caller: the tensors are already known to be
// contiguous BF16 of the declared extents and the geometry is known to be representable.
void grouped_tap_apply_launch(const Tensor& x, const Tensor& base, const Tensor& delta, Tensor& out,
                              std::int32_t side, std::int32_t group_width, std::int32_t block_size,
                              DeviceExecutionView execution);

} // namespace ninfer::ops::detail
