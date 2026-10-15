#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Plane geometry of an FP8 row-scale weight, meaningful only when validate_fp8_weight returns.

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;       // n*k E4M3 code bytes.
    std::uint64_t scale_plane_offset;     // code_plane_bytes rounded up to 256.
    std::uint64_t scale_plane_bytes;      // n BF16 row scales, two bytes per row.
    std::uint64_t required_payload_bytes; // scale_plane_offset + scale_plane_bytes.
};

// Validates an FP8_E4M3FN_ROW_BF16 row-scale weight and derives its geometry. Throws
// std::invalid_argument when the shape is not positive, the weight is not a 2D row-scale FP8 weight
// of the expected shape/strides/alignment, or its code and scale planes do not fit the payload; and
// std::overflow_error when a derived plane offset overflows. `operation` names the caller in those
// messages, so it must be a non-null, non-empty string.

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
