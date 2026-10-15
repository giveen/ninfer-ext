#include "core/checked_math.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation,
                                          bool stacked) {
    if (weight.n <= 0 || weight.k <= 0 || (weight.n % 128) != 0 || (weight.k % 64) != 0) {
        throw std::invalid_argument(std::string(operation) + ": NVFP4 requires N%128=0 and K%64=0");
    }

    const std::uint64_t elements = ninfer:: checked_mul(static_cast<std::uint64_t>(weight.n),
                                               static_cast<std::uint64_t>(weight.k), operation);
    Nvfp4WeightGeometry geometry{};
    geometry.code_plane_bytes           = elements / 2;
    geometry.scale_plane_offset         =
        ninfer:: align_up(geometry.code_plane_bytes, std::uint64_t{ 256}, operation);
    geometry.scale_plane_bytes          = elements / 16;
    const std::int32_t rows_per_divisor = weight.weight_divisor_rows;
    const bool divisor_shape = rows_per_divisor > 0 && (weight.n % rows_per_divisor) == 0 &&
                               (rows_per_divisor % 128) == 0 &&
                               (stacked || rows_per_divisor == weight.n);
    geometry.required_payload_bytes = ninfer:: checked_add(
        ninfer::
        checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation),
        ninfer::
        checked_mul(divisor_shape ? static_cast<std::uint64_t>(weight.n / rows_per_divisor) : 1,
                            static_cast<std::uint64_t>(
                    sizeof(float)), operation),
        operation);

    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.scale_dtype != DType::FP8_E4M3FN || weight.group_size != 16 || weight.group != 16 ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.qdata == nullptr || weight.scales == nullptr ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < geometry.required_payload_bytes || !divisor_shape ||
        weight.weight_divisors == nullptr || !std::isfinite(weight.weight_scale_divisor) ||
        weight.weight_scale_divisor <= 0.0F || !std::isfinite(weight.input_scale_divisor) ||
        weight.input_scale_divisor <= 0.0F || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata != payload || weight.scales != payload + geometry.scale_plane_offset ||
        weight.weight_divisors !=
            payload + geometry.scale_plane_offset + geometry.scale_plane_bytes) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    return geometry;
}

} // namespace ninfer::ops::detail
