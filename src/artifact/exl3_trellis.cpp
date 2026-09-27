#include "artifact/exl3_trellis.h"

#include <cstddef>
#include <stdexcept>

namespace ninfer::artifact {
namespace {

void validate_rate(std::uint64_t bitrate_half_bits) {
    if (bitrate_half_bits < 2 || bitrate_half_bits > 16) {
        throw std::invalid_argument("EXL3 bitrate_half_bits must be in [2, 16]");
    }
}

std::size_t tile_bytes(std::uint64_t bitrate_half_bits) {
    validate_rate(bitrate_half_bits);
    return static_cast<std::size_t>(16 * bitrate_half_bits);
}

std::size_t step_width(std::uint64_t bitrate_half_bits, std::size_t weight) {
    const auto low = static_cast<std::size_t>(bitrate_half_bits / 2);
    return low + ((bitrate_half_bits & 1U) && (weight & 1U) ? 1U : 0U);
}

std::size_t end_bit(std::uint64_t bitrate_half_bits, std::size_t weight) {
    const auto count = weight + 1;
    const auto low = static_cast<std::size_t>(bitrate_half_bits / 2);
    const auto extra = (bitrate_half_bits & 1U) ? count / 2 : 0;
    return count * low + extra;
}

std::uint32_t read_bit(std::span<const std::byte> bytes, std::size_t bit) {
    const auto word = std::to_integer<std::uint8_t>(bytes[bit / 8]);
    return (word >> (bit % 8)) & 1U;
}

void write_bit(std::span<std::byte> bytes, std::size_t bit, std::uint32_t value) {
    auto word = std::to_integer<std::uint8_t>(bytes[bit / 8]);
    const auto mask = static_cast<std::uint8_t>(1U << (bit % 8));
    word = value ? static_cast<std::uint8_t>(word | mask)
                 : static_cast<std::uint8_t>(word & static_cast<std::uint8_t>(~mask));
    bytes[bit / 8] = static_cast<std::byte>(word);
}

} // namespace

std::uint16_t exl3_trellis_state(std::span<const std::byte> tile,
                                 std::uint64_t bitrate_half_bits, std::size_t weight) {
    if (weight >= kExl3WeightsPerTile) {
        throw std::out_of_range("EXL3 tile weight index must be in [0, 255]");
    }
    if (tile.size() != tile_bytes(bitrate_half_bits)) {
        throw std::invalid_argument("EXL3 tile byte count does not match bitrate");
    }
    const auto total_bits = tile.size() * 8;
    const auto end = end_bit(bitrate_half_bits, weight);
    const auto begin = (end + total_bits - 16) % total_bits;
    std::uint16_t state = 0;
    for (std::size_t bit = 0; bit < 16; ++bit) {
        state |= static_cast<std::uint16_t>(read_bit(tile, (begin + bit) % total_bits) << bit);
    }
    return state;
}

void exl3_unpack_trellis_tile(std::span<const std::byte> tile,
                              std::uint64_t bitrate_half_bits,
                              std::span<std::uint16_t> states) {
    if (states.size() != kExl3WeightsPerTile) {
        throw std::invalid_argument("EXL3 tile must decode exactly 256 states");
    }
    for (std::size_t weight = 0; weight < states.size(); ++weight) {
        states[weight] = exl3_trellis_state(tile, bitrate_half_bits, weight);
    }
}

std::vector<std::byte> exl3_pack_trellis_tile(std::span<const std::uint16_t> states,
                                             std::uint64_t bitrate_half_bits) {
    if (states.size() != kExl3WeightsPerTile) {
        throw std::invalid_argument("EXL3 tile must contain exactly 256 states");
    }
    const auto bytes = tile_bytes(bitrate_half_bits);
    std::vector<std::byte> packed(bytes);
    std::size_t cursor = 0;
    for (std::size_t weight = 0; weight < states.size(); ++weight) {
        const auto width = step_width(bitrate_half_bits, weight);
        const auto mask = (1U << width) - 1U;
        const auto suffix = (static_cast<std::uint32_t>(states[weight]) >> (16 - width)) & mask;
        const auto previous = states[(weight + states.size() - 1) % states.size()];
        const auto expected = static_cast<std::uint16_t>(
            (previous >> width) | (suffix << (16 - width)));
        if (states[weight] != expected) {
            throw std::invalid_argument("EXL3 states do not form a circular trellis path");
        }
        for (std::size_t bit = 0; bit < width; ++bit) {
            write_bit(packed, cursor + bit, (suffix >> bit) & 1U);
        }
        cursor += width;
    }
    if (cursor != packed.size() * 8) {
        throw std::logic_error("EXL3 bitrate did not fill its tile bitstream");
    }
    return packed;
}

std::int32_t exl3_mul1_value(std::uint16_t state) noexcept {
    const auto product = static_cast<std::uint32_t>(state) * kExl3Mul1Multiplier;
    const auto sum = (product & 0xFFU) + ((product >> 8) & 0xFFU) +
                     ((product >> 16) & 0xFFU) + ((product >> 24) & 0xFFU);
    return static_cast<std::int32_t>(sum) - 510;
}

} // namespace ninfer::artifact
