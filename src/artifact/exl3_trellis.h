#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::artifact {

inline constexpr std::uint32_t kExl3Mul1Multiplier = 0x83DCD12DU;
inline constexpr std::size_t kExl3WeightsPerTile   = 256;

// The bitstream is little-endian within each byte. A decoded state's low bit is the oldest bit in
// its 16-bit circular window. Odd half-bit rates give odd-numbered weights the extra bit.
//
// Every entry point requires bitrate_half_bits in [2, 16] and tile.size() == 16 *
// bitrate_half_bits (std::invalid_argument otherwise), and a weight index below
// kExl3WeightsPerTile (std::out_of_range). The unpack/pack helpers additionally require exactly
// kExl3WeightsPerTile states.
[[nodiscard]] std::uint16_t exl3_trellis_state(std::span<const std::byte> tile,
                                               std::uint64_t bitrate_half_bits,
                                               std::size_t weight);
void exl3_unpack_trellis_tile(std::span<const std::byte> tile,
                              std::uint64_t bitrate_half_bits,
                              std::span<std::uint16_t> states);
[[nodiscard]] std::vector<std::byte>
exl3_pack_trellis_tile(std::span<const std::uint16_t> states,
                       std::uint64_t bitrate_half_bits);
[[nodiscard]] std::int32_t exl3_mul1_value(std::uint16_t state) noexcept;

} // namespace ninfer::artifact
