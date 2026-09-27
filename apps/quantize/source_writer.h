#pragma once

// Minimal safetensors writer for the quantizer's source store. Only the subset the converter
// reads is produced: named tensors with a fixed dtype, shape and contiguous little-endian data.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::quantize::app {

struct SafetensorsTensor {
    std::string name;
    std::string dtype; // "U8" or "F32"
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> data;
};

// One file holding every tensor: <u64 header length><JSON header><contiguous data>.
void write_safetensors(const std::filesystem::path& path,
                       const std::vector<SafetensorsTensor>& tensors);

// Encode one tail-biting tile's 256 states into its 16 * bitrate_half_bits byte stream. The
// caller keeps the canonical state order of the quantizer's `states` output.
[[nodiscard]] std::vector<std::byte> pack_trellis_states(const std::uint16_t* states,
                                                         int bitrate_half_bits);

} // namespace ninfer::quantize::app
