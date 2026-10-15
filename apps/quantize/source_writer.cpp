#include "source_writer.h"

#include "artifact/exl3_trellis.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::quantize::app {

void write_safetensors(const std::filesystem::path& path,
                       const std::vector<SafetensorsTensor>& tensors) {
    if (tensors.empty()) { throw std::invalid_argument("safetensors output must not be empty"); }
    nlohmann::json header = nlohmann::json::object();
    std::uint64_t offset = 0;
    for (const auto& tensor : tensors) {
        if (tensor.name.empty()) { throw std::invalid_argument("tensor name must not be empty"); }
        if (header.contains(tensor.name)) {
            // The header carries one range per name while every tensor's bytes are written, so a
            // duplicate would leave the offsets describing someone else's payload.
            throw std::invalid_argument("duplicate tensor name: " + tensor.name); }
        const std::uint64_t begin = offset;
        offset += tensor.data.size();
        header[tensor.name] = {
            {"dtype", tensor.dtype},
            {"shape", tensor.shape},
            {"data_offsets", {begin, offset}},
        };
    }
    std::string header_text = header.dump();
    // The reference format pads the header to an 8-byte boundary; JSON parsing ignores it.
    while (header_text.size() % 8 != 0) { header_text.push_back(' '); }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("cannot write safetensors: " + path.string()); }
    const std::uint64_t header_size = header_text.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    // The offsets are already known, so each tensor streams straight out: quantized shards reach
    // hundreds of megabytes and buffering them would double peak host memory for no gain.
    for (const auto& tensor : tensors) {
    out.write(reinterpret_cast<const char*>(tensor.data.data()),
              static_cast<std::streamsize>(tensor.data.size()));
    }
    if (!out) { throw std::runtime_error("safetensors write failed: " + path.string()); }
}

std::vector<std::byte> pack_trellis_states(const std::uint16_t* states, int bitrate_half_bits) {
    return artifact::exl3_pack_trellis_tile(
        std::span<const std::uint16_t>(states, artifact::kExl3WeightsPerTile),
        static_cast<std::uint64_t>(bitrate_half_bits));
}

} // namespace ninfer::quantize::app
