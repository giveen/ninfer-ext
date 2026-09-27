#include "trace_reader.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::quantize::app {
namespace {

struct Entry {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read trace: " + path.string()); }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0) { throw std::runtime_error("empty trace: " + path.string()); }
    in.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

std::vector<Entry> parse_header(const std::vector<std::byte>& bytes,
                                std::uint64_t& data_start) {
    if (bytes.size() < 8) { throw std::invalid_argument("truncated safetensors header"); }
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, bytes.data(), sizeof(header_size));
    if (header_size > bytes.size() - 8) {
        throw std::invalid_argument("safetensors header exceeds the file");
    }
    const std::string text(reinterpret_cast<const char*>(bytes.data() + 8),
                           static_cast<std::size_t>(header_size));
    const auto root    = nlohmann::json::parse(text);
    data_start         = 8 + header_size;
    const std::uint64_t data_bytes = bytes.size() - data_start;
    std::vector<Entry> entries;
    for (const auto& [name, item] : root.items()) {
        if (name == "__metadata__") { continue; }
        if (!item.contains("dtype") || !item.contains("shape") || !item.contains("data_offsets")) {
            throw std::invalid_argument(name + ": malformed safetensors entry");
        }
        Entry entry;
        entry.dtype = item.at("dtype").get<std::string>();
        for (const auto& dim : item.at("shape")) { entry.shape.push_back(dim.get<std::uint64_t>()); }
        entry.begin = item.at("data_offsets").at(0).get<std::uint64_t>();
        entry.end   = item.at("data_offsets").at(1).get<std::uint64_t>();
        if (entry.begin > entry.end || entry.end > data_bytes) {
            throw std::invalid_argument(name + ": safetensors range exceeds the file");
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

const Entry& require(const std::vector<Entry>& entries, const char* name,
                     const std::vector<std::uint64_t>& shape) {
    for (const auto& entry : entries) {
        if (entry.shape == shape && (entry.dtype == "I64")) { return entry; }
    }
    std::string expected;
    for (const auto dim : shape) { expected += " " + std::to_string(dim); }
    throw std::invalid_argument(std::string(name) + ": expected I64[" + expected + " ]");
}

} // namespace

CalibrationTrace read_calibration_trace(const std::filesystem::path& path) {
    const std::vector<std::byte> bytes = read_file(path);
    std::uint64_t data_start           = 0;
    const std::vector<Entry> entries   = parse_header(bytes, data_start);

    // input_ids is [rows, row_tokens] and lengths is [rows]. Find a consistent rows value.
    const Entry* ids     = nullptr;
    const Entry* lengths = nullptr;
    for (const auto& entry : entries) {
        if (entry.dtype != "I64") { continue; }
        if (entry.shape.size() == 2 && (ids == nullptr || entry.shape[1] > ids->shape[1])) {
            ids = &entry;
        } else if (entry.shape.size() == 1) {
            lengths = &entry;
        }
    }
    if (ids == nullptr || lengths == nullptr || ids->shape[0] != lengths->shape[0]) {
        throw std::invalid_argument(
            "trace needs I64 input_ids [rows, tokens] and matching I64 lengths [rows]");
    }
    const std::uint64_t rows       = ids->shape[0];
    const std::uint64_t row_tokens = ids->shape[1];
    if (rows == 0 || row_tokens == 0 || rows > std::numeric_limits<std::uint32_t>::max() ||
        row_tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("trace dimensions are out of range");
    }

    const auto read_i64 = [&](const Entry& entry, std::uint64_t count) {
        std::vector<std::int64_t> out(static_cast<std::size_t>(count));
        std::memcpy(out.data(), bytes.data() + data_start + entry.begin,
                    static_cast<std::size_t>(count) * sizeof(std::int64_t));
        return out;
    };
    const std::vector<std::int64_t> packed = read_i64(*ids, rows * row_tokens);
    const std::vector<std::int64_t> sizes  = read_i64(*lengths, rows);

    CalibrationTrace trace;
    trace.rows       = static_cast<std::uint32_t>(rows);
    trace.row_tokens = static_cast<std::uint32_t>(row_tokens);
    trace.input_ids.resize(packed.size());
    trace.lengths.resize(sizes.size());
    constexpr std::int64_t kMaxTokenId = std::numeric_limits<std::int32_t>::max();
    for (std::size_t index = 0; index < packed.size(); ++index) {
        if (packed[index] < 0 || packed[index] > kMaxTokenId) {
            throw std::invalid_argument("trace input_ids contains a token outside the I32 domain");
        }
        trace.input_ids[index] = static_cast<std::int32_t>(packed[index]);
    }
    for (std::size_t row = 0; row < sizes.size(); ++row) {
        if (sizes[row] < 0 || sizes[row] > static_cast<std::int64_t>(row_tokens)) {
            throw std::invalid_argument("trace lengths exceed the row width");
        }
        trace.lengths[row] = static_cast<std::int32_t>(sizes[row]);
    }
    return trace;
}

} // namespace ninfer::quantize::app
