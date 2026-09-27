#include "hessian_io.h"

#include <fstream>
#include <stdexcept>

namespace ninfer::quantize::app {

std::vector<float> read_hessian_f32(const std::filesystem::path& path, std::uint64_t k) {
    const std::uint64_t count = k * k;
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read Hessian: " + path.string()); }
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("Hessian is too short for k = " + std::to_string(k) + ": " +
                                 path.string());
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        throw std::runtime_error("Hessian has trailing bytes for k = " + std::to_string(k) + ": " +
                                 path.string());
    }
    return values;
}

void write_f32_file(const std::filesystem::path& path, std::span<const float> values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("cannot write: " + path.string()); }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) { throw std::runtime_error("write failed: " + path.string()); }
}

} // namespace ninfer::quantize::app
