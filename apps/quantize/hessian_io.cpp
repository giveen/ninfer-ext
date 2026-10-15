#include "hessian_io.h"

#include "core/checked_math.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace ninfer::quantize::app {
namespace {

// Element count of a k x k FP32 Hessian. The square is formed in 64 bits and bounded so neither it
// nor the byte size can wrap: a malformed dimension would otherwise produce an undersized buffer
// and defeat read_hessian_f32's short-file guard.
std::size_t hessian_elements(std::uint64_t k) {
    const std::uint64_t count = ninfer::checked_mul(k, k, "Hessian dimension k");
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("Hessian dimension k exceeds the host address space");
    }
    return static_cast<std::size_t>(count);
}

} // namespace

std::vector<float> read_hessian_f32(const std::filesystem::path& path, std::uint64_t k) {
    const std::size_t count = hessian_elements( k);
    std::vector<float> values(count);
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

std::vector<float> identity_hessian(std::uint64_t k) {
    std::vector<float> values(hessian_elements( k), 0.0F);
    for (std::uint64_t i = 0; i < k; ++i) {
        values[static_cast<std::size_t>(i * k + i)] = 1.0F;
    }
    return values;
}

} // namespace ninfer::quantize::app
