#include "parameter_reader.h"

#include "core/weight_view.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

namespace ninfer::quantize::app {
namespace {

float bf16_to_fp32(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float value              = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint64_t word_bytes(const WeightGeometry& geometry) {
    switch (geometry.format) {
    case QType::BF16: return 2;
    default: throw std::invalid_argument("parameter reader supports only BF16 direct tensors");
    }
}

} // namespace

std::vector<LinearParameter> enumerate_linear_parameters(const artifact::Reader& reader) {
    std::set<std::string, std::less<>> used;
    for (const auto& [key, use] : reader.directory().uses) { used.insert(key.first); }

    std::vector<LinearParameter> out;
    for (const auto& [name, binding] : reader.directory().bindings) {
        if (binding.parts.empty() || !used.contains(name)) { continue; }
        // M3 quantizes the Text and MTP stacks; Vision stays at its stored precision.
        if (!name.starts_with("text/") && !name.starts_with("mtp/")) { continue; }
        const auto& geometry = reader.geometry(binding.parts.front().object);
        if (geometry.format != QType::BF16 || geometry.layout != QuantLayout::Contiguous ||
            geometry.shape.size() != 2) {
            continue;
        }
        const std::uint64_t k = geometry.shape[1];
        if (k == 0 || binding.elements % k != 0) { continue; }
        // A binding may be a row range of a fused parent, so the logical n is its own element
        // count over the parent's K, not the parent's row count.
        const std::uint64_t n = binding.elements / k;
        if (n % 128 != 0 || k % 128 != 0) { continue; }
        out.push_back(LinearParameter{name, n, k, binding});
    }
    return out;
}

std::vector<float> read_parameter_matrix(const artifact::Reader& reader,
                                         const LinearParameter& parameter) {
    const std::uint64_t elements = parameter.n * parameter.k;
    std::vector<float> values;
    values.reserve(elements);
    for (const auto& part : parameter.binding.parts) {
        const auto& geometry = reader.geometry(part.object);
        const std::uint64_t bytes_per_word = word_bytes(geometry);
        const std::vector<std::byte> payload = reader.read_object(part.object);
        if (part.end * bytes_per_word > payload.size()) {
            throw std::runtime_error(parameter.name + ": binding range exceeds its object");
        }
        for (std::uint64_t element = part.begin; element < part.end; ++element) {
            const std::size_t at = static_cast<std::size_t>(element * bytes_per_word);
            const std::uint16_t word =
                static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(payload[at])) |
                (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(payload[at + 1])) << 8);
            values.push_back(bf16_to_fp32(word));
        }
    }
    if (values.size() != elements) {
        throw std::runtime_error(parameter.name + ": binding element count differs from its shape");
    }
    return values;
}

std::vector<float> transpose_to_kn(const std::vector<float>& nk, std::uint64_t n, std::uint64_t k) {
    if (nk.size() != n * k) { throw std::invalid_argument("transpose input size mismatch"); }
    std::vector<float> kn(n * k);
    for (std::uint64_t row = 0; row < n; ++row) {
        for (std::uint64_t col = 0; col < k; ++col) {
            kn[col * n + row] = nk[row * k + col];
        }
    }
    return kn;
}

} // namespace ninfer::quantize::app
