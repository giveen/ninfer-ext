#include "reference.h"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ninfer::perplexity {
namespace {

constexpr char kMagic[Reference::kMagicBytes] = {'N', 'I', 'N', 'F', 'K', 'L', '1', '\0'};

std::uint32_t le_u32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::string reject(const std::filesystem::path& path, std::string_view why) {
    return "reference " + path.string() + ": " + std::string(why);
}

} // namespace

void Reference::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error(reject(path, "cannot open")); }

    std::array<unsigned char, kHeaderBytes> header{};
    file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!file) { throw std::runtime_error(reject(path, "header is truncated")); }
    if (std::memcmp(header.data(), kMagic, Reference::kMagicBytes) != 0) {
        throw std::runtime_error(reject(path, "bad magic (not a NInfer KL reference)"));
    }
    if (le_u32(header.data() + 8) != kVersion) {
        throw std::runtime_error(reject(path, "unsupported version"));
    }
    vocab_size_ = le_u32(header.data() + 12);
    const std::uint32_t rows = le_u32(header.data() + 16);
    context_                 = le_u32(header.data() + 20);
    stride_                  = le_u32(header.data() + 24);
    text_digest_.assign(reinterpret_cast<const char*>(header.data() + 32), kDigestBytes);
    if (vocab_size_ == 0 || rows == 0) {
        throw std::runtime_error(reject(path, "empty vocab or row count"));
    }

    std::vector<unsigned char> position_bytes(static_cast<std::size_t>(rows) * 4U);
    file.read(reinterpret_cast<char*>(position_bytes.data()),
              static_cast<std::streamsize>(position_bytes.size()));
    if (!file) { throw std::runtime_error(reject(path, "position table is truncated")); }
    positions_.resize(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        positions_[row] = le_u32(position_bytes.data() + static_cast<std::size_t>(row) * 4U);
    }

    const std::size_t entries = static_cast<std::size_t>(rows) * vocab_size_;
    logits_.resize(entries);
    file.read(reinterpret_cast<char*>(logits_.data()),
              static_cast<std::streamsize>(entries * sizeof(std::uint16_t)));
    if (!file) { throw std::runtime_error(reject(path, "logits payload is truncated")); }

    index_.reserve(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        // A duplicate position would make row() ambiguous; refuse rather than pick one.
        if (!index_.emplace(positions_[row], row).second) {
            throw std::runtime_error(reject(path, "duplicate scored position"));
        }
    }
}

const std::uint16_t* Reference::row(std::uint32_t target) const noexcept {
    const auto found = index_.find(target);
    if (found == index_.end()) { return nullptr; }
    return logits_.data() + found->second * static_cast<std::size_t>(vocab_size_);
}

} // namespace ninfer::perplexity
