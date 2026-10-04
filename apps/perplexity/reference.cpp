#include "reference.h"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>

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

namespace {

void put_u32(std::array<unsigned char, Reference::kHeaderBytes>& header, std::size_t at,
             std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        header[at + static_cast<std::size_t>(i)] = static_cast<unsigned char>(value >> (8 * i));
    }
}

} // namespace

void write_reference(const std::filesystem::path& path, std::uint32_t vocab_size,
                     std::uint32_t context, std::uint32_t stride, std::string_view text_digest,
                     std::span<const std::uint32_t> positions,
                     std::span<const std::uint16_t> logits) {
    if (vocab_size == 0 || positions.empty() ||
        logits.size() != positions.size() * static_cast<std::size_t>(vocab_size)) {
        throw std::invalid_argument("reference rows do not match the position table");
    }
    if (text_digest.size() != Reference::kDigestBytes) {
        throw std::invalid_argument("reference text digest must be 64 hex characters");
    }
    std::array<unsigned char, Reference::kHeaderBytes> header{};
    std::memcpy(header.data(), kMagic, Reference::kMagicBytes);
    put_u32(header, 8, Reference::kVersion);
    put_u32(header, 12, vocab_size);
    put_u32(header, 16, static_cast<std::uint32_t>(positions.size()));
    put_u32(header, 20, context);
    put_u32(header, 24, stride);
    std::memcpy(header.data() + 32, text_digest.data(), Reference::kDigestBytes);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) { throw std::runtime_error(reject(path, "cannot create")); }
    file.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(positions.data()),
               static_cast<std::streamsize>(positions.size_bytes()));
    file.write(reinterpret_cast<const char*>(logits.data()),
               static_cast<std::streamsize>(logits.size_bytes()));
    if (!file) { throw std::runtime_error(reject(path, "write failed")); }
}

} // namespace ninfer::perplexity
