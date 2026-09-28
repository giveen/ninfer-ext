#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::perplexity {

// KL reference produced by an external BF16 run over the same text and window protocol.
//
// Layout is little-endian throughout: a fixed 96-byte header (8-byte magic "NINFKL1\0", u32 version,
// u32 vocab, u32 row count, u32 context, u32 stride, u32 reserved, 64-byte lowercase hex text
// sha256), then `rows` u32 scored global target indices, then `rows` rows of BF16 logits with the
// vocab contiguous per row. Loading validates every field, so a text or protocol mismatch is refused
// instead of silently mis-aligning rows.
class Reference {
public:
    static constexpr std::uint32_t kVersion   = 1;
    static constexpr std::size_t kMagicBytes  = 8;
    static constexpr std::size_t kDigestBytes = 64;
    static constexpr std::size_t kHeaderBytes = 96;

    void load(const std::filesystem::path& path);

    [[nodiscard]] std::uint32_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] std::uint32_t context() const noexcept { return context_; }
    [[nodiscard]] std::uint32_t stride() const noexcept { return stride_; }
    [[nodiscard]] const std::string& text_digest() const noexcept { return text_digest_; }
    [[nodiscard]] std::span<const std::uint32_t> positions() const noexcept { return positions_; }
    [[nodiscard]] std::size_t rows() const noexcept { return positions_.size(); }

    // BF16 row (vocab_size entries) for a global target index, or nullptr when the reference does not
    // score that position.
    [[nodiscard]] const std::uint16_t* row(std::uint32_t target) const noexcept;

private:
    std::uint32_t vocab_size_ = 0;
    std::uint32_t context_    = 0;
    std::uint32_t stride_     = 0;
    std::string text_digest_;
    std::vector<std::uint32_t> positions_;
    std::vector<std::uint16_t> logits_;
    std::unordered_map<std::uint32_t, std::size_t> index_;
};

} // namespace ninfer::perplexity
