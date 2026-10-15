#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

using Sha256Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
// Hashes `input`, calling `checkpoint` before the first complete 64-byte block and then at most
// once per MiB of complete blocks, so a caller can observe progress or cancel. It is not called for
// an input shorter than one block, and any exception it throws propagates to the caller.
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

} // namespace ninfer::models::qwen3_5::frontend
