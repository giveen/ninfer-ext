#pragma once

#include "ninfer/types.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// `auto` or a positive MiB count for the routed-expert device cache.
[[nodiscard]] inline ExpertCachePolicy parse_expert_cache(std::string_view value) {
    if (value == "auto") { return ExpertCachePolicy{}; }
    const std::string text(value);
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long mebis = std::strtoull(text.c_str(), &end, 10);
    constexpr std::uint64_t kMiB   = 1024ULL * 1024ULL;
    if (text.empty() || text.front() == '-' || errno == ERANGE || end == text.c_str() ||
        *end != '\0' || mebis == 0 || mebis > std::numeric_limits<std::size_t>::max() / kMiB) {
        throw std::invalid_argument("invalid expert-cache: " + text);
    }
    return ExpertCachePolicy::explicit_cache(static_cast<std::size_t>(mebis * kMiB));
}

[[nodiscard]] inline NgramResidency parse_ngram_residency(std::string_view value) {
    if (value == "auto") { return NgramResidency::Automatic; }
    if (value == "mapped") { return NgramResidency::Mapped; }
    if (value == "stream") { return NgramResidency::Stream; }
    throw std::invalid_argument("invalid ngram-residency: " + std::string(value));
}

[[nodiscard]] inline const char* ngram_residency_name(NgramResidency residency) noexcept {
    switch (residency) {
    case NgramResidency::Automatic:
        return "auto";
    case NgramResidency::Mapped:
        return "mapped";
    case NgramResidency::Stream:
        return "stream";
    }
    return "unknown";
}

} // namespace ninfer::product
