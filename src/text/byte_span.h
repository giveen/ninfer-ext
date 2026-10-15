#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace ninfer::text {

struct ByteSpan {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// Spans are nonempty, ordered and disjoint. A partial overlap also counts.
inline bool overlaps(std::span<const ByteSpan> spans, std::size_t begin, std::size_t end) {
    if (begin >= end) { return false; }
    const auto it =
        std::lower_bound(spans.begin(), spans.end(), begin,
                         [](const ByteSpan& span, std::size_t pos) { return span.end <= pos; });
    return it != spans.end() && it->begin < end;
}

} // namespace ninfer::text
