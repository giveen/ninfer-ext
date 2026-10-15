#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Session-local indexed membership for opaque Program target ordinals. Storage is acquired when a
// bounded search begins; discovering, assessing, and expanding targets performs no allocation and
// no linear handle scan. One instance belongs to one session and one thread: the marks are
// unsynchronized, so it must not be shared between threads.
class BoundedTargetLedger {
public:
    static constexpr std::uint8_t Discovered = 1U;
    static constexpr std::uint8_t Assessed   = 2U;
    static constexpr std::uint8_t Expanded   = 4U;

    explicit BoundedTargetLedger(std::size_t reserve_capacity = 0) {
        states_.reserve(reserve_capacity);
    }

    void reset(std::size_t capacity) { states_.assign(capacity, 0); }

    // True when every bit of `mark` has been recorded; marks are combined with `mark()`, so a
    // membership question must ask for all of them.

    [[nodiscard]] bool contains(std::uint32_t ordinal, std::uint8_t mark) const {
        require_ordinal(ordinal);
        return mark != 0 && (states_[ordinal] & mark)== mark;
    }

    void mark(std::uint32_t ordinal, std::uint8_t mark) {
        require_ordinal(ordinal);
        states_[ordinal] |= mark;
    }

private:
    void require_ordinal(std::uint32_t ordinal) const {
        if (ordinal >= states_.size()) {
            throw std::logic_error("Program target ordinal exceeds the bounded search ledger");
        }
    }

    std::vector<std::uint8_t> states_;
};

} // namespace ninfer::runtime
