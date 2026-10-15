#pragma once

// Exact unsigned 128-bit arithmetic for contract arithmetic that must neither lose a carry nor
// mistake a saturated result for an exact one. GCC and Clang provide unsigned __int128 and the
// Linux build keeps using it unchanged; MSVC has no 128-bit integer type at all, so the same value
// is carried in two 64-bit words whose addition, subtraction, truncated multiplication, shifts and
// comparisons reproduce the native type exactly: arithmetic wraps modulo 2^128 and every operation
// has the same result and the same overflow detection as its native counterpart.
//
// Only the operations the contract needs are provided; division (which MSVC can only serve through
// the _udiv128 intrinsic) is deliberately absent because no caller performs it.
//
// Defining NINFER_UINT128_PORTABLE selects the two-word implementation on any toolchain, which is
// how it is checked against the native type where the native type exists.

#include <compare>
#include <cstdint>

namespace ninfer::runtime {

#if defined(__SIZEOF_INT128__) && !defined(NINFER_UINT128_PORTABLE)

using UInt128 = unsigned __int128;

#else

class UInt128 {
public:
    constexpr UInt128() noexcept = default;

    constexpr UInt128(std::uint64_t value) noexcept : low_(value) {}

    constexpr UInt128(std::uint64_t low, std::uint64_t high) noexcept : low_(low), high_(high) {}

    // Truncation to the low word, exactly like a narrowing cast of the native type.
    [[nodiscard]] constexpr explicit operator std::uint64_t() const noexcept { return low_; }

    [[nodiscard]] constexpr std::uint64_t low() const noexcept { return low_; }

    [[nodiscard]] constexpr std::uint64_t high() const noexcept { return high_; }

    friend constexpr UInt128 operator+(UInt128 left, UInt128 right) noexcept {
        const std::uint64_t low = left.low_ + right.low_;
        return UInt128{low, left.high_ + right.high_ + (low < left.low_ ? 1U : 0U)};
    }

    friend constexpr UInt128 operator-(UInt128 left, UInt128 right) noexcept {
        const std::uint64_t low = left.low_ - right.low_;
        return UInt128{low, left.high_ - right.high_ - (left.low_ < right.low_ ? 1U : 0U)};
    }

    // The full 256-bit product truncated to 128 bits, which is what unsigned __int128
    // multiplication does: the three surviving partial products are the only ones that can reach
    // bit 127.
    friend constexpr UInt128 operator*(UInt128 left, UInt128 right) noexcept {
        const std::uint64_t low = left.low_ * right.low_;
        const std::uint64_t high =
            high_product(left.low_, right.low_) + left.high_ * right.low_ + left.low_ * right.high_;
        return UInt128{low, high};
    }

    friend constexpr UInt128 operator~(UInt128 value) noexcept {
        return UInt128{~value.low_, ~value.high_};
    }

    // Shift counts of 128 or more are undefined for the native type as well; they yield zero here
    // so that the fallback cannot invoke an undefined shift of its own.
    friend constexpr UInt128 operator<<(UInt128 value, unsigned shift) noexcept {
        if (shift == 0) { return value; }
        if (shift < 64) {
            return UInt128{value.low_ << shift,
                           (value.high_ << shift) | (value.low_ >> (64U - shift))};
        }
        if (shift < 128) { return UInt128{0, value.low_ << (shift - 64U)}; }
        return UInt128{};
    }

    friend constexpr UInt128 operator>>(UInt128 value, unsigned shift) noexcept {
        if (shift == 0) { return value; }
        if (shift < 64) {
            return UInt128{(value.low_ >> shift) | (value.high_ << (64U - shift)),
                           value.high_ >> shift};
        }
        if (shift < 128) { return UInt128{value.high_ >> (shift - 64U), 0}; }
        return UInt128{};
    }

    friend constexpr bool operator==(UInt128 left, UInt128 right) noexcept {
        return left.low_ == right.low_ && left.high_ == right.high_;
    }

    // Unsigned order: the high word decides unless the two high words are equal, exactly as the
    // native type compares its single 128-bit value. Every relational operator is rewritten from
    // this one, so they cannot disagree with each other.
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(UInt128 left,
                                                                    UInt128 right) noexcept {
        if (left.high_ != right.high_) {
            return left.high_ < right.high_ ? std::strong_ordering::less
                                            : std::strong_ordering::greater;
        }
        return left.low_ <=> right.low_;
    }

private:
    // High 64 bits of a 64x64 unsigned product, from the four 32x32 partial products of
    // a*b = a1*b1*2^64 + (a1*b0 + a0*b1)*2^32 + a0*b0. Portable and constexpr on purpose: MSVC's
    // __umulh intrinsic is neither usable in a constant expression nor available to nvcc host code.
    [[nodiscard]] static constexpr std::uint64_t high_product(std::uint64_t left,
                                                              std::uint64_t right) noexcept {
        const std::uint64_t left_low   = left & 0xFFFFFFFFULL;
        const std::uint64_t left_high  = left >> 32U;
        const std::uint64_t right_low  = right & 0xFFFFFFFFULL;
        const std::uint64_t right_high = right >> 32U;
        const std::uint64_t low_low    = left_low * right_low;
        const std::uint64_t low_high   = left_low * right_high;
        const std::uint64_t high_low   = left_high * right_low;
        const std::uint64_t high_high  = left_high * right_high;
        const std::uint64_t middle =
            (low_low >> 32U) + (low_high & 0xFFFFFFFFULL) + (high_low & 0xFFFFFFFFULL);
        return high_high + (low_high >> 32U) + (high_low >> 32U) + (middle >> 32U);
    }

    std::uint64_t low_  = 0;
    std::uint64_t high_ = 0;
};

#endif

} // namespace ninfer::runtime
