#pragma once

#include <type_traits>

#if defined(__CUDACC__)
#    define NINFER_KERNEL_HD __host__ __device__
#else
#    define NINFER_KERNEL_HD
#endif

namespace ninfer::ops {

inline constexpr float kLog2E = 1.4426950408889634F;

// Ceiling division of a non-negative x by a positive d. A negative x or a zero divisor is outside
// the contract (`d == 0` is undefined behaviour); the assert covers integrality only.

template <class T>
NINFER_KERNEL_HD constexpr T div_up(T x, T d) {
    static_assert(std::is_integral_v<T>, "div_up requires an integral type");
    return x / d + static_cast<T>(x % d != 0);
}

// Rounds a non-negative x up to the next multiple of `multiple` (which must be non-zero); it
// inherits div_up's contract.

template <class T>
NINFER_KERNEL_HD constexpr T round_up(T x, T multiple) {
    static_assert(std::is_integral_v<T>, "round_up requires an integral type");
    return div_up(x, multiple) * multiple;
}

template <auto Alignment, class T>
NINFER_KERNEL_HD constexpr T align_up(T x) {
    static_assert(std::is_integral_v<T>, "align_up requires an integral type");
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                  "align_up requires a power-of-two alignment");
    // The add is computed in the unsigned type so `x + Alignment - 1` cannot overflow the signed T.
    using U       = std::make_unsigned_t<T>;
    constexpr U a = static_cast<U>(Alignment);
    return static_cast<T>((static_cast<U> (x) + a - 1) & ~(a - 1));
}

template <int Bits>
NINFER_KERNEL_HD constexpr int sign_extend(int x) {
    static_assert(Bits > 0 && Bits < 32, "sign_extend supports bit widths in [1, 31]");
    // Mask to the low Bits first: the sign replicate below is otherwise driven by any higher bits
    // the caller happens to leave set.
    constexpr int mask = static_cast<int>((1U << Bits) - 1U);
    constexpr int sign = 1 << (Bits - 1);
    const int value    = x & mask;
    return (value ^ sign) - sign;
}

} // namespace ninfer::ops

#undef NINFER_KERNEL_HD
