#pragma once

// Small host-side integer helpers shared across subsystems: pointer alignment, overflow-checked
// arithmetic, and power-of-two rounding. Header-only and zero-cost.
//
// The kernel-facing constexpr align_up in ops/common/math.h stays separate: it takes one argument,
// performs no validation, and is usable on device.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace ninfer {

// True when `pointer` is non-null and its address is a multiple of `alignment` (a power of two).
[[nodiscard]] inline bool aligned_to(const void* pointer, std::uintptr_t alignment) noexcept {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

template <class T>
[[nodiscard]] inline T checked_add(T a, T b, std::string_view label) {
    static_assert(std::is_unsigned_v<T>, "checked_add requires an unsigned type");
    if (b > std::numeric_limits<T>::max() - a) {
        throw std::overflow_error(std::string(label) + " overflows");
    }
    return static_cast<T>(a + b);
}

template <class T>
[[nodiscard]] inline T checked_mul(T a, T b, std::string_view label) {
    static_assert(std::is_unsigned_v<T>, "checked_mul requires an unsigned type");
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflows");
    }
    return static_cast<T>(a * b);
}

template <class T>
[[nodiscard]] inline T align_up(T value, T alignment, std::string_view label) {
    static_assert(std::is_unsigned_v<T>, "align_up requires an unsigned type");
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const T mask = static_cast<T>(alignment - 1);
    return static_cast<T>(checked_add(value, mask, label) & static_cast<T>(~mask));
}

} // namespace ninfer
