#pragma once
// Kernel parameters for TMA (cp.async.bulk.tensor) descriptors.
//
// A tensormap read by cp.async.bulk.tensor must be aligned in memory, and a descriptor built from
// CUtensorMap carries a 128-byte alignment requirement. Passing that aggregate by value into a
// __global__ function is legal for nvcc's device ABI, but the host stub nvcc generates for MSVC
// rejects an over-aligned by-value parameter (C2719), so the same bytes travel as an 8-aligned
// word array and the kernel reinterprets them.
//
// Kernel parameter space starts at a 128-byte aligned address and its layout begins at offset
// zero, so a first-position parameter is aligned for TMA whatever its declared alignment is. That
// was measured on sm_120a for this exact shape (first parameter of align 8 at ...380, the same
// parameter behind an int at ...388), which is why the descriptor has to stay the kernel's first
// parameter; moving it after another parameter would break the requirement. A descriptor that did
// end up misaligned faults at the first cp.async.bulk.tensor rather than returning a wrong answer.
#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {

template <class Descriptors>
struct alignas(8) TmaDescriptorWords {
    static_assert(sizeof(Descriptors) % sizeof(std::uint64_t) == 0,
                  "a TMA descriptor aggregate must be a whole number of 64-bit words");
    std::uint64_t words[sizeof(Descriptors) / sizeof(std::uint64_t)];
};

template <class Descriptors>
inline TmaDescriptorWords<Descriptors> tma_descriptor_words(const Descriptors& descriptors) {
    TmaDescriptorWords<Descriptors> words{};
    std::memcpy(words.words, &descriptors, sizeof(descriptors));
    return words;
}

template <class Descriptors>
__device__ __forceinline__ const Descriptors&
tma_descriptors(const TmaDescriptorWords<Descriptors>& words) {
    static_assert(alignof(Descriptors) <= 128);
    return *reinterpret_cast<const Descriptors*>(words.words);
}

} // namespace ninfer::ops::detail
