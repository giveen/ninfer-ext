#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#    include <malloc.h>
#else
#    include <sys/mman.h>
#endif

namespace ninfer::core {

// Host memory that a device-visible table or a DMA staging buffer will be pinned over.
// std::aligned_alloc is C11 and absent from MSVC, whose CRT allocates aligned blocks with
// _aligned_malloc and requires _aligned_free to release them -- mixing the two corrupts the heap,
// so both ends live here and no call site names either one.
inline void* aligned_host_alloc(std::size_t bytes, std::size_t alignment) {
#if defined(_WIN32)
    return ::_aligned_malloc(bytes, alignment);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

inline void aligned_host_free(void* pointer) noexcept {
    if (pointer == nullptr) { return; }
#if defined(_WIN32)
    ::_aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

// Ask for the largest pages the OS will back this range with. Advisory everywhere: the buffer is
// correct either way, only the number of translation entries behind the pin differs. Windows needs
// SeLockMemoryPrivilege for large pages and returns them from VirtualAlloc rather than from the
// CRT, so a caller that demands them is a different design; nothing here depends on getting them.
inline void advise_large_pages([[maybe_unused]] void* pointer,
                               [[maybe_unused]] std::size_t bytes) noexcept {
#if !defined(_WIN32)
    (void)::madvise(pointer, bytes, MADV_HUGEPAGE);
#endif
}

// Touch every page before the range is pinned, so the pin is built once over the pages that will
// stay resident instead of being faulted in piecewise by the registration itself. POSIX can ask for
// exactly that; elsewhere writing the range is the portable equivalent.
inline void prefault_host_pages(void* pointer, std::size_t bytes) noexcept {
#if !defined(_WIN32)
    if (::madvise(pointer, bytes, MADV_POPULATE_WRITE) == 0) { return; }
#endif
    std::memset(pointer, 0, bytes);
}

} // namespace ninfer::core
