#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::core {

// Read-only files, opened once and read positionally from several threads at the same time: the
// artifact reader fans weight loads out over std::async readers, and the n-gram row reader keeps
// sixteen outstanding reads. This is the one place that knows how each platform spells "read at an
// offset without disturbing a shared file position".
#if defined(_WIN32)
using NativeFileHandle = void*; // HANDLE
#else
using NativeFileHandle = int;
#endif

// INVALID_HANDLE_VALUE on Windows, -1 on POSIX. A function, not a constant: the Windows spelling is
// a reinterpret_cast, which is not a constant expression.
[[nodiscard]] inline NativeFileHandle invalid_native_file_handle() noexcept {
#if defined(_WIN32)
    return reinterpret_cast<void*>(-1);
#else
    return -1;
#endif
}

// Opens an existing file for reading. `unbuffered` asks the OS to bypass its cache (O_DIRECT on
// POSIX, FILE_FLAG_NO_BUFFERING on Windows); the caller must then keep offsets and lengths aligned
// to the device's block size and read whole blocks. Throws std::system_error with the platform
// category, whose what() reads "<operation>: <platform message>", so callers stay identical on both
// platforms.
[[nodiscard]] NativeFileHandle open_read_only(const std::filesystem::path& path, bool unbuffered);

void close_file(NativeFileHandle handle) noexcept;

// Throws std::system_error if the handle cannot be queried.
[[nodiscard]] std::uint64_t file_bytes(NativeFileHandle handle);

// False for anything that is not a regular file (a directory, a pipe, a device).
[[nodiscard]] bool is_regular_file(NativeFileHandle handle);

// One positioned read attempt. Returns the bytes read, which is short only at end of file, where it
// is 0; interrupts are retried here because they are a platform detail and not a short read.
// Throws std::system_error on any other failure.
[[nodiscard]] std::size_t read_at(NativeFileHandle handle, std::uint64_t offset,
                                  std::span<std::byte> destination);

// Everything currently mapped through this file is dropped, so the pages can be faulted in again at
// some other size. Advisory on both platforms.
void advise_random_access([[maybe_unused]] const void* base,
                          [[maybe_unused]] std::size_t bytes) noexcept;

} // namespace ninfer::core
