#include "core/native_file.h"

#include <cerrno>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
// windows.h defines min and max as macros, which would swallow any std::min/std::max in this file
// or in anything it includes; this file uses neither today, and the guard is what keeps that from
// mattering.
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::core {

#if defined(_WIN32)
namespace {

// A failing Windows call reports through GetLastError(), which any intervening runtime call may
// overwrite, so the code is captured at the site and carried into the exception.
std::system_error windows_error(const char* operation) {
    return std::system_error(static_cast<int>(::GetLastError()), std::system_category(), operation);
}

} // namespace

NativeFileHandle open_read_only(const std::filesystem::path& path, bool unbuffered) {
    // FILE_FLAG_OVERLAPPED is what makes the offset per-call: a positioned read carries its offset
    // in the OVERLAPPED structure instead of moving a file pointer, which is exactly pread(2). It
    // does not make the reads asynchronous for our purposes, because every call waits for its own
    // completion before returning.
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (unbuffered) { flags |= FILE_FLAG_NO_BUFFERING; }
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { throw windows_error("open"); }
    return handle;
}

void close_file(NativeFileHandle handle) noexcept {
    if (handle != invalid_native_file_handle()) {
        (void)::CloseHandle(static_cast<HANDLE>(handle));
    }
}

std::uint64_t file_bytes(NativeFileHandle handle) {
    LARGE_INTEGER size{};
    if (::GetFileSizeEx(static_cast<HANDLE>(handle), &size) == 0) { throw windows_error("size"); }
    return static_cast<std::uint64_t>(size.QuadPart);
}

bool is_regular_file(NativeFileHandle handle) {
    BY_HANDLE_FILE_INFORMATION information{};
    if (::GetFileInformationByHandle(static_cast<HANDLE>(handle), &information) == 0) {
        throw windows_error("stat");
    }
    if ((information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) { return false; }
    return ::GetFileType(static_cast<HANDLE>(handle)) == FILE_TYPE_DISK;
}

std::size_t read_at(NativeFileHandle handle, std::uint64_t offset,
                    std::span<std::byte> destination) {
    if (destination.empty()) { return 0; }

    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

    // ReadFile takes a DWORD length, so one call covers at most 4 GiB - 1; a longer request is
    // served as a short read, which callers already loop over. Both platforms therefore agree that
    // a read may be short.
    const DWORD requested =
        static_cast<DWORD>(destination.size() > 0xFFFFFFFFULL ? 0xFFFFFFFFULL : destination.size());

    DWORD read = 0;
    if (::ReadFile(static_cast<HANDLE>(handle), destination.data(), requested, &read,
                   &overlapped) == 0) {
        const DWORD error = ::GetLastError();
        if (error == ERROR_IO_PENDING) {
            // A pending overlapped read is normal even for a disk file; wait for this request only,
            // which is why the OVERLAPPED lives on this stack frame.
            if (::GetOverlappedResult(static_cast<HANDLE>(handle), &overlapped, &read, TRUE) == 0) {
                const DWORD completion_error = ::GetLastError();
                if (completion_error == ERROR_HANDLE_EOF) { return 0; }
                ::SetLastError(completion_error);
                throw windows_error("read");
            }
            return static_cast<std::size_t>(read);
        }
        // Reading at or past end of file. Callers distinguish this from a hole inside the file.
        if (error == ERROR_HANDLE_EOF) { return 0; }
        throw windows_error("read");
    }
    return static_cast<std::size_t>(read);
}

void advise_random_access(const void*, std::size_t) noexcept {
    // No Windows equivalent worth having: MADV_RANDOM only tells the kernel's readahead to stand
    // down, and this platform's cache manager has no switch for that. Sizing is unchanged.
}

#else

namespace {

std::system_error posix_error(const char* operation) {
    return std::system_error(errno, std::generic_category(), operation);
}

} // namespace

NativeFileHandle open_read_only(const std::filesystem::path& path, bool unbuffered) {
    const int flags      = O_RDONLY | O_CLOEXEC | (unbuffered ? O_DIRECT : 0);
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) { throw posix_error("open"); }
    return descriptor;
}

void close_file(NativeFileHandle handle) noexcept {
    if (handle != invalid_native_file_handle()) { (void)::close(handle); }
}

std::uint64_t file_bytes(NativeFileHandle handle) {
    struct stat status{};
    if (::fstat(handle, &status) != 0) { throw posix_error("size"); }
    if (status.st_size < 0) { throw posix_error("size"); }
    return static_cast<std::uint64_t>(status.st_size);
}

bool is_regular_file(NativeFileHandle handle) {
    struct stat status{};
    if (::fstat(handle, &status) != 0) { throw posix_error("stat"); }
    return S_ISREG(status.st_mode);
}

std::size_t read_at(NativeFileHandle handle, std::uint64_t offset,
                    std::span<std::byte> destination) {
    if (destination.empty()) { return 0; }
    for (;;) {
        const ssize_t read =
            ::pread(handle, destination.data(), destination.size(), static_cast<off_t>(offset));
        if (read >= 0) { return static_cast<std::size_t>(read); }
        if (errno != EINTR) { throw posix_error("read"); }
    }
}

void advise_random_access(const void* base, std::size_t bytes) noexcept {
    (void)::madvise(const_cast<void*>(base), bytes, MADV_RANDOM);
}

#endif

} // namespace ninfer::core
