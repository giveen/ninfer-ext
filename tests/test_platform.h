#pragma once

// Portable process, errno and temporary-path helpers for the tests. POSIX reads /tmp and names its
// process and descriptor calls without an underscore prefix; Windows has no <unistd.h> and no /tmp,
// so the platform difference lives here instead of at every call site.

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>

#if defined(_WIN32)
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace ninfer::test {

// The test process id. Windows prefixes the POSIX call with an underscore.
inline long process_id() {
#if defined(_WIN32)
    return static_cast<long>(::_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}

// The message for an errno value. Windows has neither strerror_r nor a thread-safe strerror: its
// secure variant writes into a caller buffer, and its message text is not the POSIX one, so callers
// never format errno themselves.
inline std::string errno_message(int code) {
#if defined(_WIN32)
    char buffer[256] = {};
    return ::strerror_s(buffer, sizeof(buffer), code) == 0
               ? std::string(buffer)
               : std::string("errno ") + std::to_string(code);
#else
    return std::strerror(code);
#endif
}

// A fresh, empty directory under the system temporary directory, named for the caller and this
// process so that repeated calls and concurrent runs never share a path. std::filesystem spells the
// separators the way the platform expects, so callers only ever append with operator/.
inline std::filesystem::path temporary_directory(std::string_view name) {
    static unsigned counter = 0;
    const auto directory =
        std::filesystem::temp_directory_path() /
        (std::string(name) + "-" + std::to_string(process_id()) + "-" + std::to_string(counter++));
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory);
    return directory;
}

// The directory that the dumps the model tests write for the tools/verify reference scripts live
// in. Those scripts hardcode /tmp, which is the documented path on POSIX; Windows has no /tmp and
// uses the process temporary directory instead.
inline std::filesystem::path dump_directory() {
#if defined(_WIN32)
    return std::filesystem::temp_directory_path();
#else
    return std::filesystem::path("/tmp");
#endif
}

} // namespace ninfer::test
