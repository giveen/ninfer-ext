#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#    include <sys/types.h>
#endif

namespace ninfer::artifact {
namespace {

// The seam reports every failure as std::system_error with the platform's error category, whose
// what() reads "<operation>: <platform message>". Prefixing the path keeps the shape this file has
// always raised, "<path>: <operation>: <platform message>"; the operation name is now the seam's
// ("open", "read", "size", "stat") and the message is the platform's, exactly as strerror gave it
// on Linux.
[[noreturn]] void fail(const std::filesystem::path& path, const std::system_error& error) {
    throw ArtifactError(path.string() + ": " + error.what());
}

template <class Function>
auto attempt(const std::filesystem::path& path, Function&& function) -> decltype(function()) {
    try {
        return function();
    } catch (const std::system_error& error) { fail(path, error); }
}

// Positioned reads carry the offset in the platform's own offset type; a POSIX one is an off_t, so
// a larger offset cannot be expressed and is refused rather than truncated. Both callers reject an
// offset past the file's end first, which no file can exceed.
std::uint64_t file_offset(std::uint64_t offset) {
#if !defined(_WIN32)
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
#endif
    return offset;
}

// The longest request one positioned read may carry: pread takes an ssize_t, ReadFile a DWORD.
#if defined(_WIN32)
constexpr std::size_t kMaxReadBytes = std::numeric_limits<std::uint32_t>::max();
#else
constexpr std::size_t kMaxReadBytes = static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    // A constructor that throws never runs this object's destructor, so every failure below
    // releases the handle itself. close_file ignores the invalid handle of a failed open.
    const auto abandon = [this]() noexcept {
        core::close_file(fd_);
        fd_ = core::invalid_native_file_handle();
    };
    try {
        fd_    = core::open_read_only(path_, /*unbuffered=*/false);
        bytes_ = core::file_bytes(fd_);
        if (!core::is_regular_file(fd_)) {
            throw ArtifactError(path_.string() + ": expected a regular file");
        }
    } catch (const std::system_error& error) {
        abandon();
        fail(path_, error);
    } catch (...) {
        abandon();
        throw;
    }
}

InputFile::~InputFile() {
    core::close_file(direct_fd_);
    core::close_file(fd_);
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = attempt(path_, [this, offset, count, destination] {
            return core::read_at(fd_, file_offset(offset), destination.first(count));
        });
        // The seam reports 0 only at end of file, and this range was checked against the size, so a
        // zero read here means the file shrank underneath the reader.
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    // A direct window is a whole number of kPayloadAlignment blocks, so the file's final block may
    // be requested past its end: the caller rounds its last request up and consumes the short
    // count. That contract is the same on both platforms; what differs is the mechanism behind the
    // unbuffered handle opened below. POSIX O_DIRECT and Windows FILE_FLAG_NO_BUFFERING both bypass
    // the cache and require the offset, the length and the buffer to respect the device's block
    // size, which on Windows is the volume's sector size rather than a page size, so
    // kPayloadAlignment has to stay a multiple of it. core::read_at reports 0 at end of file and a
    // short count is legal only for that final block; this read is one request, so a short count
    // comes straight back to the caller.
    if (offset > bytes_ ||
        destination.size() > align_up(bytes_ - offset, kPayloadAlignment, "direct read bytes")) {
        throw ArtifactError(path_.string() + ": direct read exceeds file length");
    }
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > kMaxReadBytes) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    std::call_once (direct_open_once_, [this] {
        // A failed open leaves the once flag unset, so a later direct read tries again, and a
        // successful one is never repeated.
        direct_fd_ = attempt(path_, [this] { return core::open_read_only(path_, true); });
    });
    return attempt(path_, [this, offset, destination] {
        return core::read_at(direct_fd_, file_offset(offset), destination);
    });
}

} // namespace ninfer::artifact
