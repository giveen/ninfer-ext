#pragma once

#include "core/native_file.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>

namespace ninfer::artifact {

// Read-only artifact file. Both handles are read positionally, so one InputFile may be read from
// several threads at once: the materializer fans weight reads out over std::async workers and the
// row reader keeps sixteen reads outstanding.
// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    // Concurrent calls are safe: neither read moves a shared file position.
    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    core::NativeFileHandle fd_ = core::invalid_native_file_handle();
    // Opened once on the first direct read; guarded so concurrent read_direct calls are safe.
    mutable std::once_flag direct_open_once_;
    mutable core::NativeFileHandle direct_fd_ = core::invalid_native_file_handle();
    std::uint64_t bytes_   = 0;
};

} // namespace ninfer::artifact
