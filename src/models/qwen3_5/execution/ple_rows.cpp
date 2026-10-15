#include "models/qwen3_5/execution/ple_rows.h"

#include "core/host_worker_pool.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::uint64_t kPage       = 4096;
constexpr std::uint32_t kReaders    = 16; // outstanding direct reads
constexpr std::size_t kPagesPerTask = 8;

} // namespace

struct PleRowReader::File {
    std::filesystem::path path;
    int fd = -1;
    // Declared end of this volume, from the segments that live in it. A read that reaches
    // end-of-file before this offset means the volume is truncated or corrupt.
    std::uint64_t extent = 0;

    explicit File(std::filesystem::path p) : path(std::move(p)) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error(
                path.string() + ": cannot open for direct n-gram reads: " + std::strerror(errno));
        }
    }

    ~File() {
        if (fd >= 0) { ::close(fd); }
    }

    void read_page(std::uint64_t offset, std::byte* destination) const {
        std::size_t done = 0;
        while (done < kPage) {
            const ssize_t n =
                ::pread(fd, destination + done, kPage - done, static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) { continue; }
            if (n < 0) {
                throw std::runtime_error(path.string() +
                                         ": direct n-gram read failed: " + std::strerror(errno));
            }
            if (n == 0) {
                // Only the tail of the page past the volume's declared extent may be short. A zero
                // read inside that extent means the volume is truncated or corrupt, so reject it
                // instead of fabricating zeros.
                if (offset + done >= extent) {
                std::memset(destination + done, 0, kPage - done);
                return;
                }
                throw std::runtime_error(path.string() +
                                         ": direct n-gram read hit a hole inside the volume");
            }
            done += static_cast<std::size_t>(n);
        }
    }
};

PleRowReader::PleRowReader(std::span<const artifact::MappedObjectSegment> segments)
    : segments_(segments.begin(), segments.end()),
      workers_(std::make_unique<HostWorkerPool>(kReaders, 4096)) {
    if (segments_.empty()) { throw std::invalid_argument("n-gram reader has no segments"); }
    for (const auto& segment : segments_) {
        const auto it =
            std::ranges::find_if(files_, [&](const auto& f) { return f->path == segment.file; });
        const std::uint64_t end = segment.file_offset + segment.bytes;
        if (it != files_.end()) {
            segment_file_.push_back(static_cast<std::uint32_t>(it - files_.begin()));
            (*it)->extent = std::max((*it)->extent, end);
        } else {
            segment_file_.push_back(static_cast<std::uint32_t>(files_.size()));
            files_.push_back(std::make_unique<File>(segment.file));
            files_.back()->extent = end;
        }
    }
}

PleRowReader::~PleRowReader() {
    workers_.reset();
    std::free(staging_);
}

PleRowReader::Page PleRowReader::page_of(std::uint64_t object_offset, std::uint32_t bytes,
                                         std::uint32_t& pages, std::uint32_t& within) const {
    const auto it =
        std::upper_bound(segments_.begin(), segments_.end(), object_offset,
                         [](std::uint64_t offset, const artifact::MappedObjectSegment& s) {
                             return offset < s.object_offset;
                         });
    if (it == segments_.begin()) { throw std::out_of_range("n-gram offset is outside the table"); }
    const auto& segment        = *std::prev(it);
    const std::uint64_t inside = object_offset - segment.object_offset;
    if (inside + bytes > segment.bytes) {
        throw std::out_of_range("n-gram range crosses a volume segment");
    }
    const std::uint64_t file_offset = segment.file_offset + inside;
    const std::uint64_t first       = file_offset / kPage * kPage;
    pages  = static_cast<std::uint32_t>((file_offset + bytes - first + kPage - 1) / kPage);
    within = static_cast<std::uint32_t>(file_offset - first);
    return Page{segment_file_[static_cast<std::size_t>(it - segments_.begin()) - 1], first};
}

void PleRowReader::reserve(std::size_t pages) {
    if (pages <= staging_pages_) { return; }
    std::free(staging_);
    staging_       = static_cast<std::byte*>(std::aligned_alloc(kPage, pages * kPage));
    staging_pages_ = staging_ != nullptr ? pages : 0;
    if (staging_ == nullptr) { throw std::bad_alloc(); }
}

void PleRowReader::read(std::span<const Range> ranges) {
    pages_.clear();

    struct Placement {
        Page page;
        std::uint32_t within;
    };

    std::vector<Placement> placements(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        std::uint32_t count = 0, within = 0;
        const Page first = page_of(ranges[i].object_offset, ranges[i].bytes, count, within);
        placements[i]    = {first, within};
        for (std::uint32_t p = 0; p < count; ++p) {
            pages_.push_back(Page{first.file, first.offset + p * kPage});
        }
    }
    std::sort(pages_.begin(), pages_.end());
    pages_.erase(std::unique(pages_.begin(), pages_.end()), pages_.end());
    reserve(pages_.size());

    std::vector<std::future<void>> pending;
    for (std::size_t begin = 0; begin < pages_.size(); begin += kPagesPerTask) {
        const std::size_t end = std::min(pages_.size(), begin + kPagesPerTask);
        pending.push_back(workers_->submit([this, begin, end] {
            for (std::size_t p = begin; p < end; ++p) {
                files_[pages_[p].file]->read_page(pages_[p].offset, staging_ + p * kPage);
            }
        }));
    }
    // Every task is awaited before the first failure is rethrown: an abandoned task would still
    // be writing staging_ while the caller reacts to the failure.
    for (auto& task : pending) { task.wait(); }
    for (auto& task : pending) { task.get(); }

    // Consecutive pages of one range are adjacent in the sorted page list.
    located_.resize(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto it = std::lower_bound(pages_.begin(), pages_.end(), placements[i].page);
        located_[i] =
            staging_ + static_cast<std::size_t>(it - pages_.begin()) * kPage + placements[i].within;
    }
}

} // namespace ninfer::models::qwen3_5::execution
