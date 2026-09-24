#pragma once

#include "artifact/materializer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {
class HostWorkerPool;
}

namespace ninfer::models::qwen3_5::execution {

// Batched direct (O_DIRECT) reads of byte ranges of one file object, bypassing the page cache.
// Every call gathers the distinct 4 KiB pages its ranges touch, reads them concurrently into
// private aligned staging and exposes each range from there until the next call.
class PleRowReader {
public:
    explicit PleRowReader(std::span<const artifact::MappedObjectSegment> segments);
    ~PleRowReader();

    PleRowReader(const PleRowReader&)            = delete;
    PleRowReader& operator=(const PleRowReader&) = delete;

    struct Range {
        std::uint64_t object_offset = 0;
        std::uint32_t bytes         = 0;
    };

    // Reads every range; afterwards `data(i)` addresses range i. Ranges may repeat or overlap.
    void read(std::span<const Range> ranges);

    [[nodiscard]] const std::byte* data(std::size_t range) const { return located_.at(range); }

private:
    struct File;

    struct Page {
        std::uint32_t file   = 0;
        std::uint64_t offset = 0; // page-aligned file offset

        friend bool operator<(const Page& a, const Page& b) {
            return a.file != b.file ? a.file < b.file : a.offset < b.offset;
        }

        friend bool operator==(const Page&, const Page&) = default;
    };

    [[nodiscard]] Page page_of(std::uint64_t object_offset, std::uint32_t bytes,
                               std::uint32_t& pages, std::uint32_t& within) const;
    void reserve(std::size_t pages);

    std::vector<artifact::MappedObjectSegment> segments_;
    std::vector<std::unique_ptr<File>> files_;
    std::vector<std::uint32_t> segment_file_;
    std::unique_ptr<HostWorkerPool> workers_;
    std::byte* staging_        = nullptr;
    std::size_t staging_pages_ = 0;
    std::vector<Page> pages_;
    std::vector<const std::byte*> located_;
};

} // namespace ninfer::models::qwen3_5::execution
