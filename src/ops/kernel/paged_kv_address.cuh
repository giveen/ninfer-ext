#pragma once

#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

struct PagedKVDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct PagedKVBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

__device__ __forceinline__ std::int32_t paged_kv_physical_page(const std::int32_t* block_table,
                                                               std::int32_t position) {
    return block_table[position >> kPagedKVPageShift];
}

// Read-only page base of one page-major plane for a block-table word: the Device page group of
// `device`, or the Host page record read in place from `host` (KVPageRef). In-page offsets are the
// same on both arms, so a caller resolves once per page and indexes with
// paged_kv_in_page_offset. Only the address differs, never the arithmetic that consumes it.
template <typename T>
__device__ __forceinline__ const T* paged_kv_read_page(const T* device, const std::byte* host,
                                                       std::int32_t word,
                                                       std::int64_t page_elements) {
    if (word >= 0) { return device + page_elements * word; }
    return reinterpret_cast<const T*>(host + KVPageRef::from_word(word).host_offset_bytes());
}

template <int LeadingExtent, int HeadExtent, typename T>
__device__ __forceinline__ const T* paged_kv_read_page(const T* device, const std::byte* host,
                                                       std::int32_t word) {
    return paged_kv_read_page(device, host, word,
                              static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
                                  HeadExtent);
}

template <int LeadingExtent>
__device__ __forceinline__ std::int64_t
paged_kv_in_page_offset(std::int32_t head, std::int32_t page_offset, std::int32_t leading) {
    return static_cast<std::int64_t>(LeadingExtent) *
               (static_cast<std::int64_t>(kPagedKVPageSize) * head + page_offset) +
           leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(const std::int32_t* block_table, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(block_table, position), head, position & kPagedKVPageMask, leading);
}

} // namespace ninfer::ops
