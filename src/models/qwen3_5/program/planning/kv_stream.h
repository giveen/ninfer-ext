#pragma once

// KV streaming geometry (docs/maintainer/paged-kv-cache.md §6.5). With streaming, an active
// address space keeps a bounded Device window of pages; older full pages move to Host records
// that paged Ops read in place, and their Device leases return to the same window.

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// Device KV is leased on demand. An active request holds a bounded window of its remaining
// output rather than the whole client budget, and extends that window at a decode-round
// boundary; a full window is requested first and a step-sized extension is enough when the pool
// cannot spare one.
inline constexpr std::uint32_t kKVLeaseGrowthMarginTokens = 4096;

// Lease cushion: one round's Backend requirement can sit a whole draft window above the frontier
// the previous round checked, so the cushion has to absorb that jump before the lease is extended
// again.
[[nodiscard]] constexpr std::uint32_t kv_lease_cushion_pages(std::uint32_t draft_window) noexcept {
    const auto page  = static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto slack = 2U * draft_window + 2U;
    return (slack + page - 1U) / page + 1U;
}

// Leading pages that never stream: attention sinks are read by every query.
inline constexpr std::uint32_t kKVStreamSinkPages = 1;
// Pages one demotion moves at once when the window allows it: one Host extent and one copy.
inline constexpr std::uint32_t kKVStreamGroupPages = 16;

[[nodiscard]] constexpr std::uint32_t kv_stream_pages(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U
                       : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// Pages at the frontier end that never stream: the partial writer tail and one speculative
// round that may still be rejected.
[[nodiscard]] constexpr std::uint32_t kv_stream_tail_pages(std::uint32_t draft_window) noexcept {
    return 1U + kv_stream_pages(draft_window + 1U);
}

// Smallest Device window, in page groups, that streaming can always serve: sinks, the tail, one
// demotion group, and the widest single mapping step (a prefill chunk or a lease growth window)
// with its cushion.
[[nodiscard]] constexpr std::uint32_t kv_stream_window_pages(std::uint32_t prefill_width,
                                                             std::uint32_t draft_window) noexcept {
    return kKVStreamSinkPages + kKVStreamGroupPages + kv_stream_tail_pages(draft_window) +
           kv_stream_pages(std::max(prefill_width, kKVLeaseGrowthMarginTokens)) +
           kv_lease_cushion_pages(draft_window);
}

} // namespace ninfer::models::qwen3_5::detail
