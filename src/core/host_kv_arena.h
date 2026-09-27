#pragma once

#include "core/arena.h"
#include "core/kv_page_ref.h"
#include "core/paged_kv_cache.h"
#include "core/transfer_work.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

struct HostKVPlaneLayout {
    std::size_t offset             = 0; // offset inside the page's layer span
    std::size_t page_payload_bytes = 0;
    std::size_t head_payload_bytes = 0;

    friend bool operator==(const HostKVPlaneLayout&, const HostKVPlaneLayout&) = default;
};

// Host representation of one logical KV page, layer-major. Every layer of a page is a span of that
// layer's planes (each keeping the Device in-page element order, so in-page offsets match the
// Device arm); within a HostKVArena chunk, consecutive pages of one layer are adjacent, so a
// layer's Host pages form contiguous runs that paged Ops read in place and copies move in one
// piece.
struct HostKVPageLayout {
    KVPageGeometry geometry;
    std::vector<HostKVPlaneLayout> planes; // offsets repeat per layer
    std::uint32_t layer_planes = 0;
    std::uint32_t layers       = 0;
    std::size_t layer_span     = 0; // bytes of one page of one layer, a 256-byte multiple
    std::size_t page_stride    = 0; // Host bytes of one page, all layers

    [[nodiscard]] std::uint32_t layer_of(std::size_t plane) const noexcept {
        return static_cast<std::uint32_t>(plane / layer_planes);
    }

    friend bool operator==(const HostKVPageLayout&, const HostKVPageLayout&) = default;
};

[[nodiscard]] HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry);

[[nodiscard]] TransferWork plan_host_kv_transfer_work(const HostKVPageLayout& layout,
                                                      std::uint32_t pages,
                                                      std::uint32_t contiguous_runs);
[[nodiscard]] TransferWork plan_device_kv_copy_work(const HostKVPageLayout& layout,
                                                    std::uint32_t pages);

class HostKVArena;

class HostKVAllocationHandle {
public:
    HostKVAllocationHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVAllocationHandle,
                                         HostKVAllocationHandle) noexcept = default;

private:
    friend class HostKVArena;
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    HostKVAllocationHandle(const HostKVArena* owner, std::uint32_t descriptor,
                           std::uint32_t generation) noexcept
        : owner_(owner), descriptor_(descriptor), generation_(generation) {}

    const HostKVArena* owner_ = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

class HostKVAllocationView {
public:
    HostKVAllocationView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationView subview(std::uint32_t begin, std::uint32_t count) const;

    // Plane `plane` of page `page`. Pages [page, page + contiguous_pages(page)) keep each plane
    // at a fixed pitch of layout().layer_span; a chunk boundary of the arena ends such a run.
    [[nodiscard]] std::byte* plane_page(std::size_t plane, std::uint32_t page) const;
    [[nodiscard]] std::uint32_t contiguous_pages(std::uint32_t page) const;

private:
    friend class HostKVArena;
    friend class HostKVAllocationConstView;

    HostKVAllocationView(HostKVAllocationHandle handle, std::byte* region,
                         const HostKVPageLayout* layout, std::size_t chunk_bytes,
                         std::uint32_t chunk_pages, std::uint32_t first_slot,
                         std::uint32_t page_count) noexcept
        : handle_(handle), region_(region), layout_(layout), chunk_bytes_(chunk_bytes),
          chunk_pages_(chunk_pages), first_slot_(first_slot), page_count_(page_count) {}

    HostKVAllocationHandle handle_;
    std::byte* region_              = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::size_t chunk_bytes_        = 0;
    std::uint32_t chunk_pages_      = 0;
    std::uint32_t first_slot_       = 0;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocationConstView {
public:
    HostKVAllocationConstView() noexcept = default;

    HostKVAllocationConstView(HostKVAllocationView view) noexcept
        : handle_(view.handle_), region_(view.region_), layout_(view.layout_),
          chunk_bytes_(view.chunk_bytes_), chunk_pages_(view.chunk_pages_),
          first_slot_(view.first_slot_), page_count_(view.page_count_) {}

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;
    [[nodiscard]] HostKVAllocationConstView subview(std::uint32_t begin, std::uint32_t count) const;

    [[nodiscard]] const std::byte* plane_page(std::size_t plane, std::uint32_t page) const;
    [[nodiscard]] std::uint32_t contiguous_pages(std::uint32_t page) const;

private:
    friend class HostKVArena;

    HostKVAllocationHandle handle_;
    const std::byte* region_        = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::size_t chunk_bytes_        = 0;
    std::uint32_t chunk_pages_      = 0;
    std::uint32_t first_slot_       = 0;
    std::uint32_t page_count_       = 0;
};

class HostKVAllocation {
public:
    HostKVAllocation() noexcept = default;
    ~HostKVAllocation();

    HostKVAllocation(const HostKVAllocation&)            = delete;
    HostKVAllocation& operator=(const HostKVAllocation&) = delete;
    HostKVAllocation(HostKVAllocation&& other) noexcept;
    HostKVAllocation& operator=(HostKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] HostKVAllocationHandle handle() const noexcept;
    [[nodiscard]] std::uint32_t page_count() const noexcept;
    bool release() noexcept;

private:
    friend class HostKVArena;

    HostKVAllocation(HostKVArena& owner, std::uint32_t descriptor,
                     std::uint32_t generation) noexcept
        : owner_(&owner), descriptor_(descriptor), generation_(generation) {}

    void disarm() noexcept;

    HostKVArena* owner_       = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

struct HostKVAllocationRequest {
    const HostKVPageLayout* layout = nullptr;
    std::uint32_t pages            = 0;
};

struct HostKVSuballocationRelease {
    HostKVAllocationHandle allocation;
    std::uint32_t begin_page = 0;
    std::uint32_t page_count = 0;
};

class HostKVAllocationRecipe {
public:
    HostKVAllocationRecipe() noexcept                                    = default;
    HostKVAllocationRecipe(HostKVAllocationRecipe&&) noexcept            = default;
    HostKVAllocationRecipe& operator=(HostKVAllocationRecipe&&) noexcept = default;

    HostKVAllocationRecipe(const HostKVAllocationRecipe&)            = delete;
    HostKVAllocationRecipe& operator=(const HostKVAllocationRecipe&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t release_count() const noexcept { return releases_.size(); }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return targets_.size(); }

private:
    struct Target {
        std::uint32_t layout = 0;
        std::uint32_t pages  = 0;
        std::uint32_t slot   = 0;
    };

    const HostKVArena* owner_     = nullptr;
    std::uint64_t arena_revision_ = 0;
    std::vector<HostKVAllocationHandle> releases_;
    std::vector<Target> targets_;

    friend class HostKVArena;
};

// Default pages per HostKVArena chunk: a multiple of the KV streaming demotion group.
inline constexpr std::uint32_t kHostKVChunkPages = 64;

// Pinned Host pages for every supported layout. Each layout owns an equal number of page slots
// (a page of one pool pairs with a page of another). Slots come in chunks of `chunk_pages`; chunk c
// holds every layout's pages of slots [c * chunk_pages, (c + 1) * chunk_pages), each layout
// layer-major: layer l of slot s is at chunk + layout offset + l * chunk_pages * layer_span +
// (s % chunk_pages) * layer_span. First-fit allocation fills chunks from the arena start, so live
// pages stay compact however large the arena is; spreading them over a large pinned buffer slows
// both copy-engine transfers and in-place PCIe reads. Allocations are slot ranges of one layout.
class HostKVArena {
public:
    HostKVArena(std::size_t capacity_bytes, std::span<const HostKVPageLayout> supported_layouts,
                std::uint32_t chunk_pages = kHostKVChunkPages);

    HostKVArena(const HostKVArena&)            = delete;
    HostKVArena& operator=(const HostKVArena&) = delete;
    HostKVArena(HostKVArena&&)                 = delete;
    HostKVArena& operator=(HostKVArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] const HostKVPageLayout* layout_for(const KVPageGeometry& geometry) const noexcept;

    [[nodiscard]] bool can_allocate(const HostKVPageLayout& layout,
                                    std::uint32_t pages) const noexcept;
    [[nodiscard]] std::optional<HostKVAllocation> allocate(const HostKVPageLayout& layout,
                                                           std::uint32_t pages) noexcept;

    [[nodiscard]] std::optional<HostKVAllocationRecipe>
    plan_after_releases(std::span<const HostKVAllocationHandle> proposed_releases,
                        std::span<const HostKVAllocationRequest> target_allocations) const;

    [[nodiscard]] bool can_allocate_after_suballocation_releases(
        std::span<const HostKVSuballocationRelease> proposed_releases,
        std::span<const HostKVAllocationRequest> target_allocations) const;

    // The caller supplies already-sized empty outputs so successful adoption cannot allocate.
    // A false return leaves the arena and every input allocation unchanged.
    [[nodiscard]] bool apply_recipe(HostKVAllocationRecipe&& recipe,
                                    std::span<HostKVAllocation* const> proposed_releases,
                                    std::span<HostKVAllocation> target_allocations) noexcept;

    [[nodiscard]] std::pair<HostKVAllocation, HostKVAllocation> split(HostKVAllocation&& allocation,
                                                                      std::uint32_t page_offset);

    [[nodiscard]] HostKVAllocationView writable_view(HostKVAllocation& allocation);
    [[nodiscard]] HostKVAllocationConstView view(const HostKVAllocation& allocation) const;

    /**
     * Base that block-table Host words of `plane` resolve against: plane `plane` of the page with
     * word unit u is at plane_base + u * 256 (KVPageRef). Null for an empty arena.
     */
    [[nodiscard]] const std::byte* plane_base(const HostKVPageLayout& layout,
                                              std::size_t plane) const;
    /** Block-table unit of page `page` of `allocation`, read in place by paged Ops (KVPageRef). */
    [[nodiscard]] HostPageUnit page_unit(const HostKVAllocation& allocation,
                                         std::uint32_t page) const;

private:
    friend class HostKVAllocation;
    friend class HostKVAllocationView;
    friend class HostKVAllocationConstView;

    struct FreeRun {
        std::uint32_t slot  = 0;
        std::uint32_t count = 0;
    };

    struct Descriptor {
        std::uint32_t slot       = 0;
        std::uint32_t layout     = 0;
        std::uint32_t pages      = 0;
        std::uint32_t generation = 1;
        bool active              = false;
    };

    using FreeLists = std::vector<std::vector<FreeRun>>;

    [[nodiscard]] std::optional<std::uint32_t>
    find_layout(const HostKVPageLayout& layout) const noexcept;
    [[nodiscard]] static bool take_run(std::vector<FreeRun>& free, std::uint32_t pages,
                                       std::uint32_t& slot) noexcept;
    static void insert_free_run(std::vector<FreeRun>& free, FreeRun run) noexcept;
    [[nodiscard]] bool valid_handle(HostKVAllocationHandle handle) const noexcept;
    [[nodiscard]] std::uint32_t take_descriptor() noexcept;
    bool release_descriptor(std::uint32_t descriptor, std::uint32_t generation) noexcept;
    [[nodiscard]] std::byte* region_data(std::uint32_t layout) const noexcept;
    [[nodiscard]] HostKVAllocationView make_view(HostKVAllocationHandle handle,
                                                 const Descriptor& descriptor) const noexcept;
    [[nodiscard]] std::size_t descriptor_bytes(const Descriptor& descriptor) const noexcept;
    void bump_revision() noexcept;

    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_ = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<HostKVPageLayout> layouts_;
    std::vector<std::size_t> layout_offsets_; // of each layout's part inside a chunk
    std::size_t chunk_bytes_   = 0;
    std::uint32_t chunk_pages_ = 0;
    FreeLists free_;
    std::vector<Descriptor> descriptors_;
    std::vector<std::uint32_t> free_descriptors_;
    std::uint64_t revision_ = 1;
};

} // namespace ninfer
