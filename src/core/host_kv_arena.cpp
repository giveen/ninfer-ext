#include "core/host_kv_arena.h"

#include "core/dtype.h"
#include "core/kv_page_ref.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

// Arena chunks start on Huge-page boundaries when that is cheap.
constexpr std::size_t kHostKVBandAlignment = std::size_t{2} << 20U;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(std::string(label) + " alignment overflow");
    }
    return (value + mask) & ~mask;
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

} // namespace

HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry) {
    if (geometry.page_tokens == 0 || geometry.planes.empty()) {
        throw std::invalid_argument("Host KV page geometry is empty");
    }
    const std::size_t layer_planes =
        geometry.layer_planes == 0 ? geometry.planes.size() : geometry.layer_planes;
    if (geometry.planes.size() % layer_planes != 0) {
        throw std::invalid_argument("Host KV planes do not divide into whole layers");
    }

    HostKVPageLayout out;
    out.geometry     = geometry;
    out.layer_planes = static_cast<std::uint32_t>(layer_planes);
    out.layers       = static_cast<std::uint32_t>(geometry.planes.size() / layer_planes);
    out.planes.reserve(geometry.planes.size());
    std::size_t cursor = 0;
    for (std::size_t index = 0; index < geometry.planes.size(); ++index) {
        const KVPlaneGeometry& plane = geometry.planes[index];
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Host KV plane geometry must be positive");
        }
        if (index % layer_planes == 0) { cursor = 0; }
        // Device slab alignment is not part of the canonical packed Host representation.
        cursor = align_up(cursor, kHostKVPageUnitBytes, "Host KV plane");
        const std::size_t head_bytes =
            checked_mul(checked_mul(static_cast<std::size_t>(plane.leading_extent),
                                    geometry.page_tokens, "Host KV head payload overflow"),
                        dtype_size(plane.dtype), "Host KV head payload overflow");
        const std::size_t page_bytes =
            checked_mul(head_bytes, static_cast<std::size_t>(plane.head_extent),
                        "Host KV plane payload overflow");
        out.planes.push_back(HostKVPlaneLayout{
            .offset             = cursor,
            .page_payload_bytes = page_bytes,
            .head_payload_bytes = head_bytes,
        });
        cursor = checked_add(cursor, page_bytes, "Host KV page payload overflow");
        out.layer_span =
            std::max(out.layer_span, align_up(cursor, kHostKVPageUnitBytes, "Host KV layer span"));
    }
    out.page_stride = checked_mul(out.layer_span, out.layers, "Host KV page record overflow");
    return out;
}

TransferWork plan_host_kv_transfer_work(const HostKVPageLayout& layout, std::uint32_t pages,
                                        std::uint32_t contiguous_runs) {
    if (pages == 0) { return {}; }
    if (contiguous_runs == 0 || contiguous_runs > pages || layout.planes.empty() ||
        layout.planes.size() != layout.geometry.planes.size()) {
        throw std::invalid_argument("Host KV transfer geometry is invalid");
    }

    std::size_t bytes_per_page     = 0;
    std::size_t operations_per_run = 0;
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        bytes_per_page = checked_add(bytes_per_page, layout.planes[index].page_payload_bytes,
                                     "Host KV transfer payload overflow");
        const KVPlaneGeometry& plane = layout.geometry.planes[index];
        const std::size_t operations =
            layout.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor
                ? 1U
                : static_cast<std::size_t>(plane.head_extent);
        operations_per_run = checked_add(operations_per_run, operations,
                                         "Host KV transfer operation count overflow");
    }

    const std::size_t payload =
        checked_mul(bytes_per_page, pages, "Host KV transfer payload overflow");
    const std::size_t operations = checked_mul(operations_per_run, contiguous_runs,
                                               "Host KV transfer operation count overflow");
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork plan_device_kv_copy_work(const HostKVPageLayout& layout, std::uint32_t pages) {
    if (pages == 0) { return {}; }
    if (layout.planes.empty() || layout.planes.size() != layout.geometry.planes.size()) {
        throw std::invalid_argument("Device KV copy geometry is invalid");
    }

    std::size_t bytes_per_page = 0;
    for (const HostKVPlaneLayout& plane : layout.planes) {
        bytes_per_page = checked_add(bytes_per_page, plane.page_payload_bytes,
                                     "Device KV copy payload overflow");
    }
    const std::size_t payload =
        checked_mul(bytes_per_page, pages, "Device KV copy payload overflow");
    const std::size_t operations =
        checked_mul(layout.planes.size(), pages, "Device KV copy operation count overflow");
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Device KV copy operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

bool HostKVAllocationView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationView HostKVAllocationView::subview(std::uint32_t begin, std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationView(handle_, region_, layout_, chunk_bytes_, chunk_pages_,
                                first_slot_ + begin, count);
}

std::byte* HostKVAllocationView::plane_page(std::size_t plane, std::uint32_t page) const {
    if (!valid() || plane >= layout_->planes.size() || page >= page_count_) {
        throw std::out_of_range("Host KV plane page is outside its allocation");
    }
    const std::uint32_t slot = first_slot_ + page;
    return region_ + static_cast<std::size_t>(slot / chunk_pages_) * chunk_bytes_ +
           (static_cast<std::size_t>(layout_->layer_of(plane)) * chunk_pages_ +
            slot % chunk_pages_) *
               layout_->layer_span +
           layout_->planes[plane].offset;
}

std::uint32_t HostKVAllocationView::contiguous_pages(std::uint32_t page) const {
    if (!valid() || page >= page_count_) {
        throw std::out_of_range("Host KV page is outside its allocation");
    }
    return std::min(page_count_ - page, chunk_pages_ - (first_slot_ + page) % chunk_pages_);
}


bool HostKVAllocationConstView::valid() const noexcept {
    return handle_.owner_ != nullptr && handle_.owner_->valid_handle(handle_);
}

const HostKVPageLayout& HostKVAllocationConstView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationConstView HostKVAllocationConstView::subview(std::uint32_t begin,
                                                             std::uint32_t count) const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    HostKVAllocationConstView out = *this;
    out.first_slot_ += begin;
    out.page_count_ = count;
    return out;
}

const std::byte* HostKVAllocationConstView::plane_page(std::size_t plane,
                                                       std::uint32_t page) const {
    if (!valid() || plane >= layout_->planes.size() || page >= page_count_) {
        throw std::out_of_range("Host KV plane page is outside its allocation");
    }
    const std::uint32_t slot = first_slot_ + page;
    return region_ + static_cast<std::size_t>(slot / chunk_pages_) * chunk_bytes_ +
           (static_cast<std::size_t>(layout_->layer_of(plane)) * chunk_pages_ +
            slot % chunk_pages_) *
               layout_->layer_span +
           layout_->planes[plane].offset;
}

std::uint32_t HostKVAllocationConstView::contiguous_pages(std::uint32_t page) const {
    if (!valid() || page >= page_count_) {
        throw std::out_of_range("Host KV page is outside its allocation");
    }
    return std::min(page_count_ - page, chunk_pages_ - (first_slot_ + page) % chunk_pages_);
}


HostKVAllocation::~HostKVAllocation() { (void)release(); }

HostKVAllocation::HostKVAllocation(HostKVAllocation&& other) noexcept
    : owner_(other.owner_), descriptor_(other.descriptor_), generation_(other.generation_) {
    other.disarm();
}

HostKVAllocation& HostKVAllocation::operator=(HostKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_      = other.owner_;
    descriptor_ = other.descriptor_;
    generation_ = other.generation_;
    other.disarm();
    return *this;
}

HostKVAllocationHandle HostKVAllocation::handle() const noexcept {
    return valid() ? HostKVAllocationHandle(owner_, descriptor_, generation_)
                   : HostKVAllocationHandle();
}

std::uint32_t HostKVAllocation::page_count() const noexcept {
    if (!valid() || descriptor_ >= owner_->descriptors_.size()) { return 0; }
    const HostKVArena::Descriptor& descriptor = owner_->descriptors_[descriptor_];
    return descriptor.active && descriptor.generation == generation_ ? descriptor.pages : 0;
}

bool HostKVAllocation::release() noexcept {
    if (!valid()) { return false; }
    const bool released = owner_->release_descriptor(descriptor_, generation_);
    disarm();
    return released;
}

void HostKVAllocation::disarm() noexcept {
    owner_      = nullptr;
    descriptor_ = 0;
    generation_ = 0;
}

HostKVArena::HostKVArena(std::size_t capacity_bytes,
                         std::span<const HostKVPageLayout> supported_layouts,
                         std::uint32_t chunk_pages)
    : layouts_(supported_layouts.begin(), supported_layouts.end()) {
    for (std::size_t index = 0; index < layouts_.size(); ++index) {
        const HostKVPageLayout planned = plan_host_kv_page_layout(layouts_[index].geometry);
        if (planned != layouts_[index]) {
            throw std::invalid_argument("Host KV arena received an inconsistent page layout");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (layouts_[previous] == layouts_[index]) {
                throw std::invalid_argument("Host KV arena contains a duplicate page layout");
            }
        }
    }
    if (chunk_pages == 0) { throw std::invalid_argument("Host KV arena chunks must hold pages"); }
    if (capacity_bytes == 0) { return; }
    if (layouts_.empty()) {
        throw std::invalid_argument("Non-empty Host KV arena requires supported page layouts");
    }

    std::size_t stride_sum = 0;
    for (const HostKVPageLayout& layout : layouts_) {
        stride_sum = checked_add(stride_sum, layout.page_stride, "Host KV arena stride overflow");
    }
    // An arena smaller than one chunk is a single chunk of the pages it fits.
    const std::size_t packed_slots = capacity_bytes / stride_sum;
    if (packed_slots == 0) { return; }
    chunk_pages_ = static_cast<std::uint32_t>(std::min<std::size_t>(chunk_pages, packed_slots));
    layout_offsets_.reserve(layouts_.size());
    for (const HostKVPageLayout& layout : layouts_) {
        layout_offsets_.push_back(chunk_bytes_);
        chunk_bytes_ = checked_add(
            chunk_bytes_, checked_mul(layout.page_stride, chunk_pages_, "Host KV chunk overflow"),
            "Host KV chunk overflow");
    }
    // Huge-page aligned chunks when that costs at most 1/16 of a chunk.
    const std::size_t aligned = align_up(chunk_bytes_, kHostKVBandAlignment, "Host KV chunk");
    if ((aligned - chunk_bytes_) * 16U <= chunk_bytes_ && aligned <= capacity_bytes) {
        chunk_bytes_ = aligned;
    }
    // Every Host word's unit, (chunk * chunk_bytes + in-chunk offset) / 256, fits KVPageRef.
    const std::size_t addressable =
        (static_cast<std::size_t>(kMaxHostPageUnit) + 1U) * kHostKVPageUnitBytes / chunk_bytes_;
    const std::size_t chunks = std::min(capacity_bytes / chunk_bytes_, addressable);
    const std::size_t slots  = checked_mul(chunks, chunk_pages_, "Host KV arena slots");
    if (slots == 0) { return; }
    if (slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV arena slot count exceeds uint32");
    }

    capacity_bytes_ = checked_mul(stride_sum, slots, "Host KV arena capacity overflow");
    backing_.emplace(checked_mul(chunks, chunk_bytes_, "Host KV arena overflow"),
                     PinnedHostPages::Huge);
    free_.resize(layouts_.size());
    for (std::vector<FreeRun>& free : free_) {
        free.reserve(slots / 2U + 1U);
        free.push_back(FreeRun{.slot = 0, .count = static_cast<std::uint32_t>(slots)});
    }

    const std::size_t maximum_descriptors = slots * layouts_.size();
    if (maximum_descriptors > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV arena descriptor capacity exceeds uint32");
    }
    descriptors_.resize(maximum_descriptors);
    free_descriptors_.reserve(maximum_descriptors);
    for (std::size_t index = maximum_descriptors; index > 0; --index) {
        free_descriptors_.push_back(static_cast<std::uint32_t>(index - 1));
    }
}

std::optional<std::uint32_t>
HostKVArena::find_layout(const HostKVPageLayout& layout) const noexcept {
    const auto it = std::ranges::find(layouts_, layout);
    if (it == layouts_.end()) { return std::nullopt; }
    return static_cast<std::uint32_t>(it - layouts_.begin());
}

const HostKVPageLayout* HostKVArena::layout_for(const KVPageGeometry& geometry) const noexcept {
    const auto layout = std::ranges::find_if(layouts_, [&](const HostKVPageLayout& candidate) {
        return candidate.geometry == geometry;
    });
    return layout == layouts_.end() ? nullptr : &*layout;
}

bool HostKVArena::take_run(std::vector<FreeRun>& free, std::uint32_t pages,
                           std::uint32_t& slot) noexcept {
    const auto run = std::ranges::find_if(
        free, [&](const FreeRun& candidate) { return candidate.count >= pages; });
    if (run == free.end()) { return false; }
    slot = run->slot;
    run->slot += pages;
    run->count -= pages;
    if (run->count == 0) { free.erase(run); }
    return true;
}

void HostKVArena::insert_free_run(std::vector<FreeRun>& free, FreeRun run) noexcept {
    const auto position = std::ranges::lower_bound(free, run.slot, {}, &FreeRun::slot);
    auto inserted       = free.insert(position, run);
    if (inserted != free.begin()) {
        auto previous = inserted - 1;
        if (previous->slot + previous->count == inserted->slot) {
            previous->count += inserted->count;
            free.erase(inserted);
            inserted = previous;
        }
    }
    const auto next = inserted + 1;
    if (next != free.end() && inserted->slot + inserted->count == next->slot) {
        inserted->count += next->count;
        free.erase(next);
    }
}

bool HostKVArena::can_allocate(const HostKVPageLayout& layout, std::uint32_t pages) const noexcept {
    const std::optional<std::uint32_t> layout_index = find_layout(layout);
    if (pages == 0 || free_descriptors_.empty() || !layout_index || free_.empty()) { return false; }
    return std::ranges::any_of(free_[*layout_index],
                               [&](const FreeRun& run) { return run.count >= pages; });
}

std::optional<HostKVAllocation> HostKVArena::allocate(const HostKVPageLayout& layout,
                                                      std::uint32_t pages) noexcept {
    const std::optional<std::uint32_t> layout_index = find_layout(layout);
    if (!layout_index || pages == 0 || free_descriptors_.empty() || free_.empty()) {
        return std::nullopt;
    }
    std::uint32_t slot = 0;
    if (!take_run(free_[*layout_index], pages, slot)) { return std::nullopt; }

    const std::uint32_t descriptor_index = take_descriptor();
    Descriptor& descriptor               = descriptors_[descriptor_index];
    descriptor.slot                      = slot;
    descriptor.layout                    = *layout_index;
    descriptor.pages                     = pages;
    descriptor.active                    = true;
    occupied_bytes_ += descriptor_bytes(descriptor);
    bump_revision();
    return HostKVAllocation(*this, descriptor_index, descriptor.generation);
}

std::optional<HostKVAllocationRecipe> HostKVArena::plan_after_releases(
    std::span<const HostKVAllocationHandle> proposed_releases,
    std::span<const HostKVAllocationRequest> target_allocations) const {
    if (proposed_releases.empty() && target_allocations.empty()) { return std::nullopt; }
    if (target_allocations.size() > free_descriptors_.size() + proposed_releases.size()) {
        return std::nullopt;
    }

    HostKVAllocationRecipe recipe;
    recipe.owner_          = this;
    recipe.arena_revision_ = revision_;
    recipe.releases_.reserve(proposed_releases.size());
    recipe.targets_.reserve(target_allocations.size());

    FreeLists simulated = free_;
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocationHandle handle = proposed_releases[index];
        if (!valid_handle(handle) ||
            std::find(proposed_releases.begin(),
                      proposed_releases.begin() + static_cast<std::ptrdiff_t>(index),
                      handle) != proposed_releases.begin() + static_cast<std::ptrdiff_t>(index)) {
            return std::nullopt;
        }
        const Descriptor& descriptor = descriptors_[handle.descriptor_];
        insert_free_run(simulated[descriptor.layout], {descriptor.slot, descriptor.pages});
        recipe.releases_.push_back(handle);
    }

    for (const HostKVAllocationRequest& request : target_allocations) {
        if (request.layout == nullptr || request.pages == 0) { return std::nullopt; }
        const std::optional<std::uint32_t> layout_index = find_layout(*request.layout);
        std::uint32_t slot                              = 0;
        if (!layout_index || simulated.empty() ||
            !take_run(simulated[*layout_index], request.pages, slot)) {
            return std::nullopt;
        }
        recipe.targets_.push_back(HostKVAllocationRecipe::Target{
            .layout = *layout_index,
            .pages  = request.pages,
            .slot   = slot,
        });
    }
    return recipe;
}

bool HostKVArena::can_allocate_after_suballocation_releases(
    std::span<const HostKVSuballocationRelease> proposed_releases,
    std::span<const HostKVAllocationRequest> target_allocations) const {
    if (proposed_releases.empty() && target_allocations.empty()) { return true; }

    FreeLists simulated = free_;
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVSuballocationRelease& release = proposed_releases[index];
        if (!valid_handle(release.allocation) || release.page_count == 0) { return false; }
        const Descriptor& descriptor = descriptors_[release.allocation.descriptor_];
        if (release.begin_page > descriptor.pages ||
            release.page_count > descriptor.pages - release.begin_page) {
            return false;
        }
        const std::uint32_t end = release.begin_page + release.page_count;
        for (std::size_t prior = 0; prior < index; ++prior) {
            const HostKVSuballocationRelease& other = proposed_releases[prior];
            if (other.allocation != release.allocation) { continue; }
            const std::uint32_t other_end = other.begin_page + other.page_count;
            if (release.begin_page < other_end && other.begin_page < end) { return false; }
        }
        insert_free_run(simulated[descriptor.layout],
                        {descriptor.slot + release.begin_page, release.page_count});
    }

    std::size_t available_descriptors = free_descriptors_.size();
    std::size_t required_descriptors  = target_allocations.size();
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocationHandle allocation = proposed_releases[index].allocation;
        bool first                              = true;
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (proposed_releases[prior].allocation == allocation) {
                first = false;
                break;
            }
        }
        if (!first) { continue; }

        std::vector<std::pair<std::uint32_t, std::uint32_t>> intervals;
        for (const HostKVSuballocationRelease& release : proposed_releases) {
            if (release.allocation == allocation) {
                intervals.emplace_back(release.begin_page, release.begin_page + release.page_count);
            }
        }
        std::ranges::sort(intervals);
        const Descriptor& descriptor = descriptors_[allocation.descriptor_];
        std::uint32_t cursor         = 0;
        std::size_t retained_runs    = 0;
        for (const auto [begin, end] : intervals) {
            if (begin > cursor) { ++retained_runs; }
            cursor = end;
        }
        if (cursor < descriptor.pages) { ++retained_runs; }
        if (retained_runs == 0) {
            ++available_descriptors;
        } else if (retained_runs > 1) {
            required_descriptors += retained_runs - 1U;
        }
    }
    if (required_descriptors > available_descriptors) { return false; }

    for (const HostKVAllocationRequest& request : target_allocations) {
        if (request.layout == nullptr || request.pages == 0) { return false; }
        const std::optional<std::uint32_t> layout_index = find_layout(*request.layout);
        std::uint32_t slot                              = 0;
        if (!layout_index || simulated.empty() ||
            !take_run(simulated[*layout_index], request.pages, slot)) {
            return false;
        }
    }
    return true;
}

bool HostKVArena::apply_recipe(HostKVAllocationRecipe&& recipe,
                               std::span<HostKVAllocation* const> proposed_releases,
                               std::span<HostKVAllocation> target_allocations) noexcept {
    if (recipe.owner_ != this || recipe.arena_revision_ != revision_ ||
        recipe.releases_.size() != proposed_releases.size() ||
        recipe.targets_.size() != target_allocations.size()) {
        return false;
    }
    for (std::size_t index = 0; index < proposed_releases.size(); ++index) {
        const HostKVAllocation* allocation = proposed_releases[index];
        if (allocation == nullptr || allocation->handle() != recipe.releases_[index] ||
            !valid_handle(recipe.releases_[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < target_allocations.size(); ++index) {
        const HostKVAllocationRecipe::Target& target = recipe.targets_[index];
        if (target_allocations[index].valid() || target.layout >= layouts_.size() ||
            target.pages == 0) {
            return false;
        }
    }

    // All generations and outputs are validated before the first mutation. The recipe was minted
    // from this exact revision, so every operation below is an invariant-preserving adoption.
    for (HostKVAllocation* allocation : proposed_releases) {
        if (!allocation->release()) { std::terminate(); }
    }
    for (std::size_t index = 0; index < recipe.targets_.size(); ++index) {
        const HostKVAllocationRecipe::Target& target = recipe.targets_[index];
        std::optional<HostKVAllocation> allocation =
            allocate(layouts_[target.layout], target.pages);
        if (!allocation || descriptors_[allocation->descriptor_].slot != target.slot) {
            std::terminate();
        }
        target_allocations[index] = std::move(*allocation);
    }
    recipe.owner_          = nullptr;
    recipe.arena_revision_ = 0;
    recipe.releases_.clear();
    recipe.targets_.clear();
    return true;
}

std::pair<HostKVAllocation, HostKVAllocation> HostKVArena::split(HostKVAllocation&& allocation,
                                                                 std::uint32_t page_offset) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot split a stale Host KV allocation");
    }
    Descriptor& original = descriptors_[allocation.descriptor_];
    if (page_offset == 0 || page_offset >= original.pages) {
        throw std::out_of_range("Host KV split must leave two non-empty allocations");
    }
    const std::uint32_t right_index = take_descriptor();
    if (right_index == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("Host KV descriptor capacity invariant was violated");
    }

    Descriptor& right = descriptors_[right_index];
    right.slot        = original.slot + page_offset;
    right.layout      = original.layout;
    right.pages       = original.pages - page_offset;
    right.active      = true;

    increment_generation(original.generation);
    original.pages                       = page_offset;
    const std::uint32_t left_generation  = original.generation;
    const std::uint32_t right_generation = right.generation;
    const std::uint32_t left_index       = allocation.descriptor_;
    allocation.disarm();
    bump_revision();
    return {HostKVAllocation(*this, left_index, left_generation),
            HostKVAllocation(*this, right_index, right_generation)};
}

HostKVAllocationView HostKVArena::writable_view(HostKVAllocation& allocation) {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    return make_view(allocation.handle(), descriptors_[allocation.descriptor_]);
}

HostKVAllocationConstView HostKVArena::view(const HostKVAllocation& allocation) const {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    return make_view(allocation.handle(), descriptors_[allocation.descriptor_]);
}

HostKVAllocationView HostKVArena::make_view(HostKVAllocationHandle handle,
                                            const Descriptor& descriptor) const noexcept {
    return HostKVAllocationView(handle, region_data(descriptor.layout),
                                &layouts_[descriptor.layout], chunk_bytes_, chunk_pages_,
                                descriptor.slot, descriptor.pages);
}

const std::byte* HostKVArena::plane_base(const HostKVPageLayout& layout, std::size_t plane) const {
    const std::optional<std::uint32_t> index = find_layout(layout);
    if (!index || plane >= layout.planes.size()) {
        throw std::invalid_argument("Host KV plane is not part of a supported layout");
    }
    if (!backing_) { return nullptr; }
    return region_data(*index) +
           static_cast<std::size_t>(layout.layer_of(plane)) * chunk_pages_ * layout.layer_span +
           layout.planes[plane].offset;
}

HostPageUnit HostKVArena::page_unit(const HostKVAllocation& allocation, std::uint32_t page) const {
    if (!valid_handle(allocation.handle())) {
        throw std::invalid_argument("Cannot address a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    if (page >= descriptor.pages) {
        throw std::out_of_range("Host KV page is outside its allocation");
    }
    // The constructor bounds the chunks so every unit fits a block-table word.
    const std::uint32_t slot = descriptor.slot + page;
    const std::size_t offset =
        static_cast<std::size_t>(slot / chunk_pages_) * chunk_bytes_ +
        static_cast<std::size_t>(slot % chunk_pages_) * layouts_[descriptor.layout].layer_span;
    return HostPageUnit{static_cast<std::uint32_t>(offset / kHostKVPageUnitBytes)};
}

bool HostKVArena::valid_handle(HostKVAllocationHandle handle) const noexcept {
    if (handle.owner_ != this || handle.descriptor_ >= descriptors_.size()) { return false; }
    const Descriptor& descriptor = descriptors_[handle.descriptor_];
    return descriptor.active && descriptor.generation == handle.generation_;
}

std::uint32_t HostKVArena::take_descriptor() noexcept {
    if (free_descriptors_.empty()) { return std::numeric_limits<std::uint32_t>::max(); }
    const std::uint32_t out = free_descriptors_.back();
    free_descriptors_.pop_back();
    return out;
}

bool HostKVArena::release_descriptor(std::uint32_t descriptor_index,
                                     std::uint32_t generation) noexcept {
    if (descriptor_index >= descriptors_.size()) { return false; }
    Descriptor& descriptor = descriptors_[descriptor_index];
    if (!descriptor.active || descriptor.generation != generation) { return false; }

    occupied_bytes_ -= descriptor_bytes(descriptor);
    insert_free_run(free_[descriptor.layout], {descriptor.slot, descriptor.pages});
    descriptor.active = false;
    descriptor.slot   = 0;
    descriptor.pages  = 0;
    increment_generation(descriptor.generation);
    free_descriptors_.push_back(descriptor_index);
    bump_revision();
    return true;
}

std::byte* HostKVArena::region_data(std::uint32_t layout) const noexcept {
    if (!backing_) { return nullptr; }
    return static_cast<std::byte*>(backing_->data()) + layout_offsets_[layout];
}

std::size_t HostKVArena::descriptor_bytes(const Descriptor& descriptor) const noexcept {
    return static_cast<std::size_t>(descriptor.pages) * layouts_[descriptor.layout].page_stride;
}

void HostKVArena::bump_revision() noexcept {
    ++revision_;
    if (revision_ == 0) { ++revision_; }
}

} // namespace ninfer
