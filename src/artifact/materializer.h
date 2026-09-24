#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct MaterializationPlan {
    // Page-locked host parents (read at materialization) and lazily mapped file objects.
    std::vector<ObjectHandle> pinned_objects;
    std::vector<ObjectHandle> file_objects;
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t pinned_host_bytes     = 0;
    std::uint64_t mapped_file_bytes     = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
};

/** One read-only mapped piece of a file object, addressed by its object-relative offset. */
struct MappedObjectSegment {
    std::uint64_t object_offset = 0;
    std::uint64_t bytes         = 0;
    const std::byte* data       = nullptr;
};

class FileMapping;

class MaterializedArtifact {
public:
    MaterializedArtifact();
    ~MaterializedArtifact();
    MaterializedArtifact(MaterializedArtifact&&) noexcept;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept;
    MaterializedArtifact(const MaterializedArtifact&)            = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&) = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;
    [[nodiscard]] const WeightParent& pinned_parent(ObjectHandle handle) const;
    // Segments of a HostFile object in object-offset order; host access only.
    [[nodiscard]] std::span<const MappedObjectSegment> file_segments(ObjectHandle handle) const;
    [[nodiscard]] const WeightGeometry& file_geometry(ObjectHandle handle) const;

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
        std::optional<WeightParent> pinned;
        std::optional<PinnedHostBuffer> pinned_data;
        std::vector<MappedObjectSegment> file;
        std::optional<WeightGeometry> file_geometry;
    };

    std::unique_ptr<DeviceArena> arena_;
    std::vector<std::unique_ptr<FileMapping>> mappings_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
