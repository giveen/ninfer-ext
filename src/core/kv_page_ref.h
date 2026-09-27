#pragma once

// Block-table word encoding shared by the KV Store (host) and paged Ops (device). A word names
// either a Device page group or a Host page record that kernels read in place over PCIe:
//
//   word >= 0 : DevicePageGroup(word)     plane page = plane base + word * plane page elements
//   word <  0 : HostPageUnit(~word)       plane page = host plane base + ~word * 256 bytes
//
// Host pages are layer-major (HostKVPageLayout): within a HostKVArena chunk one layer's pages are
// adjacent, each plane slice in the Device in-page element order, so in-page offsets are identical
// on both arms. The unit counts 256-byte steps from the plane's base; every layer and plane of a
// page sits at the same unit offset from its own base, so one word serves them all.

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define NINFER_KV_REF_HD __host__ __device__
#else
#define NINFER_KV_REF_HD
#endif

namespace ninfer {

/** Page group index inside one DeviceKVPagePool. */
enum class DevicePageGroup : std::int32_t {};

/** Offset of a Host page from its plane's HostKVArena base, in kHostKVPageUnitBytes. */
enum class HostPageUnit : std::uint32_t {};

inline constexpr std::size_t kHostKVPageUnitBytes = 256;
inline constexpr std::uint32_t kMaxHostPageUnit   = 0x7fffffffU;

class KVPageRef {
public:
    NINFER_KV_REF_HD constexpr explicit KVPageRef(DevicePageGroup group) noexcept
        : word_(static_cast<std::int32_t>(group)) {}
    NINFER_KV_REF_HD constexpr explicit KVPageRef(HostPageUnit unit) noexcept
        : word_(~static_cast<std::int32_t>(static_cast<std::uint32_t>(unit))) {}

    [[nodiscard]] NINFER_KV_REF_HD static constexpr KVPageRef from_word(std::int32_t word) noexcept {
        return KVPageRef(word);
    }

    [[nodiscard]] NINFER_KV_REF_HD constexpr std::int32_t word() const noexcept { return word_; }
    [[nodiscard]] NINFER_KV_REF_HD constexpr bool host() const noexcept { return word_ < 0; }
    [[nodiscard]] NINFER_KV_REF_HD constexpr DevicePageGroup device_group() const noexcept {
        return static_cast<DevicePageGroup>(word_);
    }
    [[nodiscard]] NINFER_KV_REF_HD constexpr HostPageUnit host_unit() const noexcept {
        return static_cast<HostPageUnit>(static_cast<std::uint32_t>(~word_));
    }
    [[nodiscard]] NINFER_KV_REF_HD constexpr std::size_t host_offset_bytes() const noexcept {
        return static_cast<std::size_t>(static_cast<std::uint32_t>(~word_)) * kHostKVPageUnitBytes;
    }

    friend constexpr bool operator==(KVPageRef, KVPageRef) noexcept = default;

private:
    NINFER_KV_REF_HD constexpr explicit KVPageRef(std::int32_t word) noexcept : word_(word) {}

    std::int32_t word_;
};

static_assert(!KVPageRef(DevicePageGroup{0}).host());
static_assert(KVPageRef(DevicePageGroup{0x7fffffff}).word() == 0x7fffffff);
static_assert(KVPageRef(HostPageUnit{0}).word() == -1);
static_assert(KVPageRef(HostPageUnit{0}).host());
static_assert(KVPageRef(HostPageUnit{kMaxHostPageUnit}).word() == INT32_MIN);
static_assert(KVPageRef(HostPageUnit{kMaxHostPageUnit}).host_unit() == HostPageUnit{kMaxHostPageUnit});
static_assert(KVPageRef(HostPageUnit{3}).host_offset_bytes() == 3 * kHostKVPageUnitBytes);
static_assert(KVPageRef::from_word(-2).host_unit() == HostPageUnit{1});

} // namespace ninfer
