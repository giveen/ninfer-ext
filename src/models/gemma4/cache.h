#pragma once

// One sequence's attention state, per layer.
//
// Sliding layers keep a ring of `sliding_window` key and value rows, so a token's slot is its position
// modulo the window: the keys a query can still see occupy exactly the window's worth of slots.
// Global layers keep one compact row per token, written at the token's own position, so their capacity
// is the number of tokens the caller allows; the plan's paging is what will lift that bound.
//
// A slot no token has written holds `kUnwrittenPosition`, and each attention Op decides visibility by
// comparing positions, so an unwritten slot is invisible to both: the sliding Op needs
// `0 <= pq - pk < window` and the compact Op needs `0 <= pq - pk`, and a far-future key fails each.
// That is why a caller can attend over the whole cache without knowing how much of it is filled.

#include "core/tensor.h"
#include "models/gemma4/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::gemma4 {

inline constexpr std::int32_t kUnwrittenPosition = 1 << 30;

class KvCache {
public:
    KvCache() = default;
    ~KvCache();
    KvCache(const KvCache&)            = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) noexcept;
    KvCache& operator=(KvCache&&) noexcept;

    // Allocates the layers' storage lazily, as each layer is first used.
    void configure(const TextConfig& config, std::int32_t capacity);

    // Forgets every token: all slots become unwritten, keeping the allocation.
    void reset();

    // The whole key set of one layer, as the attention Ops take it: sliding [D,Hkv,window] with the
    // values alongside, global [width,Hkv,capacity] carrying values and rotated key dims together.
    [[nodiscard]] Tensor keys(std::size_t layer);
    [[nodiscard]] Tensor values(std::size_t layer);
    [[nodiscard]] Tensor positions(std::size_t layer);

    // The slot a token at `position` occupies in `layer`'s storage.
    [[nodiscard]] std::int32_t slot(std::size_t layer, std::int32_t position) const;

    // Records that the `tokens` positions from `first_position` on occupy their slots.
    void mark_range(std::size_t layer, std::int32_t first_position, std::int32_t tokens,
                    cudaStream_t stream);

    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    struct Layer {
        Tensor keys;
        Tensor values;
        Tensor positions;
        std::int32_t tokens = 0;
        bool allocated      = false;
    };

    void ensure(std::size_t layer);
    void reset_layer(std::size_t layer);

    TextConfig config_{};
    std::vector<Layer> layers_;
    std::vector<void*> owned_;
    std::int32_t capacity_ = 0;
    std::size_t bytes_     = 0;
};

} // namespace ninfer::models::gemma4
