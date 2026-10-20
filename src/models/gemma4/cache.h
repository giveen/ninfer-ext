#pragma once

// One sequence's attention state, per layer.
//
// Sliding layers keep a ring of `sliding_window + ring_slack` key and value rows, so a token's slot
// is its position modulo the ring. A pass of at most `ring_slack + 1` tokens writes its rows first:
// the rows it replaces are a window older than any of its queries. A wider pass attends to the ring
// as it was plus its own keys, and writes its last ring of keys afterwards, so it may be any width.
// The slack is what speculation needs: a verify pass writes rows for drafts that may be rejected,
// and with a slack of at least the draft count, the older row a rejected draft replaced is
// invisible to every query that follows. The rejected rows themselves are marked stale and
// forgotten before a wide pass could see them; a narrow pass overwrites them before its own queries
// reach them. Global layers keep one compact row per token, written at the token's own position, so
// their capacity is the number of tokens the caller allows; the plan's paging is what will lift
// that bound.
//
// A slot no token has written holds `kUnwrittenPosition`, and each attention Op decides visibility
// by comparing positions, so an unwritten slot is invisible to both: the sliding Op needs `0 <= pq
// - pk < window` and the compact Op needs `0 <= pq - pk`, and a far-future key fails each. That is
// why a caller can attend over the whole cache without knowing how much of it is filled.
//
// A sliding ring may instead be kept quantized (`--kv-dtype`): `keys`/`values` are then absent and
// the ring's rows live in the format's code and scale planes, which the model encodes on write and
// decodes into BF16 scratch before it attends (see `ops/dense_kv_ring.h`). FP8 also quantizes the
// global layers' 640-wide compact rows as one code plane plus one FP16 scale per (position, KV
// head); the attention decodes each staged tile with `causal_compact_attention_codes`. The other
// formats leave the global rows BF16, because no codec of theirs covers a 640-wide row.

#include "core/tensor.h"
#include "models/gemma4/config.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <algorithm>
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

    // Allocates the layers' storage lazily, as each layer is first used. A storage other than
    // BF16 quantizes the sliding rings; FP8 also quantizes the global compact rows.
    void configure(const TextConfig& config, std::int32_t capacity, std::int32_t ring_slack = 0,
                   KvCacheStorage storage = KvCacheStorage::BFloat16);

    // Allocates every layer now rather than on first use.
    void allocate();

    // Device bytes a configured cache holds once every layer is allocated.
    [[nodiscard]] static std::size_t
    device_bytes(const TextConfig& config, std::int32_t capacity, std::int32_t ring_slack = 0,
                 KvCacheStorage storage = KvCacheStorage::BFloat16);

    // Rows in a sliding layer's ring: the window plus the slack.
    [[nodiscard]] std::int32_t ring_tokens() const noexcept {
        return static_cast<std::int32_t>(config_.sliding_window) + ring_slack_;
    }

    [[nodiscard]] std::int32_t ring_slack() const noexcept { return ring_slack_; }

    // The storage the sliding rings were configured with; the global layers are always BF16.
    [[nodiscard]] KvCacheStorage storage() const noexcept { return storage_; }

    // Whether sliding `layer` holds a quantized ring rather than BF16 rows.
    // Throws like `sliding_attention` when `layer` is not one of `layer_types`'.
    [[nodiscard]] bool quantized(std::size_t layer) const {
        return storage_ != KvCacheStorage::BFloat16 && config_.sliding_attention(layer);
    }

    // Whether `layer` is a global layer whose compact rows are stored as FP8 codes.
    [[nodiscard]] bool coded_global(std::size_t layer) const {
        return global_fp8_ && !config_.sliding_attention(layer);
    }

    // Whether the global layers' compact rows are FP8-coded rather than BF16.
    [[nodiscard]] bool global_quantized() const noexcept { return global_fp8_;
    }

    // Forgets every token: all slots become unwritten, keeping the allocation.
    void reset();

    // The whole key set of one layer, as the attention Ops take it: sliding [D,Hkv,window] with the
    // values alongside, global [width,Hkv,capacity] carrying values and rotated key dims together.
    // A quantized sliding layer has no BF16 rows and refuses these: its ring is decoded from the
    // planes below.
    [[nodiscard]] Tensor keys(std::size_t layer);
    [[nodiscard]] Tensor values(std::size_t layer);
    [[nodiscard]] Tensor positions(std::size_t layer);
    // The same, for a reader that must not allocate: the layer must already exist.
    [[nodiscard]] Tensor keys(std::size_t layer) const;
    [[nodiscard]] Tensor values(std::size_t layer) const;
    [[nodiscard]] Tensor positions(std::size_t layer) const;

    // One quantized sliding layer's planes, all shaped [extent, Hkv, ring] with the format's code
    // and scale element types (`paged_kv_storage_layout`, ops/dense_kv_ring.h).
    [[nodiscard]] Tensor key_codes(std::size_t layer) const;
    [[nodiscard]] Tensor key_scales(std::size_t layer) const;
    [[nodiscard]] Tensor value_codes(std::size_t layer) const;
    [[nodiscard]] Tensor value_scales(std::size_t layer) const;
    // The same, allocating the layer on first use.
    [[nodiscard]] Tensor key_codes(std::size_t layer);
    [[nodiscard]] Tensor key_scales(std::size_t layer);
    [[nodiscard]] Tensor value_codes(std::size_t layer);
    [[nodiscard]] Tensor value_scales(std::size_t layer);

    // Rows at positions from a rollback point up to `end` were written for tokens the sequence did
    // not keep. They stay in place until a pass overwrites them or forgets them.
    void mark_stale(std::int32_t end) noexcept { stale_end_ = std::max(stale_end_, end); }

    // Makes every stale row of sliding `layer` at a position from `from` on invisible.
    void forget_stale(std::size_t layer, std::int32_t from, cudaStream_t stream);

    // The slot a token at `position` occupies in `layer`'s storage. A pass writes its rows and
    // their positions into these slots on the device (forward.h `PassIndices`).
    [[nodiscard]] std::int32_t slot(std::size_t layer, std::int32_t position) const;

    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    struct Layer {
        Tensor keys;
        Tensor values;
        Tensor positions;
        // Present instead of `keys`/`values` when the ring is quantized.
        Tensor key_codes;
        Tensor key_scales;
        Tensor value_codes;
        Tensor value_scales;
        std::int32_t tokens = 0;
        bool allocated      = false;
    };

    void ensure(std::size_t layer);
    void reset_layer(std::size_t layer);
    // Frees every device allocation this cache owns and forgets them.
    void release_owned() noexcept;

    TextConfig config_{};
    std::vector<Layer> layers_;
    std::vector<void*> owned_;
    std::int32_t capacity_   = 0;
    std::int32_t ring_slack_ = 0;
    std::int32_t stale_end_  = 0;
    std::size_t bytes_       = 0;
    KvCacheStorage storage_  = KvCacheStorage::BFloat16;
    // FP8 storage also codes the global layers' compact rows; the other formats leave them BF16.
    bool global_fp8_ = false;
};

} // namespace ninfer::models::gemma4
