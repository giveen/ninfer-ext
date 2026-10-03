#pragma once

// Qwen4ExpertPager: the one owner of how Qwen4Exp routed experts reach the device.

#include "core/tensor.h"
#include "ninfer/ops/offload_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

/**
 * Places the routed experts of one MoE call on the device. Routed experts live in pinned Host
 * banks; a call reaches them through one of two routes:
 *
 * - Cache route (narrow calls): `resolve` maps each routed expert to a slot of the device expert
 *   cache, evicting least-recently-used slots, and copies the misses from the Host bank.
 * - Staged route (calls of at least `staged_columns` columns): two whole-layer device banks are
 *   filled on the pager's own stream, the next layer's fill overlapping this layer's compute.
 *   Experts the cache holds are copied device-to-device instead of from the Host. Staged calls do
 *   not change the cache.
 *
 * The banks sit past the cache pool's slots (plus the pool's divisor tail), so a cache-route call
 * never touches them and the staged route never has to reclaim cache residency first. When the idle
 * prefill width exceeds the ordinary chunk, the top `base_slots_..decode_slots_` cache slots are a
 * lend region: `lend` reclaims them and the Program uses their bytes as the wide prefill arena;
 * `return_lend` makes them cache again.
 *
 * The two routes differ numerically (the staged route may quantize activations), so the route
 * choice is fixed by `stages`. Within a route, placement never changes a result.
 *
 * Program-owned: the Program binds the cache state from its persistent allocation and keeps the
 * pager alive while any execution that uses it is in flight. Not thread-safe; all calls come from
 * the Program's execution thread.
 */
class Qwen4ExpertPager {
public:
    static constexpr std::int32_t kBankSlots      = 2 * ops::kOffloadMoeExperts;

    // `cache.pool` holds the decode cache slots (cache plus the lent range), then two layers' weight
    // divisors; the banks begin at `bank_slot`. `base_slots` is the cache size while the wide idle
    // prefill borrows the lend range, `cache.slots` the decode-time size.
    Qwen4ExpertPager(const ops::ExpertLayout& layout, const ops::ExpertCacheState& cache,
                     std::int32_t staged_columns,
                     std::int32_t bank_slot, std::int32_t base_slots);
    ~Qwen4ExpertPager();

    Qwen4ExpertPager(const Qwen4ExpertPager&)            = delete;
    Qwen4ExpertPager& operator=(const Qwen4ExpertPager&) = delete;

    [[nodiscard]] const ops::ExpertCacheState& cache() const noexcept { return cache_; }

    // P1 lend: reclaim the lend range for the wide prefill arena, then make the cache again.
    void lend(cudaStream_t stream);
    void return_lend() noexcept { cache_.slots = decode_slots_; }

    // True when a call of `columns` columns takes the staged route.
    [[nodiscard]] bool stages(std::int32_t columns) const noexcept {
        return columns >= staged_columns_;
    }

    /**
     * Staged route. Makes `bank` (the Host bank of `layer`) ready in a staged device bank, starts
     * the fill of `next_bank` (layer `next_layer`, or none when `next_bank` is null) into the other
     * bank, and makes `stream` wait until `layer` is ready. Returns the staged addressing. Every
     * acquire is paired with `release_staged` on the same stream after the layer's last reader.
     * `forward_start` marks the first staged call of a forward: it reclaims the bank slots.
     */
    [[nodiscard]] ops::ExpertWeights acquire_staged(std::int32_t layer,
                                                    const ops::ExpertWeights& bank,
                                                    std::int32_t next_layer,
                                                    const ops::ExpertWeights* next_bank,
                                                    bool forward_start, cudaStream_t stream);
    void release_staged(std::int32_t layer, cudaStream_t stream);

    // Cache route: the most columns one `resolve` accepts.
    [[nodiscard]] std::int32_t max_resolve_columns() const noexcept;

    /**
     * Cache route. Resolves the routed experts `ids` I32 `[10,T]` of `layer` to cache slots in
     * `slot_ids` on `stream`, then copies every miss listed in `misses` from `bank` into its slot
     * on the pager's stream. The returned event completes that copy: work on `stream` may read the
     * resident slots at once and must wait on the event before reading a listed slot (and before
     * the next resolve reuses `misses`). `misses` is scratch of at least `2*(10*T)+1` words. T must
     * not exceed `max_resolve_columns`.
     */
    [[nodiscard]] cudaEvent_t resolve(const Tensor& ids, std::int32_t layer,
                                      const ops::ExpertWeights& bank, Tensor& slot_ids,
                                      Tensor& misses, cudaStream_t stream);

    // Slot-pool addressing of the cache for experts whose divisors come from `bank`.
    [[nodiscard]] ops::ExpertWeights slot_weights(const ops::ExpertWeights& bank) const {
        return ops::expert_cache_weights(cache_, bank);
    }

private:
    void snapshot_residency(cudaStream_t stream);
    void fill(std::int32_t layer, const ops::ExpertWeights& source);
    [[nodiscard]] ops::ExpertWeights staged_weights(int bank,
                                                    const ops::ExpertWeights& source) const;

    ops::ExpertLayout layout_;
    std::size_t staged_bank_bytes_ = 0; // one staged layer: its experts, then the layout tail
    ops::ExpertCacheState cache_;
    std::int32_t staged_columns_ = 0;

    // Staged route: bank b holds (or is being filled with) layer_[b] within the current staged
    // forward. A fill waits for `released_[b]` of the bank's previous reader and records
    // `ready_[b]`; the forward's first fill also waits for `reclaimed_`.
    std::byte* banks_[2]     = {};
    std::int32_t bank_slot_  = 0; // first slot the banks overlay (past the cache and lend slots)
    std::int32_t base_slots_   = 0; // cache slots while the lend range is borrowed
    std::int32_t decode_slots_ = 0; // cache slots during decode (cache plus lend)
    cudaStream_t stream_     = nullptr;
    cudaEvent_t ready_[2]    = {};
    cudaEvent_t released_[2] = {};
    cudaEvent_t reclaimed_   = nullptr;
    // Cache route: the fetch starts after `resolved_` and records `fetched_`.
    cudaEvent_t resolved_    = nullptr;
    cudaEvent_t fetched_     = nullptr;
    std::int32_t layer_[2]   = {-1, -1};
    // Pinned Host snapshot of the cache's slot_of, taken before a forward's first fill. Fills skip
    // the Host copy of experts it marks resident; while `resident_valid_` is false (a captured
    // forward cannot synchronize) they copy every expert.
    std::int32_t* resident_ = nullptr;
    bool resident_valid_    = false;
};

} // namespace ninfer::models::qwen3_5::execution
