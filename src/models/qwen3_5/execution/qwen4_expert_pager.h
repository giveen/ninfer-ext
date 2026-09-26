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
 * The two routes differ numerically (the staged route may quantize activations), so the route
 * choice is fixed by `stages`. Within a route, placement never changes a result.
 *
 * Program-owned: the Program binds the cache state and banks from its persistent allocation and
 * keeps the pager alive while any execution that uses it is in flight. Not thread-safe; all calls
 * come from the Program's execution thread.
 */
class Qwen4ExpertPager {
public:
    static constexpr std::size_t kStagedBankBytes = ops::kExpertStagedLayerBytes;

    // `staged_banks` holds two staged layers, or is null for a pager without a staged route.
    Qwen4ExpertPager(const ops::ExpertCacheState& cache, std::byte* staged_banks,
                     std::int32_t staged_columns);
    ~Qwen4ExpertPager();

    Qwen4ExpertPager(const Qwen4ExpertPager&)            = delete;
    Qwen4ExpertPager& operator=(const Qwen4ExpertPager&) = delete;

    [[nodiscard]] const ops::ExpertCacheState& cache() const noexcept { return cache_; }

    // True when a call of `columns` columns takes the staged route.
    [[nodiscard]] bool stages(std::int32_t columns) const noexcept {
        return banks_[0] != nullptr && columns >= staged_columns_;
    }

    /**
     * Staged route. Makes `bank` (the Host bank of `layer`) ready in a staged device bank, starts
     * the fill of `next_bank` (layer `next_layer`, or none when `next_bank` is null) into the other
     * bank, and makes `stream` wait until `layer` is ready. Returns the staged addressing. Every
     * acquire is paired with `release_staged` on the same stream after the layer's last reader.
     */
    [[nodiscard]] ops::ExpertWeights acquire_staged(std::int32_t layer,
                                                    const ops::ExpertWeights& bank,
                                                    std::int32_t next_layer,
                                                    const ops::ExpertWeights* next_bank,
                                                    cudaStream_t stream);
    void release_staged(std::int32_t layer, cudaStream_t stream);

    // Cache route: the most columns one `resolve` accepts.
    [[nodiscard]] std::int32_t max_resolve_columns() const noexcept;

    /**
     * Cache route. Resolves the routed experts `ids` I32 `[10,T]` of `layer` to cache slots in
     * `slot_ids` and copies every miss from `bank` into its slot, both on `stream`. `misses` is
     * scratch of at least `2*(10*T)+1` words. T must not exceed `max_resolve_columns`.
     */
    void resolve(const Tensor& ids, std::int32_t layer, const ops::ExpertWeights& bank,
                 Tensor& slot_ids, Tensor& misses, cudaStream_t stream);

    // Slot-pool addressing of the cache for experts whose divisors come from `bank`.
    [[nodiscard]] ops::ExpertWeights slot_weights(const ops::ExpertWeights& bank) const {
        return ops::expert_cache_weights(cache_, bank);
    }

private:
    void snapshot_residency(cudaStream_t stream);
    void fill(std::int32_t layer, const ops::ExpertWeights& source);
    [[nodiscard]] ops::ExpertWeights staged_weights(int bank,
                                                    const ops::ExpertWeights& source) const;

    ops::ExpertCacheState cache_;
    std::int32_t staged_columns_ = 0;

    // Staged route: bank b holds (or is being filled with) layer_[b]. A fill waits for
    // `released_[b]` of the bank's previous reader and records `ready_[b]`.
    std::byte* banks_[2]     = {};
    cudaStream_t stream_     = nullptr;
    cudaEvent_t ready_[2]    = {};
    cudaEvent_t released_[2] = {};
    std::int32_t layer_[2]   = {-1, -1};
    // Pinned Host snapshot of the cache's slot_of, taken before a forward's first fill. Fills skip
    // the Host copy of experts it marks resident; while `resident_valid_` is false (a captured
    // forward cannot synchronize) they copy every expert.
    std::int32_t* resident_ = nullptr;
    bool resident_valid_    = false;
};

} // namespace ninfer::models::qwen3_5::execution
