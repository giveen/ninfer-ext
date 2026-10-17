#pragma once

// The routed experts of one Nemotron-H model: the pool its 40 heterogeneous banks form, the device
// residency they are given, and the two routes the non-resident ones take.
//
// This family's routed banks are 37.21 GB of block-scaled NVFP4, which no 32 GB card holds, so the
// work is a split rather than a placement: some banks live on the device and are read in place, the
// rest stay in pinned host memory and reach the device one of two ways. Which one is a question
// about the *width of the call*, not about the layer:
//
//  * A call of `staged_columns` columns or more touches every expert of the layer it is on:
//    `top_k * width >= experts` holds from 56 columns for this artifact's widest bank (496
//    experts), and for a narrow bank it holds earlier still because the bank holds fewer experts
//    than the call touches. Such a call stages the layer's *whole bank* into a two-bank device
//    staging buffer as the pass reaches it, the next layer's fill overlapping this one's compute.
//    Staging cannot copy more than the call reads: every expert of the bank is used.
//  * A narrower call routes `top_k` of the bank's 240-to-496 experts. Staging a whole parent for
//    the 9 experts a one-column call reads is about 27 times the routed bytes (496 against 9 on
//    the widest bank, 27 on the narrowest), so those calls take the pager's per-expert slot cache
//    and gather only the experts they routed to.
//
// Both routes read the same stored bytes through the same decoder -- a slot pool is the layer's own
// planes at the pool's own stride -- so which one ran cannot move a result. The choice is
// `ExpertPager::stages`'s, exposed here as `RoutedExpertPool::stages`; there is no option, flag or
// policy mode for it, and both routes share the one slot pool and allocate nothing per call.
//
// The pool's *size* is the residency plan's, not the geometry's: the two staged banks are the
// floor, and the per-expert slot cache a caller's `ExpertCachePolicy::cache_slots` asks for sits
// below them, paid for out of the same device budget as the resident banks (`plan_experts.h`). A
// bigger cache is therefore fewer resident banks -- one slot's bytes for one slot's worth of bank
// not uploaded -- and a smaller one is more banks. Zero extra slots is the pool this family ran
// before the split was tunable, which is what every recorded number was taken with.
//
// This is the *execution* half. The planning half -- the pool's geometry, its size, and the split
// that decides which banks are uploaded -- is `plan_experts.h`, which the loader calls while
// resolving the artifact and this side consumes. What is left here is the device side itself:
//
//  * `expert_bank_addressing` turns two bound quantized matrices into the plane addressing the Op
//    reads, which only the bound weights can answer and so cannot be planned.
//  * `RoutedExpertPool` is the staging buffer, the pager's tables and the pager. A resident bank is
//    not in it at all -- it was uploaded once and the forward reads it where it lives.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/execution/expert_pager.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/plan_experts.h"
#include "ninfer/ops/offload_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::nemotron_h {

// Defined in forward.h. The pool is built from a model's bound banks, so it needs the model's
// weights and not only the config.
class Model;

// The plane addressing of one layer's bank as the Op reads it, from the two quantized matrices the
// model bound: plane 0/1 the gate/up codes and scales, 2/3 the down ones, expert `e` of plane `p`
// at `base[p] + e * stride[p]`, and the bank's weight divisors. `label` names the layer in the
// refusal. Throws when the matrices are not this layer's NVFP4 bank, so a bank bound at another
// layer's width is refused here rather than read out of range by a kernel.
//
// This is the one piece of the pool's addressing that cannot be planned: it reads the *bound*
// weights, which do not exist until the artifact is materialized.
[[nodiscard]] ops::ExpertWeights expert_bank_addressing(const Weight& up, const Weight& down,
                                                        std::int32_t experts, std::int32_t latent,
                                                        std::int32_t intermediate,
                                                        const char* label);

/**
 * The device side of a pool: the pager's slot pool with its tables, the pager that fills and
 * gathers it, and the one mapping from a pool layer to the bank it reads.
 *
 * Owned by the Program (or by a harness that runs the forward itself). A *resident* layer is not in
 * the pool's staging at all -- it was uploaded once at bind time and `acquire` answers with its own
 * binding, so the caller reads it in place. A non-resident layer takes `acquire`'s staged route
 * when the call is `stages(columns)` wide and `acquire_cached`'s gather route below that; the two
 * routes are the same pool, so a pass may switch between them from one width to the next without
 * reallocating anything.
 *
 * Not thread-safe: one execution thread drives it.
 */
class RoutedExpertPool {
public:
    // The pool is built from the model itself: its banks come from the config, its split and its
    // slot cache from the plan the loader bound the banks with, and its per-layer addressing from
    // the bound weights -- so a pool cannot disagree with the model it serves. `staged_columns` is
    // the width at or above which a call stages; zero takes the pool's own,
    // `ceil(max_experts / top_k)`, which is the narrowest width at which a call touches every
    // expert of the widest bank and staging therefore copies no more than the call reads. A
    // harness may pass a width of its own. The pool's *size* is not an argument: it is the
    // residency plan's (`ExpertResidencyPlan::cache_slots` below the geometry's two staged banks),
    // because the one budget that pays for the cache and for the resident banks is spent in one
    // place.
    RoutedExpertPool(const Model& model, std::int32_t top_k, std::int32_t staged_columns = 0);
    ~RoutedExpertPool();
    RoutedExpertPool(const RoutedExpertPool&)            = delete;
    RoutedExpertPool& operator=(const RoutedExpertPool&) = delete;
    RoutedExpertPool(RoutedExpertPool&&)                 = delete;
    RoutedExpertPool& operator=(RoutedExpertPool&&)      = delete;

    // One layer's bank for one staged call: the addressing to hand the Op, and whether it is a
    // staged copy the caller must release after the layer's last reader.
    struct Placed {
        ops::ExpertWeights weights;
        ops::ExpertBankSpan span;
        std::int32_t layer = -1;
        bool staged        = false;
    };

    // One layer's bank for one gathered call: the slot pool addressed with the layer's own plane
    // offsets at the pool's stride (so the Op indexes it by slot id), the layer's span in that
    // pool, and the event that completes the gather the pager enqueued on its own stream.
    struct Cached {
        ops::ExpertWeights weights;
        ops::ExpertBankSpan span;
        cudaEvent_t fetched = nullptr;
    };

    // True when a call of `columns` columns takes the staged route. Below it the same call gathers
    // the experts it routed to through the pager's slot cache instead.
    [[nodiscard]] bool stages(std::int32_t columns) const { return pager_->stages(columns); }

    // Opens a pass: the next staged fill reclaims the banks from whatever the cache left in them.
    void begin_forward() noexcept { forward_start_ = true; }

    [[nodiscard]] Placed acquire(std::int32_t layer, cudaStream_t stream);
    void release(std::int32_t layer, cudaStream_t stream);

    // The narrow route: resolve `ids` -- the layer's own routed expert ids, I32 `[top_k, columns]`
    // -- to slots of the pool in `slot_ids` (I32 `[top_k, columns]`), gather the misses from the
    // layer's pinned bank on the pager's stream, and answer the addressing to hand the Op with
    // `slot_ids` in place of `ids`. `misses` is the pager's own scratch and its contents belong to
    // the pager until the next resolve; `fetched` completes the gather, so the caller's stream must
    // wait on it before reading a slot the gather filled. Nothing is released here -- the slots
    // outlive the call and the next resolve owns their eviction. A *resident* layer is refused: it
    // is read where it was uploaded and has nothing to gather.
    [[nodiscard]] Cached acquire_cached(std::int32_t layer, const Tensor& ids, Tensor& slot_ids,
                                        Tensor& misses, cudaStream_t stream);

    [[nodiscard]] std::int32_t layers() const noexcept { return geometry_->layers(); }

    [[nodiscard]] bool resident(std::int32_t layer) const { return resident_.at(layer); }

    // What this pool has copied, over every pass it has run: the bytes its staged fills took from
    // the Host banks and the layers they took them for -- the pager's own count, taken where the
    // copies are enqueued. A prefill that touches each bank once per pass copies a layer's whole
    // bank, so this is the route's host-link traffic and the number the in-place route is measured
    // against.
    [[nodiscard]] std::size_t staged_bytes() const noexcept { return pager_->staged_bytes(); }

    [[nodiscard]] std::size_t staged_layers() const noexcept { return pager_->staged_layers(); }

    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

    // The pager's cache state: what the gathered route resolves into and gathers from. A harness
    // reads its counters (hits, misses, ids outside the bank) back after a pass it has
    // synchronized, the same way the staged route's `staged_bytes` needs no such read; both
    // counters are Device memory, which is why this answers with the state rather than numbers.
    [[nodiscard]] const ops::ExpertCacheState& cache() const noexcept { return pager_->cache(); }

    [[nodiscard]] const ops::ExpertPoolGeometry& geometry() const noexcept { return *geometry_; }

private:
    // The next layer that will be staged, so its fill can be issued with this one's; -1 when the
    // rest of the pool is resident.
    [[nodiscard]] std::int32_t next_staged(std::int32_t layer) const;

    std::vector<ops::ExpertBankShape> banks_;
    std::vector<ops::ExpertWeights> host_; // one per pool layer: the bank as it is stored
    std::vector<bool> resident_;           // one per pool layer
    std::unique_ptr<ops::ExpertPoolGeometry> geometry_;
    DeviceBuffer pool_;
    DeviceBuffer slot_of_;
    DeviceBuffer owner_;
    DeviceBuffer stamp_;
    DeviceBuffer counters_;
    std::optional<execution::ExpertPager> pager_;
    bool forward_start_ = true;
};

} // namespace ninfer::models::nemotron_h
