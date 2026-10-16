#pragma once

// The routed experts of one Nemotron-H model: the pool its 40 heterogeneous banks form, the device
// residency they are given, and the staged route the non-resident ones take.
//
// This family's routed banks are 37.21 GB of block-scaled NVFP4, which no 32 GB card holds, so the
// work is a split rather than a placement: some banks live on the device and are read in place,
// the rest stay in pinned host memory and are copied into a two-bank device staging buffer as the
// pass reaches them. Residency granularity is one *bank* -- a whole 240-to-496-expert parent, 0.42
// to 1.46 GB -- because the Op reads a bank whole and, at any chunk width this family runs, a chunk
// already touches every expert of the layer it is on (`top_k * width >= experts` holds from 56
// columns on). There is therefore no per-expert cache here: a slot cache would copy the same bytes
// and add a compaction pass to discover which experts to copy.
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
 * The device side of a pool: two staged banks, the pager's tables, the pager that fills them, and
 * the one mapping from a pool layer to the bank it reads.
 *
 * Owned by the Program (or by a harness that runs the forward itself). A *resident* layer is not in
 * the pool's staging at all -- it was uploaded once at bind time and `acquire` answers with its own
 * binding, so the caller reads it in place. A non-resident layer is staged whole into the bank the
 * layer being read is not using, and the *next* non-resident layer's fill is issued at the same
 * time, so it overlaps this layer's compute rather than following it.
 *
 * Not thread-safe: one execution thread drives it.
 */
class RoutedExpertPool {
public:
    // The pool is built from the model itself: its banks come from the config, its split from the
    // plan the loader bound the banks with, and its per-layer addressing from the bound weights --
    // so a pool cannot disagree with the model it serves. `staged_columns` is the width at or above
    // which a call stages; the pager's other route (a per-expert slot cache) is not built for this
    // family, so the pool stages every call it is given and the value is only a floor.
    RoutedExpertPool(const Model& model, std::int32_t top_k, std::int32_t staged_columns = 1);
    ~RoutedExpertPool();
    RoutedExpertPool(const RoutedExpertPool&)            = delete;
    RoutedExpertPool& operator=(const RoutedExpertPool&) = delete;
    RoutedExpertPool(RoutedExpertPool&&)                 = delete;
    RoutedExpertPool& operator=(RoutedExpertPool&&)      = delete;

    // One layer's bank for one call: the addressing to hand the Op, and whether it is a staged copy
    // the caller must release after the layer's last reader.
    struct Placed {
        ops::ExpertWeights weights;
        ops::ExpertBankSpan span;
        std::int32_t layer = -1;
        bool staged        = false;
    };

    // Opens a pass: the next staged fill reclaims the banks from whatever the cache left in them.
    void begin_forward() noexcept { forward_start_ = true; }

    [[nodiscard]] Placed acquire(std::int32_t layer, cudaStream_t stream);
    void release(std::int32_t layer, cudaStream_t stream);

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
