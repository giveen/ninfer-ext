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
// Three numbers decide everything, and each is derived rather than fixed:
//
//  * `expert_bank_geometry` is the pool: one `ExpertBankShape` per MoE block, in block order. Its
//    order is the pool layer index that every pager call is keyed on, and `expert_pool_layer` is
//    the single function that maps a block to it -- a second counting of MoE blocks would map one
//    layer's experts onto another layer's owners.
//  * `plan_expert_residency` spends what the card has left after everything that must stay free.
//  * `RoutedExpertPool` is the device side: the staging buffer, the pager's tables and the pager.
//    A resident bank is not in it at all -- it was uploaded once and the forward reads it where it
//    lives.

#include "core/arena.h"
#include "core/weight.h"
#include "models/execution/expert_pager.h"
#include "models/nemotron_h/config.h"
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

// The pool's banks, one per MoE block of `config`, in block order. Empty when the config has no MoE
// block.
[[nodiscard]] std::vector<ops::ExpertBankShape> expert_bank_geometry(const TextConfig& config);

// The pool layer index of a block, or -1 when the block is not MoE. The one mapping from the
// model's block numbering to the pool's, used by both the loader that builds the pool and the
// forward that drives it.
[[nodiscard]] std::int32_t expert_pool_layer(const TextConfig& config, std::uint32_t block);

// Device bytes a pool of these banks needs before any bank is resident: two staged banks (so the
// next layer's copy overlaps this layer's compute) and the pager's tables. Zero for an empty pool.
[[nodiscard]] std::size_t expert_staging_bytes(std::span<const ops::ExpertBankShape> banks);

// The device bytes a `RoutedExpertPool` for `config` allocates, which is what a caller that has to
// keep room for it adds to its own reservation. Zero when the config has no MoE block.
[[nodiscard]] std::size_t expert_pool_bytes(const TextConfig& config);

// The routed rank of the pool's layers: the most experts one token of any layer selects, which is
// the assignment count the pool is sized against. Zero when the config has no MoE block.
[[nodiscard]] std::int32_t expert_top_k(const TextConfig& config);

// One pass's output buffer: the head's logits, BF16 `[vocabulary, tokens]`. It is not part of
// `model_workspace_bytes` (the forward writes it into a buffer its caller owns), but it is device
// memory one pass needs, so a caller sizing what the expert cache may not take has to count it.
[[nodiscard]] std::size_t pass_output_bytes(const TextConfig& config, std::int32_t tokens);

// Device memory the runtime itself holds once a model of this family is loaded and its state
// planned: the CUDA context, the math libraries' handles and the scratch they keep. It is reserved
// rather than measured because the expert cache has to be sized *before* materialization -- the
// banks' residency is a property of the binding, so it is decided before the weights are uploaded,
// and the only other measurement available is taken after. The number is a load's own measurement
// on the documented 32 GiB machine; `--expert-cache` overrides the cache it feeds either way.
inline constexpr std::size_t kRuntimeReservationBytes = 824ULL * 1024 * 1024;

// Bytes one bank of the pool occupies when it is bound device-resident: its experts' planes.
[[nodiscard]] std::int64_t expert_bank_bytes(const ops::ExpertBankShape& bank);

// The plane addressing of one layer's bank as the Op reads it, from the two quantized matrices the
// model bound: plane 0/1 the gate/up codes and scales, 2/3 the down ones, expert `e` of plane `p`
// at `base[p] + e * stride[p]`, and the bank's weight divisors. `label` names the layer in the
// refusal. Throws when the matrices are not this layer's NVFP4 bank, so a bank bound at another
// layer's width is refused here rather than read out of range by a kernel.
[[nodiscard]] ops::ExpertWeights expert_bank_addressing(const Weight& up, const Weight& down,
                                                        std::int32_t experts, std::int32_t latent,
                                                        std::int32_t intermediate,
                                                        const char* label);

// How the experts were placed on the device: the split itself, and the numbers it implies.
//
// `resident` is one flag per pool layer. `staging_bytes` is zero when nothing is staged, which is
// the case either because every bank is resident or because the card affords no staging buffer --
// and in the second case the non-resident banks are read in place over UVA, which is the route the
// family ran before this split existed.
struct ExpertResidencyPlan {
    std::size_t budget_bytes      = 0; // device bytes the routed experts occupy in total
    std::size_t staging_bytes     = 0; // the two staged banks and the pager's tables
    std::size_t resident_bytes    = 0; // the banks bound device-resident
    std::int32_t resident_experts = 0; // experts resident: the cache's "slots"
    std::int32_t pool_experts     = 0; // expert count of the whole pool
    std::vector<bool> resident{};      // one flag per pool layer
    // An explicit size was larger than the card can afford, so it was re-derived smaller. The
    // request is never honoured by starving the state and K/V reservation, and never by
    // over-committing the card: both numbers are reported for the caller to see.
    bool clamped                = false;
    std::size_t requested_bytes = 0;

    [[nodiscard]] double resident_fraction() const noexcept {
        return pool_experts == 0
                   ? 0.0
                   : static_cast<double>(resident_experts) / static_cast<double>(pool_experts);
    }

    // Expert bytes left in host memory: the total less what is resident.
    [[nodiscard]] std::size_t pinned_bytes(std::size_t total_bank_bytes) const noexcept {
        return total_bank_bytes - resident_bytes;
    }
};

/**
 * Spend the device memory the routed experts may have.
 *
 * `available_bytes` is what the card has free for *resident* banks: free device memory less the
 * dense weights, less the lane states and one pass's workspace, less the runtime's own reservation
 * and its headroom, less the staging buffer. It is derived by the caller from the engine's own
 * resolved plan (contexts, lanes, K/V dtype, chunk) rather than from a constant, because it is the
 * K/V that must not give way: a cache sized before the K/V resolution would take the memory the
 * K/V needs and fail a request later.
 *
 * `explicit_size` marks a caller-set cap, which is clamped to `available_bytes` rather than
 * over-committed. Banks are taken largest first, so the resident set is the one that keeps the most
 * bytes off the host link; the order is a total order on (bytes, pool index) and therefore
 * reproducible run to run.
 */
[[nodiscard]] ExpertResidencyPlan plan_expert_residency(std::span<const ops::ExpertBankShape> banks,
                                                        std::size_t available_bytes,
                                                        bool explicit_size,
                                                        std::size_t explicit_bytes);

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
