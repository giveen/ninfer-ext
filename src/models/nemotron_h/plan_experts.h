#pragma once

// The planning half of the Nemotron-H routed experts: the pool its 40 heterogeneous banks form, and
// how much of it the card can hold.
//
// This lives on the loading side because that is what it is -- pure config-to-plan arithmetic, run
// while the artifact is being resolved and before a byte is uploaded. The loader decides each bank's
// `Residency` from this plan, so the functions have to be callable from `models/nemotron_h/load.cpp`
// and must not pull in the execution side; the runtime consumes them from here (it links the loading
// library, not the other way round).
//
// The numbers that decide everything, each derived rather than fixed:
//
//  * `expert_bank_geometry` is the pool: one `ExpertBankShape` per MoE block, in block order. Its
//    order is the pool layer index that every pager call is keyed on, and `expert_pool_layer` is the
//    single function that maps a block to it -- a second counting of MoE blocks would map one
//    layer's experts onto another layer's owners.
//  * `plan_expert_residency` spends what the card has left after everything that must stay free.
//
// The execution side of the same split is `RoutedExpertPool` in experts.h: the staging buffer, the
// pager's tables and the pager. A resident bank is not in it at all -- it was uploaded once and the
// forward reads it where it lives.

#include "models/nemotron_h/config.h"
#include "ninfer/ops/offload_moe.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::nemotron_h {

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

} // namespace ninfer::models::nemotron_h
