// Implements: models/nemotron_h/plan_experts.h
//
// Pure config-to-plan arithmetic, so it is compiled into the loading library: the loader calls it
// while resolving the artifact, and the execution side consumes the plan it produces. Nothing here
// touches a device or an execution type.
#include "models/nemotron_h/plan_experts.h"

#include <algorithm>
#include <numeric>
#include <cstddef>

namespace ninfer::models::nemotron_h {
namespace {

// The number of staged banks, which is also what the pool's device bytes reserve. It mirrors
// `execution::ExpertPager::kBankCount` -- one layer being read while the next one is being filled --
// and is stated here rather than included from there because the loading side must not depend on an
// execution header. A change to one without the other would put the staged banks out of the pool.
constexpr std::size_t kStagedBanks = 2;

} // namespace

std::vector<ops::ExpertBankShape> expert_bank_geometry(const TextConfig& config) {
    std::vector<ops::ExpertBankShape> banks;
    banks.reserve(config.num_hidden_layers());
    for (std::uint32_t block = 0; block < config.num_hidden_layers(); ++block) {
        if (config.block_kind(block) != BlockKind::Moe) { continue; }
        const MoeBankConfig& bank = config.moe_bank(block);
        const std::int32_t latent =
            static_cast<std::int32_t>(bank.expert_input_width(config.hidden_size));
        banks.push_back(ops::nvfp4_expert_bank(static_cast<std::int32_t>(bank.num_experts), latent,
                                               static_cast<std::int32_t>(bank.intermediate_size),
                                               1));
    }
    return banks;
}

std::int32_t expert_pool_layer(const TextConfig& config, std::uint32_t block) {
    if (block >= config.num_hidden_layers() || config.block_kind(block) != BlockKind::Moe) {
        return -1;
    }
    std::int32_t layer = 0;
    for (std::uint32_t before = 0; before < block; ++before) {
        if (config.block_kind(before) == BlockKind::Moe) { ++layer; }
    }
    return layer;
}

std::size_t expert_staging_bytes(std::span<const ops::ExpertBankShape> banks) {
    if (banks.empty()) { return 0; }
    // One `ExpertPoolGeometry` states the numbers once; the buffer the pager binds into is two
    // banks each as wide as the pool's widest expert, and the two divisor tails past them, which is
    // the layout `execution::ExpertPager` requires (`bank_slot + bank_slots == slots`).
    const ops::ExpertPoolGeometry geometry(banks);
    return kStagedBanks * static_cast<std::size_t>(geometry.max_experts()) *
               static_cast<std::size_t>(geometry.pool_stride()) +
           kStagedBanks * static_cast<std::size_t>(geometry.max_tail_bytes());
}

std::size_t expert_pool_bytes(const TextConfig& config) {
    const std::vector<ops::ExpertBankShape> banks = expert_bank_geometry(config);
    if (banks.empty()) { return 0; }
    // The staging buffer, then the pager's tables: one slot table entry per expert of the pool and
    // one owner, stamp and counter set per slot of the two banks.
    const ops::ExpertPoolGeometry geometry(banks);
    const std::size_t slots = kStagedBanks * static_cast<std::size_t>(geometry.max_experts());
    return expert_staging_bytes(banks) +
           static_cast<std::size_t>(geometry.slot_entries()) * sizeof(std::int32_t) +
           slots * (sizeof(std::int32_t) + sizeof(std::uint64_t)) + 4 * sizeof(std::uint64_t);
}

std::int32_t expert_top_k(const TextConfig& config) {
    std::int32_t top_k = 0;
    for (std::uint32_t block = 0; block < config.num_hidden_layers(); ++block) {
        if (config.block_kind(block) != BlockKind::Moe) { continue; }
        top_k = std::max(top_k, static_cast<std::int32_t>(config.moe_bank(block).experts_per_token));
    }
    return top_k;
}

std::size_t pass_output_bytes(const TextConfig& config, std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    return static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(tokens) *
           sizeof(std::uint16_t);
}

std::int64_t expert_bank_bytes(const ops::ExpertBankShape& bank) {
    return static_cast<std::int64_t>(bank.experts) * bank.layout.slot_bytes;
}

ExpertResidencyPlan plan_expert_residency(std::span<const ops::ExpertBankShape> banks,
                                          std::size_t available_bytes, bool explicit_size,
                                          std::size_t explicit_bytes) {
    ExpertResidencyPlan plan;
    plan.resident.assign(banks.size(), false);
    if (banks.empty()) { return plan; }

    const std::size_t staging = expert_staging_bytes(banks);
    for (const ops::ExpertBankShape& bank : banks) { plan.pool_experts += bank.experts; }

    // The budget is the whole expert footprint -- staging and residents together -- and it is the
    // caller's cap or everything the card has, whichever is smaller. A cap is clamped rather than
    // over-committed, and the request is reported beside the grant.
    plan.requested_bytes = explicit_size ? explicit_bytes : available_bytes;
    plan.budget_bytes = explicit_size ? std::min(explicit_bytes, available_bytes) : available_bytes;
    plan.clamped      = explicit_size && explicit_bytes > available_bytes;

    // The staging buffer is what makes a non-resident bank readable on the device at all, so it is
    // taken first. A card that cannot afford it keeps every bank in host memory and reads them in
    // place, which is the route this family ran before the split existed.
    if (plan.budget_bytes < staging) {
        plan.budget_bytes  = 0;
        plan.staging_bytes = 0;
        return plan;
    }
    plan.staging_bytes = staging;

    std::size_t resident_budget = plan.budget_bytes - staging;
    // Largest bank first, so the resident set is the one that keeps the most bytes off the host
    // link; ties break on the pool index. The order is total, so two runs of the same load pick the
    // same set.
    std::vector<std::int32_t> order(banks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
        const std::int64_t left  = expert_bank_bytes(banks[a]);
        const std::int64_t right = expert_bank_bytes(banks[b]);
        return left != right ? left > right : a < b;
    });
    for (const std::int32_t layer : order) {
        const std::size_t bytes = static_cast<std::size_t>(expert_bank_bytes(banks[layer]));
        if (bytes > resident_budget) { continue; }
        plan.resident[layer] = true;
        plan.resident_bytes += bytes;
        plan.resident_experts += banks[layer].experts;
        resident_budget -= bytes;
    }
    return plan;
}

} // namespace ninfer::models::nemotron_h
