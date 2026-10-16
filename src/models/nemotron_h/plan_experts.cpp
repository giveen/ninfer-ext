// Implements: models/nemotron_h/plan_experts.h
//
// Pure config-to-plan arithmetic, so it is compiled into the loading library: the loader calls it
// while resolving the artifact, and the execution side consumes the plan it produces. Nothing here
// touches a device or an execution type.
#include "models/nemotron_h/plan_experts.h"

#include <algorithm>
#include <numeric>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ninfer::models::nemotron_h {
namespace {

// The number of staged banks, which is also what the pool's device bytes reserve. It mirrors
// `execution::ExpertPager::kBankCount` -- one layer being read while the next one is being filled
// -- and is stated here rather than included from there because the loading side must not depend on
// an execution header. A change to one without the other would put the staged banks out of the
// pool.
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

namespace {

// The staged banks' slots -- the pool the pager's two banks overlay -- and, for a pool with a slot
// cache, the tables one slot costs: `owner` (I32) and `stamp` (U64) per slot. The tables are stated
// once here because both the pool's own device bytes and the *sizing* of a cache slot's share of a
// budget are functions of them, and a 12-byte term solved for differently in two places is how a
// grant and an allocation come out one slot apart. `slot_entries` is the pool's expert space, which
// no cache slot changes.
constexpr std::size_t kSlotTableBytes = sizeof(std::int32_t) + sizeof(std::uint64_t);
constexpr std::size_t kCounterBytes   = 4 * sizeof(std::uint64_t);

// Bytes a pool's tables need for `slots` slots.
std::size_t pool_table_bytes(std::int32_t slot_entries, std::int32_t slots) {
    return static_cast<std::size_t>(slot_entries) * sizeof(std::int32_t) +
           static_cast<std::size_t>(slots) * kSlotTableBytes + kCounterBytes;
}

} // namespace

std::size_t expert_pool_bytes(std::span<const ops::ExpertBankShape> banks,
                              std::int32_t cache_slots) {
    if (banks.empty()) { return 0; }
    if (cache_slots < 0) {
        throw std::invalid_argument("nemotron_h expert pool: a slot cache cannot have fewer than "
                                    "zero slots");
    }
    const ops::ExpertPoolGeometry geometry(banks);
    const std::int32_t bank_slots =
        static_cast<std::int32_t>(kStagedBanks) * geometry.max_experts();
    const std::int32_t slots = bank_slots + cache_slots;
    return expert_staging_bytes(banks) +
           static_cast<std::size_t>(cache_slots) *
               static_cast<std::size_t>(geometry.pool_stride()) +
           pool_table_bytes(geometry.slot_entries(), slots);
}

std::size_t expert_pool_bytes(const TextConfig& config, std::int32_t cache_slots) {
    return expert_pool_bytes(expert_bank_geometry(config), cache_slots);
}

std::int32_t expert_top_k(const TextConfig& config) {
    std::int32_t top_k = 0;
    for (std::uint32_t block = 0; block < config.num_hidden_layers(); ++block) {
        if (config.block_kind(block) != BlockKind::Moe) { continue; }
        top_k =
            std::max(top_k, static_cast<std::int32_t>(config.moe_bank(block).experts_per_token));
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
                                          std::size_t explicit_bytes,
                                          std::int32_t requested_slots) {
    ExpertResidencyPlan plan;
    plan.resident.assign(banks.size(), false);
    if (banks.empty()) { return plan; }
    if (requested_slots < 0) {
        throw std::invalid_argument("nemotron_h expert residency: a slot cache cannot have fewer "
                                    "than zero slots");
    }

    const ops::ExpertPoolGeometry geometry(banks);
    const std::int32_t bank_slots =
        static_cast<std::int32_t>(kStagedBanks) * geometry.max_experts();
    const std::size_t bank_banks = expert_staging_bytes(banks);
    for (const ops::ExpertBankShape& bank : banks) { plan.pool_experts += bank.experts; }

    // The budget is the whole expert footprint -- the pool and the residents together -- and it is
    // the caller's cap or everything the card has, whichever is smaller. A cap is clamped rather
    // than over-committed, and the request is reported beside the grant.
    plan.requested_bytes = explicit_size ? explicit_bytes : available_bytes;
    plan.budget_bytes = explicit_size ? std::min(explicit_bytes, available_bytes) : available_bytes;
    plan.clamped      = explicit_size && explicit_bytes > available_bytes;
    plan.requested_slots = requested_slots;

    // The two staged banks are what makes a non-resident bank readable on the device at all, so
    // they and their tables are taken first. A card that cannot afford them keeps every bank in
    // host memory and reads them in place, which is the route this family ran before the split
    // existed.
    const std::size_t bank_tables = pool_table_bytes(geometry.slot_entries(), bank_slots);
    if (plan.budget_bytes < bank_banks + bank_tables) {
        plan.budget_bytes  = 0;
        plan.staging_bytes = 0;
        return plan;
    }

    // The slot cache is the next claim on the budget, and the only elastic term of the pool: a slot
    // costs its bytes of pool *and* its twelve bytes of table, and every slot granted here is one
    // slot's worth of bank that cannot be made resident. So the count the card can afford solves
    // `slots * (stride + table) <= what the banks and their tables leave`, and a request above it
    // is granted at the cap and reported rather than refused -- a slot request is a share of the
    // budget, not a demand.
    const std::size_t room     = plan.budget_bytes - bank_banks - bank_tables;
    const std::size_t per_slot = static_cast<std::size_t>(geometry.pool_stride()) + kSlotTableBytes;
    const std::size_t affordable =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) < room / per_slot
            ? static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
            : room / per_slot;
    const std::size_t granted = std::min(static_cast<std::size_t>(requested_slots), affordable);
    plan.cache_slots          = static_cast<std::int32_t>(granted);
    plan.slots_clamped        = granted != static_cast<std::size_t>(requested_slots);
    plan.staging_bytes        = bank_banks + granted * per_slot + bank_tables;

    std::size_t resident_budget = plan.budget_bytes - plan.staging_bytes;
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
