// Implements: models/nemotron_h/experts.h
#include "models/nemotron_h/experts.h"

#include "models/nemotron_h/forward.h" // Model: the pool is built from its bound banks

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ninfer::models::nemotron_h {
namespace {

// Both routed matrices of this family are block-scaled NVFP4 with one weight divisor per bank
// matrix. Anything else is another conversion's bank and has no route through this Op.
void require_nvfp4_bank(const Weight& weight, std::int64_t rows, std::int64_t columns,
                        const char* label) {
    const auto fail = [label](const std::string& why) {
        throw std::invalid_argument(std::string("nemotron_h routed experts: ") + label + " " + why);
    };
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group_size != 16) {
        fail("must be block-scaled NVFP4 (K16 M128x4)");
    }
    if (weight.padded_shape[1] != 0 && weight.padded_shape[1] != columns) {
        fail("must keep the whole parent's K");
    }
    if (static_cast<std::int64_t>(weight.n) != rows ||
        static_cast<std::int64_t>(weight.k) != columns) {
        fail("is not the layer's own bank shape");
    }
    if (weight.qdata == nullptr || weight.scales == nullptr || weight.weight_divisors == nullptr ||
        weight.weight_divisor_rows <= 0 || (rows % weight.weight_divisor_rows) != 0) {
        fail("has no resident planes or no weight divisors covering its rows");
    }
}

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
    return static_cast<std::size_t>(execution::ExpertPager::bank_slots(geometry)) *
               static_cast<std::size_t>(geometry.pool_stride()) +
           2 * static_cast<std::size_t>(geometry.max_tail_bytes());
}

std::size_t expert_pool_bytes(const TextConfig& config) {
    const std::vector<ops::ExpertBankShape> banks = expert_bank_geometry(config);
    if (banks.empty()) { return 0; }
    // The staging buffer, then the pager's tables: one slot table entry per expert of the pool and
    // one owner, stamp and counter set per slot of the two banks.
    const ops::ExpertPoolGeometry geometry(banks);
    const std::size_t slots = static_cast<std::size_t>(execution::ExpertPager::bank_slots(geometry));
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

ops::ExpertWeights expert_bank_addressing(const Weight& up, const Weight& down,
                                          std::int32_t experts, std::int32_t latent,
                                          std::int32_t intermediate, const char* label) {
    require_nvfp4_bank(up, static_cast<std::int64_t>(experts) * intermediate, latent, label);
    require_nvfp4_bank(down, static_cast<std::int64_t>(experts) * latent, intermediate, label);
    const ops::ExpertBankShape shape = ops::nvfp4_expert_bank(experts, latent, intermediate, 1);
    ops::ExpertWeights bank;
    bank.layout  = shape.layout;
    bank.base[0] = static_cast<const std::byte*>(up.qdata);
    bank.base[1] = static_cast<const std::byte*>(up.scales);
    bank.base[2] = static_cast<const std::byte*>(down.qdata);
    bank.base[3] = static_cast<const std::byte*>(down.scales);
    for (std::int32_t plane = 0; plane < shape.layout.planes; ++plane) {
        bank.stride[plane] = shape.layout.plane_bytes[plane];
    }
    bank.gate_up_divisors     = static_cast<const float*>(up.weight_divisors);
    bank.gate_up_divisor_rows = up.weight_divisor_rows;
    bank.down_divisors        = static_cast<const float*>(down.weight_divisors);
    bank.down_divisor_rows    = down.weight_divisor_rows;
    return bank;
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

RoutedExpertPool::RoutedExpertPool(const Model& model, std::int32_t top_k,
                                   std::int32_t staged_columns)
    : banks_(expert_bank_geometry(model.config())), resident_(model.expert_residency().resident) {
    if (banks_.empty()) {
        throw std::invalid_argument("nemotron_h routed experts: a pool needs at least one bank");
    }
    if (resident_.size() != banks_.size()) {
        throw std::invalid_argument(
            "nemotron_h routed experts: the residency plan does not describe this pool's layers");
    }
    geometry_                = std::make_unique<ops::ExpertPoolGeometry>(banks_);
    const std::int32_t slots = execution::ExpertPager::bank_slots(*geometry_);
    if (slots < top_k) {
        throw std::invalid_argument("nemotron_h routed experts: the staged banks hold fewer slots "
                                    "than one call assigns");
    }
    // The one mapping from a pool layer to its stored bank: the same `expert_pool_layer` the loader
    // planned with, walked in block order, so a pool whose layers drifted from the model's is
    // refused here rather than reading another layer's experts.
    const TextConfig& config = model.config();
    host_.reserve(banks_.size());
    for (std::uint32_t block = 0; block < config.num_hidden_layers(); ++block) {
        if (config.block_kind(block) != BlockKind::Moe) { continue; }
        const std::optional<MoeWeights>& moe = model.weights().text.layers[block].moe;
        if (!moe) {
            throw std::invalid_argument("nemotron_h routed experts: a MoE block has no bank");
        }
        const MoeBankConfig& bank = config.moe_bank(block);
        const std::int32_t latent =
            static_cast<std::int32_t>(bank.expert_input_width(config.hidden_size));
        host_.push_back(expert_bank_addressing(
            model.matrix(moe->experts_up), model.matrix(moe->experts_down),
            static_cast<std::int32_t>(bank.num_experts), latent,
            static_cast<std::int32_t>(bank.intermediate_size), "routed experts/up"));
        const ops::ExpertBankShape& shape = banks_[host_.size() - 1];
        if (shape.experts != static_cast<std::int32_t>(bank.num_experts) ||
            shape.gate_up_rows != static_cast<std::int64_t>(bank.num_experts) *
                                      static_cast<std::int64_t>(bank.intermediate_size) ||
            shape.down_rows != static_cast<std::int64_t>(bank.num_experts) * latent) {
            throw std::invalid_argument("nemotron_h routed experts: the pool's bank for block " +
                                        std::to_string(block) + " is not the layer's own");
        }
    }
    if (host_.size() != banks_.size()) {
        throw std::invalid_argument(
            "nemotron_h routed experts: the pool has a bank no MoE block of the model owns");
    }
    // Two whole-layer banks: bank b holds (or is being filled with) the layer the pass is on, while
    // the other bank takes the next staged layer's copy. The divisor tails past the slot region are
    // what keep the second bank's divisors in bounds.
    const std::size_t pool_bytes =
        static_cast<std::size_t>(slots) * static_cast<std::size_t>(geometry_->pool_stride()) +
        2 * static_cast<std::size_t>(geometry_->max_tail_bytes());
    pool_ = DeviceBuffer(pool_bytes);
    pool_.fill(0);
    slot_of_ =
        DeviceBuffer(static_cast<std::size_t>(geometry_->slot_entries()) * sizeof(std::int32_t));
    owner_    = DeviceBuffer(static_cast<std::size_t>(slots) * sizeof(std::int32_t));
    stamp_    = DeviceBuffer(static_cast<std::size_t>(slots) * sizeof(std::uint64_t));
    counters_ = DeviceBuffer(4 * sizeof(std::uint64_t));
    // -1 is "absent": every expert is uncached, every slot unowned. A staged fill reads these to
    // skip experts the cache holds, so they must start in the state the pool is actually in.
    slot_of_.fill(0xFF);
    owner_.fill(0xFF);
    stamp_.fill(0);
    counters_.fill(0);

    const ops::ExpertCacheState cache{.slot_of = static_cast<std::int32_t*>(slot_of_.p),
                                      .owner   = static_cast<std::int32_t*>(owner_.p),
                                      .stamp   = static_cast<unsigned long long*>(stamp_.p),
                                      .clock   = static_cast<unsigned long long*>(counters_.p),
                                      .statistics =
                                          static_cast<unsigned long long*>(counters_.p) + 1,
                                      .slots        = slots,
                                      .layers       = geometry_->layers(),
                                      .slot_entries = geometry_->slot_entries(),
                                      .pool_stride  = geometry_->pool_stride(),
                                      .pool         = static_cast<std::byte*>(pool_.p)};
    // The staged banks are the whole slot range, so the pager's cache route has no slots of its own
    // and `resolve` is never called: this family stages a bank whole and has no per-expert cache.
    pager_.emplace(*geometry_, cache, staged_columns, top_k, 0, slots);
}

RoutedExpertPool::~RoutedExpertPool() = default;

std::int32_t RoutedExpertPool::next_staged(std::int32_t layer) const {
    for (std::int32_t after = layer + 1; after < geometry_->layers(); ++after) {
        if (!resident_[after]) { return after; }
    }
    return -1;
}

RoutedExpertPool::Placed RoutedExpertPool::acquire(std::int32_t layer, cudaStream_t stream) {
    if (layer < 0 || layer >= geometry_->layers()) {
        throw std::out_of_range("nemotron_h routed experts: no such pool layer");
    }
    // A resident bank was uploaded once and is read where it lives: nothing to stage, nothing to
    // release.
    if (resident_[layer]) { return Placed{host_[layer], geometry_->span(layer), layer, false}; }
    const bool forward_start        = forward_start_;
    forward_start_                  = false;
    const std::int32_t next         = next_staged(layer);
    const ops::ExpertWeights staged = pager_->acquire_staged(
        layer, host_[layer], next, next >= 0 ? &host_[next] : nullptr, forward_start, stream);
    return Placed{staged, geometry_->span(layer), layer, true};
}

void RoutedExpertPool::release(std::int32_t layer, cudaStream_t stream) {
    pager_->release_staged(layer, stream);
}

std::size_t RoutedExpertPool::allocated_bytes() const noexcept {
    return pool_.bytes + slot_of_.bytes + owner_.bytes + stamp_.bytes + counters_.bytes;
}

} // namespace ninfer::models::nemotron_h
