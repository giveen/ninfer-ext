// Implements: models/nemotron_h/experts.h
#include "models/nemotron_h/experts.h"

#include "models/nemotron_h/forward.h" // Model: the pool is built from its bound banks

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
