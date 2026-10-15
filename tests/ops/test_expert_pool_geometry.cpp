// ExpertPoolGeometry over the bank tables the two served families actually have.
//
// The pool's whole job is to turn a table of per-layer banks into addresses, and every one of those
// addresses is silent when it is wrong: a slot table one expert short, a slot stride taken from the
// narrowest bank, or an owner decoded by a division reads and writes the wrong bytes and produces a
// plausible model. So the cases below check the three decoded quantities directly rather than
// through an Op:
//
//   * the offset table. Every layer's `slot_base` is the prefix sum of the banks before it, checked
//     at both ends of every layer, so an implementation that assumed a global expert count fails on
//     the first layer whose count differs from its predecessor's.
//   * the owner round trip. Every (layer, expert) pair of the pool is encoded and decoded back, and
//     the decode is checked to be the *layer's*, not a division by any one bank's width.
//   * the per-layer byte counts. A tile is sized from the widest bank and addressed from the
//     layer's own, so the two disagree exactly when one of them was taken from the wrong bank.
//
// The Nemotron table is the released checkpoint's own `block_configs` order -- 40 MoE layers of 240
// to 496 experts at four intermediate widths, twelve (experts, intermediate) pairs -- because a
// synthetic table of equal counts is the case that hides all three bugs. Qwen4Exp's uniform pool is
// the second case: it is the regression that says the general path still reduces to the one it
// replaced, where `slot_base(layer) == layer * experts` and the pool stride is the single
// `layout.slot_bytes`.
#include "ninfer/ops/offload_moe.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::ops::ExpertBankShape;
using ninfer::ops::ExpertPoolGeometry;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "expert_pool_geometry: " << what << '\n';
        ++failures;
    }
}

void check_equal(std::int64_t got, std::int64_t want, const std::string& what) {
    if (got != want) {
        std::cerr << "expert_pool_geometry: " << what << ": got " << got << ", want " << want
                  << '\n';
        ++failures;
    }
}

// The NVFP4 expert of a bank whose experts read the `latent` width and write `intermediate`: codes
// then scales for `gate_up_matrices` matrices, then the same for one down matrix. Stated here from
// the format rather than read from the builder, which is the point of comparing them.
[[nodiscard]] std::int64_t expected_slot_bytes(std::int64_t latent, std::int64_t intermediate,
                                               std::int64_t gate_up_matrices) {
    return gate_up_matrices * intermediate * latent / 2 +
           gate_up_matrices * intermediate * latent / 16 + latent * intermediate / 2 +
           latent * intermediate / 16;
}

// One weight divisor per expert and matrix, FP32, after the planes.
[[nodiscard]] std::int64_t expected_tail_bytes(std::int64_t experts,
                                               std::int64_t gate_up_matrices) {
    return (gate_up_matrices + 1) * experts * 4;
}

struct Layer {
    std::int32_t experts;
    std::int32_t intermediate;
};

// Nemotron-3.5-Bolt-74B-A9B's 40 MoE layers in block order, as the released checkpoint's
// `block_configs` declares them: (n_routed_experts, moe_intermediate_size) per MoE block. The
// layers are not sorted and not grouped, so the prefix sums are the only thing that addresses them.
constexpr std::array<Layer, 40> kNemotronLayers = {{
    {240, 2048}, {240, 1536}, {240, 2048}, {304, 2304}, {240, 2304}, {240, 2048}, {240, 2048},
    {304, 2304}, {304, 2048}, {432, 2560}, {432, 2304}, {240, 2048}, {432, 2304}, {304, 2560},
    {304, 2304}, {432, 2304}, {368, 2304}, {368, 2048}, {368, 2304}, {432, 2560}, {368, 2560},
    {496, 2560}, {432, 2560}, {240, 2304}, {368, 2304}, {368, 2560}, {304, 2048}, {240, 2304},
    {240, 2304}, {240, 2048}, {240, 2048}, {368, 2048}, {368, 2304}, {432, 2304}, {432, 2560},
    {496, 2560}, {432, 2560}, {432, 2560}, {496, 2560}, {368, 2560},
}};

// The width the experts of this family read: the checkpoint's `moe_latent_size`, so every layer's
// bank is bound at the same input width and the experts are not gated (`gate_up_matrices` is one).
constexpr std::int32_t kNemotronLatent  = 1024;
constexpr std::int32_t kNemotronTotal   = 13824;         // experts over the 40 layers
constexpr std::int64_t kNemotronStride  = 1152LL * 2560; // the widest bank's expert
constexpr std::int32_t kNemotronWidest  = 496;
constexpr std::int64_t kNemotronMaxTail = 8LL * 496;

[[nodiscard]] std::vector<ExpertBankShape> nemotron_banks() {
    std::vector<ExpertBankShape> banks;
    for (const Layer& layer : kNemotronLayers) {
        banks.push_back(
            ninfer::ops::nvfp4_expert_bank(layer.experts, kNemotronLatent, layer.intermediate, 1));
    }
    return banks;
}

// The Qwen4Exp pool: every one of its cache layers is the same bank, which is what its constructor
// refuses a checkpoint for failing to be.
std::vector<ExpertBankShape> uniform_banks(std::int32_t layers) {
    return std::vector<ExpertBankShape>(static_cast<std::size_t>(layers),
                                        ninfer::ops::kQwen4ExpertBank);
}

int nvfp4_bank_case() {
    // The one bank shape the Qwen4Exp pool is built from must be exactly the constants its closed
    // geometry is written against, or the pool and the Op would disagree about one expert.
    const ExpertBankShape qwen4 = ninfer::ops::nvfp4_expert_bank(512, 2560, 640, 2);
    check_equal(qwen4.experts, ninfer::ops::kOffloadMoeExperts, "qwen4 experts");
    check_equal(qwen4.layout.planes, 4, "qwen4 planes");
    check_equal(qwen4.layout.plane_bytes[0], ninfer::ops::kExpertGateUpCodeBytes,
                "qwen4 gate/up codes");
    check_equal(qwen4.layout.plane_bytes[1], ninfer::ops::kExpertGateUpScaleBytes,
                "qwen4 gate/up scales");
    check_equal(qwen4.layout.plane_bytes[2], ninfer::ops::kExpertDownCodeBytes, "qwen4 down codes");
    check_equal(qwen4.layout.plane_bytes[3], ninfer::ops::kExpertDownScaleBytes,
                "qwen4 down scales");
    check_equal(qwen4.layout.slot_bytes, ninfer::ops::kExpertSlotBytes, "qwen4 slot bytes");
    check_equal(qwen4.layout.tail_bytes, ninfer::ops::kExpertDivisorBytes, "qwen4 divisor bytes");
    check(qwen4.layout == ninfer::ops::nvfp4_expert_layout(), "qwen4 layout is the constant one");
    check_equal(qwen4.layout.staged_layer_bytes(512), ninfer::ops::kExpertStagedLayerBytes,
                "qwen4 staged layer bytes");
    // Gate and up share the first plane pair, so the divisor tail holds two per expert there.
    check_equal(qwen4.gate_up_divisors, 2 * 512, "qwen4 gate/up divisors");
    check_equal(qwen4.down_divisors, 512, "qwen4 down divisors");
    check_equal(qwen4.gate_up_rows, 512LL * 2 * 640, "qwen4 gate/up rows");
    check_equal(qwen4.down_rows, 512LL * 2560, "qwen4 down rows");
    return 0;
}

int uniform_pool_case() {
    // 41 layers: the 40 MoE blocks plus the MTP block, which the cache addresses as one more layer.
    constexpr std::int32_t kLayers  = 41;
    constexpr std::int32_t kExperts = ninfer::ops::kOffloadMoeExperts;
    const ExpertPoolGeometry pool(uniform_banks(kLayers));

    check_equal(pool.layers(), kLayers, "uniform layers");
    check_equal(pool.slot_entries(), std::int64_t(kLayers) * kExperts, "uniform slot entries");
    check_equal(pool.max_experts(), kExperts, "uniform max experts");
    check_equal(pool.pool_stride(), ninfer::ops::kExpertSlotBytes, "uniform pool stride");
    check_equal(pool.max_staged_bytes(), ninfer::ops::kExpertStagedLayerBytes,
                "uniform max staged bytes");
    check_equal(pool.max_tail_bytes(), ninfer::ops::kExpertDivisorBytes, "uniform max tail bytes");

    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        // The uniform pool's offsets are the arithmetic the cache used before it had a table, so
        // this is the regression that says the general one still produces them.
        check_equal(pool.slot_base(layer), std::int64_t(layer) * kExperts, "uniform slot base");
        check_equal(pool.span(layer).experts, kExperts, "uniform span experts");
        check_equal(pool.staged_bytes(layer), ninfer::ops::kExpertStagedLayerBytes,
                    "uniform staged bytes");
        for (const std::int32_t expert : {0, 1, kExperts / 2, kExperts - 1}) {
            const std::int32_t owner = pool.owner(layer, expert);
            check_equal(owner, std::int64_t(layer) * kExperts + expert, "uniform owner");
            check_equal(pool.owner_layer(owner), layer, "uniform owner layer");
            check_equal(pool.owner_expert(owner), expert, "uniform owner expert");
        }
    }
    return 0;
}

int nemotron_pool_case() {
    const ExpertPoolGeometry pool(nemotron_banks());

    check_equal(pool.layers(), static_cast<std::int64_t>(kNemotronLayers.size()),
                "nemotron layers");
    check_equal(pool.slot_entries(), kNemotronTotal, "nemotron slot entries");
    check_equal(pool.max_experts(), kNemotronWidest, "nemotron widest bank");
    // The pool's own stride is its widest bank's, so the narrowest layer's expert is a short run
    // inside a slot rather than a second pool.
    check_equal(pool.pool_stride(), kNemotronStride, "nemotron pool stride");
    check_equal(pool.max_tail_bytes(), kNemotronMaxTail, "nemotron max tail bytes");

    std::int64_t base          = 0;
    std::int64_t widest_staged = 0;
    for (std::size_t layer = 0; layer < kNemotronLayers.size(); ++layer) {
        const Layer& bank     = kNemotronLayers[layer];
        const auto index      = static_cast<std::int32_t>(layer);
        const auto slot_bytes = expected_slot_bytes(kNemotronLatent, bank.intermediate, 1);
        const auto staged     = bank.experts * slot_bytes + expected_tail_bytes(bank.experts, 1);
        widest_staged         = std::max(widest_staged, staged);

        check_equal(pool.slot_base(index), base, "nemotron slot base");
        check_equal(pool.span(index).first_expert, base, "nemotron span first expert");
        check_equal(pool.span(index).experts, bank.experts, "nemotron span experts");
        check_equal(pool.experts(index), bank.experts, "nemotron layer experts");
        check_equal(pool.staged_bytes(index), staged, "nemotron staged bytes");
        check(pool.staged_bytes(index) % 16 == 0, "nemotron staged layer stays 16-byte aligned");

        // Both ends of the layer, because a boundary is where a division by any single bank's width
        // first goes wrong -- and the two ends differ whenever the neighbours differ.
        for (const std::int32_t expert : {0, 1, bank.experts / 2, bank.experts - 1}) {
            const std::int32_t owner = pool.owner(index, expert);
            check_equal(owner, base + expert, "nemotron owner");
            check_equal(pool.owner_layer(owner), index, "nemotron owner layer");
            check_equal(pool.owner_expert(owner), expert, "nemotron owner expert");
        }
        base += bank.experts;
    }
    check_equal(base, kNemotronTotal, "nemotron prefix sum ends at the pool's experts");
    check_equal(pool.max_staged_bytes(), widest_staged, "nemotron max staged bytes");

    // The banks overlay the top slots of the pool, so the pool must be wide enough to hold them:
    // two staged layers of the widest bank, plus the two divisor tails the slot region does not
    // cover. Equal here, which is what makes the overlay exact rather than merely sufficient.
    check(2 * pool.max_staged_bytes() <=
              2 * std::int64_t(pool.max_experts()) * pool.pool_stride() + 2 * pool.max_tail_bytes(),
          "the staged banks fit the top of the pool");

    // Every expert of the pool decodes back to the layer that owns it. A division by the widest
    // bank would report layer 27 for expert 13456 -- layer 39's first expert -- so this sweep is
    // what distinguishes the prefix sum from a stride.
    for (std::int32_t owner = 0; owner < pool.slot_entries(); ++owner) {
        const std::int32_t layer = pool.owner_layer(owner);
        if (layer < 0 || static_cast<std::size_t>(layer) >= kNemotronLayers.size() ||
            pool.span(layer).first_expert > owner ||
            owner >= pool.span(layer).first_expert + pool.span(layer).experts) {
            check(false, "nemotron owner sweep found a layer that does not own its expert");
            break;
        }
    }
    check_equal(pool.owner_layer(kNemotronTotal - 1), 39,
                "the last expert belongs to the last layer");
    check_equal(pool.owner_expert(kNemotronTotal - 1), 367, "the last expert's index");
    return 0;
}

int refusal_cases() {
    const auto refuses = [](std::vector<ExpertBankShape> banks, const std::string& what) {
        try {
            const ExpertPoolGeometry pool(banks);
            check(false, what + ": a pool that cannot be addressed was accepted");
        } catch (const std::invalid_argument&) {}
    };
    refuses({}, "empty pool");
    refuses({{0, 0, 0, 0, 0, ninfer::ops::nvfp4_expert_layout()}}, "a bank with no experts");

    // A staged layer that is not a multiple of the alignment its planes are read at would put the
    // next bank's first plane off that alignment.
    ExpertBankShape misaligned = ninfer::ops::nvfp4_expert_bank(240, kNemotronLatent, 1536, 1);
    misaligned.layout.tail_bytes += 4;
    refuses({misaligned}, "a staged layer off the alignment");

    const ExpertPoolGeometry pool(nemotron_banks());
    const auto rejects_owner = [&](std::int32_t owner) {
        try {
            (void)pool.owner_layer(owner);
            check(false, "an owner outside the pool was accepted");
        } catch (const std::invalid_argument&) {}
    };
    rejects_owner(-1);
    rejects_owner(kNemotronTotal);
    try {
        (void)pool.bank(40);
        check(false, "a layer outside the pool was accepted");
    } catch (const std::invalid_argument&) {}
    return 0;
}

} // namespace

int main() {
    nvfp4_bank_case();
    uniform_pool_case();
    nemotron_pool_case();
    refusal_cases();
    if (failures == 0) {
        std::cout << "OK: the pool's offsets, owner decode and per-layer staged sizes match the "
                     "banks of both served families\n";
    }
    return failures == 0 ? 0 : 1;
}
