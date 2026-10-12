// ninfer::ops::gemma_moe_experts against its FP64 oracle.
//
// The second case is the one worth having: the expert MLP is stored padded to the block-scaled
// layout's row multiple, and the padding is only arithmetically invisible if the gate and up
// padding rows and the down padding columns are all zero. The Op is run at the stored width over a
// bank whose upper rows and columns are zero and is required to agree with the oracle evaluated at
// the checkpoint's own width over the truncated bank.
#include "ninfer/ops/gemma_moe_experts.h"

#include "ops/direct_bf16_weight.h"
#include "ops/gemma_moe_experts_ref.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using ninfer::test::gemma_moe_experts_ref::run;

namespace {

// The returned row is BF16, whose own relative resolution is about 3.9e-3, and the Op's private
// accumulator is more precise than the oracle's target, so the criterion is that resolution with a
// floor for near-zero outputs.
constexpr PointwiseCriterion kCriterion = {/*absolute*/ 1e-3, /*relative*/ 1e-2};

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

// A patterned bank, with every row in [kept_rows, rows) and every column in [kept_columns, columns)
// forced to zero: that is what the artifact's padding looks like.
direct_bf16_weight::HostWeight padded_bank(std::int32_t rows, std::int32_t columns,
                                           std::int32_t kept_rows, std::int32_t kept_columns,
                                           std::uint32_t seed) {
    direct_bf16_weight::HostWeight bank = direct_bf16_weight::make_patterned(rows, columns, seed);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            if (row >= kept_rows || column >= kept_columns) {
                bank.bits[static_cast<std::size_t>(row) * columns + column] = f32_to_bf16(0.0F);
            }
        }
    }
    return bank;
}

// The first `kept_rows` rows and `kept_columns` columns of a bank, as its own narrower bank.
direct_bf16_weight::HostWeight truncate(const direct_bf16_weight::HostWeight& bank,
                                        std::int32_t kept_rows, std::int32_t kept_columns) {
    direct_bf16_weight::HostWeight narrow;
    narrow.n = kept_rows;
    narrow.k = kept_columns;
    narrow.bits.resize(static_cast<std::size_t>(kept_rows) * kept_columns);
    for (std::int32_t row = 0; row < kept_rows; ++row) {
        for (std::int32_t column = 0; column < kept_columns; ++column) {
            narrow.bits[static_cast<std::size_t>(row) * kept_columns + column] =
                bank.bits[static_cast<std::size_t>(row) * bank.k + column];
        }
    }
    return narrow;
}

int run_case(const char* label, std::int32_t hidden, std::int32_t experts, std::int32_t top_k,
             std::int32_t stored_intermediate, const direct_bf16_weight::HostWeight& gate_up,
             const direct_bf16_weight::HostWeight& down,
             const direct_bf16_weight::HostWeight& oracle_gate_up,
             const direct_bf16_weight::HostWeight& oracle_down, std::int32_t oracle_intermediate,
             std::int32_t tokens) {
    std::vector<float> x(static_cast<std::size_t>(hidden) * tokens);
    fill_uniform(x, 21, -3.0F, 3.0F);
    round_to_bf16(x);
    const auto x_bits = encode(x);

    // A deterministic selection over distinct experts, and weights that do not sum to one.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> routing(static_cast<std::size_t>(top_k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * tokens + token;
            ids[at]              = (rank * 3 + token * 5) % experts;
            routing[at]          = 0.2F + 0.1F * static_cast<float>((rank + token) % 4);
        }
    }

    const auto reference = run(x_bits, hidden, tokens, ids, routing, oracle_gate_up, oracle_down,
                               oracle_intermediate, top_k);

    direct_bf16_weight::DeviceWeight gate_up_device(gate_up);
    direct_bf16_weight::DeviceWeight down_device(down);
    GuardedDeviceBuffer x_device(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer routing_device(routing.size() * sizeof(float));
    GuardedDeviceBuffer out_device(x_bits.size() * sizeof(std::uint16_t));
    x_device.copy_from_host(x_bits.data(), x_device.bytes());
    ids_device.copy_from_host(ids.data(), ids_device.bytes());
    routing_device.copy_from_host(routing.data(), routing_device.bytes());

    Tensor x_tensor(x_device.data(), DType::BF16, {hidden, tokens});
    Tensor ids_tensor(ids_device.data(), DType::I32, {top_k, tokens});
    Tensor routing_tensor(routing_device.data(), DType::FP32, {top_k, tokens});
    Tensor out_tensor(out_device.data(), DType::BF16, {hidden, tokens});

    ops::GemmaMoeExpertWeights weights;
    weights.gate_up = gate_up_device.view();
    weights.down    = down_device.view();
    const ops::GemmaMoeExpertGeometry geometry{hidden, experts, top_k, stored_intermediate};

    // The output buffer starts as a poisoned pattern, so a region the Op fails to write shows up.
    std::vector<std::uint16_t> poison(x_bits.size(), f32_to_bf16(-12345.0F));
    out_device.copy_from_host(poison.data(), out_device.bytes());
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(out_device.data(), static_cast<std::int64_t>(x_bits.size()));
    int failures =
        verify_pointwise(std::string(label) + " out", std::vector<double>(got.begin(), got.end()),
                         reference, kCriterion);
    failures += gate_up_device.verify_preserved(label);
    failures += down_device.verify_preserved(label);
    return failures;
}

// A geometry big enough to exercise the expert-major row arithmetic without a gigabyte of banks.
int run_medium_case() {
    constexpr std::int32_t kHidden       = 512;
    constexpr std::int32_t kExperts      = 16;
    constexpr std::int32_t kTopK         = 4;
    constexpr std::int32_t kIntermediate = 256;
    constexpr std::int32_t kTokens       = 8;
    const auto gate_up =
        direct_bf16_weight::make_patterned(kExperts * 2 * kIntermediate, kHidden, 3);
    const auto down = direct_bf16_weight::make_patterned(kExperts * kHidden, kIntermediate, 4);
    return run_case("medium", kHidden, kExperts, kTopK, kIntermediate, gate_up, down, gate_up, down,
                    kIntermediate, kTokens);
}

// The checkpoint's width is 32 here and the stored width is 48; the Op is run at 48 over a bank
// whose rows and columns [32, 48) are zero and must reproduce the oracle at 32.
int run_padding_case() {
    constexpr std::int32_t kHidden  = 64;
    constexpr std::int32_t kExperts = 4;
    constexpr std::int32_t kTopK    = 2;
    constexpr std::int32_t kStored  = 48;
    constexpr std::int32_t kLogical = 32;
    constexpr std::int32_t kTokens  = 4;
    const auto gate_up =
        padded_bank(kExperts * 2 * kStored, kHidden, kExperts * 2 * kLogical, kHidden, 5);
    const auto down = padded_bank(kExperts * kHidden, kStored, kExperts * kHidden, kLogical, 6);

    // The oracle's own view of the same numbers: the logical width of each expert's rows.
    direct_bf16_weight::HostWeight narrow_gate_up;
    narrow_gate_up.n = kExperts * 2 * kLogical;
    narrow_gate_up.k = kHidden;
    narrow_gate_up.bits.resize(static_cast<std::size_t>(narrow_gate_up.n) * kHidden);
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        for (std::int32_t half = 0; half < 2; ++half) {
            for (std::int32_t i = 0; i < kLogical; ++i) {
                const std::size_t from =
                    (static_cast<std::size_t>(expert) * 2 * kStored + half * kStored + i) * kHidden;
                const std::size_t to =
                    (static_cast<std::size_t>(expert) * 2 * kLogical + half * kLogical + i) *
                    kHidden;
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    narrow_gate_up.bits[to + h] = gate_up.bits[from + h];
                }
            }
        }
    }
    const auto narrow_down = truncate(down, kExperts * kHidden, kLogical);
    return run_case("padded width", kHidden, kExperts, kTopK, kStored, gate_up, down,
                    narrow_gate_up, narrow_down, kLogical, kTokens);
}

} // namespace

int main() {
    int failures = 0;
    try {
        failures += run_medium_case();
        failures += run_padding_case();
    } catch (const std::exception& error) {
        std::cerr << "gemma_moe_experts: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) {
        std::cout << "OK: gemma_moe_experts matches its oracle and the padded width is inert\n";
    }
    return failures == 0 ? 0 : 1;
}
