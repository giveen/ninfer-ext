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

#include <algorithm>
#include <cstring>
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
    Tensor no_workspace;
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           no_workspace, nullptr);
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

// The artifact's own bank format. Each expert is a separately quantised source with its own
// divisor, and the fixture writes distinct divisors per source, so a reader that ignored the
// divisor array would disagree here. At the 26B's own H, stored I and K the Op takes its streamed
// route, which is also run twice and must reproduce itself exactly.
int run_nvfp4_case(const char* label, std::int32_t hidden, std::int32_t experts, std::int32_t top_k,
                   std::int32_t intermediate, std::int32_t tokens) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 2.0F;
    options.input_scale_divisor  = 1.0F;
    // One divisor per source matrix: an expert's gate rows, its up rows, then its down rows.
    options.weight_divisor_rows = intermediate;
    const auto gate_up          = quantized_weight::make_patterned_weight(
        QType::NVFP4, experts * 2 * intermediate, hidden, 31, options);
    options.weight_divisor_rows = hidden;
    const auto down = quantized_weight::make_patterned_weight(QType::NVFP4, experts * hidden,
                                                              intermediate, 32, options);

    std::vector<float> x(static_cast<std::size_t>(hidden) * tokens);
    fill_uniform(x, 41, -2.0F, 2.0F);
    round_to_bf16(x);
    const auto x_bits = encode(x);

    // Distinct experts within a token, and weights that do not sum to one.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> routing(static_cast<std::size_t>(top_k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * tokens + token;
            ids[at]              = (rank * 3 + 2 * token) % experts;
            routing[at]          = 0.3F + 0.2F * static_cast<float>(rank % 3);
        }
    }

    const auto reference = gemma_moe_experts_ref::run(x_bits, hidden, tokens, ids, routing,
                                                      gate_up, down, intermediate, top_k);

    GuardedDeviceBuffer gate_up_device(gate_up.payload.size());
    GuardedDeviceBuffer down_device(down.payload.size());
    gate_up_device.copy_from_host(gate_up.payload.data(), gate_up_device.bytes());
    down_device.copy_from_host(down.payload.data(), down_device.bytes());
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
    weights.gate_up = gate_up.device_weight(gate_up_device.data());
    weights.down    = down.device_weight(down_device.data());
    const ops::GemmaMoeExpertGeometry geometry{hidden, experts, top_k, intermediate};

    const std::size_t workspace_bytes =
        ops::gemma_moe_experts_workspace_bytes(geometry, QType::NVFP4, tokens);
    GuardedDeviceBuffer workspace_device(std::max<std::size_t>(workspace_bytes, 1));
    Tensor workspace(workspace_bytes > 0 ? workspace_device.data() : nullptr, DType::U8,
                     {static_cast<std::int32_t>(std::max<std::size_t>(workspace_bytes, 1))});

    std::vector<std::uint16_t> poison(x_bits.size(), f32_to_bf16(-12345.0F));
    out_device.copy_from_host(poison.data(), out_device.bytes());
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           workspace, nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(out_device.data(), static_cast<std::int64_t>(x_bits.size()));
    int failures   = verify_pointwise(std::string(label) + " out",
                                      std::vector<double>(got.begin(), got.end()), reference,
                                      kCriterion);

    // The same call again over a re-poisoned output: the route has no order the scheduler picks.
    out_device.copy_from_host(poison.data(), out_device.bytes());
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           workspace, nullptr);
    cuda_synchronize();
    const auto again =
        from_device_bf16(out_device.data(), static_cast<std::int64_t>(x_bits.size()));
    failures += verify_exact((std::string(label) + " run-to-run").c_str(), again, got);

    // Read only: the codes, scales and divisor planes come back exactly as they went in.
    failures +=
        verify_exact((std::string(label) + " gate_up preserved").c_str(),
                     from_device<std::uint8_t>(gate_up_device.data(),
                                               static_cast<std::int64_t>(gate_up_device.bytes())),
                     gate_up.payload);
    failures +=
        verify_exact((std::string(label) + " down preserved").c_str(),
                     from_device<std::uint8_t>(down_device.data(),
                                               static_cast<std::int64_t>(down_device.bytes())),
                     down.payload);
    return failures;
}


// An int4 group-32 row bank in row_group_v1: codes in [-8,7] stored as code + 8, one binary16
// scale per 32 elements of a row. The oracle reads it through its own decode below.
struct Q4RowsBank {
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::vector<std::int8_t> codes;    // [n, k]
    std::vector<std::uint16_t> scales; // [n, k / 32], binary16

    double value(std::int32_t row, std::int32_t column) const {
        return static_cast<double>(codes[static_cast<std::size_t>(row) * k + column]) *
               static_cast<double>(quantized_weight::detail::f16_to_f32(
                   scales[static_cast<std::size_t>(row) * (k / 32) + column / 32]));
    }

    std::size_t scale_offset() const {
        return (static_cast<std::size_t>(n) * k / 2 + 255) / 256 * 256;
    }

    std::vector<std::uint8_t> payload() const {
        std::vector<std::uint8_t> bytes(scale_offset() + scales.size() * 2, 0);
        for (std::size_t i = 0; i < codes.size(); i += 2) {
            bytes[i / 2] = static_cast<std::uint8_t>((codes[i] + 8) | ((codes[i + 1] + 8) << 4));
        }
        std::memcpy(bytes.data() + scale_offset(), scales.data(), scales.size() * 2);
        return bytes;
    }

    Weight device_weight(void* device_payload, std::size_t bytes) const {
        Weight w;
        w.payload       = device_payload;
        w.payload_bytes = bytes;
        w.qtype         = QType::Q4_G32_FP16_ROWS;
        w.layout        = QuantLayout::RowGroup;
        w.group_size    = 32;
        w.group         = 32;
        w.scale_dtype   = DType::FP16;
        w.ndim          = 2;
        w.shape[0] = w.padded_shape[0] = n;
        w.shape[1] = w.padded_shape[1] = k;
        w.n                            = n;
        w.k                            = k;
        w.qdata                        = device_payload;
        w.scales = static_cast<std::uint8_t*>(device_payload) + scale_offset();
        return w;
    }
};

Q4RowsBank make_q4_rows_bank(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    Q4RowsBank bank;
    bank.n = n;
    bank.k = k;
    bank.codes.resize(static_cast<std::size_t>(n) * k);
    bank.scales.resize(static_cast<std::size_t>(n) * (k / 32));
    std::uint32_t state = seed * 2654435761U + 1U;
    auto next           = [&] {
        state = state * 1664525U + 1013904223U;
        return state >> 8;
    };
    for (auto& code : bank.codes) { code = static_cast<std::int8_t>(static_cast<int>(next() % 16) - 8); }
    for (auto& scale : bank.scales) {
        // Learned-looking steps across two octaves, and some that are not powers of two.
        scale = quantized_weight::detail::f32_to_f16(0.004F + 0.0001F * static_cast<float>(next() % 80));
    }
    return bank;
}

int run_q4_rows_case(const char* label, std::int32_t hidden, std::int32_t experts,
                     std::int32_t top_k, std::int32_t intermediate, std::int32_t tokens,
                     gemma_moe_experts_ref::Boundaries boundaries = {}) {
    const Q4RowsBank gate_up = make_q4_rows_bank(experts * 2 * intermediate, hidden, 51);
    const Q4RowsBank down    = make_q4_rows_bank(experts * hidden, intermediate, 52);

    std::vector<float> x(static_cast<std::size_t>(hidden) * tokens);
    fill_uniform(x, 43, -2.0F, 2.0F);
    round_to_bf16(x);
    const auto x_bits = encode(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> routing(static_cast<std::size_t>(top_k) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const std::size_t at = static_cast<std::size_t>(rank) * tokens + token;
            ids[at]              = (rank * 5 + 3 * token) % experts;
            routing[at]          = 0.25F + 0.15F * static_cast<float>(rank % 4);
        }
    }
    std::vector<double> magnitude;
    const auto reference = gemma_moe_experts_ref::run(
        [&](int row, int column) { return gate_up.value(row, column); },
        [&](int row, int column) { return down.value(row, column); }, x_bits, hidden, tokens, ids,
        routing, experts, intermediate, top_k, boundaries, &magnitude);

    const auto gate_up_bytes = gate_up.payload();
    const auto down_bytes    = down.payload();
    GuardedDeviceBuffer gate_up_device(gate_up_bytes.size());
    GuardedDeviceBuffer down_device(down_bytes.size());
    gate_up_device.copy_from_host(gate_up_bytes.data(), gate_up_bytes.size());
    down_device.copy_from_host(down_bytes.data(), down_bytes.size());
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
    weights.gate_up = gate_up.device_weight(gate_up_device.data(), gate_up_bytes.size());
    weights.down    = down.device_weight(down_device.data(), down_bytes.size());
    const ops::GemmaMoeExpertGeometry geometry{hidden, experts, top_k, intermediate};
    const std::size_t workspace_bytes =
        ops::gemma_moe_experts_workspace_bytes(geometry, QType::Q4_G32_FP16_ROWS, tokens);
    GuardedDeviceBuffer workspace_device(std::max<std::size_t>(workspace_bytes, 1));
    Tensor workspace(workspace_bytes > 0 ? workspace_device.data() : nullptr, DType::U8,
                     {static_cast<std::int32_t>(std::max<std::size_t>(workspace_bytes, 1))});

    std::vector<std::uint16_t> poison(x_bits.size(), f32_to_bf16(-12345.0F));
    out_device.copy_from_host(poison.data(), out_device.bytes());
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           workspace, nullptr);
    cuda_synchronize();
    const auto got = from_device_bf16(out_device.data(), static_cast<std::int64_t>(x_bits.size()));
    int failures = 0;
    if (!boundaries.expert_out_bf16) {
        failures += verify_pointwise(std::string(label) + " out",
                                     std::vector<double>(got.begin(), got.end()), reference,
                                     kCriterion);
    } else {
        // A route that rounds each expert's output to BF16 may land one ulp away from the
        // oracle's rounding of the same value, since their accumulations differ; where the
        // outputs cancel, that is a flip of up to 2^-7 of each summand's magnitude.
        std::size_t worst  = 0;
        double worst_ratio = 0.0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const double bound = kCriterion.absolute + kCriterion.relative * std::abs(reference[i]) +
                                 std::ldexp(magnitude[i], -7);
            const double ratio = std::abs(static_cast<double>(got[i]) - reference[i]) / bound;
            if (!(ratio <= worst_ratio)) {
                worst_ratio = ratio;
                worst       = i;
            }
        }
        std::cout << label << ": worst error " << worst_ratio << " of its bound\n";
        if (!(worst_ratio <= 1.0)) {
            std::cerr << label << " out: error " << worst_ratio << " of its bound at " << worst
                      << " (got " << got[worst] << ", reference " << reference[worst] << ")\n";
            ++failures;
        }
    }
    out_device.copy_from_host(poison.data(), out_device.bytes());
    ops::gemma_moe_experts(x_tensor, ids_tensor, routing_tensor, weights, geometry, out_tensor,
                           workspace, nullptr);
    cuda_synchronize();
    const auto again =
        from_device_bf16(out_device.data(), static_cast<std::int64_t>(x_bits.size()));
    failures += verify_exact((std::string(label) + " run-to-run").c_str(), again, got);
    failures += verify_exact(
        (std::string(label) + " gate_up preserved").c_str(),
        from_device<std::uint8_t>(gate_up_device.data(),
                                  static_cast<std::int64_t>(gate_up_bytes.size())),
        gate_up_bytes);
    return failures;
}
} // namespace

int main() {
    int failures = 0;
    try {
        failures += run_medium_case();
        failures += run_padding_case();
        failures += run_nvfp4_case("nvfp4 per-token", 256, 4, 2, 128, 4);
        failures += run_q4_rows_case("q4 rows per-token", 256, 4, 2, 128, 4);
        // The 26B's H, stored I and K over a 16-expert bank: the streamed route, at one token, at
        // a batched-decode width and at an MTP verify width.
        for (const std::int32_t tokens : {1, 3, 8}) {
            failures += run_nvfp4_case(("nvfp4 streamed T=" + std::to_string(tokens)).c_str(), 2816,
                                       16, 8, 768, tokens);
            failures += run_q4_rows_case(("q4 rows streamed T=" + std::to_string(tokens)).c_str(),
                                         2816, 16, 8, 768, tokens);
        }
        // From 16 tokens the int4 group-32 pair takes the grouped route: a single partial job per
        // expert, ragged jobs, several full jobs an expert, and a pass longer than one slice.
        for (const std::int32_t tokens : {16, 37, 300, 1030}) {
            failures += run_q4_rows_case(("q4 rows grouped T=" + std::to_string(tokens)).c_str(),
                                         2816, 16, 8, 768, tokens, {true, true, true});
        }
    } catch (const std::exception& error) {
        std::cerr << "gemma_moe_experts: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) {
        std::cout << "OK: gemma_moe_experts matches its oracle over a BF16 bank, over the stored "
                     "width with inert padding, and over stacked NVFP4 banks whose per-source "
                     "divisors matter, over int4 group-32 row banks on all three routes, and the "
                     "streamed and grouped routes reproduce themselves\n";
    }
    return failures == 0 ? 0 : 1;
}
