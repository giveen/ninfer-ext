// Qwen4Exp hyper-connection Ops against FP64 oracles at the Qwen3.8-Flash-Next geometry
// (4 streams of 2560, low rank 320).
#include "ninfer/ops/hyper_connection.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStreams = 4;
constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kWide    = kStreams * kHidden;
constexpr std::int32_t kRank    = 320;
constexpr float kEps            = 1.0e-6F;

// BF16 outputs: one output rounding (2^-8 relative) over an FP64 reference.
constexpr PointwiseCriterion kBf16Output{1.0e-6, 1.0 / 128.0};
constexpr PointwiseCriterion kFp32Output{1.0e-7, 1.0e-6};

std::vector<double> read_f32(const void* device, std::size_t n) {
    const std::vector<float> v = from_device<float>(device, n);
    return {v.begin(), v.end()};
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

int grouped_norm_case(std::int32_t tokens, std::uint32_t seed) {
    const auto x      = random_bf16(static_cast<std::size_t>(kWide) * tokens, seed, -3.0F, 3.0F);
    const auto weight = random_bf16(kWide, seed + 1U, -0.5F, 0.5F);
    std::vector<double> reference(x.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t g = 0; g < kStreams; ++g) {
            const std::size_t base = static_cast<std::size_t>(t) * kWide + g * kHidden;
            double sum             = 0.0;
            for (std::int32_t h = 0; h < kHidden; ++h) { sum += double(x[base + h]) * x[base + h]; }
            const double inv = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (std::int32_t h = 0; h < kHidden; ++h) {
                reference[base + h] =
                    double(x[base + h]) * inv * (1.0 + double(weight[g * kHidden + h]));
            }
        }
    }
    DeviceBuffer dx = to_device(bits(x));
    DeviceBuffer dw = to_device(bits(weight));
    GuardedDeviceBuffer out(x.size() * 2);
    Tensor tx(dx.p, DType::BF16, {kWide, tokens});
    Tensor tw(dw.p, DType::BF16, {kWide});
    Tensor to(out.data(), DType::BF16, {kWide, tokens});
    ops::grouped_offset_rmsnorm(tx, tw, kStreams, kEps, to, nullptr);
    cuda_synchronize();
    const std::string label = "grouped_offset_rmsnorm T=" + std::to_string(tokens);
    return verify_pointwise(label, from_device_bf16(out.data(), x.size()), reference, kBf16Output) +
           out.verify_guards(label);
}

int gates_case(std::int32_t tokens, bool inject, std::uint32_t seed) {
    const std::int32_t rows = kRank + (inject ? kStreams : 0);
    const auto p = random_bf16(static_cast<std::size_t>(rows) * tokens, seed, -12.0F, 12.0F);
    std::vector<double> lowrank(static_cast<std::size_t>(kRank) * tokens);
    std::vector<double> gates(static_cast<std::size_t>(kStreams) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < kRank; ++r) {
            const double v                                   = double(p[t * rows + r]) / kStreams;
            lowrank[static_cast<std::size_t>(t) * kRank + r] = v * sigmoid(v);
        }
        for (std::int32_t s = 0; inject && s < kStreams; ++s) {
            gates[static_cast<std::size_t>(t) * kStreams + s] =
                2.0 * sigmoid(double(p[t * rows + kRank + s]) / kStreams);
        }
    }
    DeviceBuffer dp = to_device(bits(p));
    GuardedDeviceBuffer low(lowrank.size() * 2);
    GuardedDeviceBuffer inj(gates.size() * 4);
    Tensor tp(dp.p, DType::BF16, {rows, tokens});
    Tensor tl(low.data(), DType::BF16, {kRank, tokens});
    Tensor ti(inj.data(), DType::FP32, {kStreams, tokens});
    ops::hyper_connection_gates(tp, kStreams, tl, inject ? &ti : nullptr, nullptr);
    cuda_synchronize();
    const std::string label =
        "hyper_connection_gates T=" + std::to_string(tokens) + (inject ? " inject" : " head");
    int failures =
        verify_pointwise(label + " lowrank", from_device_bf16(low.data(), lowrank.size()), lowrank,
                         kBf16Output) +
        low.verify_guards(label);
    if (inject) {
        failures += verify_pointwise(label + " inject", read_f32(inj.data(), gates.size()), gates,
                                     kFp32Output) +
                    inj.verify_guards(label);
    }
    return failures;
}

int collapse_combine_expand_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const auto up            = random_bf16(wide, seed, -6.0F, 6.0F);
    const auto normalized    = random_bf16(wide, seed + 1U, -3.0F, 3.0F);
    const auto y             = random_bf16(narrow, seed + 2U, -2.0F, 2.0F);
    const auto residual      = random_bf16(wide, seed + 3U, -4.0F, 4.0F);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * tokens);
    fill_uniform(inject, seed + 4U, 0.0F, 2.0F);

    std::vector<double> mixed(narrow), combined(wide), expanded(wide);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kHidden; ++h) {
            double sum = 0.0;
            for (std::int32_t s = 0; s < kStreams; ++s) {
                const std::size_t i = static_cast<std::size_t>(t) * kWide + s * kHidden + h;
                sum += sigmoid(double(up[i])) * double(normalized[i]);
                combined[i] = double(residual[i]) +
                              double(inject[t * kStreams + s]) * double(y[t * kHidden + h]);
                expanded[i] = double(y[t * kHidden + h]);
            }
            mixed[static_cast<std::size_t>(t) * kHidden + h] = sum / kStreams;
        }
    }
    DeviceBuffer du = to_device(bits(up));
    DeviceBuffer dn = to_device(bits(normalized));
    DeviceBuffer dy = to_device(bits(y));
    DeviceBuffer di = to_device(inject);
    DeviceBuffer dr = to_device(bits(residual));
    GuardedDeviceBuffer mixed_out(narrow * 2);
    GuardedDeviceBuffer expand_out(wide * 2);
    Tensor tu(du.p, DType::BF16, {kWide, tokens});
    Tensor tn(dn.p, DType::BF16, {kWide, tokens});
    Tensor ty(dy.p, DType::BF16, {kHidden, tokens});
    Tensor ti(di.p, DType::FP32, {kStreams, tokens});
    Tensor tr(dr.p, DType::BF16, {kWide, tokens});
    Tensor tm(mixed_out.data(), DType::BF16, {kHidden, tokens});
    Tensor te(expand_out.data(), DType::BF16, {kWide, tokens});
    ops::hyper_connection_collapse(tu, tn, kStreams, tm, nullptr);
    ops::hyper_connection_combine(ty, ti, tr, nullptr);
    ops::hyper_connection_expand(ty, kStreams, te, nullptr);
    cuda_synchronize();
    const std::string label = "hyper_connection T=" + std::to_string(tokens);
    int failures            = 0;
    failures += verify_pointwise(label + " collapse", from_device_bf16(mixed_out.data(), narrow),
                                 mixed, kBf16Output);
    failures +=
        verify_pointwise(label + " combine", from_device_bf16(dr, wide), combined, kBf16Output);
    failures += verify_pointwise(label + " expand", from_device_bf16(expand_out.data(), wide),
                                 expanded, PointwiseCriterion{0.0, 0.0});
    failures += mixed_out.verify_guards(label + " collapse");
    failures += expand_out.verify_guards(label + " expand");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const std::int32_t tokens : {1, 3, 72, 1024}) {
        failures += grouped_norm_case(tokens, 0x100U + tokens);
        failures += gates_case(tokens, true, 0x200U + tokens);
        failures += gates_case(tokens, false, 0x300U + tokens);
        failures += collapse_combine_expand_case(tokens, 0x400U + tokens);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " hyper_connection correctness\n";
    return failures == 0 ? 0 : 1;
}
