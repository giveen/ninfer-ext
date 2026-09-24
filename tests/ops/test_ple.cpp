// Qwen4Exp PLE Ops against FP64 oracles at the Qwen3.8-Flash-Next geometry: 4 streams of 2560,
// a kernel-4 causal convolution with dilation 3 (nine history columns per slot).
#include "ninfer/ops/ple.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStreams  = 4;
constexpr std::int32_t kHidden   = 2560;
constexpr std::int32_t kChannels = kStreams * kHidden;
constexpr std::int32_t kTaps     = 4;
constexpr std::int32_t kDilation = 3;
constexpr std::int32_t kHistory  = (kTaps - 1) * kDilation;
constexpr std::int32_t kSlots    = 3;

constexpr PointwiseCriterion kBf16Output{1.0e-6, 1.0 / 128.0};

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

int gate_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t wide = static_cast<std::size_t>(kChannels) * tokens;
    const auto key         = random_bf16(wide, seed, -2.0F, 2.0F);
    auto query             = random_bf16(wide, seed + 1U, -2.0F, 2.0F);
    const auto value =
        random_bf16(static_cast<std::size_t>(kHidden) * tokens, seed + 2U, -3.0F, 3.0F);
    // Column 0 stream 1 has an exactly zero dot product, exercising the sign(0) and 1e-6 floor.
    for (std::int32_t h = 0; h < kHidden; ++h) { query[kHidden + h] = 0.0F; }
    std::vector<double> reference(wide);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t s = 0; s < kStreams; ++s) {
            const std::size_t base = static_cast<std::size_t>(t) * kChannels + s * kHidden;
            double dot             = 0.0;
            for (std::int32_t h = 0; h < kHidden; ++h) {
                dot += double(key[base + h]) * double(query[base + h]);
            }
            const double g    = dot / std::sqrt(double(kHidden));
            const double sign = g > 0.0 ? 1.0 : (g < 0.0 ? -1.0 : 0.0);
            const double gate = sigmoid(sign * std::sqrt(std::max(std::abs(g), 1.0e-6)));
            for (std::int32_t h = 0; h < kHidden; ++h) {
                reference[base + h] =
                    gate * double(value[static_cast<std::size_t>(t) * kHidden + h]);
            }
        }
    }
    DeviceBuffer dk = to_device(bits(key));
    DeviceBuffer dq = to_device(bits(query));
    DeviceBuffer dv = to_device(bits(value));
    GuardedDeviceBuffer out(wide * 2);
    Tensor tk(dk.p, DType::BF16, {kChannels, tokens});
    Tensor tq(dq.p, DType::BF16, {kChannels, tokens});
    Tensor tv(dv.p, DType::BF16, {kHidden, tokens});
    Tensor to(out.data(), DType::BF16, {kChannels, tokens});
    ops::ple_gate(tk, tq, tv, kStreams, to, nullptr);
    cuda_synchronize();
    const std::string label = "ple_gate T=" + std::to_string(tokens);
    return verify_pointwise(label, from_device_bf16(out.data(), wide), reference, kBf16Output) +
           out.verify_guards(label);
}

// Column m of lane b in [history ; normed]: history rows are oldest first.
double history_value(const std::vector<float>& states, const std::vector<float>& normed,
                     std::int32_t slot, std::int32_t lane, std::int32_t width, std::int32_t m,
                     std::int32_t c) {
    if (m >= 0) { return normed[(static_cast<std::size_t>(lane) * width + m) * kChannels + c]; }
    const std::int32_t row = kHistory + m;
    return states[(static_cast<std::size_t>(slot) * kHistory + row) * kChannels + c];
}

int conv_case(std::int32_t width, std::int32_t lanes, std::uint32_t seed) {
    const std::size_t columns = static_cast<std::size_t>(width) * lanes;
    const auto normed         = random_bf16(columns * kChannels, seed, -3.0F, 3.0F);
    const auto gated          = random_bf16(columns * kChannels, seed + 1U, -2.0F, 2.0F);
    const auto residual       = random_bf16(columns * kChannels, seed + 2U, -4.0F, 4.0F);
    const auto weight =
        random_bf16(static_cast<std::size_t>(kTaps) * kChannels, seed + 3U, -0.5F, 0.5F);
    const auto states = random_bf16(static_cast<std::size_t>(kSlots) * kHistory * kChannels,
                                    seed + 4U, -3.0F, 3.0F);
    const std::vector<int> sources      = {2, 0};
    const std::vector<int> destinations = {1, 2};
    const std::vector<int> valid        = {width, width > 1 ? width - 1 : 1};

    std::vector<double> reference(columns * kChannels);
    std::vector<float> advanced = states;
    for (std::int32_t lane = 0; lane < lanes; ++lane) {
        const std::int32_t slot = sources[lane];
        for (std::int32_t w = 0; w < width; ++w) {
            for (std::int32_t c = 0; c < kChannels; ++c) {
                double sum = 0.0;
                for (std::int32_t j = 0; j < kTaps; ++j) {
                    const std::int32_t m = w - (kTaps - 1 - j) * kDilation;
                    sum += double(weight[static_cast<std::size_t>(j) * kChannels + c]) *
                           history_value(states, normed, slot, lane, width, m, c);
                }
                const std::size_t i = (static_cast<std::size_t>(lane) * width + w) * kChannels + c;
                reference[i]        = double(residual[i]) + double(gated[i]) + sum * sigmoid(sum);
            }
        }
        // History after valid[lane] columns: the last kHistory of [state ; normed[0:count]].
        const std::int32_t count = valid[lane];
        for (std::int32_t r = 0; r < kHistory; ++r) {
            for (std::int32_t c = 0; c < kChannels; ++c) {
                advanced[(static_cast<std::size_t>(destinations[lane]) * kHistory + r) * kChannels +
                         c] = static_cast<float>(history_value(states, normed, slot, lane, width,
                                                               count + r - kHistory, c));
            }
        }
    }

    DeviceBuffer dn   = to_device(bits(normed));
    DeviceBuffer dg   = to_device(bits(gated));
    DeviceBuffer dr   = to_device(bits(residual));
    DeviceBuffer dw   = to_device(bits(weight));
    DeviceBuffer ds   = to_device(bits(states));
    DeviceBuffer dsrc = to_device_i32(std::vector<int>(sources.begin(), sources.begin() + lanes));
    DeviceBuffer ddst =
        to_device_i32(std::vector<int>(destinations.begin(), destinations.begin() + lanes));
    DeviceBuffer dvalid = to_device_i32(std::vector<int>(valid.begin(), valid.begin() + lanes));
    Tensor tn(dn.p, DType::BF16, {kChannels, width, lanes});
    Tensor tg(dg.p, DType::BF16, {kChannels, width, lanes});
    Tensor tr(dr.p, DType::BF16, {kChannels, width, lanes});
    Tensor tw(dw.p, DType::BF16, {kTaps, kChannels});
    Tensor ts(ds.p, DType::BF16, {kHistory * kChannels, kSlots});
    Tensor tsrc(dsrc.p, DType::I32, {lanes});
    Tensor tdst(ddst.p, DType::I32, {lanes});
    Tensor tvalid(dvalid.p, DType::I32, {lanes});
    ops::ple_dilated_conv(tn, tg, tw, kDilation, ts, tsrc, tr, nullptr);
    cuda_synchronize();
    const std::string label =
        "ple_dilated_conv W=" + std::to_string(width) + " B=" + std::to_string(lanes);
    int failures =
        verify_pointwise(label, from_device_bf16(dr, reference.size()), reference, kBf16Output);
    // The convolution reads but never writes history.
    failures += verify_exact((label + " state immutable").c_str(),
                             from_device<std::uint16_t>(ds, states.size()), bits(states));

    ops::ple_conv_advance(tn, &tvalid, kHistory, ts, tsrc, tdst, nullptr);
    cuda_synchronize();
    failures += verify_exact(
        ("ple_conv_advance W=" + std::to_string(width) + " B=" + std::to_string(lanes)).c_str(),
        from_device<std::uint16_t>(ds, states.size()), bits(advanced));
    return failures;
}

int in_place_advance_case(std::uint32_t seed) {
    // Decode updates a slot in place: source equals destination for width one.
    const auto normed = random_bf16(kChannels, seed, -3.0F, 3.0F);
    const auto states = random_bf16(static_cast<std::size_t>(kSlots) * kHistory * kChannels,
                                    seed + 1U, -3.0F, 3.0F);
    std::vector<float> expected = states;
    const std::size_t base      = static_cast<std::size_t>(1) * kHistory * kChannels;
    for (std::int32_t r = 0; r + 1 < kHistory; ++r) {
        for (std::int32_t c = 0; c < kChannels; ++c) {
            expected[base + static_cast<std::size_t>(r) * kChannels + c] =
                states[base + static_cast<std::size_t>(r + 1) * kChannels + c];
        }
    }
    for (std::int32_t c = 0; c < kChannels; ++c) {
        expected[base + static_cast<std::size_t>(kHistory - 1) * kChannels + c] = normed[c];
    }
    DeviceBuffer dn   = to_device(bits(normed));
    DeviceBuffer ds   = to_device(bits(states));
    DeviceBuffer slot = to_device_i32({1});
    Tensor tn(dn.p, DType::BF16, {kChannels, 1, 1});
    Tensor ts(ds.p, DType::BF16, {kHistory * kChannels, kSlots});
    Tensor tslot(slot.p, DType::I32, {1});
    ops::ple_conv_advance(tn, nullptr, kHistory, ts, tslot, tslot, nullptr);
    cuda_synchronize();
    return verify_exact("ple_conv_advance in place", from_device<std::uint16_t>(ds, states.size()),
                        bits(expected));
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const std::int32_t tokens : {1, 5, 256}) { failures += gate_case(tokens, 0x10U + tokens); }
    failures += conv_case(1, 1, 0x100U);
    failures += conv_case(1, 2, 0x200U);
    failures += conv_case(8, 2, 0x300U);
    failures += conv_case(13, 1, 0x400U);
    failures += in_place_advance_case(0x500U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " ple correctness\n";
    return failures == 0 ? 0 : 1;
}
