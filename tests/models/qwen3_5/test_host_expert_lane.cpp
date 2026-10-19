// The decode miss split's host lane end to end: a published call whose assignments all resolve to
// the host, computed by the lane's team from a bank in registered Host memory, merged by
// moe_experts. The lane quantizes activations to Q8_0, so the FP64 oracle over the stored bank
// carries that cost (about 1% relative rms); the point of the comparison is the lane's own glue --
// which assignments reach which expert and column, and how their weighted rows land.
#include "models/execution/host_expert_lane.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t H = ops::kOffloadMoeHidden;
constexpr std::int32_t E = ops::kOffloadMoeExperts;
constexpr std::int32_t K = ops::kOffloadMoeTopK;
constexpr std::int32_t I = ops::kOffloadMoeIntermediate;

// The lane's own Q8_0 activations cost about 1% relative rms against the exact weights.
constexpr ReductionCriterion kLaneOutput{3.0e-2, 0.0, 6.0e-2};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

double e2m1(std::uint8_t code) {
    static constexpr double values[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v                    = values[code & 7];
    return (code & 8) != 0 ? -v : v;
}

double e4m3(std::uint8_t word) {
    const int e    = (word >> 3) & 0xF;
    const int m    = word & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return (word & 0x80) != 0 ? -v : v;
}

double decode(const std::uint8_t* codes, const std::uint8_t* scales, std::int32_t cols,
              std::int32_t n, std::int32_t k, double divisor) {
    const std::uint8_t byte = codes[(static_cast<std::size_t>(n) * cols + k) / 2];
    const std::uint8_t code = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    const std::int32_t g = k / 16, k_tiles = cols / 64;
    const std::size_t offset = (static_cast<std::size_t>(n / 128) * k_tiles + g / 4) * 512 +
                               (n % 128 % 32) * 16 + (n % 128 / 32) * 4 + g % 4;
    return e2m1(code) * e4m3(scales[offset]) / divisor;
}

// A bank in registered Host memory: the lane requires every plane to be a mapped allocation whose
// device pointer is its Host pointer.
struct MappedBank {
    void* planes[4] = {};
    void* divisors  = nullptr;
    ops::ExpertWeights weights{};

    MappedBank(const std::vector<std::uint8_t> (&data)[4], const std::vector<float>& gate_up,
               const std::vector<float>& down) {
        const std::int64_t strides[4] = {ops::kExpertGateUpCodeBytes, ops::kExpertGateUpScaleBytes,
                                         ops::kExpertDownCodeBytes, ops::kExpertDownScaleBytes};
        for (int p = 0; p < 4; ++p) {
            cuda_check(cudaHostAlloc(&planes[p], data[p].size(), cudaHostAllocMapped),
                       "bank plane allocation");
            std::memcpy(planes[p], data[p].data(), data[p].size());
            weights.base[p]   = static_cast<const std::byte*>(planes[p]);
            weights.stride[p] = strides[p];
        }
        const std::size_t divisor_bytes = (gate_up.size() + down.size()) * sizeof(float);
        cuda_check(cudaHostAlloc(&divisors, divisor_bytes, cudaHostAllocMapped),
                   "bank divisor allocation");
        auto* floats = static_cast<float*>(divisors);
        std::copy(gate_up.begin(), gate_up.end(), floats);
        std::copy(down.begin(), down.end(), floats + gate_up.size());
        weights.gate_up_divisors     = floats;
        weights.gate_up_divisor_rows = I;
        weights.down_divisors        = floats + gate_up.size();
        weights.down_divisor_rows    = H;
    }

    ~MappedBank() {
        for (void*& plane : planes) {
            if (plane != nullptr) { (void)cudaFreeHost(plane); }
        }
        if (divisors != nullptr) { (void)cudaFreeHost(divisors); }
    }

    MappedBank(const MappedBank&)            = delete;
    MappedBank& operator=(const MappedBank&) = delete;
};

struct HostBank {
    std::vector<std::uint8_t> planes[4];
    std::vector<float> gate_up_divisors;
    std::vector<float> down_divisors;
};

HostBank make_bank(std::uint32_t seed) {
    HostBank bank;
    const std::int64_t sizes[4] = {ops::kExpertGateUpCodeBytes, ops::kExpertGateUpScaleBytes,
                                   ops::kExpertDownCodeBytes, ops::kExpertDownScaleBytes};
    std::uint32_t state         = seed | 1U;
    for (int p = 0; p < 4; ++p) {
        bank.planes[p].resize(static_cast<std::size_t>(sizes[p]) * E);
        for (auto& byte : bank.planes[p]) {
            const std::uint32_t r = next(state);
            byte = (p == 1 || p == 3) ? static_cast<std::uint8_t>(0x20 + (r % 0x28))
                                      : static_cast<std::uint8_t>(r);
        }
    }
    bank.gate_up_divisors.resize(2 * E);
    bank.down_divisors.resize(E);
    for (auto& d : bank.gate_up_divisors) {
        d = 1.0F + static_cast<float>(next(state) % 1024) / 256.0F;
    }
    for (auto& d : bank.down_divisors) {
        d = 1.0F + static_cast<float>(next(state) % 1024) / 256.0F;
    }
    return bank;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<double> reference(const HostBank& bank, const std::vector<float>& x,
                              const std::vector<int>& ids, const std::vector<float>& weights,
                              std::int32_t tokens) {
    std::vector<double> y(static_cast<std::size_t>(H) * tokens, 0.0);
    std::vector<double> act(I);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const float* xt = x.data() + static_cast<std::size_t>(t) * H;
        for (std::int32_t k = 0; k < K; ++k) {
            const int e = ids[t * K + k];
            const std::uint8_t* gu_codes =
                bank.planes[0].data() + static_cast<std::size_t>(e) * ops::kExpertGateUpCodeBytes;
            const std::uint8_t* gu_scales =
                bank.planes[1].data() + static_cast<std::size_t>(e) * ops::kExpertGateUpScaleBytes;
            const std::uint8_t* d_codes =
                bank.planes[2].data() + static_cast<std::size_t>(e) * ops::kExpertDownCodeBytes;
            const std::uint8_t* d_scales =
                bank.planes[3].data() + static_cast<std::size_t>(e) * ops::kExpertDownScaleBytes;
            for (std::int32_t r = 0; r < I; ++r) {
                double g = 0.0, u = 0.0;
                for (std::int32_t h = 0; h < H; ++h) {
                    g += decode(gu_codes, gu_scales, H, r, h, bank.gate_up_divisors[2 * e]) * xt[h];
                    u +=
                        decode(gu_codes, gu_scales, H, I + r, h, bank.gate_up_divisors[2 * e + 1]) *
                        xt[h];
                }
                act[r] = g * sigmoid(g) * u;
            }
            for (std::int32_t h = 0; h < H; ++h) {
                double sum = 0.0;
                for (std::int32_t r = 0; r < I; ++r) {
                    sum += decode(d_codes, d_scales, I, h, r, bank.down_divisors[e]) * act[r];
                }
                y[static_cast<std::size_t>(t) * H + h] += double(weights[t * K + k]) * sum;
            }
        }
    }
    return y;
}

int lane_case(const HostBank& mapped_host, const MappedBank& mapped, std::int32_t tokens,
              std::int32_t threads, std::uint32_t seed) {
    const auto x = random_bf16(static_cast<std::size_t>(H) * tokens, seed, -0.05F, 0.05F);
    std::vector<int> ids(static_cast<std::size_t>(K) * tokens);
    std::vector<float> weights(ids.size());
    std::uint32_t state = seed | 1U;
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<int> chosen;
        float total = 0.0F;
        for (std::int32_t k = 0; k < K; ++k) {
            int e;
            do {
                e = static_cast<int>(next(state) % E);
            } while (std::find(chosen.begin(), chosen.end(), e) != chosen.end());
            chosen.push_back(e);
            ids[t * K + k]     = e;
            weights[t * K + k] = 0.1F + static_cast<float>(next(state) % 100) / 100.0F;
            total += weights[t * K + k];
        }
        for (std::int32_t k = 0; k < K; ++k) { weights[t * K + k] /= total; }
    }
    const std::string label =
        "host_expert_lane T=" + std::to_string(tokens) + " threads=" + std::to_string(threads);
    const std::vector<double> expected = reference(mapped_host, x, ids, weights, tokens);

    DeviceBuffer dx     = to_device(bits(x));
    DeviceBuffer dids   = to_device_i32(ids);
    DeviceBuffer dw     = to_device(weights);
    DeviceBuffer dslots = to_device_i32(std::vector<int>(ids.size(), -1));
    DeviceBuffer dgate  = to_device(std::vector<float>(static_cast<std::size_t>(tokens), 0.0F));
    DeviceBuffer dshared =
        to_device(bits(std::vector<float>(static_cast<std::size_t>(H) * tokens, 0.0F)));
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    Tensor tids(dids.p, DType::I32, {K, tokens});
    Tensor tw(dw.p, DType::FP32, {K, tokens});
    Tensor tslots(dslots.p, DType::I32, {K, tokens});
    Tensor tgate(dgate.p, DType::FP32, {tokens});
    Tensor tshared(dshared.p, DType::BF16, {H, tokens});

    models::execution::HostExpertLane lane({mapped.weights}, tokens, threads);
    ops::moe_host_publish(tx, tids, tslots, tw, 0, lane.ring(), nullptr);
    cuda_check(cudaStreamSynchronize(nullptr), "publish");

    const std::uint32_t ticket = from_device<std::uint32_t>(lane.ring().ticket, 1).front();
    if (ticket == 0) {
        std::cerr << label << ": every assignment is a host job, but publish rang no ticket\n";
        return 1;
    }
    // Wait for the lane's answer (a device read of the mapped word works under UVA and without).
    const auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::uint32_t answered = 0;
    do {
        cuda_check(cudaMemcpy(&answered, const_cast<std::uint32_t*>(lane.ring().answered),
                              sizeof(answered), cudaMemcpyDeviceToHost),
                   "answer word");
        if (answered == ticket) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    if (answered != ticket) {
        std::cerr << label << ": the lane did not answer ticket " << ticket << '\n';
        return 1;
    }

    GuardedDeviceBuffer out(static_cast<std::size_t>(H) * tokens * 2);
    Tensor ty(out.data(), DType::BF16, {H, tokens});
    {
        WorkspaceArena workspace(
            ops::moe_experts_workspace_bytes(tokens, E, ops::nvfp4_expert_layout()));
        ops::moe_experts(tx, tids, tslots, tw, tgate, tshared, mapped.weights, E, workspace, ty,
                         nullptr, nullptr, nullptr, &lane.ring());
        cuda_check(cudaStreamSynchronize(nullptr), "moe_experts");
    }
    int failures =
        verify_reduction(label, from_device_bf16(out.data(), static_cast<std::size_t>(H) * tokens),
                         expected, kLaneOutput);
    failures += out.verify_guards(label);
    if (lane.calls() != 1 || lane.assignments() != static_cast<std::uint64_t>(ids.size())) {
        std::cerr << label << ": the lane reports " << lane.calls() << " calls and "
                  << lane.assignments() << " assignments\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const HostBank host = make_bank(0x51A7E5U);
    const MappedBank mapped(host.planes, host.gate_up_divisors, host.down_divisors);
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 8}) {
        for (const std::int32_t threads : {1, 4}) {
            failures += lane_case(host, mapped, tokens, threads,
                                  0xE00U + static_cast<std::uint32_t>(tokens) * 16U +
                                      static_cast<std::uint32_t>(threads));
        }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " host_expert_lane correctness\n";
    return failures == 0 ? 0 : 1;
}
