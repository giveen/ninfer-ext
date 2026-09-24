// Qwen4Exp offloaded MoE Ops: routing against an FP64 oracle, expert-cache residency semantics,
// and routed expert SwiGLU against an FP64 oracle that decodes each NVFP4 weight independently.
// The staged-bank and cache routes must produce identical bits.
#include "ninfer/ops/offload_moe.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t H = ops::kOffloadMoeHidden;
constexpr std::int32_t E = ops::kOffloadMoeExperts;
constexpr std::int32_t K = ops::kOffloadMoeTopK;
constexpr std::int32_t I = ops::kOffloadMoeIntermediate;

constexpr PointwiseCriterion kRouteWeights{1.0e-6, 1.0e-5};
// Expert outputs cross a private BF16 intermediate and BF16 output rounding.
constexpr ReductionCriterion kExpertOutput{8.0e-3, 2.0e-2, 2.0e-2};

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

std::vector<double> read_f32(const void* device, std::size_t n) {
    const std::vector<float> v = from_device<float>(device, n);
    return {v.begin(), v.end()};
}

// ---- routing ---------------------------------------------------------------------------------

int route_case(std::int32_t tokens, std::uint32_t seed) {
    const auto x      = random_bf16(static_cast<std::size_t>(H) * tokens, seed, -1.0F, 1.0F);
    const auto router = random_bf16(static_cast<std::size_t>(E + 1) * H, seed + 1U, -0.1F, 0.1F);
    std::vector<int> ids(static_cast<std::size_t>(K) * tokens);
    std::vector<double> weights(ids.size()), shared(tokens);
    std::vector<std::vector<double>> logits(tokens, std::vector<double>(E + 1));
    for (std::int32_t t = 0; t < tokens; ++t) {
        auto& l = logits[t];
        for (std::int32_t e = 0; e <= E; ++e) {
            double sum = 0.0;
            for (std::int32_t h = 0; h < H; ++h) {
                sum += double(router[static_cast<std::size_t>(e) * H + h]) *
                       double(x[static_cast<std::size_t>(t) * H + h]);
            }
            l[e] = sum;
        }
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
        double selected = 0.0;
        for (std::int32_t k = 0; k < K; ++k) { selected += std::exp(l[order[k]] - l[order[0]]); }
        for (std::int32_t k = 0; k < K; ++k) {
            ids[t * K + k]     = order[k];
            weights[t * K + k] = std::exp(l[order[k]] - l[order[0]]) / selected;
        }
        shared[t] = sigmoid(l[E]);
    }
    DeviceBuffer dx = to_device(bits(x));
    DeviceBuffer dr = to_device(bits(router));
    GuardedDeviceBuffer out_ids(ids.size() * 4), out_w(ids.size() * 4), out_s(tokens * 4);
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    Tensor tr(dr.p, DType::BF16, {E + 1, H});
    Tensor ti(out_ids.data(), DType::I32, {K, tokens});
    Tensor tw(out_w.data(), DType::FP32, {K, tokens});
    Tensor ts(out_s.data(), DType::FP32, {tokens});
    ops::moe_route(tx, tr, ti, tw, ts, nullptr);
    cuda_synchronize();
    const std::string label = "moe_route T=" + std::to_string(tokens);
    int failures            = 0;
    // Selection is a set per column; a near-tied tenth/eleventh logit may legitimately swap.
    const std::vector<int> got = from_device<int>(out_ids.data(), ids.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::set<int> a(got.begin() + t * K, got.begin() + (t + 1) * K);
        std::set<int> b(ids.begin() + t * K, ids.begin() + (t + 1) * K);
        if (a != b) {
            std::vector<double> sorted(logits[t].begin(), logits[t].begin() + E);
            std::sort(sorted.rbegin(), sorted.rend());
            if (sorted[K - 1] - sorted[K] > 1.0e-4) {
                std::cerr << label << ": expert set mismatch at column " << t << '\n';
                ++failures;
            }
        } else {
            std::vector<double> got_w(K), ref_w(K);
            const std::vector<double> device_w = read_f32(out_w.data(), weights.size());
            for (std::int32_t k = 0; k < K; ++k) {
                const auto it =
                    std::find(ids.begin() + t * K, ids.begin() + (t + 1) * K, got[t * K + k]);
                got_w[k] = device_w[t * K + k];
                ref_w[k] = weights[static_cast<std::size_t>(it - ids.begin())];
            }
            failures += verify_pointwise(label + " weights", got_w, ref_w, kRouteWeights);
        }
    }
    failures += verify_pointwise(label + " shared gate", read_f32(out_s.data(), tokens), shared,
                                 kRouteWeights);
    failures +=
        out_ids.verify_guards(label) + out_w.verify_guards(label) + out_s.verify_guards(label);
    return failures;
}

// ---- NVFP4 expert banks ----------------------------------------------------------------------

struct HostBank {
    std::vector<std::uint8_t> planes[4];
    std::vector<float> gate_up_divisors; // one per gate or up matrix: 2 * E
    std::vector<float> down_divisors;    // one per expert
};

std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

HostBank make_bank(std::uint32_t seed) {
    HostBank bank;
    const std::int64_t sizes[4] = {ops::kExpertGateUpCodeBytes, ops::kExpertGateUpScaleBytes,
                                   ops::kExpertDownCodeBytes, ops::kExpertDownScaleBytes};
    std::uint32_t state         = seed | 1U;
    for (int p = 0; p < 4; ++p) {
        bank.planes[p].resize(static_cast<std::size_t>(sizes[p]) * E);
        for (auto& byte : bank.planes[p]) {
            const std::uint32_t r = next(state);
            // Scale words: positive E4M3FN exponents 4..8 (0.125..~3.75); codes: every E2M1 pair.
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

// Weight [n,k] of one expert matrix with `rows` rows and `cols` columns, whose code and scale
// planes start at the given pointers (per-expert block_scale_k16_m128x4 layout).
double decode(const std::uint8_t* codes, const std::uint8_t* scales, std::int32_t cols,
              std::int32_t n, std::int32_t k, double divisor) {
    const std::uint8_t byte = codes[(static_cast<std::size_t>(n) * cols + k) / 2];
    const std::uint8_t code = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    const std::int32_t g = k / 16, k_tiles = cols / 64;
    const std::size_t offset = (static_cast<std::size_t>(n / 128) * k_tiles + g / 4) * 512 +
                               (n % 128 % 32) * 16 + (n % 128 / 32) * 4 + g % 4;
    return e2m1(code) * e4m3(scales[offset]) / divisor;
}

std::vector<double> expert_oracle(const HostBank& bank, const std::vector<float>& x,
                                  const std::vector<int>& ids, const std::vector<float>& weights,
                                  const std::vector<float>& shared_gate,
                                  const std::vector<float>& shared, std::int32_t tokens) {
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
        for (std::int32_t h = 0; h < H; ++h) {
            y[static_cast<std::size_t>(t) * H + h] +=
                double(shared_gate[t]) * double(shared[static_cast<std::size_t>(t) * H + h]);
        }
    }
    return y;
}

struct DeviceBank {
    DeviceBuffer planes[4];
    DeviceBuffer gate_up_divisors, down_divisors;
    ops::ExpertWeights weights;
};

DeviceBank upload(const HostBank& host) {
    DeviceBank out;
    const std::int64_t strides[4] = {ops::kExpertGateUpCodeBytes, ops::kExpertGateUpScaleBytes,
                                     ops::kExpertDownCodeBytes, ops::kExpertDownScaleBytes};
    for (int p = 0; p < 4; ++p) {
        out.planes[p]         = to_device(host.planes[p]);
        out.weights.base[p]   = static_cast<const std::byte*>(out.planes[p].p);
        out.weights.stride[p] = strides[p];
    }
    out.gate_up_divisors             = to_device(host.gate_up_divisors);
    out.down_divisors                = to_device(host.down_divisors);
    out.weights.gate_up_divisors     = static_cast<const float*>(out.gate_up_divisors.p);
    out.weights.gate_up_divisor_rows = I;
    out.weights.down_divisors        = static_cast<const float*>(out.down_divisors.p);
    out.weights.down_divisor_rows    = H;
    return out;
}

struct DeviceCache {
    DeviceBuffer slot_of, owner, stamp, counters, pool;
    ops::ExpertCacheState state;

    DeviceCache(std::int32_t slots, std::int32_t layers)
        : slot_of(to_device_i32(std::vector<int>(static_cast<std::size_t>(layers) * E, -1))),
          owner(to_device_i32(std::vector<int>(slots, -1))),
          stamp(to_device(std::vector<unsigned long long>(slots, 0))),
          counters(to_device(std::vector<unsigned long long>(3, 0))),
          pool(to_device(std::vector<std::uint8_t>(
              static_cast<std::size_t>(slots) * ops::kExpertSlotBytes, 0))) {
        state =
            ops::ExpertCacheState{.slot_of    = static_cast<std::int32_t*>(slot_of.p),
                                  .owner      = static_cast<std::int32_t*>(owner.p),
                                  .stamp      = static_cast<unsigned long long*>(stamp.p),
                                  .clock      = static_cast<unsigned long long*>(counters.p),
                                  .statistics = static_cast<unsigned long long*>(counters.p) + 1,
                                  .slots      = slots,
                                  .layers     = layers,
                                  .pool       = static_cast<std::byte*>(pool.p)};
    }

    std::vector<int> table() const { return from_device<int>(slot_of, slot_of_elements()); }

    std::size_t slot_of_elements() const { return static_cast<std::size_t>(state.layers) * E; }
};

// Resolves and fetches `experts` (one column per 10) for `layer`; returns slot ids.
std::vector<int> resolve(DeviceCache& cache, const DeviceBank& bank, std::int32_t layer,
                         const std::vector<int>& experts) {
    const auto columns  = static_cast<std::int32_t>(experts.size() / K);
    DeviceBuffer ids    = to_device_i32(experts);
    DeviceBuffer slots  = to_device_i32(std::vector<int>(experts.size(), -7));
    DeviceBuffer misses = to_device_i32(std::vector<int>(2 * experts.size() + 1, -9));
    Tensor ti(ids.p, DType::I32, {K, columns});
    Tensor ts(slots.p, DType::I32, {K, columns});
    Tensor tm(misses.p, DType::I32, {static_cast<std::int32_t>(2 * experts.size() + 1)});
    ops::expert_cache_resolve(ti, layer, cache.state, ts, tm, nullptr);
    ops::expert_cache_fetch(bank.weights, tm, static_cast<std::int32_t>(experts.size()),
                            cache.state, nullptr);
    cuda_synchronize();
    return from_device<int>(slots, experts.size());
}

int check_residency(const char* label, const DeviceCache& cache, const HostBank& bank,
                    std::int32_t layer, const std::vector<int>& experts,
                    const std::vector<int>& slot_ids) {
    int failures                 = 0;
    const std::vector<int> table = cache.table();
    const std::vector<int> owner = from_device<int>(cache.owner, cache.state.slots);
    std::set<int> used;
    for (std::size_t i = 0; i < experts.size(); ++i) {
        const int slot = slot_ids[i];
        if (slot < 0 || slot >= cache.state.slots || owner[slot] != layer * E + experts[i] ||
            table[layer * E + experts[i]] != slot) {
            std::cerr << label << ": assignment " << i << " is not resident in its slot\n";
            return failures + 1;
        }
        used.insert(experts[i]);
        std::vector<std::uint8_t> bytes(ops::kExpertSlotBytes);
        cuda_check(cudaMemcpy(bytes.data(),
                              static_cast<const std::uint8_t*>(cache.pool.p) +
                                  static_cast<std::size_t>(slot) * ops::kExpertSlotBytes,
                              bytes.size(), cudaMemcpyDeviceToHost),
                   "read expert slot");
        std::size_t offset          = 0;
        const std::int64_t sizes[4] = {ops::kExpertGateUpCodeBytes, ops::kExpertGateUpScaleBytes,
                                       ops::kExpertDownCodeBytes, ops::kExpertDownScaleBytes};
        for (int p = 0; p < 4; ++p) {
            if (std::memcmp(bytes.data() + offset,
                            bank.planes[p].data() + static_cast<std::size_t>(experts[i]) * sizes[p],
                            static_cast<std::size_t>(sizes[p])) != 0) {
                std::cerr << label << ": slot " << slot << " plane " << p << " differs\n";
                return failures + 1;
            }
            offset += static_cast<std::size_t>(sizes[p]);
        }
    }
    // Distinct experts occupy distinct slots.
    std::set<int> slots_of_used;
    for (const int e : used) { slots_of_used.insert(table[layer * E + e]); }
    if (slots_of_used.size() != used.size()) {
        std::cerr << label << ": two experts share a slot\n";
        ++failures;
    }
    return failures;
}

std::vector<int> expert_range(int first, int count) {
    std::vector<int> v(count);
    std::iota(v.begin(), v.end(), first);
    return v;
}

int cache_case(const HostBank& host, const DeviceBank& bank) {
    int failures = 0;
    DeviceCache cache(20, 2);
    const auto a = expert_range(0, 10), b = expert_range(10, 10), c = expert_range(100, 10);
    failures +=
        check_residency("expert cache fill A", cache, host, 0, a, resolve(cache, bank, 0, a));
    failures +=
        check_residency("expert cache fill B", cache, host, 0, b, resolve(cache, bank, 0, b));
    failures +=
        check_residency("expert cache hit A", cache, host, 0, a, resolve(cache, bank, 0, a));
    const auto counters = from_device<unsigned long long>(cache.counters, 3);
    if (counters[1] != 10 || counters[2] != 20) {
        std::cerr << "expert cache statistics: hits=" << counters[1] << " misses=" << counters[2]
                  << " (expected 10 and 20)\n";
        ++failures;
    }
    // A full cache evicts the least recently used experts: B, not the just-used A.
    failures +=
        check_residency("expert cache evict", cache, host, 0, c, resolve(cache, bank, 0, c));
    const std::vector<int> table = cache.table();
    for (const int e : a) {
        if (table[e] < 0) {
            std::cerr << "expert cache evicted recently used expert " << e << '\n';
            ++failures;
        }
    }
    for (const int e : b) {
        if (table[e] >= 0) {
            std::cerr << "expert cache kept least recently used expert " << e << '\n';
            ++failures;
        }
    }
    // Layers key separately; duplicate experts in one call share one slot.
    const std::vector<int> repeated = {5, 5, 7, 7, 9, 9, 11, 11, 13, 13};
    failures += check_residency("expert cache layer 1 duplicates", cache, host, 1, repeated,
                                resolve(cache, bank, 1, repeated));
    return failures;
}

// `pool` > 0 routes every column within `pool` experts, so one expert job carries many tokens
// (real prefill routing concentrates); 0 spreads routing over all experts.
int experts_case(const HostBank& host, const DeviceBank& bank, std::int32_t tokens,
                 std::uint32_t seed, int pool = 0) {
    const auto x      = random_bf16(static_cast<std::size_t>(H) * tokens, seed, -0.05F, 0.05F);
    const auto shared = random_bf16(static_cast<std::size_t>(H) * tokens, seed + 1U, -4.0F, 4.0F);
    std::vector<int> ids(static_cast<std::size_t>(K) * tokens);
    std::vector<float> weights(ids.size()), shared_gate(tokens);
    std::uint32_t state = seed | 1U;
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::set<int> chosen;
        float total = 0.0F;
        for (std::int32_t k = 0; k < K; ++k) {
            int e;
            do {
                e = static_cast<int>(next(state) % (pool > 0 ? pool : E)) * (pool > 0 ? 37 : 1) % E;
            } while (!chosen.insert(e).second);
            ids[t * K + k]     = e;
            weights[t * K + k] = 0.1F + static_cast<float>(next(state) % 100) / 100.0F;
            total += weights[t * K + k];
        }
        for (std::int32_t k = 0; k < K; ++k) { weights[t * K + k] /= total; }
        shared_gate[t] = static_cast<float>(next(state) % 1000) / 1000.0F;
    }
    const std::vector<double> reference =
        expert_oracle(host, x, ids, weights, shared_gate, shared, tokens);

    DeviceBuffer dx      = to_device(bits(x));
    DeviceBuffer dshared = to_device(bits(shared));
    DeviceBuffer dids    = to_device_i32(ids);
    DeviceBuffer dw      = to_device(weights);
    DeviceBuffer dsg     = to_device(shared_gate);
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    Tensor tshared(dshared.p, DType::BF16, {H, tokens});
    Tensor tids(dids.p, DType::I32, {K, tokens});
    Tensor tw(dw.p, DType::FP32, {K, tokens});
    Tensor tsg(dsg.p, DType::FP32, {tokens});

    // Staged route: the bank itself, indexed by expert id.
    GuardedDeviceBuffer staged(static_cast<std::size_t>(H) * tokens * 2);
    Tensor ty_staged(staged.data(), DType::BF16, {H, tokens});
    {
        WorkspaceArena workspace(ops::moe_experts_workspace_bytes(tokens, E));
        ops::moe_experts(tx, tids, tids, tw, tsg, tshared, bank.weights, E, workspace, ty_staged,
                         nullptr);
        cuda_synchronize();
    }
    // Cache route: slot placement differs from expert ids.
    const std::int32_t slots = std::max(10 * tokens, 40);
    DeviceCache cache(slots, 1);
    // Occupy low slots with unrelated experts so routed experts land at permuted slots.
    (void)resolve(cache, bank, 0, expert_range(E - 10, 10));
    const std::vector<int> slot_ids = resolve(cache, bank, 0, ids);
    DeviceBuffer dslots             = to_device_i32(slot_ids);
    Tensor tslots(dslots.p, DType::I32, {K, tokens});
    GuardedDeviceBuffer cached(static_cast<std::size_t>(H) * tokens * 2);
    Tensor ty_cached(cached.data(), DType::BF16, {H, tokens});
    {
        WorkspaceArena workspace(ops::moe_experts_workspace_bytes(tokens, slots));
        ops::moe_experts(tx, tids, tslots, tw, tsg, tshared,
                         ops::expert_cache_weights(cache.state, bank.weights), slots, workspace,
                         ty_cached, nullptr);
        cuda_synchronize();
    }
    const std::string label =
        "moe_experts T=" + std::to_string(tokens) + (pool > 0 ? " concentrated" : "");
    const std::size_t n = static_cast<std::size_t>(H) * tokens;
    int failures =
        verify_reduction(label, from_device_bf16(staged.data(), n), reference, kExpertOutput);
    failures += verify_exact((label + " cache route bits").c_str(),
                             from_device<std::uint16_t>(cached.data(), n),
                             from_device<std::uint16_t>(staged.data(), n));
    failures += staged.verify_guards(label) + cached.verify_guards(label);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const std::int32_t tokens : {1, 4, 33}) { failures += route_case(tokens, 0x51U + tokens); }
    const HostBank host   = make_bank(0xC0FFEEU);
    const DeviceBank bank = upload(host);
    failures += cache_case(host, bank);
    for (const std::int32_t tokens : {1, 3, 16}) {
        failures += experts_case(host, bank, tokens, 0x900U + tokens);
    }
    // Concentrated routing: 12 experts serve every column, so jobs hold up to 64 tokens and the
    // decode GEMV walks several 8-token groups per expert.
    for (const std::int32_t tokens : {9, 22, 64}) {
        failures += experts_case(host, bank, tokens, 0xA00U + tokens, 12);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " offload_moe correctness\n";
    return failures == 0 ? 0 : 1;
}
