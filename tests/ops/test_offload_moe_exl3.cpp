// Routed EXL3 experts (Qwen4Exp layout) against an independent FP64 oracle. The oracle reads the
// stored bank exactly as docs/maintainer/storage-layouts.md section 9 and tensor-formats.md
// section 3.5 define it: it reconstructs every matrix Z from the circular trellis windows, applies
// the normalized 128-point Hadamards, the stored scale words and the SwiGLU in FP64, and merges
// the experts in slot order. Banks hold random trellis bits, so every state sequence a tile can
// carry is exercised; rates cover the register-window decodes (4-bit, even, fast half) and the
// bitwise fallback. The cache-slot route must reproduce the bank-indexed route bit for bit.
#include "ninfer/ops/offload_moe.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t H = ops::kOffloadMoeHidden;
constexpr std::int32_t E = ops::kOffloadMoeExperts;
constexpr std::int32_t K = ops::kOffloadMoeTopK;
constexpr std::int32_t I = ops::kOffloadMoeIntermediate;

// Output is BF16 (relative rounding about 1.1e-3 RMS) plus the FP16 rotated activations of both
// GEMM stages (about 3e-4 each); the oracle keeps everything in FP64.
constexpr ReductionCriterion kExl3Output{3.0e-3, 0.0, 1.0e-2};

double silu(double x) { return x / (1.0 + std::exp(-x)); }

// ---- stored-format decode (the oracle) -------------------------------------------------------

// Z[k][n] of a [N,K] matrix whose tiles start at `tiles` (tiles_k = K / 16).
std::vector<std::int16_t> decode_z(const std::uint8_t* tiles, int half_bits, int n_total, int k_total) {
    const int tiles_k = k_total / 16;
    const int total   = 128 * half_bits;
    std::vector<int> end(256);
    for (int t = 0, at = 0; t < 256; ++t) {
        at += half_bits / 2 + ((half_bits & 1) && (t & 1) ? 1 : 0);
        end[t] = at;
    }
    std::vector<std::int16_t> z(static_cast<std::size_t>(k_total) * n_total);
    for (int nt = 0; nt < n_total / 16; ++nt) {
        for (int kt = 0; kt < tiles_k; ++kt) {
            const std::uint8_t* tile = tiles + (static_cast<std::size_t>(nt) * tiles_k + kt) * 16 * half_bits;
            for (int t = 0; t < 256; ++t) {
                std::uint32_t state = 0;
                for (int b = 0; b < 16; ++b) {
                    const int position = ((end[t] - 16 + b) % total + total) % total;
                    state |= (static_cast<std::uint32_t>(tile[position >> 3] >> (position & 7)) & 1U) << b;
                }
                const std::uint32_t product = state * 0x83DCD12DU;
                const int sum = static_cast<int>((product & 0xFF) + ((product >> 8) & 0xFF) +
                                                 ((product >> 16) & 0xFF) + (product >> 24));
                const int lane = t >> 3, j = t & 7;
                const int g = lane >> 2, q = lane & 3, h = j >> 2, r = j & 3;
                const int k = 2 * q + (r & 1) + 8 * (r >> 1);
                const int n = g + 8 * h;
                z[static_cast<std::size_t>(kt * 16 + k) * n_total + nt * 16 + n] =
                    static_cast<std::int16_t>(sum - 510);
            }
        }
    }
    return z;
}

// Orthonormal block-diagonal Hadamard over 128-element blocks.
void hadamard_blocks(std::vector<double>& v) {
    for (std::size_t base = 0; base < v.size(); base += 128) {
        for (int stride = 1; stride < 128; stride <<= 1) {
            for (int i = 0; i < 128; ++i) {
                if (i & stride) { continue; }
                const double a = v[base + i], b = v[base + i + stride];
                v[base + i]          = a + b;
                v[base + i + stride] = a - b;
            }
        }
        for (int i = 0; i < 128; ++i) { v[base + i] /= std::sqrt(128.0); }
    }
}

// y[n] = (svh . H(Z^T (H(suh . x))))[n] for one stored matrix.
std::vector<double> apply_matrix(const std::vector<std::int16_t>& z, int k_total, int n_total,
                                 const float* suh, const float* svh, std::vector<double> x) {
    for (int k = 0; k < k_total; ++k) { x[k] *= suh[k]; }
    hadamard_blocks(x);
    std::vector<double> y(n_total, 0.0);
    for (int k = 0; k < k_total; ++k) {
        const std::int16_t* row = &z[static_cast<std::size_t>(k) * n_total];
        for (int n = 0; n < n_total; ++n) { y[n] += x[k] * row[n]; }
    }
    hadamard_blocks(y);
    for (int n = 0; n < n_total; ++n) { y[n] *= svh[n]; }
    return y;
}

// ---- host bank --------------------------------------------------------------------------------

struct HostBank {
    ops::ExpertLayout layout;
    std::vector<std::uint8_t> planes[6];
    const std::uint8_t* at(int plane, int expert) const {
        return planes[plane].data() + static_cast<std::size_t>(expert) * layout.plane_bytes[plane];
    }
    const float* scales(int plane, int expert) const {
        return reinterpret_cast<const float*>(at(plane, expert));
    }
};

HostBank make_bank(int gate_up_half_bits, int down_half_bits, std::uint64_t seed) {
    HostBank bank;
    bank.layout = ops::exl3_expert_layout(gate_up_half_bits, down_half_bits);
    std::mt19937_64 rng(seed);
    for (int p = 0; p < 6; ++p) {
        bank.planes[p].resize(static_cast<std::size_t>(bank.layout.plane_bytes[p]) * E);
    }
    for (int p : {0, 3}) {
        auto& bytes = bank.planes[p];
        for (std::size_t i = 0; i + 8 <= bytes.size(); i += 8) {
            const std::uint64_t word = rng();
            std::memcpy(&bytes[i], &word, 8);
        }
    }
    // Scale words: signed, near one; output scales set the layer's magnitude so the result is O(1).
    std::uniform_real_distribution<float> magnitude(0.5F, 1.5F);
    const float svh_gate_up = 1.5e-4F, svh_down = 5.0e-4F;
    for (int e = 0; e < E; ++e) {
        for (int p : {1, 4, 2, 5}) {
            auto* s = reinterpret_cast<float*>(bank.planes[p].data() +
                                               static_cast<std::size_t>(e) * bank.layout.plane_bytes[p]);
            const std::size_t count = bank.layout.plane_bytes[p] / 4;
            for (std::size_t i = 0; i < count; ++i) {
                float v = magnitude(rng);
                if (p == 1 || p == 4) { v *= (rng() & 1) ? 1.0F : -1.0F; }
                if (p == 2) { v *= svh_gate_up; }
                if (p == 5) { v *= svh_down; }
                s[i] = v;
            }
        }
    }
    return bank;
}

struct DeviceBank {
    std::vector<DeviceBuffer> buffers;
    ops::ExpertWeights weights;
};

// Device copy with expert e stored at index placement[e].
DeviceBank upload(const HostBank& bank, const std::vector<int>& placement) {
    DeviceBank out;
    out.weights.layout = bank.layout;
    for (int p = 0; p < 6; ++p) {
        const std::size_t stride = static_cast<std::size_t>(bank.layout.plane_bytes[p]);
        std::vector<std::uint8_t> permuted(stride * E);
        for (int e = 0; e < E; ++e) {
            std::copy_n(bank.planes[p].data() + stride * e, stride,
                        permuted.data() + stride * placement[e]);
        }
        out.buffers.push_back(to_device(permuted));
        out.weights.base[p]   = static_cast<const std::byte*>(out.buffers.back().p);
        out.weights.stride[p] = static_cast<std::int64_t>(stride);
    }
    return out;
}

// ---- oracle for one call ----------------------------------------------------------------------

struct ExpertZ {
    std::vector<std::int16_t> gate, up, down; // Z[k][n]
};

std::vector<double> oracle(const HostBank& bank, const std::vector<float>& x,
                           const std::vector<int>& ids, const std::vector<float>& weights,
                           const std::vector<float>& shared_gate, const std::vector<float>& shared,
                           int tokens) {
    std::map<int, ExpertZ> decoded;
    const int hg = bank.layout.gate_up_half_bits, hd = bank.layout.down_half_bits;
    const auto z_of = [&](int e) -> const ExpertZ& {
        auto it = decoded.find(e);
        if (it != decoded.end()) { return it->second; }
        const std::uint8_t* gu = bank.at(0, e);
        ExpertZ z;
        z.gate = decode_z(gu, hg, I, H);
        z.up   = decode_z(gu + static_cast<std::size_t>(I / 16) * (H / 16) * 16 * hg, hg, I, H);
        z.down = decode_z(bank.at(3, e), hd, H, I);
        return decoded.emplace(e, std::move(z)).first->second;
    };
    std::vector<double> y(static_cast<std::size_t>(H) * tokens);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> acc(H, 0.0);
        const std::vector<double> xt(x.begin() + static_cast<std::ptrdiff_t>(t) * H,
                                     x.begin() + static_cast<std::ptrdiff_t>(t + 1) * H);
        for (int k = 0; k < K; ++k) {
            const int e = ids[static_cast<std::size_t>(t) * K + k];
            const ExpertZ& z = z_of(e);
            const float* suh = bank.scales(1, e);   // [2][H]
            const float* svh = bank.scales(2, e);   // [2][I]
            const float* sd  = bank.scales(4, e);   // [I]
            const float* svd = bank.scales(5, e);   // [H]
            const auto g = apply_matrix(z.gate, H, I, suh, svh, xt);
            const auto u = apply_matrix(z.up, H, I, suh + H, svh + I, xt);
            std::vector<double> act(I);
            for (int n = 0; n < I; ++n) { act[n] = silu(g[n]) * u[n]; }
            const std::vector<double> down =
                apply_matrix(z.down, I, H, sd, svd, act);
            for (int h = 0; h < H; ++h) {
                acc[h] += static_cast<double>(weights[static_cast<std::size_t>(t) * K + k]) * down[h];
            }
        }
        for (int h = 0; h < H; ++h) {
            y[static_cast<std::size_t>(t) * H + h] =
                acc[h] + static_cast<double>(shared_gate[t]) *
                             static_cast<double>(shared[static_cast<std::size_t>(t) * H + h]);
        }
    }
    return y;
}


// ---- cases ------------------------------------------------------------------------------------

std::vector<std::uint16_t> run(const DeviceBank& bank, const std::vector<int>& slot_ids,
                               const std::vector<int>& ids, const std::vector<float>& x,
                               const std::vector<float>& weights,
                               const std::vector<float>& shared_gate,
                               const std::vector<float>& shared, int tokens) {
    DeviceBuffer dx = to_device_bf16(x), dshared = to_device_bf16(shared);
    DeviceBuffer dids = to_device_i32(ids), dslots = to_device_i32(slot_ids);
    DeviceBuffer dw = to_device(weights), dsg = to_device(shared_gate);
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    Tensor tshared(dshared.p, DType::BF16, {H, tokens});
    Tensor tids(dids.p, DType::I32, {K, tokens});
    Tensor tslots(dslots.p, DType::I32, {K, tokens});
    Tensor tw(dw.p, DType::FP32, {K, tokens});
    Tensor tsg(dsg.p, DType::FP32, {tokens});
    GuardedDeviceBuffer out(static_cast<std::size_t>(H) * tokens * 2);
    Tensor ty(out.data(), DType::BF16, {H, tokens});
    WorkspaceArena workspace(ops::moe_experts_workspace_bytes(tokens, E, bank.weights.layout));
    ops::moe_experts(tx, tids, tslots, tw, tsg, tshared, bank.weights, E, workspace, ty, nullptr);
    cuda_synchronize();
    return from_device<std::uint16_t>(out.data(), static_cast<std::size_t>(H) * tokens);
}

// The real cache route: some of the routed experts are resident, the rest are fetched (all six
// planes) on another stream while the resident ones already run. The pool starts as 0xFF so a slot
// read before its fetch lands changes the bits.
std::vector<std::uint16_t> run_cached(const DeviceBank& bank, const std::vector<int>& ids,
                                      const std::vector<float>& x, const std::vector<float>& weights,
                                      const std::vector<float>& shared_gate,
                                      const std::vector<float>& shared, int tokens,
                                      int* misses_out) {
    const std::int32_t slots = std::max(K * tokens, 40);
    DeviceBuffer slot_of = to_device_i32(std::vector<int>(E, -1));
    DeviceBuffer owner   = to_device_i32(std::vector<int>(slots, -1));
    DeviceBuffer stamp(static_cast<std::size_t>(slots) * 8);
    DeviceBuffer counters(3 * 8);
    DeviceBuffer pool(static_cast<std::size_t>(slots) * bank.weights.layout.slot_bytes);
    CUDA_CHECK(cudaMemset(stamp.p, 0, stamp.bytes));
    CUDA_CHECK(cudaMemset(counters.p, 0, counters.bytes));
    CUDA_CHECK(cudaMemset(pool.p, 0xFF, pool.bytes));
    ops::ExpertCacheState cache;
    cache.slot_of    = static_cast<std::int32_t*>(slot_of.p);
    cache.owner      = static_cast<std::int32_t*>(owner.p);
    cache.stamp      = static_cast<unsigned long long*>(stamp.p);
    cache.clock      = static_cast<unsigned long long*>(counters.p);
    cache.statistics = static_cast<unsigned long long*>(counters.p) + 1;
    cache.slots      = slots;
    cache.layers     = 1;
    cache.pool       = static_cast<std::byte*>(pool.p);

    cudaStream_t main = nullptr, fetch = nullptr;
    cudaEvent_t resolved = nullptr, fetched = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&fetch, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&resolved, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&fetched, cudaEventDisableTiming));
    const auto resolve = [&](const std::vector<int>& want, DeviceBuffer& slot_ids,
                             DeviceBuffer& misses) {
        DeviceBuffer d_ids = to_device_i32(want);
        Tensor t_ids(d_ids.p, DType::I32, {static_cast<std::int32_t>(want.size()), 1});
        Tensor t_slots(slot_ids.p, DType::I32, {static_cast<std::int32_t>(want.size()), 1});
        Tensor t_misses(misses.p, DType::I32, {static_cast<std::int32_t>(2 * want.size() + 1)});
        ops::expert_cache_resolve(t_ids, 0, cache, t_slots, t_misses, main);
        CUDA_CHECK(cudaEventRecord(resolved, main));
        CUDA_CHECK(cudaStreamWaitEvent(fetch, resolved));
        ops::expert_cache_fetch(bank.weights, t_misses, static_cast<std::int32_t>(want.size()), cache,
                                fetch);
        CUDA_CHECK(cudaEventRecord(fetched, fetch));
        CUDA_CHECK(cudaStreamSynchronize(fetch));
        CUDA_CHECK(cudaStreamSynchronize(main));
    };
    // Warm half of the first token's experts plus experts it does not route.
    std::vector<int> warm(ids.begin(), ids.begin() + K / 2);
    for (int e = 0; static_cast<int>(warm.size()) < K; ++e) {
        if (std::find(ids.begin(), ids.end(), e) == ids.end()) { warm.push_back(e); }
    }
    {
        DeviceBuffer s = to_device_i32(std::vector<int>(warm.size(), 0));
        DeviceBuffer m = to_device_i32(std::vector<int>(2 * warm.size() + 1, 0));
        resolve(warm, s, m);
    }

    DeviceBuffer dx = to_device_bf16(x), dshared = to_device_bf16(shared);
    DeviceBuffer dids = to_device_i32(ids);
    DeviceBuffer dw = to_device(weights), dsg = to_device(shared_gate);
    DeviceBuffer dslots = to_device_i32(std::vector<int>(ids.size(), -7));
    DeviceBuffer dmisses = to_device_i32(std::vector<int>(2 * ids.size() + 1, -9));
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    Tensor tshared(dshared.p, DType::BF16, {H, tokens});
    Tensor tids(dids.p, DType::I32, {K, tokens});
    Tensor tslots(dslots.p, DType::I32, {K, tokens});
    Tensor tmisses(dmisses.p, DType::I32, {static_cast<std::int32_t>(2 * ids.size() + 1)});
    Tensor tw(dw.p, DType::FP32, {K, tokens});
    Tensor tsg(dsg.p, DType::FP32, {tokens});
    GuardedDeviceBuffer out(static_cast<std::size_t>(H) * tokens * 2);
    Tensor ty(out.data(), DType::BF16, {H, tokens});
    WorkspaceArena workspace(ops::moe_experts_workspace_bytes(tokens, slots, bank.weights.layout));
    ops::expert_cache_resolve(tids, 0, cache, tslots, tmisses, main);
    CUDA_CHECK(cudaEventRecord(resolved, main));
    CUDA_CHECK(cudaStreamWaitEvent(fetch, resolved));
    ops::expert_cache_fetch(bank.weights, tmisses, static_cast<std::int32_t>(ids.size()), cache, fetch);
    CUDA_CHECK(cudaEventRecord(fetched, fetch));
    const ops::MoeExpertsPending wait{&tmisses, fetched};
    ops::moe_experts(tx, tids, tslots, tw, tsg, tshared, ops::expert_cache_weights(cache, bank.weights),
                     slots, workspace, ty, main, &wait);
    CUDA_CHECK(cudaStreamSynchronize(main));
    CUDA_CHECK(cudaStreamSynchronize(fetch));
    *misses_out = from_device<int>(dmisses, 1)[0];
    CUDA_CHECK(cudaEventDestroy(fetched));
    CUDA_CHECK(cudaEventDestroy(resolved));
    CUDA_CHECK(cudaStreamDestroy(fetch));
    CUDA_CHECK(cudaStreamDestroy(main));
    return from_device<std::uint16_t>(out.data(), static_cast<std::size_t>(H) * tokens);
}

// `concentrated`: every token routes the same ten experts, so each expert's job carries every row.
int exl3_case(int gate_up_half_bits, int down_half_bits, int tokens, bool concentrated, bool isolate,
              std::uint64_t seed) {
    const HostBank bank = make_bank(gate_up_half_bits, down_half_bits, seed);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::vector<int> ids(static_cast<std::size_t>(K) * tokens);
    std::vector<float> weights(ids.size());
    std::vector<int> pool(E);
    for (int e = 0; e < E; ++e) { pool[e] = e; }
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int t = 0; t < tokens; ++t) {
        if (!concentrated) { std::shuffle(pool.begin(), pool.end(), rng); }
        float total = 0.0F;
        for (int k = 0; k < K; ++k) {
            ids[static_cast<std::size_t>(t) * K + k] = pool[k];
            weights[static_cast<std::size_t>(t) * K + k] = 0.05F + static_cast<float>(rng() % 100) / 100.0F;
            total += weights[static_cast<std::size_t>(t) * K + k];
        }
        for (int k = 0; k < K; ++k) { weights[static_cast<std::size_t>(t) * K + k] /= total; }
    }
    std::vector<float> x(static_cast<std::size_t>(H) * tokens), shared(x.size()), shared_gate(tokens);
    fill_uniform(x, static_cast<std::uint32_t>(seed) + 1, -2.0F, 2.0F);
    fill_uniform(shared, static_cast<std::uint32_t>(seed) + 2, -1.0F, 1.0F);
    fill_uniform(shared_gate, static_cast<std::uint32_t>(seed) + 3, 0.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(shared);
    if (isolate) {
        std::fill(shared_gate.begin(), shared_gate.end(), 0.0F);
        for (std::size_t i = 0; i < weights.size(); ++i) { weights[i] = i % K == 0 ? 1.0F : 0.0F; }
    }

    std::vector<int> identity(E), reversed(E);
    for (int e = 0; e < E; ++e) {
        identity[e] = e;
        reversed[e] = E - 1 - e;
    }
    std::vector<int> moved(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) { moved[i] = reversed[ids[i]]; }

    const std::string label = "exl3 moe_experts h=" + std::to_string(gate_up_half_bits) + "/" +
                              std::to_string(down_half_bits) + " T=" + std::to_string(tokens) +
                              (concentrated ? " concentrated" : "") + (isolate ? " single expert" : "");
    std::vector<std::uint16_t> by_expert, repeat, by_slot, by_cache;
    int misses = 0;
    {
        const DeviceBank device = upload(bank, identity);
        by_expert = run(device, ids, ids, x, weights, shared_gate, shared, tokens);
        repeat    = run(device, ids, ids, x, weights, shared_gate, shared, tokens);
        by_cache  = run_cached(device, ids, x, weights, shared_gate, shared, tokens, &misses);
    }
    {
        // The same experts stored at other slots must give the same bits.
        const DeviceBank device = upload(bank, reversed);
        by_slot = run(device, moved, ids, x, weights, shared_gate, shared, tokens);
    }
    const std::vector<double> reference = oracle(bank, x, ids, weights, shared_gate, shared, tokens);
    std::vector<double> got(by_expert.size());
    for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32(by_expert[i]); }
    {
        const ReductionStats stats = compute_reduction_stats(got.data(), reference.data(),
                                                             static_cast<std::int64_t>(got.size()));
        std::cout << label << ": relative_l2=" << stats.relative_l2
                  << " max_abs_error=" << stats.maximum_absolute_error
                  << " reference_rms=" << stats.reference_root_mean_square << '\n';
    }
    int failures = verify_reduction(label, got, reference, kExl3Output);
    failures += verify_exact((label + " repeat bits").c_str(), repeat, by_expert);
    failures += verify_exact((label + " slot placement bits").c_str(), by_slot, by_expert);
    failures += verify_exact((label + " cache route with pending fetch bits").c_str(), by_cache, by_expert);
    if (misses <= 0) {
        std::cerr << label << ": the pending fetch case needs missed experts, got " << misses << '\n';
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
    int failures = 0;
    failures += exl3_case(8, 8, 1, false, true, 0x150U);
    // Shipped 4.0 bpw: one column, a speculative verify width, a mid batch, and a call wider than
    // one internal pass (32 columns) whose tail pass is a single column.
    for (const int tokens : {1, 2, 8, 33}) { failures += exl3_case(8, 8, tokens, false, false, 0xE31U + tokens); }
    // Every row on the same experts: each job spans two row groups of 16.
    failures += exl3_case(8, 8, 32, true, false, 0xC0DEU);
    // Other rates: half-bit with the fast window (9, 7, 5), even with a 3-bit step (6), 5-bit
    // step (10), and the half-bit bitwise fallback (11).
    failures += exl3_case(9, 7, 4, false, false, 0x971U);
    failures += exl3_case(6, 10, 3, false, false, 0x610U);
    failures += exl3_case(11, 5, 2, false, false, 0xB05U);
    if (failures == 0) { std::cout << "OK offload_moe_exl3 correctness\n"; }
    return failures == 0 ? 0 : 1;
}
