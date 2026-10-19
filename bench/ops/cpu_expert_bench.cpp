// Host NVFP4 routed-expert benchmark (Qwen4Exp geometry) for the decode miss split.
//
// A bank of random experts larger than the last-level cache sits in 2 MiB-page memory; every call
// computes E distinct random experts for T tokens on a team of spinning threads (the caller is one
// of them): Q8_0 of x, gate/up rows split across the team, Q8_0 of h, down rows split across the
// team. Timing is the median wall time of the whole call. GB/s counts the experts' weight bytes.
//
//   ninfer_cpu_expert_bench [--threads 1,4,8,16] [--experts 1,2,4] [--tokens 1] [--iters N]
//
// Before timing, one call is checked against an FP64 reference over the dequantized weights: on
// the kernel's own Q8_0 activations (the arithmetic's own error) and on the unrounded ones (the
// cost of quantizing activations).
#include "ops/offload_moe/cpu_nvfp4.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::ops::cpu::kNvfp4Hidden;
using ninfer::ops::cpu::kNvfp4Intermediate;
using ninfer::ops::cpu::Nvfp4Expert;
using ninfer::ops::cpu::Q8Row;

constexpr int H = kNvfp4Hidden, I = kNvfp4Intermediate;
constexpr std::int64_t kGuCode  = std::int64_t(2) * I * H / 2;
constexpr std::int64_t kGuScale = std::int64_t(2) * I * H / 16;
constexpr std::int64_t kDnCode  = std::int64_t(H) * I / 2;
constexpr std::int64_t kDnScale = std::int64_t(H) * I / 16;
constexpr std::int64_t kExpertBytes = kGuCode + kGuScale + kDnCode + kDnScale;
constexpr int kMaxT = 8;

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p != '\0';) {
        out.push_back(std::atoi(p));
        while (*p != '\0' && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

double e4m3(std::uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const double v = e == 0 ? m / 512.0 : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}

double e2m1(int nibble) {
    static const double v[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (nibble & 8) ? -v[nibble & 7] : v[nibble & 7];
}

std::int64_t scale_word(int row, int block, int blocks) {
    return (std::int64_t(row >> 7) * blocks + block) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4;
}

// Dequantized weight (row, col) of a plane with `cols` columns.
double weight(const std::uint8_t* codes, const std::uint8_t* scales, int cols, int row, int col) {
    const std::uint8_t byte = codes[std::int64_t(row) * (cols / 2) + col / 2];
    const int nibble = (col & 1) ? byte >> 4 : byte & 0xF;
    const std::uint8_t s = scales[scale_word(row, col / 64, cols / 64) + (col % 64) / 16];
    return e2m1(nibble) * e4m3(s);
}

struct Bank {
    std::uint8_t* bytes = nullptr;
    int experts = 0;
    std::vector<float> inv; // 3 per expert

    Nvfp4Expert expert(int e) const {
        std::uint8_t* base = bytes + std::int64_t(e) * kExpertBytes;
        Nvfp4Expert x;
        x.gate_up_codes  = base;
        x.gate_up_scales = base + kGuCode;
        x.down_codes     = base + kGuCode + kGuScale;
        x.down_scales    = base + kGuCode + kGuScale + kDnCode;
        x.inv_gate = inv[3 * e]; x.inv_up = inv[3 * e + 1]; x.inv_down = inv[3 * e + 2];
        return x;
    }
};

Bank make_bank(int experts, std::mt19937& rng) {
    Bank bank;
    bank.experts = experts;
    const std::size_t bytes = ((std::size_t(experts) * kExpertBytes) + (2u << 20) - 1) & ~((2u << 20) - 1);
    bank.bytes = static_cast<std::uint8_t*>(std::aligned_alloc(2u << 20, bytes));
    madvise(bank.bytes, bytes, MADV_HUGEPAGE);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> scale_exp(5, 9), mant(0, 7);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* base = bank.bytes + std::int64_t(e) * kExpertBytes;
        for (std::int64_t i = 0; i < kGuCode; ++i) { base[i] = static_cast<std::uint8_t>(byte(rng)); }
        for (std::int64_t i = 0; i < kGuScale; ++i) {
            base[kGuCode + i] = static_cast<std::uint8_t>((scale_exp(rng) << 3) | mant(rng));
        }
        std::uint8_t* dn = base + kGuCode + kGuScale;
        for (std::int64_t i = 0; i < kDnCode; ++i) { dn[i] = static_cast<std::uint8_t>(byte(rng)); }
        for (std::int64_t i = 0; i < kDnScale; ++i) {
            dn[kDnCode + i] = static_cast<std::uint8_t>((scale_exp(rng) << 3) | mant(rng));
        }
    }
    std::uniform_real_distribution<float> div(0.05F, 0.2F);
    bank.inv.resize(std::size_t(experts) * 3);
    for (float& v : bank.inv) { v = div(rng); }
    return bank;
}

// A team of `n` threads (the caller is thread 0) that runs one function per phase and joins.
class Team {
public:
    explicit Team(int n) : n_(n) {
        for (int i = 1; i < n; ++i) { workers_.emplace_back([this, i] { loop(i); }); }
    }
    ~Team() {
        stop_.store(true);
        generation_.fetch_add(1, std::memory_order_release);
        for (auto& w : workers_) { w.join(); }
    }
    template <class F>
    void run(F&& f) {
        task_ = [&f](int i) { f(i); };
        done_.store(0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        task_(0);
        done_.fetch_add(1, std::memory_order_acq_rel);
        while (done_.load(std::memory_order_acquire) < n_) { _mm_pause_(); }
    }
    int size() const { return n_; }

private:
    static void _mm_pause_() { __builtin_ia32_pause(); }
    void loop(int i) {
        std::uint64_t seen = 0;
        for (;;) {
            std::uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) { _mm_pause_(); }
            seen = g;
            if (stop_.load()) { return; }
            task_(i);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    int n_;
    std::vector<std::thread> workers_;
    std::function<void(int)> task_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
};

struct Buffers {
    std::vector<std::int8_t> xq;
    std::vector<float> xd, h, hd, out;
    std::vector<std::int8_t> hq;
    Buffers(int experts, int tokens)
        : xq(std::size_t(tokens) * H), xd(std::size_t(tokens) * H / 32),
          h(std::size_t(experts) * tokens * I), hd(std::size_t(experts) * tokens * I / 32),
          out(std::size_t(experts) * tokens * H), hq(std::size_t(experts) * tokens * I) {}
};

// One call: E experts (ids) for T tokens of x.
void call(Team& team, const Bank& bank, const int* ids, int E, int T, const float* x, Buffers& b) {
    for (int t = 0; t < T; ++t) {
        ninfer::ops::cpu::quantize_q8_0(x + std::size_t(t) * H, H, b.xq.data() + std::size_t(t) * H,
                                        b.xd.data() + std::size_t(t) * H / 32);
    }
    Q8Row xr[kMaxT];
    for (int t = 0; t < T; ++t) { xr[t] = {b.xq.data() + std::size_t(t) * H, b.xd.data() + std::size_t(t) * H / 32}; }
    const int n = team.size();
    team.run([&](int i) {
        const int total = E * I, r0 = int(std::int64_t(total) * i / n), r1 = int(std::int64_t(total) * (i + 1) / n);
        for (int e = r0 / I; e < E && e * I < r1; ++e) {
            const int a = std::max(r0, e * I) - e * I, z = std::min(r1, (e + 1) * I) - e * I;
            float* hp[kMaxT];
            for (int t = 0; t < T; ++t) { hp[t] = b.h.data() + (std::size_t(e) * T + t) * I; }
            ninfer::ops::cpu::nvfp4_gate_up_rows(bank.expert(ids[e]), a, z, xr, T, hp);
        }
    });
    team.run([&](int i) {
        for (int k = i; k < E * T; k += n) {
            ninfer::ops::cpu::quantize_q8_0(b.h.data() + std::size_t(k) * I, I, b.hq.data() + std::size_t(k) * I,
                                            b.hd.data() + std::size_t(k) * I / 32);
        }
    });
    team.run([&](int i) {
        const int total = E * H, r0 = int(std::int64_t(total) * i / n), r1 = int(std::int64_t(total) * (i + 1) / n);
        for (int e = r0 / H; e < E && e * H < r1; ++e) {
            const int a = std::max(r0, e * H) - e * H, z = std::min(r1, (e + 1) * H) - e * H;
            Q8Row hr[kMaxT];
            float* op[kMaxT];
            for (int t = 0; t < T; ++t) {
                const std::size_t k = std::size_t(e) * T + t;
                hr[t] = {b.hq.data() + k * I, b.hd.data() + k * I / 32};
                op[t] = b.out.data() + k * H;
            }
            ninfer::ops::cpu::nvfp4_down_rows(bank.expert(ids[e]), a, z, hr, T, op);
        }
    });
}

double rel_rms(const std::vector<double>& ref, const float* got) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

// FP64 reference of one expert and token: gate/up from `xin`, down from `hin` (or from the
// reference's own h when `hin` is null). Returns h and out.
void reference(const Bank& bank, int id, const std::vector<double>& xin, const std::vector<double>* hin,
               std::vector<double>& h, std::vector<double>& out) {
    const Nvfp4Expert e = bank.expert(id);
    h.assign(I, 0.0);
    for (int r = 0; r < I; ++r) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
            g += weight(e.gate_up_codes, e.gate_up_scales, H, r, c) * xin[c];
            u += weight(e.gate_up_codes, e.gate_up_scales, H, r + I, c) * xin[c];
        }
        g *= e.inv_gate; u *= e.inv_up;
        h[r] = g / (1.0 + std::exp(-g)) * u;
    }
    const std::vector<double>& hv = hin != nullptr ? *hin : h;
    out.assign(H, 0.0);
    for (int r = 0; r < H; ++r) {
        double s = 0;
        for (int c = 0; c < I; ++c) { s += weight(e.down_codes, e.down_scales, I, r, c) * hv[c]; }
        out[r] = s * e.inv_down;
    }
}

int check(const Bank& bank, std::mt19937& rng) {
    Team team(4);
    Buffers b(1, 1);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(H);
    for (float& v : x) { v = normal(rng); }
    const int id = 3;
    call(team, bank, &id, 1, 1, x.data(), b);
    std::vector<double> xq(H), hq(I), h, out, h_exact, out_exact;
    for (int c = 0; c < H; ++c) { xq[c] = double(b.xq[c]) * b.xd[c / 32]; }
    for (int c = 0; c < I; ++c) { hq[c] = double(b.hq[c]) * b.hd[c / 32]; }
    reference(bank, id, xq, &hq, h, out);
    std::vector<double> xd(x.begin(), x.end());
    reference(bank, id, xd, nullptr, h_exact, out_exact);
    const double eh = rel_rms(h, b.h.data()), eo = rel_rms(out, b.out.data());
    const double ehx = rel_rms(h_exact, b.h.data()), eox = rel_rms(out_exact, b.out.data());
    std::printf("check (isa %s): own-rounding rel rms h %.2e out %.2e | vs unrounded h %.2e out %.2e\n",
                ninfer::ops::cpu::nvfp4_host_isa(), eh, eo, ehx, eox);
    // The kernel's own arithmetic is FP32 over exact int32 group sums; the Q8_0 cost is ~1e-2.
    return (eh < 1e-4 && eo < 1e-4 && eox < 5e-2) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> threads = {1, 2, 4, 8, 12, 16}, experts = {1, 2, 4}, tokens = {1};
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) { threads = parse_list(argv[++i]); }
        else if (a == "--experts" && i + 1 < argc) { experts = parse_list(argv[++i]); }
        else if (a == "--tokens" && i + 1 < argc) { tokens = parse_list(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc) { iters = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "usage: %s [--threads L] [--experts L] [--tokens L] [--iters N]\n", argv[0]);
            return 2;
        }
    }
    if (!ninfer::ops::cpu::nvfp4_host_supported()) {
        std::fprintf(stderr, "this CPU lacks AVX2/FMA/F16C\n");
        return 2;
    }
    std::mt19937 rng(1234);
    constexpr int kBankExperts = 128; // 354 MB, ten times the L3
    const Bank bank = make_bank(kBankExperts, rng);
    if (check(bank, rng) != 0) { std::fprintf(stderr, "check FAILED\n"); return 1; }

    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(std::size_t(kMaxT) * H);
    for (float& v : x) { v = normal(rng); }
    std::printf("%8s %8s %7s %10s %10s %9s\n", "threads", "experts", "tokens", "median_us", "p90_us", "GB/s");
    for (int n : threads) {
        Team team(n);
        for (int T : tokens) {
            for (int E : experts) {
                Buffers b(E, T);
                std::vector<double> us;
                std::vector<int> ids(E);
                for (int it = 0; it < iters + 20; ++it) {
                    // distinct random experts, so each call streams its weights from DRAM
                    for (int e = 0; e < E; ++e) {
                        int v;
                        do { v = int(rng() % kBankExperts); } while (std::find(ids.begin(), ids.begin() + e, v) != ids.begin() + e);
                        ids[e] = v;
                    }
                    const auto t0 = std::chrono::steady_clock::now();
                    call(team, bank, ids.data(), E, T, x.data(), b);
                    const auto t1 = std::chrono::steady_clock::now();
                    if (it >= 20) { us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()); }
                }
                std::sort(us.begin(), us.end());
                const double med = us[us.size() / 2], p90 = us[us.size() * 9 / 10];
                std::printf("%8d %8d %7d %10.1f %10.1f %9.1f\n", n, E, T, med, p90,
                            double(E) * kExpertBytes / (med * 1e3));
            }
        }
    }
    std::free(bank.bytes);
    return 0;
}
