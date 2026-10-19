#include "models/execution/host_expert_lane.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/offload_moe/cpu_nvfp4.h"

#include <cuda_runtime.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::execution {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int H              = ops::kOffloadMoeHidden;
constexpr int I              = ops::kOffloadMoeIntermediate;
constexpr int K              = ops::kOffloadMoeTopK;
constexpr int kTokensPerPass = 8; // ops::cpu kernels take at most 8 tokens per call

// How long an idle leader keeps spinning before it parks, and how often a parked leader polls.
constexpr auto kSpinWindow = std::chrono::milliseconds(20);
constexpr auto kParkedPoll = std::chrono::microseconds(50);
// A worker spins only long enough to catch the next phase of the call it is serving; between
// decode calls it parks, so a full-size team (one thread per pinned core) does not starve the
// process's own threads.
constexpr auto kTeamSpinWindow = std::chrono::microseconds(100);

constexpr std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

void pause() noexcept { __builtin_ia32_pause(); }

std::uint32_t load_volatile(const std::uint32_t* p) noexcept {
    return *reinterpret_cast<const volatile std::uint32_t*>(p);
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

// [lo, hi) of `total` items for team member `i` of `n`.
std::pair<int, int> share_of(int total, int i, int n) {
    return {static_cast<int>(static_cast<std::int64_t>(total) * i / n),
            static_cast<int>(static_cast<std::int64_t>(total) * (i + 1) / n)};
}

ops::cpu::Nvfp4Expert expert_of(const ops::ExpertWeights& bank, int e) {
    const auto plane = [&](int p) {
        return reinterpret_cast<const std::uint8_t*>(bank.base[p]) +
               static_cast<std::int64_t>(e) * bank.stride[p];
    };
    const std::int64_t gate_row = static_cast<std::int64_t>(e) * 2 * I;
    const std::int64_t down_row = static_cast<std::int64_t>(e) * H;
    ops::cpu::Nvfp4Expert x;
    x.gate_up_codes  = plane(0);
    x.gate_up_scales = plane(1);
    x.down_codes     = plane(2);
    x.down_scales    = plane(3);
    x.inv_gate       = 1.0F / bank.gate_up_divisors[gate_row / bank.gate_up_divisor_rows];
    x.inv_up         = 1.0F / bank.gate_up_divisors[(gate_row + I) / bank.gate_up_divisor_rows];
    x.inv_down       = 1.0F / bank.down_divisors[down_row / bank.down_divisor_rows];
    return x;
}

bool host_readable(const void* p) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeHost && attributes.hostPointer == p;
}

// CPUs this process may run on, or 0 when the query fails. A team that fills the allowance would
// leave the Program's own threads (the main decode thread included) no core between phases, which
// measured as a 40-60% decode loss on a 16-thread allowance with a 16-thread team.
int allowed_cpus() noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) { return 0; }
    return CPU_COUNT(&set);
}

} // namespace

HostExpertLane::HostExpertLane(std::vector<ops::ExpertWeights> banks, std::int32_t max_columns,
                               std::int32_t threads)
    : banks_(std::move(banks)), max_columns_(max_columns), threads_(threads) {
    if (banks_.empty() || max_columns_ <= 0 || threads_ <= 0) {
        throw std::invalid_argument("host expert lane needs banks, columns and threads");
    }
    const int allowed         = allowed_cpus();
    const int reserve         = std::min(2, std::max(allowed - 1, 0));
    const std::int32_t usable = std::max(1, allowed - reserve);
    if (allowed > 0 && threads_ > usable) {
        std::fprintf(stderr,
                     "ninfer: host expert lane: %d threads requested, using %d of %d allowed "
                     "CPUs (leaving %d for the process)\n",
                     threads_, usable, allowed, reserve);
        threads_ = usable;
    }
    if (!ops::cpu::nvfp4_host_supported()) {
        throw std::runtime_error("host expert lane needs a CPU with AVX2, FMA and F16C");
    }
    for (const ops::ExpertWeights& bank : banks_) {
        if (bank.layout != ops::nvfp4_expert_layout() || bank.gate_up_divisors == nullptr ||
            bank.down_divisors == nullptr) {
            throw std::invalid_argument("host expert lane serves the Qwen4Exp NVFP4 bank only");
        }
        // The kernels read the stored bank and its divisors in place, so every plane must be a
        // registered Host range whose device pointer is its Host pointer.
        for (int p = 0; p < 4; ++p) {
            if (!host_readable(bank.base[p])) {
                throw std::invalid_argument("host expert lane needs Host-resident expert planes");
            }
        }
        if (!host_readable(bank.gate_up_divisors) || !host_readable(bank.down_divisors)) {
            throw std::invalid_argument("host expert lane needs Host-resident expert divisors");
        }
    }

    const std::size_t assignments  = static_cast<std::size_t>(max_columns_) * K;
    const std::size_t columns      = static_cast<std::size_t>(max_columns_);
    const std::size_t off_doorbell = 64, off_answered = 128, off_jobs = 256;
    const std::size_t off_x    = align256(off_jobs + assignments * sizeof(ops::MoeHostJob));
    const std::size_t off_rows = align256(off_x + columns * H * sizeof(std::uint16_t));
    const std::size_t bytes    = align256(off_rows + columns * H * sizeof(float));
    CUDA_CHECK(cudaHostAlloc(&mapped_, bytes, cudaHostAllocMapped));
    std::memset(mapped_, 0, bytes);
    void* device = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&device, mapped_, 0));
    auto* host_base   = static_cast<std::byte*>(mapped_);
    auto* device_base = static_cast<std::byte*>(device);
    CUDA_CHECK(cudaMalloc(&device_words_, 256));
    CUDA_CHECK(cudaMemset(device_words_, 0, 256));

    ring_.header      = reinterpret_cast<ops::MoeHostHeader*>(device_base);
    ring_.doorbell    = reinterpret_cast<std::uint32_t*>(device_base + off_doorbell);
    ring_.answered    = reinterpret_cast<const std::uint32_t*>(device_base + off_answered);
    ring_.jobs        = reinterpret_cast<ops::MoeHostJob*>(device_base + off_jobs);
    ring_.x           = reinterpret_cast<std::uint16_t*>(device_base + off_x);
    ring_.rows        = reinterpret_cast<const float*>(device_base + off_rows);
    ring_.sequence    = static_cast<std::uint32_t*>(device_words_);
    ring_.ticket      = static_cast<std::uint32_t*>(device_words_) + 1;
    ring_.max_columns = max_columns_;
    answered_         = reinterpret_cast<std::uint32_t*>(host_base + off_answered);

    jobs_.resize(assignments);
    xq_.resize(columns * H);
    xd_.resize(columns * H / 32);
    h_.resize(assignments * I);
    hq_.resize(assignments * I);
    hd_.resize(assignments * I / 32);
    out_.resize(assignments * H);
    groups_.reserve(assignments);

    for (int i = 1; i < threads_; ++i) {
        workers_.emplace_back([this, i] { worker(i); });
    }
    leader_ = std::thread([this] { leader(); });
}

HostExpertLane::~HostExpertLane() {
    stop_.store(true);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    if (leader_.joinable()) { leader_.join(); }
    for (auto& w : workers_) { w.join(); }
    if (device_words_ != nullptr) { (void)cudaFree(device_words_); }
    if (mapped_ != nullptr) { (void)cudaFreeHost(mapped_); }
}

void HostExpertLane::run(const std::function<void(int)>& phase) {
    phase_ = &phase;
    done_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    if (parked_.load(std::memory_order_acquire) > 0) { generation_.notify_all(); }
    phase(0);
    done_.fetch_add(1, std::memory_order_acq_rel);
    while (done_.load(std::memory_order_acquire) < threads_) { pause(); }
}

void HostExpertLane::worker(int index) {
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t generation = generation_.load(std::memory_order_acquire);
        if (generation == seen) {
            const auto spin_start = Clock::now();
            int polls             = 0;
            while ((generation = generation_.load(std::memory_order_acquire)) == seen) {
                pause();
                if (++polls % 256 == 0 && Clock::now() - spin_start > kTeamSpinWindow) {
                    parked_.fetch_add(1, std::memory_order_acq_rel);
                    generation_.wait(seen, std::memory_order_acquire);
                    parked_.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
        }
        seen = generation;
        if (stop_.load()) { return; }
        (*phase_)(index);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void HostExpertLane::leader() {
    const auto* doorbell =
        reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(mapped_) + 64);
    std::uint32_t seen = 0;
    auto last_call     = Clock::now();
    while (!stop_.load(std::memory_order_relaxed)) {
        const std::uint32_t ring = load_volatile(doorbell);
        if (ring != seen) {
            seen = ring;
            std::atomic_thread_fence(std::memory_order_acquire);
            serve(ring);
            last_call = Clock::now();
            continue;
        }
        if (Clock::now() - last_call < kSpinWindow) {
            pause();
        } else {
            std::this_thread::sleep_for(kParkedPoll);
        }
    }
}

void HostExpertLane::serve(std::uint32_t sequence) {
    const auto* base        = static_cast<const std::byte*>(mapped_);
    const auto* header      = reinterpret_cast<const volatile ops::MoeHostHeader*>(base);
    const int layer         = header->layer;
    const int columns       = header->columns;
    const int count         = header->jobs;
    const auto* mapped_jobs = reinterpret_cast<const ops::MoeHostJob*>(base + 256);
    const auto* x_bf16      = reinterpret_cast<const std::uint16_t*>(
        base + (reinterpret_cast<const std::byte*>(ring_.x) -
                reinterpret_cast<const std::byte*>(ring_.header)));
    auto* rows = reinterpret_cast<float*>(const_cast<std::byte*>(base) +
                                          (reinterpret_cast<const std::byte*>(ring_.rows) -
                                           reinterpret_cast<const std::byte*>(ring_.header)));

    const bool valid = layer >= 0 && static_cast<std::size_t>(layer) < banks_.size() &&
                       columns > 0 && columns <= max_columns_ && count > 0 && count <= columns * K;
    if (!valid) {
        // Answer anyway: the device is waiting. Zero rows drop the host share of this call.
        std::fprintf(stderr,
                     "ninfer: host expert lane: malformed call (layer %d, %d columns, %d "
                     "jobs); its experts contribute nothing\n",
                     layer, columns, count);
        std::memset(rows, 0, static_cast<std::size_t>(std::max(columns, 0)) * H * sizeof(float));
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<volatile std::uint32_t*>(answered_) = sequence;
        return;
    }
    std::memcpy(jobs_.data(), mapped_jobs,
                static_cast<std::size_t>(count) * sizeof(ops::MoeHostJob));
    const ops::ExpertWeights& bank = banks_[static_cast<std::size_t>(layer)];

    // Group the jobs by expert, keeping published order inside a group; mark the columns used.
    std::array<std::int32_t, ops::kOffloadMoeExperts> group_of;
    group_of.fill(-1);
    groups_.clear();
    std::vector<std::int32_t> used_columns;
    std::vector<std::int32_t> column_begin(static_cast<std::size_t>(columns) + 1, 0);
    for (int j = 0; j < count; ++j) {
        const ops::MoeHostJob& job = jobs_[static_cast<std::size_t>(j)];
        std::int32_t& g            = group_of[static_cast<std::size_t>(job.expert)];
        if (g < 0) {
            g = static_cast<std::int32_t>(groups_.size());
            groups_.push_back(Group{job.expert, {}});
        }
        groups_[static_cast<std::size_t>(g)].jobs.push_back(j);
        ++column_begin[static_cast<std::size_t>(job.column) + 1];
    }
    for (int t = 0; t < columns; ++t) {
        if (column_begin[static_cast<std::size_t>(t) + 1] > 0) { used_columns.push_back(t); }
        column_begin[static_cast<std::size_t>(t) + 1] += column_begin[static_cast<std::size_t>(t)];
    }
    const int group_count = static_cast<int>(groups_.size());
    const int n           = threads_;

    run([&](int i) {
        // Q8_0 of every column a job reads.
        const auto [lo, hi] = share_of(static_cast<int>(used_columns.size()), i, n);
        float wide[H];
        for (int c = lo; c < hi; ++c) {
            const int t              = used_columns[static_cast<std::size_t>(c)];
            const std::uint16_t* src = x_bf16 + static_cast<std::size_t>(t) * H;
            for (int v = 0; v < H; ++v) { wide[v] = bf16_to_float(src[v]); }
            ops::cpu::quantize_q8_0(wide, H, xq_.data() + static_cast<std::size_t>(t) * H,
                                    xd_.data() + static_cast<std::size_t>(t) * H / 32);
        }
    });
    run([&](int i) {
        // Gate/up rows of every group, split across the team.
        const auto [r0, r1] = share_of(group_count * I, i, n);
        for (int g = r0 / I; g < group_count && g * I < r1; ++g) {
            const Group& group = groups_[static_cast<std::size_t>(g)];
            const auto expert  = expert_of(bank, group.expert);
            const int a = std::max(r0, g * I) - g * I, z = std::min(r1, (g + 1) * I) - g * I;
            for (std::size_t p = 0; p < group.jobs.size(); p += kTokensPerPass) {
                const int tokens =
                    static_cast<int>(std::min<std::size_t>(kTokensPerPass, group.jobs.size() - p));
                ops::cpu::Q8Row x[kTokensPerPass];
                float* h[kTokensPerPass];
                for (int k = 0; k < tokens; ++k) {
                    const std::int32_t j = group.jobs[p + static_cast<std::size_t>(k)];
                    const std::size_t t =
                        static_cast<std::size_t>(jobs_[static_cast<std::size_t>(j)].column);
                    x[k] = {xq_.data() + t * H, xd_.data() + t * H / 32};
                    h[k] = h_.data() + static_cast<std::size_t>(j) * I;
                }
                ops::cpu::nvfp4_gate_up_rows(expert, a, z, x, tokens, h);
            }
        }
    });
    run([&](int i) {
        const auto [lo, hi] = share_of(count, i, n);
        for (int j = lo; j < hi; ++j) {
            const std::size_t at = static_cast<std::size_t>(j) * I;
            ops::cpu::quantize_q8_0(h_.data() + at, I, hq_.data() + at, hd_.data() + at / 32);
        }
    });
    run([&](int i) {
        // Down rows of every group, split across the team.
        const auto [r0, r1] = share_of(group_count * H, i, n);
        for (int g = r0 / H; g < group_count && g * H < r1; ++g) {
            const Group& group = groups_[static_cast<std::size_t>(g)];
            const auto expert  = expert_of(bank, group.expert);
            const int a = std::max(r0, g * H) - g * H, z = std::min(r1, (g + 1) * H) - g * H;
            for (std::size_t p = 0; p < group.jobs.size(); p += kTokensPerPass) {
                const int tokens =
                    static_cast<int>(std::min<std::size_t>(kTokensPerPass, group.jobs.size() - p));
                ops::cpu::Q8Row hq[kTokensPerPass];
                float* out[kTokensPerPass];
                for (int k = 0; k < tokens; ++k) {
                    const std::size_t j =
                        static_cast<std::size_t>(group.jobs[p + static_cast<std::size_t>(k)]);
                    hq[k]  = {hq_.data() + j * I, hd_.data() + j * I / 32};
                    out[k] = out_.data() + j * H;
                }
                ops::cpu::nvfp4_down_rows(expert, a, z, hq, tokens, out);
            }
        }
    });
    run([&](int i) {
        // Each column's jobs, in published (rank) order, weighted into the mapped rows.
        const auto [h0, h1] = share_of(H, i, n);
        for (int t = 0; t < columns; ++t) {
            float* row  = rows + static_cast<std::size_t>(t) * H;
            const int b = column_begin[static_cast<std::size_t>(t)];
            const int e = column_begin[static_cast<std::size_t>(t) + 1];
            for (int v = h0; v < h1; ++v) {
                float sum = 0.0F;
                for (int j = b; j < e; ++j) {
                    sum += jobs_[static_cast<std::size_t>(j)].weight *
                           out_[static_cast<std::size_t>(j) * H + static_cast<std::size_t>(v)];
                }
                row[v] = sum;
            }
        }
    });
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile std::uint32_t*>(answered_) = sequence;
    calls_.fetch_add(1, std::memory_order_relaxed);
    assignments_.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
}

} // namespace ninfer::models::execution
