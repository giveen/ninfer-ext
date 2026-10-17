// Routed-expert cache fetch: the SM-issued host->device copy at controlled miss counts, against the
// copy engine on the same bytes, and what the host bank's allocation and layout do to both.
//
//   ninfer_expert_fetch_bench [--bank malloc-host|register|arena-default|arena-huge]
//                             [--layout planes|slot-major] [--misses 1,2,4,8,16,32] [--iters 50]
//
// The engine's expert bank lives in host memory and its cache fetch reads that memory with SM loads
// (expert_cache_fetch_kernel, a fixed 64-CTA grid by design: more CTAs contend and lose PCIe
// throughput). host_kv and the artifact materializer allocate through PinnedHostBuffer, whose Huge
// mode is advisory (MADV_HUGEPAGE) — without transparent huge pages it lands in the registered-4KiB
// regime, which is far slower than plain cudaMallocHost. This bench separates the per-launch and
// per-miss costs, measures the copy engine on the same bytes, and compares the four allocations.
#include "ninfer/ops/offload_moe.h"

#include "core/arena.h"  // PinnedHostBuffer, PinnedHostPages
#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t E      = ops::kOffloadMoeExperts; // experts in one layer's bank
constexpr std::int32_t kSlots = 64;                      // pool slots the misses land in

// The stored bank keeps one array per plane (stride = that plane's per-expert bytes). A slot-major
// bank keeps whole slots, so plane `p` of expert `e` starts at `base[p] + e * slot_bytes`; the
// fetch kernel and a single per-miss memcpy both accept that addressing.
enum class Layout { Planes, SlotMajor };
enum class Bank { MallocHost, Registered, ArenaDefault, ArenaHuge };

struct HostBank {
    ops::ExpertWeights weights;
    std::vector<PinnedHostBuffer> buffers; // arena allocations own their memory
    std::vector<void*> raw;                // raw allocations, freed here
    Bank mode = Bank::MallocHost;

    HostBank()                           = default;
    HostBank(const HostBank&)            = delete;
    HostBank& operator=(const HostBank&) = delete;

    ~HostBank() {
        for (void* p : raw) {
            if (mode == Bank::MallocHost) {
                (void)cudaFreeHost(p);
            } else {
                (void)cudaHostUnregister(p);
                std::free(p);
            }
        }
    }
};

void* allocate(Bank mode, std::size_t bytes) {
    switch (mode) {
    case Bank::MallocHost: {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, bytes));
        return ptr;
    }
    case Bank::Registered: {
        void* ptr = std::malloc(bytes);
        if (ptr == nullptr) {
            std::fprintf(stderr, "malloc of %zu bytes failed\n", bytes);
            std::exit(1);
        }
        CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault));
        return ptr;
    }
    case Bank::ArenaDefault:
    case Bank::ArenaHuge:
        return nullptr; // the arena path allocates through PinnedHostBuffer, not here
    }
    return nullptr;
}

void fill_bank(HostBank& out, Layout layout, Bank mode) {
    const ops::ExpertLayout plan = ops::nvfp4_expert_layout();
    out.mode                     = mode;
    out.weights.layout           = plan;
    for (std::int32_t p = 0; p < plan.planes; ++p) {
        const std::size_t bytes = layout == Layout::SlotMajor
                                      ? static_cast<std::size_t>(plan.slot_bytes) * E
                                      : static_cast<std::size_t>(plan.plane_bytes[p]) * E;
        void* host              = nullptr;
        if (mode == Bank::ArenaDefault || mode == Bank::ArenaHuge) {
            out.buffers.emplace_back(bytes, mode == Bank::ArenaHuge ? PinnedHostPages::Huge
                                                                    : PinnedHostPages::Default);
            host = out.buffers.back().data();
        } else {
            host = allocate(mode, bytes);
            out.raw.push_back(host);
        }
        std::memset(host, 0x38, bytes); // E4M3 1.0 codes, and finite floats in the scale planes
        if (layout == Layout::SlotMajor) {
            out.weights.base[p]   = static_cast<const std::byte*>(host) + plan.plane_offset(p);
            out.weights.stride[p] = plan.slot_bytes;
        } else {
            out.weights.base[p]   = static_cast<const std::byte*>(host);
            out.weights.stride[p] = plan.plane_bytes[p];
        }
    }
}

std::vector<std::int32_t> parse_list(const std::string_view text) {
    std::vector<std::int32_t> out;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t comma = text.find(',', at);
        out.push_back(std::stoi(
            std::string(text.substr(at, comma == std::string_view::npos ? comma : comma - at))));
        if (comma == std::string_view::npos) { break; }
        at = comma + 1;
    }
    return out;
}

// Median of `iters` timed calls (the first five are warm-up), measured with events on `stream`.
double median_us(cudaStream_t stream, int iters, const std::function<void()>& body,
                 cudaEvent_t begin, cudaEvent_t end) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iters));
    for (int it = 0; it < iters + 5; ++it) {
        CUDA_CHECK(cudaEventRecord(begin, stream));
        body();
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        if (it >= 5) { times.push_back(static_cast<double>(ms) * 1000.0); }
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    Layout layout                    = Layout::Planes;
    Bank mode                        = Bank::MallocHost;
    std::vector<std::int32_t> counts = {1, 2, 4, 8, 16, 32};
    int iters                        = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const auto next          = [&]() -> std::string_view {
            if (++i >= argc) {
                std::fprintf(stderr, "%s needs a value\n", argv[i - 1]);
                std::exit(2);
            }
            return argv[i];
        };
        if (a == "--bank") {
            const std::string_view v = next();
            if (v == "malloc-host") {
                mode = Bank::MallocHost;
            } else if (v == "register") {
                mode = Bank::Registered;
            } else if (v == "arena-default") {
                mode = Bank::ArenaDefault;
            } else if (v == "arena-huge") {
                mode = Bank::ArenaHuge;
            } else {
                std::fprintf(stderr,
                             "bank is malloc-host, register, arena-default or arena-huge\n");
                return 2;
            }
        } else if (a == "--layout") {
            const std::string_view v = next();
            if (v == "planes") {
                layout = Layout::Planes;
            } else if (v == "slot-major") {
                layout = Layout::SlotMajor;
            } else {
                std::fprintf(stderr, "layout is planes or slot-major\n");
                return 2;
            }
        } else if (a == "--misses") {
            counts = parse_list(next());
        } else if (a == "--iters") {
            iters = std::stoi(std::string(next()));
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }

    const ops::ExpertLayout plan = ops::nvfp4_expert_layout();
    HostBank bank;
    fill_bank(bank, layout, mode);

    void* pool = nullptr;
    CUDA_CHECK(cudaMalloc(&pool, static_cast<std::size_t>(plan.slot_bytes) * kSlots));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t begin = nullptr, end = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    const std::int32_t max_misses = *std::max_element(counts.begin(), counts.end());
    std::vector<std::int32_t> host_misses(static_cast<std::size_t>(2 * max_misses + 1), 0);
    std::int32_t* dev_misses = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_misses, host_misses.size() * sizeof(std::int32_t)));
    ops::ExpertCacheState cache;
    cache.pool  = static_cast<std::byte*>(pool);
    cache.slots = kSlots;
    cache.pool_stride = plan.slot_bytes;

    static const char* const kBankNames[] = {"cudaMallocHost", "cudaHostRegister",
                                             "PinnedHostBuffer Default", "PinnedHostBuffer Huge"};
    std::printf("nvfp4 expert slot %lld B, %d experts/layer, %s bank, %s layout\n",
                static_cast<long long>(plan.slot_bytes), E, kBankNames[static_cast<int>(mode)],
                layout == Layout::SlotMajor ? "slot-major" : "plane-major");
    std::printf("%6s %22s %22s %22s\n", "misses", "sm fetch", "memcpy per plane",
                "memcpy per slot");

    for (const std::int32_t n : counts) {
        host_misses[0] = n;
        for (std::int32_t j = 0; j < n; ++j) {
            host_misses[1 + 2 * j] = j % kSlots;      // a distinct slot per miss
            host_misses[2 + 2 * j] = (7 * j + 3) % E; // scattered experts, as the router gives
        }
        CUDA_CHECK(cudaMemcpy(dev_misses, host_misses.data(),
                              host_misses.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
        Tensor tmiss(dev_misses, DType::I32, {static_cast<std::int32_t>(host_misses.size())});

        const auto copy_planes = [&] {
            for (std::int32_t j = 0; j < n; ++j) {
                const std::int32_t slot   = host_misses[1 + 2 * j];
                const std::int32_t expert = host_misses[2 + 2 * j];
                for (std::int32_t p = 0; p < plan.planes; ++p) {
                    CUDA_CHECK(
                        cudaMemcpyAsync(static_cast<std::byte*>(pool) +
                                            static_cast<std::int64_t>(slot) * plan.slot_bytes +
                                            plan.plane_offset(p),
                                        bank.weights.base[p] + static_cast<std::int64_t>(expert) *
                                                                   bank.weights.stride[p],
                                        static_cast<std::size_t>(plan.plane_bytes[p]),
                                        cudaMemcpyHostToDevice, stream));
                }
            }
        };
        const auto copy_slots = [&] {
            for (std::int32_t j = 0; j < n; ++j) {
                const std::int32_t slot   = host_misses[1 + 2 * j];
                const std::int32_t expert = host_misses[2 + 2 * j];
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::byte*>(pool) +
                        static_cast<std::int64_t>(slot) * plan.slot_bytes,
                    bank.weights.base[0] + static_cast<std::int64_t>(expert) * plan.slot_bytes,
                    static_cast<std::size_t>(plan.slot_bytes), cudaMemcpyHostToDevice, stream));
            }
        };

        // Clear the destination first so a failed copy cannot read as a pass.
        CUDA_CHECK(
            cudaMemsetAsync(pool, 0, static_cast<std::size_t>(plan.slot_bytes) * kSlots, stream));
        const double sm = median_us(
            stream, iters,
            [&] { ops::expert_cache_fetch(bank.weights, tmiss, max_misses, cache, stream); }, begin,
            end);
        const double plane = median_us(stream, iters, copy_planes, begin, end);
        const double slot  = median_us(stream, iters, copy_slots, begin, end);

        const double bytes = static_cast<double>(n) * plan.slot_bytes;
        std::printf("%6d %10.2f us %8.1f GB/s %10.2f us %8.1f GB/s %10.2f us %8.1f GB/s\n", n, sm,
                    bytes / (sm * 1e-6) / 1e9, plane, bytes / (plane * 1e-6) / 1e9, slot,
                    bytes / (slot * 1e-6) / 1e9);
    }

    // One check that the paths actually move the bytes: fetch a slot with the SM kernel, then with
    // a single slot-major memcpy, and compare both against a host read of the bank.
    host_misses[0] = 1;
    host_misses[1] = 0;
    host_misses[2] = 5;
    CUDA_CHECK(cudaMemcpy(dev_misses, host_misses.data(), host_misses.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    const auto first_word_matches = [&](const std::byte* host_plane) {
        std::vector<std::byte> got(16);
        CUDA_CHECK(cudaMemcpy(got.data(), static_cast<std::byte*>(pool) + plan.plane_offset(0), 16,
                              cudaMemcpyDeviceToHost));
        return std::memcmp(got.data(),
                           host_plane + static_cast<std::int64_t>(5) * bank.weights.stride[0],
                           16) == 0;
    };
    CUDA_CHECK(
        cudaMemsetAsync(pool, 0, static_cast<std::size_t>(plan.slot_bytes) * kSlots, stream));
    {
        Tensor tmiss(dev_misses, DType::I32, {static_cast<std::int32_t>(host_misses.size())});
        ops::expert_cache_fetch(bank.weights, tmiss, 1, cache, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const bool sm_ok = first_word_matches(bank.weights.base[0]);
    CUDA_CHECK(
        cudaMemsetAsync(pool, 0, static_cast<std::size_t>(plan.slot_bytes) * kSlots, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        pool, bank.weights.base[0] + static_cast<std::int64_t>(5) * bank.weights.stride[0],
        static_cast<std::size_t>(plan.slot_bytes), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const bool memcpy_ok = first_word_matches(bank.weights.base[0]);
    std::printf("verify: sm fetch %s, slot memcpy %s\n", sm_ok ? "ok" : "MISMATCH",
                memcpy_ok ? "ok" : "MISMATCH");

    CUDA_CHECK(cudaFree(dev_misses));
    CUDA_CHECK(cudaFree(pool));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return sm_ok && memcpy_ok ? 0 : 1;
}
