#pragma once

// HostExpertLane: the host side of the decode miss split. A cache-route call whose resolve admits
// only part of its misses publishes the rest through `ring()` (`ops::moe_host_publish`); the lane
// computes those experts on a team of host threads straight from the registered Host banks and
// answers with the FP32 contribution the call's merge adds (`ops::moe_experts` with the ring).
//
// The experts run in `ops::cpu` arithmetic: Q8_0 activations and exact int32 group sums, so a host
// expert rounds differently from the device kernels and decode output depends on which experts
// the cache held when the call ran (accepted for this route; see docs/flash-next.md).
//
// Program-owned. The team spins while calls keep arriving and parks after an idle interval; a
// parked leader polls the doorbell at a coarse interval, so the first call after idling waits for
// at most that interval. Not copyable; destruction stops and joins the threads, and the caller must
// not destroy it while a published call is unanswered on the device.

#include "ninfer/ops/offload_moe.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace ninfer::models::execution {

class HostExpertLane {
public:
    /**
     * `banks[layer]` is cache layer `layer`'s NVFP4 bank (the stored bank: its planes and divisors
     * in registered Host memory). `max_columns` bounds one call; `threads` is the team size, the
     * leader included (at least 1).
     */
    HostExpertLane(std::vector<ops::ExpertWeights> banks, std::int32_t max_columns,
                   std::int32_t threads);
    ~HostExpertLane();

    HostExpertLane(const HostExpertLane&)            = delete;
    HostExpertLane& operator=(const HostExpertLane&) = delete;

    [[nodiscard]] const ops::MoeHostRing& ring() const noexcept { return ring_; }

    [[nodiscard]] std::int32_t threads() const noexcept { return threads_; }

    /** Calls answered and assignments computed so far (measurement only). */
    [[nodiscard]] std::uint64_t calls() const noexcept { return calls_.load(); }

    [[nodiscard]] std::uint64_t assignments() const noexcept { return assignments_.load(); }

private:
    void leader();
    void worker(int index);
    // Run `phase(i)` on every team member i (the leader is 0) and return when all are done.
    void run(const std::function<void(int)>& phase);
    void serve(std::uint32_t sequence);

    std::vector<ops::ExpertWeights> banks_;
    std::int32_t max_columns_ = 0;
    std::int32_t threads_     = 0;
    ops::MoeHostRing ring_{};

    // Mapped Host words and buffers (cudaHostAlloc), and the two device words.
    void* mapped_         = nullptr;
    void* device_words_   = nullptr;
    std::uint32_t* answered_ = nullptr; // Host view of ring_.answered

    // Per-call scratch, sized for 10 * max_columns assignments.
    std::vector<ops::MoeHostJob> jobs_;
    std::vector<std::int8_t> xq_, hq_;
    std::vector<float> xd_, hd_, h_, out_;
    struct Group {
        std::int32_t expert = 0;
        std::vector<std::int32_t> jobs; // indices into jobs_, in published order
    };
    std::vector<Group> groups_;

    // Team: generation-counted phases; workers spin, then park on `generation_`.
    std::vector<std::thread> workers_;
    const std::function<void(int)>* phase_ = nullptr;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<std::int32_t> done_{0};
    std::atomic<std::int32_t> parked_{0};
    std::atomic<bool> stop_{false};
    std::thread leader_;

    std::atomic<std::uint64_t> calls_{0};
    std::atomic<std::uint64_t> assignments_{0};
};

} // namespace ninfer::models::execution
