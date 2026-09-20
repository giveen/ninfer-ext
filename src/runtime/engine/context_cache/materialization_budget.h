#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>

namespace ninfer::runtime {

template <class Clock = std::chrono::steady_clock>
[[nodiscard]] std::uint64_t planning_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

// One Engine admission boundary, including all head/backfill inspections. Correctness work and
// sealing cannot be cancelled by this allowance, but their elapsed time reduces optional headroom.
struct PlanningAllowance {
    std::uint64_t started_ns              = planning_now_ns();
    std::uint64_t limit_ns                = 50'000'000;
    std::uint32_t affected_requests       = 1;
    const std::atomic<bool>* cancellation = nullptr;
    std::uint64_t control_deadline_ns     = std::numeric_limits<std::uint64_t>::max();

    [[nodiscard]] static PlanningAllowance
    boundary(std::uint32_t other_runnable, std::uint64_t now = planning_now_ns()) noexcept {
        return {.started_ns        = now,
                .limit_ns          = other_runnable == 0 ? 50'000'000ULL : 10'000'000ULL,
                .affected_requests = 1U + other_runnable};
    }

    [[nodiscard]] std::uint64_t remaining(std::uint64_t now) const noexcept {
        if ((cancellation && cancellation->load(std::memory_order_relaxed)) ||
            now >= control_deadline_ns) {
            return 0;
        }
        const auto elapsed = now > started_ns ? now - started_ns : 0;
        return std::min(elapsed < limit_ns ? limit_ns - elapsed : 0, control_deadline_ns - now);
    }
};

// Integer timestamps make wall-budget decisions reproducible without sleeping in policy tests.
// Estimates control optional work only; they never certify feasibility or a search bound.
class MaterializationSearchBudget {
public:
    MaterializationSearchBudget(PlanningAllowance allowance, std::uint64_t started,
                                std::uint64_t initial_cost) noexcept
        : allowance_(allowance), started_(started),
          granted_(std::min(base_grant(initial_cost), allowance.remaining(started))) {}

    [[nodiscard]] bool allow(std::uint64_t now, std::uint64_t next_operation_ns,
                             std::uint64_t completion_ns, std::uint64_t gain_ns,
                             bool complete_prediction, std::uint64_t progress,
                             bool discovery_eligible = true) noexcept {
        boundary_limited_             = false;
        const std::uint64_t elapsed   = now > started_ ? now - started_ : 0;
        const std::uint64_t remaining = allowance_.remaining(now);
        next_operation_ns             = std::max<std::uint64_t>(1, next_operation_ns);
        completion_ns                 = std::max(next_operation_ns, completion_ns);
        if (next_operation_ns > remaining) {
            boundary_limited_ = true;
            reason_           = MaterializationStopReason::TimeBudget;
            return false;
        }
        if (elapsed < granted_ && next_operation_ns <= granted_ - elapsed) { return true; }
        if (completion_ns > remaining || completion_ns > economic(gain_ns)) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        // Unknown forecasts get one bounded discovery episode, not a fresh grant for every node.
        if (!complete_prediction && (!discovery_eligible || discovery_used_)) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        if (renewals_ != 0 && progress <= renewal_progress_) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        const auto hard = elapsed + remaining; // bounded by the allowance's representable duration
        const auto growth = std::max(granted_, next_operation_ns);
        auto added        = std::min(growth, hard > granted_ ? hard - granted_ : 0);
        if (!complete_prediction) { added = std::min<std::uint64_t>(added, 5'000'000); }
        if (added == 0 || elapsed + next_operation_ns > granted_ + added) {
            reason_ = MaterializationStopReason::TimeBudget;
            return false;
        }
        granted_ += added;
        ++renewals_;
        renewal_progress_ = progress;
        discovery_used_   = discovery_used_ || !complete_prediction;
        return true;
    }

    [[nodiscard]] std::uint64_t granted_ns() const noexcept { return granted_; }

    [[nodiscard]] std::uint32_t renewals() const noexcept { return renewals_; }

    [[nodiscard]] bool boundary_limited() const noexcept { return boundary_limited_; }

    [[nodiscard]] bool discovery_used() const noexcept { return discovery_used_; }

    [[nodiscard]] MaterializationStopReason stop_reason() const noexcept { return reason_; }

    [[nodiscard]] std::uint64_t overshoot(std::uint64_t now) const noexcept {
        const auto elapsed = now > started_ ? now - started_ : 0;
        return elapsed > granted_ ? elapsed - granted_ : 0;
    }

private:
    // Base search grant in ns. Scales with the incumbent's value at stake (its total machine-work
    // cost) instead of a fixed 5 ms, so an expensive incumbent -- where a better eviction plan
    // saves a lot -- earns a longer search, while a cheap one still gets the 5 ms floor. A
    // saturated (UINT64_MAX) cost is not evidence of large value, so it earns no base grant.
    static constexpr std::uint64_t kMinSearchGrantNs  = 5'000'000;   // 5 ms floor
    static constexpr std::uint64_t kMaxSearchGrantNs  = 250'000'000; // 250 ms cap
    static constexpr std::uint64_t kSearchCostDivisor = 20;

    [[nodiscard]] static std::uint64_t base_grant(std::uint64_t incumbent_cost) noexcept {
        if (incumbent_cost == std::numeric_limits<std::uint64_t>::max()) { return 0; }
        const std::uint64_t scaled = incumbent_cost / kSearchCostDivisor;
        return std::min(kMaxSearchGrantNs, std::max(kMinSearchGrantNs, scaled));
    }

    [[nodiscard]] std::uint64_t economic(std::uint64_t gain) const noexcept {
        if (gain == std::numeric_limits<std::uint64_t>::max()) { return 0; }
        return gain / kSearchCostDivisor / std::max(1U, allowance_.affected_requests);
    }

    bool boundary_limited_ = false;
    PlanningAllowance allowance_;
    std::uint64_t started_            = 0;
    std::uint64_t granted_            = 0;
    std::uint64_t renewal_progress_   = 0;
    std::uint32_t renewals_           = 0;
    bool discovery_used_              = false;
    MaterializationStopReason reason_ = MaterializationStopReason::TimeBudget;
};

} // namespace ninfer::runtime
