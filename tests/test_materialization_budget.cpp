#include "runtime/engine/context_cache/materialization_budget.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::runtime;

static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    constexpr std::uint64_t ms = 1'000'000;
    try {
        auto idle = PlanningAllowance::boundary(0, 0);
        auto busy = PlanningAllowance::boundary(2, 0);

        // The base grant scales with the incumbent's value at stake (its total machine-work cost):
        // a cheap incumbent gets the 5 ms floor, a medium one cost/20, an expensive one the whole
        // planning allowance, and a saturated cost earns nothing.
        require(MaterializationSearchBudget(idle, 0, 1 * ms).granted_ns() == 5 * ms,
                "cheap request lost its 5 ms floor grant");
        require(MaterializationSearchBudget(idle, 0, 400 * ms).granted_ns() == 20 * ms,
                "base grant did not scale with the incumbent cost");
        require(MaterializationSearchBudget(idle, 0, 80'000 * ms).granted_ns() == idle.limit_ns,
                "expensive grant did not reach the planning allowance");
        require(MaterializationSearchBudget(idle, 0, UINT64_MAX).granted_ns() == 0,
                "saturated cost was used as evidence of large value");
        // A cheap request is still economically bounded once it spends its floor.
        {
            MaterializationSearchBudget cheap(idle, 0, 1 * ms);
            require(!cheap.allow(5 * ms, ms, ms, 1 * ms, false, 1),
                    "a cheap request renewed past its floor on a low gain");
        }

        // A valuable completion extends past the base grant; cumulative accounting is retained,
        // and a later cheap gain is denied with the economic (not wall) stop reason.
        {
            MaterializationSearchBudget renew(idle, 0, 400 * ms);
            require(renew.allow(20 * ms, 2 * ms, 3 * ms, 70'000 * ms, true, 1),
                    "valuable completion could not extend past the base grant");
            require(renew.granted_ns() == 40 * ms && renew.renewals() == 1,
                    "extension did not retain cumulative accounting");
            require(!renew.allow(40 * ms, 2 * ms, 2 * ms, 1 * ms, true, 2),
                    "an improved incumbent retained the expensive root's budget");
            require(renew.stop_reason() ==
                        ninfer::MaterializationStopReason::InsufficientExpectedGain,
                    "economic stopping was reported as wall exhaustion");
        }

        // An unknown candidate gets one bounded discovery episode, then a complete prediction
        // may continue.
        {
            MaterializationSearchBudget discovery(idle, 0, 400 * ms);
            require(discovery.allow(20 * ms, ms, 4 * ms, 70'000 * ms, false, 1),
                    "unknown candidate could not receive bounded discovery");
            require(!discovery.allow(25 * ms, ms, ms, 70'000 * ms, false, 2),
                    "unknown candidate repeatedly renewed discovery");
            require(discovery.allow(25 * ms, ms, ms, 70'000 * ms, true, 2),
                    "complete prediction could not continue after discovery");
        }

        // Mandatory (busy) work consumes the boundary allowance; the base grant never exceeds it.
        {
            MaterializationSearchBudget first(busy, busy.limit_ns - 4 * ms, 80'000 * ms);
            require(first.granted_ns() == 4 * ms, "mandatory work did not consume boundary time");
            MaterializationSearchBudget backfill(busy, busy.limit_ns - ms, 80'000 * ms);
            require(backfill.granted_ns() == ms, "backfill reset the boundary allowance");
            require(!backfill.allow(busy.limit_ns, 1, 1, 70'000 * ms, true, 1),
                    "search delayed runnable requests beyond their boundary");
            require(backfill.overshoot(busy.limit_ns + ms) == ms,
                    "indivisible overrun was not measured");
        }

        // A busy boundary that already spent time on setup still leaves room for the first
        // complete reuse assessment (the case the fixed 5 ms base starved).
        {
            MaterializationSearchBudget restore_after_setup(busy, 3 * ms + ms / 2, 80'000 * ms);
            require(restore_after_setup.allow(3 * ms + ms / 2, 2 * ms, 2 * ms, 70'000 * ms, true, 1),
                    "mandatory setup starved the first complete reuse assessment in a busy boundary");
        }

        // An already-seeded candidate cannot renew on an incomplete estimate, but a complete
        // profitable refinement may.
        {
            MaterializationSearchBudget seeded(idle, 0, 400 * ms);
            require(!seeded.allow(20 * ms, ms, ms, 70'000 * ms, false, 1, false),
                    "an already-seeded candidate renewed solely on an incomplete optimistic estimate");
            require(seeded.allow(20 * ms, ms, ms, 70'000 * ms, true, 1, false),
                    "a complete profitable refinement was denied after seeding");
        }

        // Stalled work (no progress) does not renew its allowance.
        {
            MaterializationSearchBudget stalled(idle, 0, 400 * ms);
            require(stalled.allow(20 * ms, ms, ms, 70'000 * ms, true, 7), "first forecast grant failed");
            require(!stalled.allow(40 * ms, ms, ms, 70'000 * ms, true, 7),
                    "stalled work renewed its allowance");
        }

        // The control deadline and cancellation constrain the boundary directly.
        {
            std::atomic<bool> cancelled{false};
            auto controlled                = PlanningAllowance::boundary(0, 0);
            controlled.cancellation        = &cancelled;
            controlled.control_deadline_ns = 3 * ms;
            require(controlled.remaining(2 * ms) == ms, "control deadline did not constrain planning");
            cancelled.store(true);
            require(controlled.remaining(0) == 0, "cancelled request kept optional planning headroom");
        }

        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
