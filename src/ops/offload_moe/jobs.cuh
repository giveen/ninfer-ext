#pragma once

// Job view shared by the offloaded-MoE projection kernels: a job is one run of <= kJobTokens
// assignments that share a cache slot (see moe_jobs_kernel).

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kJobTokens = 64;

struct JobView {
    int begin;
    int count;
    int slot;
    int expert;
};

// Selects which jobs a projection pass runs: all of them, or only the slots that are (not) listed
// in a resolve miss list.
struct JobFilter {
    const std::int32_t* misses; // [count, (slot, expert)...]; null runs every job
    bool missed;                // run the listed slots (true) or the others (false)

    __device__ __forceinline__ bool admits(int slot) const {
        if (misses == nullptr) { return true; }
        const int count = misses[0];
        bool listed     = false;
        for (int j = 0; j < count && !listed; ++j) { listed = misses[1 + 2 * j] == slot; }
        return listed == missed;
    }
};

// First assignment and slot of this CTA's job; count 0 when the CTA has no job or `filter` skips
// it.
__device__ __forceinline__ JobView job_head(const std::int32_t* jobs, const std::int32_t* job_count,
                                            const std::int32_t* sorted_slot, JobFilter filter) {
    JobView view{-1, 0, -1, -1};
    const int job = static_cast<int>(blockIdx.x);
    if (job >= *job_count) { return view; }
    view.begin = jobs[job];
    view.slot  = sorted_slot[view.begin];
    if (!filter.admits(view.slot)) { return view; }
    view.count = 1;
    return view;
}

// Token count and expert of a job whose head is known.
__device__ __forceinline__ void job_tail(JobView& view, const std::int32_t* sorted_slot,
                                         const std::int32_t* sorted_assign,
                                         const std::int32_t* expert_ids, int assignments) {
    int count = 0;
    while (count < kJobTokens && view.begin + count < assignments &&
           sorted_slot[view.begin + count] == view.slot) {
        ++count;
    }
    view.count  = count;
    view.expert = expert_ids[sorted_assign[view.begin]];
}

__device__ __forceinline__ JobView load_job(const std::int32_t* jobs, const std::int32_t* job_count,
                                            const std::int32_t* sorted_slot,
                                            const std::int32_t* sorted_assign,
                                            const std::int32_t* expert_ids, int assignments,
                                            JobFilter filter) {
    JobView view = job_head(jobs, job_count, sorted_slot, filter);
    if (view.count == 0) { return view; }
    job_tail(view, sorted_slot, sorted_assign, expert_ids, assignments);
    return view;
}

} // namespace ninfer::ops::detail
