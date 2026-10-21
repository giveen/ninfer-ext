#pragma once

// Prefill read-ahead for the Qwen4Exp n-gram (PLE) rows.
//
// The gather is O(tokens) host work - a scalar E4M3-to-BF16 decode of sixteen table rows per token,
// over a table far larger than the page cache leaves resident - and a chunk's rows are needed before
// its first layer reads them. Gathering them inside the chunk leaves the device idle for the whole
// gather: profiling a 64,512-token prompt put 92% of the prefill's CPU samples under
// PleGather::decode_row, self time and page-fault frames together, while `program_submit` ran
// 13.8 s against a 1.5 s device wait.
//
// One extra pinned plane, owned by the Program: the worker gathers the next chunk into it while the
// device runs the current one, and waits for the event of the device copy that read the plane
// before it so a gather never overwrites rows a transfer is still reading. Chunks are equal-sized
// by construction, so the next chunk's window is this chunk's width; any other window falls back to
// the caller's synchronous gather.

#include "models/qwen3_5/execution/ple_gather.h"

#include <cstdint>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5::execution {

class PleReadAhead {
public:
    // `plane` is a pinned BF16 `[width, capacity]` plane, owned by the caller and outliving this
    // object; it must not be the plane a synchronous gather or a decode round writes.
    PleReadAhead(PleGather& gather, std::uint16_t* plane, std::uint32_t width, std::int64_t capacity);
    ~PleReadAhead();

    PleReadAhead(const PleReadAhead&)            = delete;
    PleReadAhead& operator=(const PleReadAhead&) = delete;

    /**
     * The plane when it holds at least `[first, first + count)`, or null when no read-ahead window
     * matches (the caller gathers synchronously). Awaits any gather in flight.
     */
    [[nodiscard]] std::uint16_t* take(std::uint32_t first, std::uint32_t count);

    /**
     * After the device copy from the plane `take` returned, records the event the next gather waits
     * for. Call it before `start`; it is a no-op when nothing was taken.
     */
    void record_copy(cudaStream_t stream);

    /**
     * Starts gathering `[first, first + count)` of `tokens` into the plane. Awaits the previous
     * gather first; a window the plane already holds is not gathered again.
     */
    void start(std::span<const std::int32_t> tokens, std::uint32_t first, std::uint32_t count);

    /** Awaits any gather. `start` and `take` do this themselves; the destructor does too. */
    void join();

    [[nodiscard]] std::int64_t capacity() const noexcept { return capacity_; }

private:
    void promote();

    PleGather* gather_      = nullptr;
    std::uint16_t* plane_   = nullptr;
    std::uint32_t width_    = 0;
    std::int64_t capacity_  = 0;

    std::uint32_t resident_first_ = 0; // the window `plane` holds, once `promote` has run
    std::uint32_t resident_count_ = 0;
    std::uint32_t pending_first_  = 0; // what the worker is filling
    std::uint32_t pending_count_  = 0;
    bool pending_active_          = false;
    bool handed_                  = false; // the last `take` returned the plane
    std::thread worker_;
    std::exception_ptr error_;
    cudaEvent_t copied_ = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
