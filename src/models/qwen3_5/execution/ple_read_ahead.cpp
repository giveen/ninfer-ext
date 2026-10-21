#include "models/qwen3_5/execution/ple_read_ahead.h"

#include "core/device.h" // CUDA_CHECK

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {

PleReadAhead::PleReadAhead(PleGather& gather, std::uint16_t* plane, std::uint32_t width,
                           std::int64_t capacity)
    : gather_(&gather), plane_(plane), width_(width), capacity_(capacity) {
    if (width == 0 || capacity <= 0 || plane == nullptr) {
        throw std::invalid_argument("PLE read-ahead needs a plane, a width and a positive capacity");
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
}

PleReadAhead::~PleReadAhead() {
    // A gather may still be writing the caller's plane, so let it finish before the plane goes.
    try {
        join();
    } catch (...) {
        // A destructor reports nothing; `take` already raised the failure to the request.
    }
    if (copied_ != nullptr) { (void)cudaEventDestroy(copied_); }
    copied_ = nullptr;
}

void PleReadAhead::promote() {
    if (worker_.joinable()) { worker_.join(); }
    if (pending_active_) {
        resident_first_ = pending_first_;
        resident_count_ = pending_count_;
        pending_active_ = false;
    }
    if (error_ != nullptr) {
        auto error      = std::exchange(error_, nullptr);
        resident_count_ = 0;
        std::rethrow_exception(error);
    }
}

void PleReadAhead::join() { promote(); }

std::uint16_t* PleReadAhead::take(std::uint32_t first, std::uint32_t count) {
    promote();
    if (count == 0 || count > static_cast<std::uint64_t>(capacity_) || resident_count_ < count ||
        resident_first_ != first) {
        handed_ = false;
        return nullptr;
    }
    handed_ = true;
    return plane_;
}

void PleReadAhead::record_copy(cudaStream_t stream) {
    if (!handed_ || stream == nullptr) { return; }
    CUDA_CHECK(cudaEventRecord(copied_, stream));
}

void PleReadAhead::start(std::span<const std::int32_t> tokens, std::uint32_t first,
                         std::uint32_t count) {
    promote();
    handed_ = false;
    if (count == 0 || count > static_cast<std::uint64_t>(capacity_)) { return; }
    if (first + count > tokens.size()) {
        throw std::invalid_argument("PLE read-ahead window exceeds the prompt");
    }
    if (resident_count_ >= count && resident_first_ == first) { return; }
    pending_first_  = first;
    pending_count_  = count;
    pending_active_ = true;
    // The gather addresses the prompt by absolute position, so it takes its own copy of the tokens:
    // the caller's prompt may be released before the next chunk asks for the result.
    std::vector<std::int32_t> copy(tokens.begin(), tokens.end());
    std::uint16_t* const out  = plane_;
    const std::uint32_t width = width_;
    cudaEvent_t reuse_after   = copied_;
    worker_ = std::thread([this, copy = std::move(copy), first, count, out, width, reuse_after] {
        try {
            // The device may still be copying the rows this plane held; wait for that copy rather
            // than overwrite them.
            CUDA_CHECK(cudaEventSynchronize(reuse_after));
            gather_->gather(copy, first, count,
                            std::span<std::uint16_t>(out, static_cast<std::size_t>(width) * count));
        } catch (...) {
            error_ = std::current_exception();
        }
    });
}

} // namespace ninfer::models::qwen3_5::execution
