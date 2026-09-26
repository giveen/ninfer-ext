#include "models/qwen3_5/execution/qwen4_expert_pager.h"

#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

Qwen4ExpertPager::Qwen4ExpertPager(const ops::ExpertCacheState& cache, std::byte* staged_banks,
                                   std::int32_t staged_columns)
    : cache_(cache), staged_columns_(staged_columns) {
    if (cache_.slots < ops::kOffloadMoeTopK) {
        throw std::logic_error("expert cache holds fewer slots than one column");
    }
    if (staged_banks == nullptr) { return; }
    banks_[0] = staged_banks;
    banks_[1] = staged_banks + kStagedBankBytes;
    CUDA_CHECK(cudaMallocHost(&resident_, static_cast<std::size_t>(cache_.layers) *
                                              ops::kOffloadMoeExperts * sizeof(std::int32_t)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&released_[b], cudaEventDisableTiming));
    }
}

Qwen4ExpertPager::~Qwen4ExpertPager() {
    // A fill may still be in flight into Program-owned memory.
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    for (int b = 0; b < 2; ++b) {
        if (ready_[b] != nullptr) { (void)cudaEventDestroy(ready_[b]); }
        if (released_[b] != nullptr) { (void)cudaEventDestroy(released_[b]); }
    }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    if (resident_ != nullptr) { (void)cudaFreeHost(resident_); }
}

// Staged-bank addressing of `source`: its four planes back to back, then its weight divisors.
ops::ExpertWeights Qwen4ExpertPager::staged_weights(int bank,
                                                    const ops::ExpertWeights& source) const {
    ops::ExpertWeights staged = source;
    std::byte* cursor         = banks_[bank];
    for (int plane = 0; plane < 4; ++plane) {
        staged.base[plane] = cursor;
        cursor += static_cast<std::size_t>(source.stride[plane]) * ops::kOffloadMoeExperts;
    }
    auto* divisors          = reinterpret_cast<float*>(cursor);
    staged.gate_up_divisors = divisors;
    staged.down_divisors    = divisors + 2 * ops::kOffloadMoeExperts;
    return staged;
}

// Snapshot the cache's residency for the fills of this forward. Staged calls leave the cache
// untouched, so the snapshot stays exact until a cache-route call; a stale one only costs
// bandwidth (expert_cache_stage re-checks the device state). Capture cannot synchronize, so a
// captured forward copies every expert.
void Qwen4ExpertPager::snapshot_residency(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    resident_valid_ = false;
    if (capture != cudaStreamCaptureStatusNone) { return; }
    // Earlier fills may still read the previous snapshot.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpyAsync(resident_, cache_.slot_of,
                               static_cast<std::size_t>(cache_.layers) * ops::kOffloadMoeExperts *
                                   sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    resident_valid_ = true;
}

// Fill bank layer&1 with `layer` unless it already holds it. The fill waits until the bank's
// previous reader has released it and marks the bank ready when done. With a residency snapshot,
// Host copies cover only the runs of uncached experts and the cached ones come from their slots.
void Qwen4ExpertPager::fill(std::int32_t layer, const ops::ExpertWeights& source) {
    const int b = layer & 1;
    if (layer_[b] == layer) { return; }
    CUDA_CHECK(cudaStreamWaitEvent(stream_, released_[b]));
    const ops::ExpertWeights staged = staged_weights(b, source);
    const std::int32_t* resident =
        resident_valid_ ? resident_ + static_cast<std::ptrdiff_t>(layer) * ops::kOffloadMoeExperts
                        : nullptr;
    if (resident != nullptr) {
        ops::expert_cache_stage(cache_, layer, resident_, source, staged, stream_);
    }
    for (int plane = 0; plane < 4; ++plane) {
        const auto stride = static_cast<std::size_t>(source.stride[plane]);
        auto* target      = const_cast<std::byte*>(staged.base[plane]);
        for (int first = 0; first < ops::kOffloadMoeExperts;) {
            if (resident != nullptr && resident[first] >= 0) {
                ++first;
                continue;
            }
            int last = first + 1;
            while (last < ops::kOffloadMoeExperts && (resident == nullptr || resident[last] < 0)) {
                ++last;
            }
            CUDA_CHECK(cudaMemcpyAsync(target + first * stride, source.base[plane] + first * stride,
                                       (last - first) * stride, cudaMemcpyDefault, stream_));
            first = last;
        }
    }
    // A bank stores one weight divisor per `divisor_rows` rows: per expert matrix, or as few as
    // one per bank (the MTP layer's banks).
    const auto divisors = [](std::int64_t rows, std::int32_t divisor_rows, std::int64_t room) {
        if (divisor_rows <= 0 || rows % divisor_rows != 0 || rows / divisor_rows > room) {
            throw std::logic_error("Qwen4Exp staged bank cannot hold the layer's weight divisors");
        }
        return static_cast<std::size_t>(rows / divisor_rows) * sizeof(float);
    };
    constexpr std::int64_t kGateUpRows =
        std::int64_t(ops::kOffloadMoeExperts) * 2 * ops::kOffloadMoeIntermediate;
    constexpr std::int64_t kDownRows = std::int64_t(ops::kOffloadMoeExperts) * ops::kOffloadMoeHidden;
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<float*>(staged.gate_up_divisors), source.gate_up_divisors,
        divisors(kGateUpRows, source.gate_up_divisor_rows, 2 * ops::kOffloadMoeExperts),
        cudaMemcpyDefault, stream_));
    CUDA_CHECK(cudaMemcpyAsync(const_cast<float*>(staged.down_divisors), source.down_divisors,
                               divisors(kDownRows, source.down_divisor_rows,
                                        ops::kOffloadMoeExperts),
                               cudaMemcpyDefault, stream_));
    CUDA_CHECK(cudaEventRecord(ready_[b], stream_));
    layer_[b] = layer;
}

ops::ExpertWeights Qwen4ExpertPager::acquire_staged(std::int32_t layer,
                                                    const ops::ExpertWeights& bank,
                                                    std::int32_t next_layer,
                                                    const ops::ExpertWeights* next_bank,
                                                    cudaStream_t stream) {
    if (banks_[0] == nullptr) { throw std::logic_error("Qwen4Exp pager has no staged route"); }
    // A forward's first staged call fills its own layer; later layers were prefetched.
    const int b = layer & 1;
    if (layer_[b] != layer) { snapshot_residency(stream); }
    fill(layer, bank);
    if (next_bank != nullptr) { fill(next_layer, *next_bank); }
    CUDA_CHECK(cudaStreamWaitEvent(stream, ready_[b]));
    return staged_weights(b, bank);
}

void Qwen4ExpertPager::release_staged(std::int32_t layer, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(released_[layer & 1], stream));
}

std::int32_t Qwen4ExpertPager::max_resolve_columns() const noexcept {
    return std::min(cache_.slots / ops::kOffloadMoeTopK, ops::moe_experts_max_columns(cache_.slots));
}

void Qwen4ExpertPager::resolve(const Tensor& ids, std::int32_t layer,
                               const ops::ExpertWeights& bank, Tensor& slot_ids, Tensor& misses,
                               cudaStream_t stream) {
    const auto columns = static_cast<std::int32_t>(ids.ne[1]);
    if (columns > max_resolve_columns()) {
        throw std::logic_error("Qwen4Exp resolve exceeds the expert cache's column capacity");
    }
    ops::expert_cache_resolve(ids, layer, cache_, slot_ids, misses, stream);
    ops::expert_cache_fetch(bank, misses, columns * ops::kOffloadMoeTopK, cache_, stream);
}

} // namespace ninfer::models::qwen3_5::execution
