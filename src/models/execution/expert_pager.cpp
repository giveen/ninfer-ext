#include "models/execution/expert_pager.h"

#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::execution {

ExpertPager::ExpertPager(const ops::ExpertPoolGeometry& pool, const ops::ExpertCacheState& cache,
                         std::int32_t staged_columns, std::int32_t top_k, std::int32_t bank_slot,
                         std::int32_t base_slots)
    : pool_(pool), cache_(cache), staged_columns_(staged_columns), top_k_(top_k),
      bank_slot_(bank_slot), base_slots_(base_slots), decode_slots_(cache.slots) {
    // Every route still needs room for one call's assignments, and the banks must sit past the
    // cache (and, when the idle prefill is wider than the chunk, past its lent top range).
    if (cache_.slots < top_k_ || base_slots_ < top_k_) {
        throw std::logic_error("expert cache is too small for one resolve");
    }
    // The banks are the top of the slot range. A `base_slots_` equal to the whole range means the
    // cache route has no slots of its own -- a family that stages every layer and keeps no
    // per-expert cache -- and then there is no lent range to keep clear of the banks; otherwise the
    // lent range must end below them.
    if ((base_slots_ > bank_slot_ && base_slots_ < cache_.slots) ||
        bank_slot_ + bank_slots(pool_) != cache_.slots) {
        throw std::logic_error("staged banks must be the top of the expert cache slots");
    }
    if (static_cast<std::int32_t>(pool_.layers()) != cache_.layers ||
        pool_.slot_entries() != cache_.slot_entries || pool_.pool_stride() != cache_.pool_stride) {
        throw std::logic_error("the cache state does not describe the pool it serves");
    }
    // The pool's slots are its widest bank's, and the banks are the top of them, so one bank's byte
    // range is `bank_slot_` slots on plus the two layers' divisor tails past the slot region.
    banks_[0] = cache_.pool + static_cast<std::size_t>(bank_slot_) * cache_.pool_stride;
    banks_[1] = banks_[0] + static_cast<std::size_t>(pool_.max_staged_bytes());
    CUDA_CHECK(cudaMallocHost(&resident_, static_cast<std::size_t>(cache_.slot_entries) *
                                              sizeof(std::int32_t)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (int b = 0; b < kBankCount; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&released_[b], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&reclaimed_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&resolved_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&fetched_, cudaEventDisableTiming));
}

ExpertPager::~ExpertPager() {
    // A fill may still be in flight into Program-owned memory.
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    for (int b = 0; b < kBankCount; ++b) {
        if (ready_[b] != nullptr) { (void)cudaEventDestroy(ready_[b]); }
        if (released_[b] != nullptr) { (void)cudaEventDestroy(released_[b]); }
    }
    for (cudaEvent_t event : {reclaimed_, resolved_, fetched_}) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    if (resident_ != nullptr) { (void)cudaFreeHost(resident_); }
}

// Staged-bank addressing of `source`: its planes back to back, then the layout tail (NVFP4: its
// weight divisors). Both the per-plane stride and the tail's split into the layer's two divisor
// arrays follow the layer's own bank, so a layer that is narrower than the pool's widest, or holds
// a different number of experts, lands where its own contract says.
ops::ExpertWeights ExpertPager::staged_weights(int bank, const ops::ExpertBankShape& shape,
                                               const ops::ExpertWeights& source) const {
    ops::ExpertWeights staged = source;
    std::byte* cursor         = banks_[bank];
    for (int plane = 0; plane < shape.layout.planes; ++plane) {
        staged.base[plane] = cursor;
        cursor += static_cast<std::size_t>(source.stride[plane]) * shape.experts;
    }
    if (shape.layout.format != ops::ExpertFormat::Nvfp4) { return staged; }
    auto* divisors          = reinterpret_cast<float*>(cursor);
    staged.gate_up_divisors = divisors;
    staged.down_divisors    = divisors + shape.gate_up_divisors;
    return staged;
}

// Snapshot the cache's residency for the fills of this forward. Staged calls leave the cache
// untouched, so the snapshot stays exact until a cache-route call; a stale one only costs
// bandwidth (expert_cache_stage re-checks the device state). Capture cannot synchronize, so a
// captured forward copies every expert.
void ExpertPager::snapshot_residency(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    resident_valid_ = false;
    if (capture != cudaStreamCaptureStatusNone) { return; }
    // Earlier fills may still read the previous snapshot.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpyAsync(resident_, cache_.slot_of,
                               static_cast<std::size_t>(cache_.slot_entries) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    resident_valid_ = true;
}

// Fill staged bank `bank` with `layer` unless it already holds it. The fill waits until the bank's
// previous reader has released it and marks the bank ready when done. With a residency snapshot,
// Host copies cover only the runs of uncached experts and the cached ones come from their slots.
void ExpertPager::fill_into(std::int32_t bank, std::int32_t layer,
                            const ops::ExpertWeights& source) {
    const int b = bank;
    if (layer_[b] == layer) { return; }
    CUDA_CHECK(cudaStreamWaitEvent(stream_, released_[b]));
    const ops::ExpertBankShape& shape = pool_.bank(layer);
    const ops::ExpertBankSpan span    = pool_.span(layer);
    const ops::ExpertWeights staged   = staged_weights(b, shape, source);
    const std::int32_t* resident =
        resident_valid_ ? resident_ + static_cast<std::ptrdiff_t>(span.first_expert) : nullptr;
    if (resident != nullptr) {
        ops::expert_cache_stage(cache_, span, resident_, source, staged, stream_);
    }
    // One copy per run of uncached experts per plane, plus the layer's weight divisors: the bytes
    // this fill takes from the Host banks, which is what `staged_bytes` reports.
    std::size_t copied = 0;
    for (int plane = 0; plane < shape.layout.planes; ++plane) {
        const auto stride = static_cast<std::size_t>(source.stride[plane]);
        auto* target      = const_cast<std::byte*>(staged.base[plane]);
        for (int first = 0; first < span.experts;) {
            if (resident != nullptr && resident[first] >= 0) {
                ++first;
                continue;
            }
            int last = first + 1;
            while (last < span.experts && (resident == nullptr || resident[last] < 0)) { ++last; }
            CUDA_CHECK(cudaMemcpyAsync(target + first * stride, source.base[plane] + first * stride,
                                       (last - first) * stride, cudaMemcpyDefault, stream_));
            copied += static_cast<std::size_t>(last - first) * stride;
            first = last;
        }
    }
    if (shape.layout.format == ops::ExpertFormat::Nvfp4) {
        // A bank stores one weight divisor per `divisor_rows` rows: per expert matrix, or as few as
        // one per bank (the MTP layer's banks). The staged tail reserves one per expert and matrix
        // of the layer's own bank, so a bank whose divisors do not fit its layer is refused.
        const auto divisors = [](std::int64_t rows, std::int32_t divisor_rows, std::int64_t room) {
            if (divisor_rows <= 0 || rows % divisor_rows != 0 || rows / divisor_rows > room) {
                throw std::logic_error("the staged bank cannot hold the layer's weight divisors");
            }
            return static_cast<std::size_t>(rows / divisor_rows) * sizeof(float);
        };
        const std::size_t gate_up =
            divisors(shape.gate_up_rows, source.gate_up_divisor_rows, shape.gate_up_divisors);
        const std::size_t down =
            divisors(shape.down_rows, source.down_divisor_rows, shape.down_divisors);
        CUDA_CHECK(cudaMemcpyAsync(const_cast<float*>(staged.gate_up_divisors),
                                   source.gate_up_divisors, gate_up, cudaMemcpyDefault, stream_));
        CUDA_CHECK(cudaMemcpyAsync(const_cast<float*>(staged.down_divisors), source.down_divisors,
                                   down, cudaMemcpyDefault, stream_));
        copied += gate_up + down;
    }
    CUDA_CHECK(cudaEventRecord(ready_[b], stream_));
    layer_[b] = layer;
    staged_bytes_ += copied;
    ++staged_layers_;
}

ops::ExpertWeights ExpertPager::acquire_staged(std::int32_t layer, const ops::ExpertWeights& bank,
                                               std::int32_t next_layer,
                                               const ops::ExpertWeights* next_bank,
                                               bool forward_start, cudaStream_t stream) {
    if (forward_start) {
        // The banks double as cache slots between staged forwards, so reclaim whatever the cache
        // placed in them. A wide idle step has already shrunk the cache below the banks and
        // reclaimed the lend range itself, so skip it there.
        if (bank_slot_ < cache_.slots) {
            ops::expert_cache_reclaim(cache_, bank_slot_, stream);
            CUDA_CHECK(cudaEventRecord(reclaimed_, stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream_, reclaimed_));
        }
        layer_[0] = layer_[1] = -1;
        last_bank_            = -1;
    }
    // Which bank holds (or is being filled with) this layer. A layer the previous call prefetched
    // is already in one; otherwise the fill takes whichever bank the last staged layer did not use,
    // so a pass that stages only some layers -- every other bank resident, say -- never fills the
    // bank the layer being read is in. `layer & 1` would do that here, and a gap of one resident
    // layer is enough to make the two collide.
    int b = layer_[0] == layer ? 0 : layer_[1] == layer ? 1 : -1;
    if (b < 0) {
        b = last_bank_ < 0 ? 0 : 1 - last_bank_;
        // A forward's first staged call fills its own layer; later ones were prefetched.
        snapshot_residency(stream);
        fill_into(b, layer, bank);
    }
    if (next_bank != nullptr) { fill_into(1 - b, next_layer, *next_bank); }
    CUDA_CHECK(cudaStreamWaitEvent(stream, ready_[b]));
    last_bank_ = b;
    return staged_weights(b, pool_.bank(layer), bank);
}

void ExpertPager::release_staged(std::int32_t layer, cudaStream_t stream) {
    // The bank the layer is in, not its parity: with resident layers in between, a layer's parity
    // and its bank are different questions.
    const int b = layer_[0] == layer ? 0 : layer_[1] == layer ? 1 : -1;
    if (b < 0) { throw std::logic_error("expert pager: releasing a layer that was never staged"); }
    CUDA_CHECK(cudaEventRecord(released_[b], stream));
}

// P1 lend: the wide idle prefill borrows the top cache slots' bytes as its arena. Invalidate them
// so a later resolve cannot trust a slot whose bytes the arena overwrote, then shrink the cache to
// the base count. `expert_cache_reclaim` reads `cache_.slots` at launch, so it covers the full lend
// range before the shrink.
void ExpertPager::lend(cudaStream_t stream) {
    if (cache_.slots == base_slots_) { return; }
    ops::expert_cache_reclaim(cache_, base_slots_, stream);
    cache_.slots = base_slots_;
}

std::int32_t ExpertPager::max_resolve_columns() const noexcept {
    return std::min(cache_.slots / top_k_, ops::moe_experts_max_columns(cache_.slots));
}

cudaEvent_t ExpertPager::resolve(const Tensor& ids, std::int32_t layer,
                                 const ops::ExpertWeights& bank, Tensor& slot_ids, Tensor& misses,
                                 cudaStream_t stream) {
    const auto columns = static_cast<std::int32_t>(ids.ne[1]);
    if (columns > max_resolve_columns()) {
        throw std::logic_error("expert cache resolve exceeds the cache's column capacity");
    }
    const ops::ExpertBankSpan span = pool_.span(layer);
    ops::expert_cache_resolve(ids, span, cache_, slot_ids, misses, stream);
    // The fetch writes only the slots it lists, which resident-slot work on `stream` never reads.
    CUDA_CHECK(cudaEventRecord(resolved_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_, resolved_));
    ops::expert_cache_fetch(bank, misses, columns * top_k_, cache_, stream_);
    CUDA_CHECK(cudaEventRecord(fetched_, stream_));
    return fetched_;
}

} // namespace ninfer::models::execution
