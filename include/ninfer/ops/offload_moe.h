#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Closed Qwen4Exp routed-expert geometry: hidden 2560, 512 experts, top-10, expert width 640,
 * NVFP4 (block_scale_k16_m128x4_v1) gate/up and down matrices with BF16 activations.
 */
inline constexpr std::int32_t kOffloadMoeHidden       = 2560;
inline constexpr std::int32_t kOffloadMoeExperts      = 512;
inline constexpr std::int32_t kOffloadMoeTopK         = 10;
inline constexpr std::int32_t kOffloadMoeIntermediate = 640;

/** Bytes of each plane of one expert and of one complete expert slot. */
inline constexpr std::int64_t kExpertGateUpCodeBytes =
    std::int64_t(2) * kOffloadMoeIntermediate * kOffloadMoeHidden / 2;
inline constexpr std::int64_t kExpertGateUpScaleBytes =
    std::int64_t(2) * kOffloadMoeIntermediate * kOffloadMoeHidden / 16;
inline constexpr std::int64_t kExpertDownCodeBytes =
    std::int64_t(kOffloadMoeHidden) * kOffloadMoeIntermediate / 2;
inline constexpr std::int64_t kExpertDownScaleBytes =
    std::int64_t(kOffloadMoeHidden) * kOffloadMoeIntermediate / 16;
inline constexpr std::int64_t kExpertSlotBytes =
    kExpertGateUpCodeBytes + kExpertGateUpScaleBytes + kExpertDownCodeBytes + kExpertDownScaleBytes;

/**
 * Addressing of routed expert weights. Expert (or slot) index i of plane p begins at
 * `base[p] + i * stride[p]`; planes are gate/up codes, gate/up scales, down codes, down scales.
 * A stored bank uses the plane layout (stride = per-expert plane bytes), a slot pool uses one
 * stride of kExpertSlotBytes with plane offsets inside the slot. Every expert's gate rows precede
 * its up rows. `divisors` is FP32 `[512, 3]` (gate, up, down weight divisors by expert id).
 */
struct ExpertWeights {
    const std::byte* base[4] = {};
    std::int64_t stride[4]   = {};
    const float* divisors    = nullptr;
};

/** Device-resident mutable state of the expert cache; owned by one Program. */
struct ExpertCacheState {
    std::int32_t* slot_of          = nullptr; // [layers*512], -1 when absent
    std::int32_t* owner            = nullptr; // [slots], layer*512+expert or -1
    unsigned long long* stamp      = nullptr; // [slots], last use
    unsigned long long* clock      = nullptr; // [1]
    unsigned long long* statistics = nullptr; // [2]: hits, misses
    std::int32_t slots             = 0;
    std::int32_t layers            = 0;
    std::byte* pool                = nullptr; // [slots, kExpertSlotBytes]
};

/**
 * Router, top-k selection and shared-expert gate for `x [2560,T]`.
 *
 * `router` is BF16 `[513, 2560]`: 512 expert rows then the shared-expert score row. For every
 * column: probabilities are the softmax of the 512 expert logits, the 10 largest are selected
 * (lower expert id wins an exact tie) and renormalized to `weights` (FP32 `[10,T]`), their ids go
 * to `ids` (I32 `[10,T]`), and `shared_gate[t] = sigmoid(shared logit)` (FP32 `[T]`).
 */
void moe_route(const Tensor& x, const Tensor& router, Tensor& ids, Tensor& weights,
               Tensor& shared_gate, cudaStream_t stream);

/**
 * Resolve routed experts of one layer to cache slots, choosing least-recently-used victims for
 * misses. `ids` I32 `[10,T]` in; `slot_ids` I32 `[10,T]` out; `misses` I32 `[2*(10*T) + 1]`
 * receives the miss count followed by (slot, expert) pairs. The cache must hold at least 10*T
 * slots. Every slot touched by this call is protected from eviction within it.
 */
void expert_cache_resolve(const Tensor& ids, std::int32_t layer, const ExpertCacheState& cache,
                          Tensor& slot_ids, Tensor& misses, cudaStream_t stream);

/**
 * Copy every missing expert listed in `misses` from its bank into its slot. The bank planes must
 * be device-addressable (for a host bank, mapped pinned memory); the grid covers `max_misses`
 * entries and idles past the recorded count, so the call is graph-capturable.
 */
void expert_cache_fetch(const ExpertWeights& bank, const Tensor& misses, std::int32_t max_misses,
                        const ExpertCacheState& cache, cudaStream_t stream);

/** Slot-pool addressing of a cache for the given expert divisors. */
[[nodiscard]] ExpertWeights expert_cache_weights(const ExpertCacheState& cache,
                                                 const float* divisors);

[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t tokens, std::int32_t slots);

/**
 * Routed expert SwiGLU and down projections, routing-weight merge and shared expert:
 *
 *   y[:,t] = sum_k weights[k,t] * down_e(silu(gate_e x_t) * up_e x_t) + shared_gate[t] *
 * shared[:,t]
 *
 * where e = expert of assignment (k,t) and `slot_ids [10,T]` locate it in `weights_source`
 * (a cache slot pool or a staged bank indexed by expert id; `slots` bounds those indices).
 * `expert_ids` supply the divisors. `shared` is the shared expert output BF16 `[2560,T]`; `y` is
 * BF16 `[2560,T]`. The oracle decodes each NVFP4 weight exactly and evaluates in FP64; routing
 * terms are summed in k order, so results do not depend on cache placement.
 */
void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& slot_ids,
                 const Tensor& weights, const Tensor& shared_gate, const Tensor& shared,
                 const ExpertWeights& weights_source, std::int32_t slots, WorkspaceArena& workspace,
                 Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
