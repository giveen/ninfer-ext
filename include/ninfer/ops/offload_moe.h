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
 * its up rows. Weight divisors are read from the stored banks by expert id: gate/up bank row r
 * uses `gate_up_divisors[r / gate_up_divisor_rows]` and down bank row r uses
 * `down_divisors[r / down_divisor_rows]`. The activation divisors are the layer's NVFP4 input
 * divisors of gate/up and down, positive when the checkpoint permits A4 inputs and zero otherwise.
 */
struct ExpertWeights {
    const std::byte* base[4]          = {};
    std::int64_t stride[4]            = {};
    const float* gate_up_divisors     = nullptr;
    std::int32_t gate_up_divisor_rows = 0;
    const float* down_divisors        = nullptr;
    std::int32_t down_divisor_rows    = 0;
    float gate_up_input_divisor       = 0.0F;
    float down_input_divisor          = 0.0F;
};

/** Weight divisors of one layer: gate and up of every expert, then down of every expert. */
inline constexpr std::int64_t kExpertDivisorBytes =
    std::int64_t(3) * kOffloadMoeExperts * sizeof(float);

/** One staged layer: every expert's planes in the bank layout, then the layer's divisors. */
inline constexpr std::int64_t kExpertStagedLayerBytes =
    std::int64_t(kOffloadMoeExperts) * kExpertSlotBytes + kExpertDivisorBytes;

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
void moe_route(const Tensor& x, const Tensor& router, WorkspaceArena& workspace, Tensor& ids,
               Tensor& weights, Tensor& shared_gate, cudaStream_t stream);

/** Workspace of moe_route: the FP32 router logits of `tokens` columns. */
[[nodiscard]] std::size_t moe_route_workspace_bytes(std::int32_t tokens);

/**
 * Give the padding columns of a speculative batch their lane's routing, so they fetch no expert
 * of their own. `ids` is I32 `[10, W*B]` with column `w + W*b` belonging to lane b, and
 * `valid_columns` is I32 `[B]`. For every lane with `1 <= valid_columns[b] < W`:
 *
 *   ids[:, w + W*b] = ids[:, valid_columns[b]-1 + W*b]   for valid_columns[b] <= w < W.
 *
 * Other columns are unchanged. The outputs of padding columns are never consumed, so only which
 * experts they touch changes, never a valid column's result.
 */
void moe_route_share_padding(Tensor& ids, const Tensor& valid_columns, std::int32_t width,
                             cudaStream_t stream);

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

/**
 * Fill the experts of `layer` that `resident` marks as cached into the staged bank `staged`.
 * `resident` is a device-readable (for example mapped pinned) snapshot of `cache.slot_of`; the
 * caller copies every expert it marks absent itself. A marked expert comes from its cache slot
 * while the device state still holds it and otherwise from `bank`, so a stale snapshot costs
 * bandwidth, never correctness.
 */
void expert_cache_stage(const ExpertCacheState& cache, std::int32_t layer,
                        const std::int32_t* resident, const ExpertWeights& bank,
                        const ExpertWeights& staged, cudaStream_t stream);

/**
 * Evict every expert held in slots `[first_slot, cache.slots)`: their `slot_of` entries become -1,
 * their owners -1 and their stamps 0, so the range may be overwritten. Other slots are unchanged.
 */
void expert_cache_reclaim(const ExpertCacheState& cache, std::int32_t first_slot,
                          cudaStream_t stream);

/** Slot-pool addressing of a cache whose experts come from `bank` (divisors stay in the bank). */
[[nodiscard]] ExpertWeights expert_cache_weights(const ExpertCacheState& cache,
                                                 const ExpertWeights& bank);

[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t tokens, std::int32_t slots);

/**
 * Most columns one moe_experts call over a pool of `slots` slots accepts: job building sorts a
 * call's assignments in shared memory, or counts per slot when the pool has at most 4096 slots.
 */
[[nodiscard]] std::int32_t moe_experts_max_columns(std::int32_t slots) noexcept;

/**
 * Routed expert SwiGLU and down projections, routing-weight merge and shared expert:
 *
 *   y[:,t] = sum_k weights[k,t] * down_e(silu(gate_e x_t) * up_e x_t) + shared_gate[t] *
 * shared[:,t]
 *
 * where e = expert of assignment (k,t) and `slot_ids [10,T]` locate it in `weights_source`
 * (a cache slot pool or a staged bank indexed by expert id; `slots` bounds those indices).
 * `expert_ids` supply the divisors. `shared` is the shared expert output BF16 `[2560,T]`; `y` is
 * BF16 `[2560,T]`. Code planes of `weights_source` are 16-byte aligned (base and stride), scale
 * planes 2-byte aligned. The oracle decodes each NVFP4 weight exactly and evaluates in FP64;
 * routing terms are summed in k order, so results do not depend on cache placement.
 *
 * With `pending`, the slots listed in `pending->misses` (an expert_cache_resolve miss list) may
 * still be filling: the other assignments run first, then `stream` waits on `pending->fetched`
 * before the listed slots are read. The result is identical to a call made after the fetch.
 */
struct MoeExpertsPending {
    const Tensor* misses;
    cudaEvent_t fetched;
};

void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& slot_ids,
                 const Tensor& weights, const Tensor& shared_gate, const Tensor& shared,
                 const ExpertWeights& weights_source, std::int32_t slots, WorkspaceArena& workspace,
                 Tensor& y, cudaStream_t stream, const MoeExpertsPending* pending = nullptr);

[[nodiscard]] std::size_t moe_experts_a4_workspace_bytes(std::int32_t tokens);

/**
 * moe_experts over a device-staged layer bank indexed by expert id, with A4 activations: `x` is
 * quantized to NVFP4 with `staged.gate_up_input_divisor` and each SwiGLU intermediate with
 * `staged.down_input_divisor`, and both projections run on NVFP4 tensor cores. The formula and
 * its FP64 oracle are moe_experts'; the activation quantization is covered by this route's output
 * criterion. The staged planes use the bank layout and, like both weight-divisor arrays, must be
 * device memory; both activation divisors must be positive.
 */
void moe_experts_a4(const Tensor& x, const Tensor& expert_ids, const Tensor& weights,
                    const Tensor& shared_gate, const Tensor& shared, const ExpertWeights& staged,
                    WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
