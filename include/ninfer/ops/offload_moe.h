#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::ops {

/**
 * Closed Qwen4Exp routed-expert geometry: hidden 2560, 512 experts, top-10, expert width 640,
 * NVFP4 (block_scale_k16_m128x4_v1) gate/up and down matrices with BF16 activations.
 */
inline constexpr std::int32_t kOffloadMoeHidden       = 2560;
inline constexpr std::int32_t kOffloadMoeExperts      = 512;
inline constexpr std::int32_t kOffloadMoeTopK         = 10;
inline constexpr std::int32_t kOffloadMoeIntermediate = 640;

/** Most planes one expert stores (NVFP4: 4, EXL3: 6). */
inline constexpr std::int32_t kMaxExpertPlanes = 6;

/**
 * Most experts one *bank* may hold. `expert_cache_resolve` keeps one shared-memory table entry per
 * expert, so a wider bank has nowhere to resolve; both served families are far under it (Qwen4Exp
 * 512, and a 74B MoE family whose layers run 240 to 496).
 */
inline constexpr std::int32_t kMaxBankExperts = 1024;

enum class ExpertFormat : std::uint8_t {
    Nvfp4, // block-scaled NVFP4 codes and scales, divisors in the staged layer's tail
    Exl3,  // EXL3 mul1 trellis, with per-expert scale sets
};

/**
 * How one expert is laid out in a bank, a cache slot and a staged layer. A slot holds an expert's
 * planes back to back in plane order; a staged layer holds every expert's slot-ordered planes in
 * the bank layout (plane p of expert e at `base[p] + e * stride[p]`), then `tail_bytes` of
 * per-layer data. The pager, cache and planner size everything from this table, so the Op's
 * geometry is stated once, by the bound bank. How many of these experts a bank holds is the bank's
 * own (see `ExpertBankShape`): this table describes one expert of it.
 */
struct ExpertLayout {
    ExpertFormat format                        = ExpertFormat::Nvfp4;
    std::int32_t planes                        = 0;
    std::int64_t plane_bytes[kMaxExpertPlanes] = {}; // per expert
    std::int64_t slot_bytes                    = 0;  // sum of plane_bytes
    std::int64_t tail_bytes                    = 0;  // per staged layer, after the planes
    std::int32_t gate_up_half_bits             = 0;  // EXL3 trellis rates; 0 for NVFP4
    std::int32_t down_half_bits                = 0;

    [[nodiscard]] bool operator==(const ExpertLayout&) const = default;

    /** Offset of plane `p` inside a cache slot; `p` must be in [0,planes), which is at most
     * kMaxExpertPlanes. */
    [[nodiscard]] constexpr std::int64_t plane_offset(std::int32_t p) const {
        std::int64_t offset = 0;
        for (std::int32_t i = 0; i < p; ++i) { offset += plane_bytes[i]; }
        return offset;
    }

    /** One staged layer of `experts` experts. */
    [[nodiscard]] constexpr std::int64_t staged_layer_bytes(std::int32_t experts) const {
        return std::int64_t(experts) * slot_bytes + tail_bytes;
    }
};

/**
 * One layer's routed-expert bank: how many experts it holds, how one expert of it is laid out, and
 * how a staged layer of it stores that layer's weight divisors.
 *
 * Both the expert count and the per-expert width vary between the layers of a pool, so a pool
 * describes itself with one of these per layer rather than with a single geometry. A uniform pool
 * is the case where every entry is equal, which is why no consumer needs a separate path for it.
 *
 * `gate_up_rows`/`down_rows` are the whole-bank row extent of the layout's two matrix groups (for
 * NVFP4 its gate/up planes and its down planes; for a non-gated bank its up and down matrices), and
 * `gate_up_divisors`/`down_divisors` are the weight divisors a staged layer of this bank reserves
 * for them: one per expert and matrix of the group. The rows are what the staged fill checks a
 * stored bank's own divisor granularity against, and the counts are what it copies, so a bank that
 * is one expert short, one matrix wider, or declares a divisor array that does not fit its tail is
 * refused instead of read out of range.
 *
 * A layout whose staged form carries no divisors (EXL3, whose per-expert scale sets are planes)
 * leaves the tail empty: `tail_bytes` is zero and none of the four are consulted.
 */
struct ExpertBankShape {
    std::int32_t experts          = 0;
    std::int64_t gate_up_rows     = 0;
    std::int32_t gate_up_divisors = 0;
    std::int64_t down_rows        = 0;
    std::int32_t down_divisors    = 0;
    ExpertLayout layout{};
};

/**
 * The NVFP4 routed-expert bank of one layer: `experts` experts, each `gate_up_matrices` matrices of
 * `intermediate` rows into `input_width` columns (gate and up for a gated bank, the up matrix alone
 * for a non-gated one) followed by one down matrix of `input_width` rows and `intermediate`
 * columns, codes then scales, with one weight divisor per expert and matrix in the staged tail.
 */
[[nodiscard]] constexpr ExpertBankShape nvfp4_expert_bank(std::int32_t experts,
                                                          std::int32_t input_width,
                                                          std::int32_t intermediate,
                                                          std::int32_t gate_up_matrices) {
    ExpertBankShape out;
    out.experts           = experts;
    ExpertLayout& layout  = out.layout;
    layout.format         = ExpertFormat::Nvfp4;
    layout.planes         = 4;
    layout.plane_bytes[0] = std::int64_t(gate_up_matrices) * intermediate * input_width / 2;
    layout.plane_bytes[1] = std::int64_t(gate_up_matrices) * intermediate * input_width / 16;
    layout.plane_bytes[2] = std::int64_t(input_width) * intermediate / 2;
    layout.plane_bytes[3] = std::int64_t(input_width) * intermediate / 16;
    for (std::int32_t p = 0; p < layout.planes; ++p) { layout.slot_bytes += layout.plane_bytes[p]; }
    layout.tail_bytes    = std::int64_t(gate_up_matrices + 1) * experts * sizeof(float);
    out.gate_up_rows     = std::int64_t(gate_up_matrices) * experts * intermediate;
    out.gate_up_divisors = gate_up_matrices * experts;
    out.down_rows        = std::int64_t(experts) * input_width;
    out.down_divisors    = experts;
    return out;
}

/** The Qwen4Exp routed-expert bank, and the per-expert and staged sizes that follow from it. */
inline constexpr ExpertBankShape kQwen4ExpertBank =
    nvfp4_expert_bank(kOffloadMoeExperts, kOffloadMoeHidden, kOffloadMoeIntermediate, 2);
inline constexpr std::int64_t kExpertGateUpCodeBytes  = kQwen4ExpertBank.layout.plane_bytes[0];
inline constexpr std::int64_t kExpertGateUpScaleBytes = kQwen4ExpertBank.layout.plane_bytes[1];
inline constexpr std::int64_t kExpertDownCodeBytes    = kQwen4ExpertBank.layout.plane_bytes[2];
inline constexpr std::int64_t kExpertDownScaleBytes   = kQwen4ExpertBank.layout.plane_bytes[3];
inline constexpr std::int64_t kExpertSlotBytes        = kQwen4ExpertBank.layout.slot_bytes;
inline constexpr std::int64_t kExpertDivisorBytes     = kQwen4ExpertBank.layout.tail_bytes;

/** The NVFP4 routed-expert layout: gate/up codes, gate/up scales, down codes, down scales. */
[[nodiscard]] constexpr ExpertLayout nvfp4_expert_layout() { return kQwen4ExpertBank.layout; }

/**
 * The EXL3 mul1 routed-expert layout at the given trellis rates (half-bits per weight, 8 = 4.0
 * bpw). Planes: gate/up trellis, gate/up `suh`, gate/up `svh`, down trellis, down `suh`, down
 * `svh`. An expert's gate and up matrices are two adjacent scale sets of the gate/up bank, so each
 * plane is contiguous per expert. Scale planes are FP32 and do not depend on the rate.
 */
[[nodiscard]] constexpr ExpertLayout exl3_expert_layout(std::int32_t gate_up_half_bits,
                                                        std::int32_t down_half_bits) {
    ExpertLayout layout;
    layout.format            = ExpertFormat::Exl3;
    layout.planes            = 6;
    layout.gate_up_half_bits = gate_up_half_bits;
    layout.down_half_bits    = down_half_bits;
    const std::int64_t gate_up_tiles =
        std::int64_t(2) * kOffloadMoeIntermediate / 16 * (kOffloadMoeHidden / 16);
    const std::int64_t down_tiles =
        std::int64_t(kOffloadMoeHidden) / 16 * (kOffloadMoeIntermediate / 16);
    layout.plane_bytes[0] = gate_up_tiles * 16 * gate_up_half_bits;
    layout.plane_bytes[1] = std::int64_t(2) * kOffloadMoeHidden * 4;
    layout.plane_bytes[2] = std::int64_t(2) * kOffloadMoeIntermediate * 4;
    layout.plane_bytes[3] = down_tiles * 16 * down_half_bits;
    layout.plane_bytes[4] = std::int64_t(kOffloadMoeIntermediate) * 4;
    layout.plane_bytes[5] = std::int64_t(kOffloadMoeHidden) * 4;
    for (std::int32_t p = 0; p < layout.planes; ++p) { layout.slot_bytes += layout.plane_bytes[p]; }
    return layout;
}

/** One staged NVFP4 layer: every expert's planes in the bank layout, then the layer's divisors. */
inline constexpr std::int64_t kExpertStagedLayerBytes =
    nvfp4_expert_layout().staged_layer_bytes(kOffloadMoeExperts);

/**
 * Addressing of routed expert weights. Expert (or slot) index i of plane p begins at
 * `base[p] + i * stride[p]`; for NVFP4 the planes are gate/up codes, gate/up scales, down codes,
 * down scales. A stored bank uses the plane layout (stride = per-expert plane bytes), a slot pool
 * uses one stride of the pool's `pool_stride` with the layer's plane offsets inside the slot
 * (which is `layout.slot_bytes` when every layer's bank is equally wide). Every expert's gate
 * rows precede its up rows. NVFP4 weight divisors are read from the stored banks by expert id:
 * gate/up bank row r uses `gate_up_divisors[r / gate_up_divisor_rows]` and down bank row r uses
 * `down_divisors[r / down_divisor_rows]`. The activation divisors are the layer's NVFP4 input
 * divisors of gate/up and down, positive when the checkpoint permits A4 inputs and zero otherwise.
 */
struct ExpertWeights {
    ExpertLayout layout                     = nvfp4_expert_layout();
    const std::byte* base[kMaxExpertPlanes] = {};
    std::int64_t stride[kMaxExpertPlanes]   = {};
    const float* gate_up_divisors           = nullptr;
    std::int32_t gate_up_divisor_rows       = 0;
    const float* down_divisors              = nullptr;
    std::int32_t down_divisor_rows          = 0;
    float gate_up_input_divisor             = 0.0F;
    float down_input_divisor                = 0.0F;
};

/**
 * One layer's bank as it sits in a pool: the expert count and the offset of its first expert in the
 * pool's expert space (`slot_of`): the two numbers every per-layer cache call needs and the only
 * ones that differ between one layer's bank and another's. `ExpertPoolGeometry::span` produces it.
 */
struct ExpertBankSpan {
    std::int32_t first_expert = 0;
    std::int32_t experts      = 0;
};

/**
 * Host-side addressing of a routed-expert pool: what a pool of `ExpertBankShape`s means once the
 * layers are placed. The pool's slots are one array with one stride — the widest bank's
 * `slot_bytes`, so a narrower layer's expert is a shorter run inside its slot — and its experts are
 * one array in layer order, so a layer's expert `e` is the pool's expert `slot_base(layer) + e` and
 * an `owner` is exactly that pool-expert index. That makes an owner decodable without a search only
 * for a uniform pool, which is why the decode is here and not at the call sites.
 *
 * The type is what lets one cache serve pools that differ per layer: every derived number is a
 * function of the table, and the uniform case is the one where the table's entries are equal, so
 * `slot_base(layer) == layer * experts` and `pool_stride() == layout(0).slot_bytes` fall out rather
 * than being a second path.
 */
class ExpertPoolGeometry {
public:
    /** `banks` are the pool's layers in order: the order `slot_of`, `owner` and the staged banks
     * use. Throws if the table cannot be addressed (empty, a bank with no experts, or a bank whose
     * staged layer is not 16-byte aligned so its successor would start misaligned). */
    explicit ExpertPoolGeometry(std::span<const ExpertBankShape> banks);

    [[nodiscard]] std::int32_t layers() const noexcept {
        return static_cast<std::int32_t>(banks_.size());
    }

    [[nodiscard]] const ExpertBankShape& bank(std::int32_t layer) const;

    [[nodiscard]] const ExpertLayout& layout(std::int32_t layer) const {
        return bank(layer).layout;
    }

    [[nodiscard]] std::int32_t experts(std::int32_t layer) const { return bank(layer).experts; }

    /** Where this layer's experts start in the pool's expert space. */
    [[nodiscard]] std::int32_t slot_base(std::int32_t layer) const;

    [[nodiscard]] ExpertBankSpan span(std::int32_t layer) const {
        return ExpertBankSpan{slot_base(layer), experts(layer)};
    }

    /** Entries in `slot_of`, and the highest expert id + 1 the pool can hold. */
    [[nodiscard]] std::int32_t slot_entries() const noexcept {
        return slot_base_.empty() ? 0 : slot_base_.back();
    }

    [[nodiscard]] std::int32_t max_experts() const noexcept { return max_experts_; }

    /** Bytes between two cache slots: the widest bank's per-expert size. */
    [[nodiscard]] std::int64_t pool_stride() const noexcept { return pool_stride_; }

    /** One staged layer of `layer`: its experts, then its layout's tail. */
    [[nodiscard]] std::int64_t staged_bytes(std::int32_t layer) const {
        return layout(layer).staged_layer_bytes(experts(layer));
    }

    [[nodiscard]] std::int64_t max_staged_bytes() const noexcept { return max_staged_bytes_; }

    [[nodiscard]] std::int64_t max_tail_bytes() const noexcept { return max_tail_bytes_; }

    /** Owner of one layer's expert, and the two halves of that decode. */
    [[nodiscard]] std::int32_t owner(std::int32_t layer, std::int32_t expert) const {
        return slot_base(layer) + expert;
    }

    [[nodiscard]] std::int32_t owner_layer(std::int32_t owner) const;

    [[nodiscard]] std::int32_t owner_expert(std::int32_t owner) const {
        return owner - slot_base(owner_layer(owner));
    }

private:
    std::vector<ExpertBankShape> banks_;
    std::vector<std::int32_t> slot_base_; // layers() + 1 prefix sums, the last one slot_entries()
    std::int32_t max_experts_      = 0;
    std::int64_t pool_stride_      = 0;
    std::int64_t max_staged_bytes_ = 0;
    std::int64_t max_tail_bytes_   = 0;
};

/** Device-resident mutable state of the expert cache; owned by one Program. */
struct ExpertCacheState {
    std::int32_t* slot_of          = nullptr; // [slot_entries], -1 when absent
    std::int32_t* owner            = nullptr; // [slots], an index into slot_of, or -1
    unsigned long long* stamp      = nullptr; // [slots], last use
    unsigned long long* clock      = nullptr; // [1]
    unsigned long long* statistics = nullptr; // [3]: hits, misses, ids outside the bank
    std::int32_t slots             = 0;
    std::int32_t layers            = 0;
    std::int32_t slot_entries      = 0; // experts over the pool's layers: the length of slot_of
    std::int64_t pool_stride = 0; // bytes between two slots; the widest bank's per-expert size
    std::byte* pool          = nullptr; // [slots, pool_stride]
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
 * misses. `ids` I32 `[K,T]` in; `slot_ids` I32 `[K,T]` out; `misses` I32 `[2*(K*T) + 1]` receives
 * the miss count followed by (slot, expert) pairs. K is the caller's own rank count -- the route
 * works per assignment and `moe_experts` is the call that fixes 10 -- and `bank` is the layer's
 * span in the pool (`ExpertPoolGeometry::span`), which is what places its experts and what an
 * `owner` records. The cache must hold at least K*T slots. Every slot touched by this call is
 * protected from eviction within it.
 */
void expert_cache_resolve(const Tensor& ids, ExpertBankSpan bank, const ExpertCacheState& cache,
                          Tensor& slot_ids, Tensor& misses, cudaStream_t stream);

/**
 * Copy every missing expert listed in `misses` from its bank into its slot. The bank planes must
 * be device-addressable (for a host bank, mapped pinned memory); the grid covers `max_misses`
 * entries and idles past the recorded count, so the call is graph-capturable. `bank` is the bank
 * the misses came from, so a pool whose layers differ in width fills each layer's slot from its own
 * layout.
 */
void expert_cache_fetch(const ExpertWeights& bank, const Tensor& misses, std::int32_t max_misses,
                        const ExpertCacheState& cache, cudaStream_t stream);

/**
 * Fill the experts of one layer that `resident` marks as cached into the staged bank `staged`, in
 * whole expert runs for the experts it leaves unmarked. `bank` is the layer's span in the pool, so
 * its `slot_of` entries, the residency snapshot's offsets and the per-expert count are all the
 * layer's own. `resident` is a device-readable (for example mapped pinned) snapshot of
 * `cache.slot_of`; the caller copies the unmarked experts itself. A marked expert comes from its
 * cache slot while the device state still holds it and otherwise from `bank`, so a stale snapshot
 * costs bandwidth, never correctness.
 */
void expert_cache_stage(const ExpertCacheState& cache, ExpertBankSpan bank,
                        const std::int32_t* resident, const ExpertWeights& weights,
                        const ExpertWeights& staged, cudaStream_t stream);

/**
 * Evict every expert held in slots `[first_slot, cache.slots)`: their `slot_of` entries become -1,
 * their owners -1 and their stamps 0, so the range may be overwritten. Other slots are unchanged.
 * An owner is an index into `slot_of`, so the entry to clear is the owner itself whatever the
 * pool's layers are.
 */
void expert_cache_reclaim(const ExpertCacheState& cache, std::int32_t first_slot,
                          cudaStream_t stream);

/**
 * Slot-pool addressing of a cache whose experts come from `bank` (divisors stay in the bank): the
 * bank's plane offsets inside a slot, at the pool's own stride. A layer narrower than the pool's
 * widest bank occupies a shorter run at the front of its slot, so this is the addressing every
 * layer of such a pool uses.
 */
[[nodiscard]] ExpertWeights expert_cache_weights(const ExpertCacheState& cache,
                                                 const ExpertWeights& bank);

/** Workspace of moe_experts for `tokens` columns over `slots` slots of experts in `layout`. */
[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t tokens, std::int32_t slots,
                                                      const ExpertLayout& layout);

/**
 * Most columns one moe_experts call over a pool of `slots` slots accepts: job building sorts a
 * call's assignments in shared memory, or counts per slot when the pool has at most 4096 slots.
 */
[[nodiscard]] std::int32_t moe_experts_max_columns(std::int32_t slots) noexcept;

struct MoeExpertsPending {
    const Tensor* misses;
    cudaEvent_t fetched;
};

/**
 * Offline-calibration tap. After each token chunk of moe_experts, `tap` receives the BF16 SwiGLU
 * intermediates `act` [640, 10*columns] (assignment a = column*10 + k is column a) and the chunk's
 * expert ids [10, columns]; both live in the call's workspace and are valid only inside the call.
 * Serving never passes a tap.
 */
using MoeActivationTap =
    std::function<void(const Tensor& act, const Tensor& expert_ids, cudaStream_t stream)>;

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

void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& slot_ids,
                 const Tensor& weights, const Tensor& shared_gate, const Tensor& shared,
                 const ExpertWeights& weights_source, std::int32_t slots, WorkspaceArena& workspace,
                 Tensor& y, cudaStream_t stream, const MoeExpertsPending* pending = nullptr,
                 const MoeActivationTap* tap = nullptr);

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
