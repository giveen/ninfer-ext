#pragma once

// The mutable execution state of up to `lanes` independent Nemotron-H sequences, and the entry
// points the runtime drives a lane through.
//
// The family's state is exactly three classes, and this Program owns all of them:
//
//  * the Mamba-2 recurrent plane per Mamba layer per lane, in FP32, exactly as
//    `ops::mamba2_scan` declares it (`[128, value_dim, H, 1]`, key extent fastest), in
//    `LinearAttentionStatePool` -- the pool the plan records as reused unchanged;
//  * the Mamba-2 convolution history per Mamba layer per lane, the pool's conv plane, whose Op
//    state is BF16 `[C, 3]` oldest to newest;
//  * the attention K/V of the GQA blocks, in `AttentionState`: one dense BF16 plane per attention
//    layer per lane, exactly the operand `ops::sliding_causal_attention` takes, plus the lane's own
//    positions. That module owns the route as well as the storage (append, then attend with
//    `window == 0`, the Op's unbounded causal route), so this Program composes it rather than
//    describing a second K/V layout.
//
// Three facts decide the numbers this Program owns, and each is verified in place rather than
// assumed:
//
//  * The recurrent plane's key extent is `ops::mamba2_scan`'s own constant, 128, not the config's
//    `state_size`: the Op's state extent is a property of its kernel, and a narrower configured
//    state is carried by zero-extending the key axis (forward.h), which is exact. A state *wider*
//    than 128 has no route and is refused here rather than shrunk. The plane the pool plans is
//    `{key_head_dim = 128, value_head_dim = head_dim, value_heads = num_heads}`, which is the
//    shape the Op's state argument is declared with, in the Op's own axis order.
//  * The conv plane is `{conv_channels, conv_kernel - 1, lanes}` -- the history the depthwise
//    convolution consumes, `conv_channels()` derived by the config (the inner width plus one B and
//    one C vector per group).
//  * The attention state is the config's own geometry: `head_dim` wide, `num_key_value_heads`
//    heads, one plane per attention layer per lane, `capacity` tokens each, no padding and no
//    page rounding. The capacity a lane is sized for is one request window,
//    `EngineOptions::max_context`, as the sibling families' instances also resolve it.
//
// Execution: the landed forward is a *zero-state prefill* -- a Mamba mixer starts from a zero state
// and the pass keeps no state between calls (forward.h) -- so the tokens a lane consumes are not
// reflected in the state this Program owns. Every entry point that would compute therefore refuses
// (`kMissingStatePersistence`) instead of returning: a prefill or decode driven through the landed
// forward would produce the output of a zero-state model and quietly discard every token's
// contribution, which is a wrong model rather than a missing feature. The state the composition
// would have to write into is here, and the lane lifecycle (create, finish, abort) and the capacity
// reporting do work, because they touch only state this Program owns.

#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "models/nemotron_h/attention_state.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/forward.h"
#include "ninfer/types.h" // EngineOptions

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::nemotron_h {

// The key extent `ops::mamba2_scan`'s state plane is pinned to. A configured state narrower than
// this is carried by zero-extending the key axis (exact); a wider one is refused at construction.
inline constexpr std::int32_t kMambaScanStateExtent = 128;

// Tokens one pass of this family's forward covers when `EngineOptions::prefill_chunk` is empty. The
// routed experts of this model are `Residency::HostPinned`, and a chunk streams each host-resident
// expert layer once, so the documented default for such a model is the widest candidate (see
// `EngineOptions::prefill_chunk`). Nothing here allocates it: it sizes the reported workspace.
inline constexpr std::uint32_t kDefaultPrefillChunk = 4096;

// What the state this Program owns cannot do yet, and the reason every computing entry point names
// it. Public so a caller (the runtime, a test) identifies the gap without matching free-form text.
inline constexpr std::string_view kMissingStatePersistence =
    "the landed forward is the zero-state prefill of models/nemotron_h/forward.h: a Mamba mixer "
    "starts from a zero state and writes nothing back, and no attention K/V is written, so driving "
    "a "
    "lane through it would compute a wrong model rather than refuse a missing feature. The state "
    "itself is here -- the Mamba pool's slot and AttentionState, whose append/attend route runs "
    "the "
    "attention over a lane's own K/V -- and what is missing is the composition wiring it into "
    "forward.{h,cpp}: each Mamba layer's convolution history and fp32 recurrent plane saved into "
    "the lane's slot, and each attention layer's K/V appended before its query";

// The device bytes this Program's state costs, by class. The three class values are the planner's
// own region bytes -- the cost of the geometry, with no inter-region or inter-allocation padding --
// while `state_allocation_bytes` is what the two allocations actually hold.
struct StateMemorySummary {
    std::size_t mamba_recurrent_bytes = 0;
    std::size_t mamba_conv_bytes      = 0;
    std::size_t attention_kv_bytes    = 0;
    // The forward's transient arena for one pass of `kDefaultPrefillChunk` tokens (or the
    // configured chunk), from `model_workspace_bytes`. Not allocated here: the pass's caller owns
    // the arena (forward.h takes one). It is reported because a caller admitting a request has to
    // hold it beside the state.
    std::size_t workspace_bytes = 0;
    // The device allocations the three state classes live in: the Mamba pool's backing and the
    // attention state's, each rounded up to its own alignment.
    std::size_t state_allocation_bytes = 0;

    [[nodiscard]] std::size_t state_bytes() const noexcept {
        return mamba_recurrent_bytes + mamba_conv_bytes + attention_kv_bytes;
    }

    // What the Engine must have free for the state and one pass's workspace together.
    [[nodiscard]] std::size_t device_bytes() const noexcept {
        return state_allocation_bytes + workspace_bytes;
    }
};

class Program {
public:
    // `options.max_concurrency` fixes the lanes and `options.max_context` the tokens each lane's
    // K/V holds, both at construction: no request discovers half way through that the state it
    // needs was never allocated. `options.kv_cache` must be BF16, which is the dtype the attention
    // Op's operands are declared with; a quantized K/V would have to be decoded into BF16 scratch
    // per call, which is a route this family does not have yet. `device` supplies the stream the
    // initial empty state is written on; the Program is otherwise bound to streaming, not to a
    // stream, since every state-touching call takes a `DeviceExecutionView`.
    //
    // Release is the destructor rather than an entry point: every device allocation is an RAII
    // member and the pools own no storage of their own, so there is no partially built object to
    // unwind and nothing to hand back early. `finish` and `abort` release a lane.
    Program(const Model& model, const EngineOptions& options, DeviceContext& device);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // The state plus one pass's workspace for these options, so a caller can check the fit before
    // constructing one. Pure: it plans the same allocations the constructor makes and allocates
    // nothing.
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config,
                                                  const EngineOptions& options);

    // Sequences that can be admitted at once: one per lane, because a lane owns a state slot and
    // its own K/V planes.
    [[nodiscard]] std::int32_t admission_capacity() const noexcept { return lanes(); }

    [[nodiscard]] std::int32_t lanes() const noexcept { return lanes_; }

    // Tokens of one sequence one lane's K/V holds (its request window).
    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }

    // The state, by class, as the layouts planned it.
    [[nodiscard]] StateMemorySummary memory_summary() const noexcept;

    // How the routed banks were split between device-resident uploads and pinned host memory, and
    // the numbers that follow from it. Reported by the loader and carried here so a caller can see
    // the residency beside the state it shared the card with.
    [[nodiscard]] const ExpertResidencyPlan& expert_residency() const noexcept {
        return model_->expert_residency();
    }

    // The staged expert banks, or null when the cache policy asked for none. A forward this Program
    // drives stages its non-resident banks through it.
    [[nodiscard]] const RoutedExpertPool* expert_pool() const noexcept {
        return experts_ ? &*experts_ : nullptr;
    }

    // The model whose mixers this state belongs to.
    [[nodiscard]] const Model& model() const noexcept { return *model_; }

    // The Mamba state pool, or null when the config has no Mamba block. Non-owning; the pool
    // outlives every call and is bound to one lane per slot.
    [[nodiscard]] const LinearAttentionStatePool* mamba_state() const noexcept {
        return mamba_ ? &*mamba_ : nullptr;
    }

    [[nodiscard]] LinearAttentionStatePool* mamba_state() noexcept {
        return mamba_ ? &*mamba_ : nullptr;
    }

    // The attention K/V state, or null when the config has no attention block.
    [[nodiscard]] const AttentionState* attention_state() const noexcept {
        return attention_ ? &*attention_ : nullptr;
    }

    [[nodiscard]] AttentionState* attention_state() noexcept {
        return attention_ ? &*attention_ : nullptr;
    }

    // ---- Request lifecycle. These touch only state this Program owns. ----

    // Occupies `lane` for one request and empties its state on `execution.stream`: the Mamba
    // frontier is zeroed (the zero state the landed prefill forward starts from) and the lane's
    // attention positions become unwritten, so a lane reissued to a new sequence never inherits the
    // previous one's convolution history, recurrent plane or K/V. Refuses a lane that already holds
    // a request.
    void create(std::int32_t lane, DeviceExecutionView execution);

    // The request completed: the lane is free for the next one.
    void finish(std::int32_t lane);

    // The request was cancelled: the lane is free for the next one. Identical to `finish` here
    // because nothing of a request survives outside its lane state.
    void abort(std::int32_t lane);

    // ---- Execution. Every one of these refuses: see kMissingStatePersistence. ----

    // Consumes the tokens of a prefill pass and leaves the lane advanced.
    void prefill(std::int32_t lane, std::span<const std::int32_t> ids,
                 DeviceExecutionView execution);

    // Consumes one token for each of `lanes` (distinct, one id each) in one pass.
    void decode(std::span<const std::int32_t> lanes, std::span<const std::int32_t> ids,
                DeviceExecutionView execution);
    void decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution);

    // Keeps the first `kept` tokens of the lane's uncommitted round, exactly as the sibling
    // family's commit_round does for speculation.
    void commit(std::int32_t lane, std::int32_t kept, DeviceExecutionView execution);

    // Drops the lane's uncommitted round.
    void discard(std::int32_t lane);

private:
    // One lane's mutable bookkeeping: whether a request holds it. The frontier lives in the state
    // pool and in AttentionState, not here.
    struct Lane {
        bool in_use = false;
    };

    [[nodiscard]] Lane& lane_at(std::int32_t lane);
    [[nodiscard]] const Lane& lane_at(std::int32_t lane) const;

    // A lane holding a request. Every entry point that would consume tokens or commit them takes
    // its lane this way: a lane with no request has no sequence to write into, which is a caller
    // error rather than the missing state persistence.
    [[nodiscard]] Lane& occupied_lane(std::int32_t lane);

    // Frees the lane for the next request, or refuses when it was never in use.
    void release_lane(std::int32_t lane);

    // The Mamba pool's backing, then the pools that bind into it. `mamba_backing_` is declared
    // first so it outlives every non-owning pool bound to its storage.
    DeviceBuffer mamba_backing_;
    std::optional<LinearAttentionStatePool> mamba_;
    std::optional<AttentionState> attention_;
    // The Mamba layout the byte report is read from, kept because a summary must report what was
    // planned rather than re-derive it.
    LinearAttentionStatePoolLayout mamba_layout_{};
    std::vector<Lane> lanes_state_;
    // The staged expert banks and their split, when the cache policy affords them. Declared after
    // the state so a partially built Program destroys the pool before the state it was staged into.
    std::optional<RoutedExpertPool> experts_;
    const Model* model_                 = nullptr;
    std::size_t workspace_bytes_        = 0;
    std::size_t state_allocation_bytes_ = 0;
    std::int32_t capacity_              = 0;
    std::int32_t lanes_                 = 0;
};

} // namespace ninfer::models::nemotron_h
