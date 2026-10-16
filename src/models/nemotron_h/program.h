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
// Execution: a pass is one sequence's tokens over the composition, and its state is this Program's.
// `prefill` consumes a prompt and leaves the lane advanced, `decode` consumes one token per lane,
// and `sample` draws the next token from the logits the lane's last pass left -- so a lane's
// generation is a decode pass per token, each advancing the same state the prefill wrote. The
// composition itself is `models/nemotron_h/forward.h`: a state-driven pass appends every attention
// block's K/V before its query runs and leaves every Mamba block's convolution history and FP32
// recurrent plane advanced in place.
//
// The state lives in three places and this Program owns the binding between them: the Mamba pool's
// slot (one lane per slot), `AttentionState`'s per-layer planes, and the lane cursor
// (`Lane::position`, the tokens the lane has consumed, which is the absolute position of its next
// token). A pass is handed the three through `PassState`, and `forward_model` refuses a pass whose
// stated first position disagrees with the attention state's own frontier, so a cursor that drifts
// is an error rather than attention at a position no token occupied.

#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "models/nemotron_h/attention_state.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/forward.h"
#include "ninfer/ops/sampling.h"
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

// The device bytes this Program's state costs, by class. The three class values are the planner's
// own region bytes -- the cost of the geometry, with no inter-region or inter-allocation padding --
// while `state_allocation_bytes` is what the two allocations actually hold.
struct StateMemorySummary {
    std::size_t mamba_recurrent_bytes = 0;
    std::size_t mamba_conv_bytes      = 0;
    std::size_t attention_kv_bytes    = 0;
    // The forward's transient arena for one pass of `kDefaultPrefillChunk` tokens (or the
    // configured chunk), from `model_workspace_bytes`. It is allocated here now, because this
    // Program drives the passes: it is the arena every `prefill`/`decode` call runs against.
    std::size_t workspace_bytes = 0;
    // One pass's operands and output this Program owns: the `[chunk]` token ids it uploads and the
    // `[vocabulary, lanes]` logits the head writes, one column per lane.
    std::size_t pass_bytes = 0;
    // The sampler's own storage: one device `ops::SamplingConfig` per lane, its `[vocabulary]`
    // occurrence counts, the drawn/position vectors and the Op's workspace.
    std::size_t sampling_bytes = 0;
    // The device allocations the three state classes live in: the Mamba pool's backing and the
    // attention state's, each rounded up to its own alignment.
    std::size_t state_allocation_bytes = 0;

    [[nodiscard]] std::size_t state_bytes() const noexcept {
        return mamba_recurrent_bytes + mamba_conv_bytes + attention_kv_bytes;
    }

    // What the Engine must have free for the state, one pass's transient storage and the sampler
    // together.
    [[nodiscard]] std::size_t device_bytes() const noexcept {
        return state_allocation_bytes + workspace_bytes + pass_bytes + sampling_bytes;
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

    // The routed experts' own device bytes under these options' cache policy: the pool the
    // constructor allocates for them, and the elastic half of a load. It is included in
    // `device_bytes` and it is *inside* the budget `plan_load` spends, so a caller deriving that
    // budget subtracts this term from `device_bytes` -- reserving it in both places reserves it
    // twice and hands the resident banks one pool's worth less than the card can afford. Zero when
    // the policy asks for no pool at all.
    [[nodiscard]] static std::size_t expert_pool_device_bytes(const TextConfig& config,
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

    // ---- Request lifecycle. ----

    // Occupies `lane` for one request and empties its state on `execution.stream`: the Mamba
    // frontier is zeroed (the zero state a sequence's first pass starts from) and the lane's
    // attention positions become unwritten, so a lane reissued to a new sequence never inherits the
    // previous one's convolution history, recurrent plane or K/V. The lane's cursor returns to
    // zero. Refuses a lane that already holds a request.
    void create(std::int32_t lane, DeviceExecutionView execution);

    // Installs the sampler a lane draws with, and clears the occurrence counts its penalties read.
    // `create` does this implicitly with a greedy sampler; a request that samples installs its own
    // here, before its first pass. The `token_counts` the caller's struct names are replaced by the
    // lane's own count array, so two lanes' counts can never alias.
    void configure(std::int32_t lane, const ops::SamplingConfig& sampling,
                   DeviceExecutionView execution);

    // The request completed: the lane is free for the next one.
    void finish(std::int32_t lane);

    // The request was cancelled: the lane is free for the next one. Identical to `finish` here
    // because nothing of a request survives outside its lane state.
    void abort(std::int32_t lane);

    // ---- Execution. ----

    // Consumes the tokens of a prefill pass and leaves the lane advanced: every block's state is
    // written, the lane's cursor moves past the tokens, and the logits of the last token are left
    // for `sample`. The tokens are run in passes of at most the configured prefill chunk, so a
    // prompt wider than one pass is the same computation in more calls. Synchronizes.
    void prefill(std::int32_t lane, std::span<const std::int32_t> ids,
                 DeviceExecutionView execution);

    // Consumes one token for each of `lanes` (distinct, one id each) in one pass, leaving each
    // lane's logits for `sample`. Synchronizes.
    void decode(std::span<const std::int32_t> lanes, std::span<const std::int32_t> ids,
                DeviceExecutionView execution);
    void decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution);

    // Draws the token after the one each of `lanes` consumed last, from the logits its last pass
    // left, with the lane's sampler: the counter-based RNG is keyed by the lane's position, so a
    // draw is reproducible from the sequence and the seed alone. Synchronizes.
    [[nodiscard]] std::vector<std::int32_t> sample(std::span<const std::int32_t> lanes,
                                                   DeviceExecutionView execution);
    [[nodiscard]] std::int32_t sample(std::int32_t lane, DeviceExecutionView execution);

    // Tokens the lane has consumed, which is also the absolute position its next token occupies.
    [[nodiscard]] std::int32_t position(std::int32_t lane) const;

    // The BF16 [vocabulary, 1] logits the lane's last pass left, which `sample` draws from.
    [[nodiscard]] Tensor logits(std::int32_t lane) const;

private:
    // One lane's mutable bookkeeping: whether a request holds it, where its logits column is, and
    // the tokens it has consumed. The frontier itself lives in the state pool and in
    // AttentionState, not here.
    struct Lane {
        bool in_use           = false;
        std::int32_t position = 0;
    };

    [[nodiscard]] Lane& lane_at(std::int32_t lane);
    [[nodiscard]] const Lane& lane_at(std::int32_t lane) const;

    // A lane holding a request. Every entry point that would consume tokens or commit them takes
    // its lane this way: a lane with no request has no sequence to write into, which is a caller
    // error rather than the missing state persistence.
    [[nodiscard]] Lane& occupied_lane(std::int32_t lane);

    // Frees the lane for the next request, or refuses when it was never in use.
    void release_lane(std::int32_t lane);

    // The lane's state, as the pass that consumes tokens from the lane's own cursor takes it: one
    // entry per block, the Mamba pool's slot for each Mamba layer and `AttentionState`'s binding
    // for each attention layer. The vector is the caller's scratch, reused so a pass binds no
    // storage.
    void bind_state(std::int32_t lane, std::vector<BlockState>& blocks);

    // The Op's key-split workspace a stateful attention call may use, sized for a one-token step.
    // A wider pass that would prefer a larger one takes the Op's one-pass route instead, which
    // computes the same result; empty when the Op reports it needs none.
    [[nodiscard]] Tensor attention_workspace() const { return attention_workspace_; }

    // The lane's logits column: the head's `[vocabulary, 1]` output for the token the lane's last
    // pass consumed. One column per lane, so a batched round leaves every row's draw available.
    [[nodiscard]] Tensor logits_column(std::int32_t lane) const;

    // Runs `tokens` tokens of `lane` as one pass from the lane's current cursor, uploading `ids`.
    // Leaves the logits of the pass's last token in the lane's column.
    void run_pass(std::int32_t lane, const std::int32_t* ids, std::int32_t tokens,
                  DeviceExecutionView execution);

    // The Mamba pool's backing, then the pools that bind into it. `mamba_backing_` is declared
    // first so it outlives every non-owning pool bound to its storage.
    DeviceBuffer mamba_backing_;
    std::optional<LinearAttentionStatePool> mamba_;
    std::optional<AttentionState> attention_;
    // The Mamba layout the byte report is read from, kept because a summary must report what was
    // planned rather than re-derive it.
    LinearAttentionStatePoolLayout mamba_layout_{};
    std::vector<Lane> lanes_state_;
    // The pass's block bindings, reused so a pass allocates none: the vector is written by
    // `bind_state` and read through `PassState`, and the passes of one Program are sequential.
    std::vector<BlockState> pass_blocks_;
    // The staged expert banks and their split, when the cache policy affords them. Declared after
    // the state so a partially built Program destroys the pool before the state it was staged into.
    std::optional<RoutedExpertPool> experts_;

    // The pass's own storage. `arena_` is the transient workspace every pass runs against and
    // `ids_`/`logits_` are what a pass reads and writes: a lane's logits are its own column, so one
    // batched decode leaves every row's draw available to `sample`.
    DeviceArena arena_;
    DeviceBuffer ids_backing_;
    DeviceBuffer logits_backing_;
    DeviceBuffer attention_workspace_backing_;
    Tensor ids_;
    Tensor logits_;
    Tensor attention_workspace_;
    // The sampler: one device `ops::SamplingConfig` per lane (with the lane's own occurrence
    // counts beside it), a drawn-token and a position vector for one call, and the Op's workspace.
    DeviceBuffer sampling_backing_;
    DeviceBuffer counts_backing_;
    DeviceBuffer drawn_backing_;
    DeviceBuffer positions_backing_;
    DeviceArena sampling_workspace_;
    std::vector<std::int32_t> sampled_host_;
    const Model* model_                 = nullptr;
    std::size_t workspace_bytes_        = 0;
    std::size_t pass_bytes_             = 0;
    std::size_t sampling_bytes_         = 0;
    std::size_t state_allocation_bytes_ = 0;
    std::int32_t capacity_              = 0;
    std::int32_t lanes_                 = 0;
    std::int32_t chunk_                 = 0;
};

} // namespace ninfer::models::nemotron_h
