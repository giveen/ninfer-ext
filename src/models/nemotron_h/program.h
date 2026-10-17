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
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "models/nemotron_h/attention_state.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/execution/draft.h"
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

// The target's own residual stream at the blocks a speculative draft taps: the one input a draft
// cannot compute for itself. A LiLiCorr draft's `fc` consumes exactly these blocks' outputs,
// concatenated in the artifact's recorded order, as one feature per position, so the draft's
// context is the target's activations rather than a second copy of its own.
//
// The composition publishes that tensor as `BlockStage::Block`, and a reported stage borrows the
// pass's own storage and is valid for the callback only -- so the Program copies it while the pass
// runs, on the pass's stream. `features` therefore holds the columns of *one* pass: a consumer that
// needs a sequence's worth takes them from each pass as it completes, which is also how a draft's
// ring is filled.
struct PassTaps {
    // The block ids, ascending, in the order the concatenation takes them.
    std::vector<std::uint32_t> layers;
    // BF16 `[layers.size() * hidden_size, tokens]`: one `[hidden_size, tokens]` block per tap, in
    // `layers` order, each laid out as the residual it copies is (the token axis strided, the
    // channels contiguous).
    Tensor features;
    // The columns the last pass wrote, and the absolute position of its first column.
    std::int32_t tokens         = 0;
    std::int32_t first_position = 0;

    [[nodiscard]] bool armed() const noexcept { return !layers.empty(); }
};

// One speculative round's outcome: what the drafter proposed, what the target accepted, and what
// each phase cost.
//
// `accepted` counts the draft slots the target's own greedy argmax agreed with, so the tokens a
// round commits are `accepted + 1`: the anchor the round consumed at position `F` plus the target's
// own next token, which becomes the next round's anchor. That is the reference's own accounting
// (`tools/verify/lilicorr_acceptance.py`), measured against this engine's target rather than a
// recorded stream.
struct DraftRoundReport {
    // The draft's tokens, one per mask slot (block_size - 1 of them), in slot order.
    std::vector<std::int32_t> proposed;
    // The tokens the round consumed from the target: the anchor followed by the accepted drafts.
    // Their positions are `[anchor_position, anchor_position + accepted]`.
    std::vector<std::int32_t> committed;
    std::int32_t accepted        = 0;
    std::int32_t anchor          = 0;
    std::int32_t anchor_position = 0;
    // The target's own greedy argmax at every row of the verify block, in row order: row `r` is the
    // token the target predicts for the position `anchor_position + r + 1`, which is the token a
    // decode would commit. Slots are accepted against it, and the argument of this vector at
    // `accepted` is `bonus`.
    std::vector<std::int32_t> verified;
    // The target's own argmax at the last accepted row: the token a greedy decode would commit
    // next. A draw from the committed state reproduces it unless the verify pass's width and the
    // commit pass's select different kernel routes near a tie, so a caller can compare the two to
    // see whether the round's own arithmetic agreed with itself.
    std::int32_t bonus = 0;
    // Milliseconds, each measured around one phase of the round on the caller's stream. The five
    // do not add up to the round only because argument checks and host bookkeeping sit outside
    // them; `draft_ms` is the drafter's own forward and `verify_ms` the target's block.
    double draft_ms   = 0.0;
    double verify_ms  = 0.0;
    double accept_ms  = 0.0;
    double restore_ms = 0.0;
    double commit_ms  = 0.0;
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

    // The same for an artifact that carries a LiLiCorr draft, plus the draft's own state: the
    // context ring, one forward's scratch, the Mamba snapshot a round rolls back through and the
    // verify block's logits. This is the figure a caller that will `engage_draft` must reserve --
    // the draft is not part of the load budget the runtime plans, because the runtime drives no
    // drafter for this family and never allocates one. Zero draft bytes when the config carries no
    // draft.
    [[nodiscard]] static std::size_t device_bytes(const Config& config,
                                                  const EngineOptions& options);
    [[nodiscard]] static std::size_t device_bytes(const Model& model, const EngineOptions& options);

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

    // Arms the tap capture: every pass from here on copies its tap blocks' residual outputs into
    // `taps()`, whose buffer is grown to the widest pass and never shrunk, so a wide prefill chunk
    // sizes it once. `layers` is ascending and unique; an empty span disarms.
    void capture_taps(std::span<const std::uint32_t> layers);

    // Disarms the capture and forgets the columns the last pass wrote. The buffer stays allocated,
    // because a caller that re-arms for every request should not pay for it twice.
    void stop_capturing_taps() noexcept;

    // What the last armed pass captured: empty layers, and `tokens == 0`, when nothing is armed or
    // no pass has run since.
    [[nodiscard]] const PassTaps& taps() const noexcept { return taps_; }

    // ---- The draft. ----
    // The draft's execution state, or null when none is engaged. The constructor does not allocate
    // it: this artifact carries a draft, but the runtime drives no drafter for this family, so a
    // Program that never drafts must not pay ~250 MB of ring, snapshot and verify logits for it.
    [[nodiscard]] const DraftContext* draft() const noexcept { return draft_ ? &*draft_ : nullptr; }

    // Allocates the draft's state from the model's spec and arms the tap capture with the draft's
    // own `target_layer_ids`, so the conditioning cannot be armed with the wrong blocks. Every
    // committed pass from here on conditions the draft's ring; `draft_round` proposes, verifies and
    // commits. Refuses a model with no draft and a second call is a no-op. The ring is written on
    // the construction path's default stream, so this synchronizes the device once.
    void engage_draft(DeviceExecutionView execution);

    // One proposal round, at the reference's accounting: the drafter proposes `block_size - 1`
    // tokens conditioned on the last committed positions; the target runs the whole block (the
    // already-drawn `anchor` plus the mask rows) as one verify pass; the longest prefix the
    // target's own argmax agrees with is accepted, and the round commits `accepted + 1` tokens --
    // the anchor and the accepted drafts -- leaving the target's state exactly there and its own
    // next token available through `sample`. The lane's cursor moves by `accepted + 1`.
    //
    // `anchor` is the token occupying the lane's current position: the id `sample` drew from the
    // lane's last pass. The round needs the capture armed with the draft's taps and that capture's
    // last column to be the frontier (which every committed pass leaves true). Synchronizes.
    // `observer` is the stage-diff hook, off by default: it is notified for every stage the draft's
    // forward writes under the reference's own stage names, which is what lets a measurement feed the
    // oracle the engine's exact inputs (`body.norm_out`, `body.anchor_hidden`, the candidate ids and
    // their log-probabilities) instead of comparing two forwards on separate inputs.
    [[nodiscard]] DraftRoundReport draft_round(std::int32_t lane, std::int32_t anchor,
                                               DeviceExecutionView execution,
                                               const DraftObserver* observer = nullptr,
                                               bool single_row_commit = false,
                                               std::int32_t slots = 0);

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

    // The target over one speculative block: `tokens` columns of every column's logits into
    // `draft_logits_`, and the state advanced by them in place. Unlike `run_pass` it neither moves
    // the lane's cursor nor captures taps -- the caller rolls the state back to whatever prefix it
    // accepts and runs the committed pass, which is the pass whose capture and cursor are real.
    void verify_pass(std::int32_t lane, const std::int32_t* ids, std::int32_t tokens,
                     DeviceExecutionView execution);
    // The round's commit without a pass: the accepted prefix's Mamba state, replayed from the
    // captured transition operands, and the attention frontier, the head's logits, the taps and the
    // lane's cursor moved to where the round committed. The caller has already restored the
    // snapshot and rewound attention to the pre-round frontier.
    void replay_commit(std::int32_t lane, std::int32_t accepted, DeviceExecutionView execution);

    // The one Mamba lane's image a round rolls back through: every layer's convolution history and
    // FP32 recurrent plane, packed in layer order. The attention K/V needs no image -- a causal
    // state hides and then overwrites the discarded slots (`AttentionState::rewind`).
    void snapshot_state(std::int32_t lane, cudaStream_t stream);
    void restore_state(std::int32_t lane, cudaStream_t stream);

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
    // The draft's conditioning capture, and the bytes it borrows: the buffer follows the widest
    // pass this Program has run with the capture armed.
    DeviceBuffer taps_backing_;
    PassTaps taps_;
    // The columns `taps_backing_` holds: the widest pass the capture has seen, so a capture grows
    // once and a later narrower pass re-views the same bytes.
    std::int32_t taps_capacity_ = 0;
    // The rows those bytes hold, which the armed layer count decides: a re-arm onto a wider set of
    // taps reallocates rather than re-viewing.
    std::int32_t taps_rows_ = 0;
    // The draft, when a caller engaged one. Declared after the state it conditions, so it is
    // destroyed first; `draft_rollback_` is one lane's Mamba image and `draft_logits_` is the
    // verify block's `[vocabulary, block_size]` logits, which no other pass writes.
    std::optional<DraftContext> draft_;
    DeviceBuffer draft_rollback_;
    // The Mamba state replay's operands and scratch: the conv input and time steps of the verify
    // pass, per Mamba block, and the buffers one layer's replayed convolution/scan writes. Planned
    // for the widest block a round can issue, and allocated only when a draft is engaged.
    // One captured verify pass per lane: the block's whole forward, the tap copies and the head
    // over its columns, launched as one graph. It reads the fixed id and logits buffers, so a
    // round only writes the ids before launching. Only when the banks are read where they live --
    // a staging pool makes host decisions a capture cannot replay.
    std::vector<DecodeGraphDefinition> verify_graphs_;
    std::vector<DecodeGraphExecutable> verify_execs_;
    MambaReplay replay_capture_;
    MambaReplayScratch replay_scratch_;
    DeviceBuffer replay_capture_backing_;
    DeviceBuffer replay_scratch_backing_;
    bool replay_ready_ = false;
    DeviceBuffer draft_logits_backing_;
    Tensor draft_logits_;
    // The verify block's greedy draws: one `ops::SamplingConfig` per column (all greedy, with no
    // occurrence-count array, so verifying never touches a request's counts), the block's absolute
    // positions and the drawn ids.
    DeviceBuffer draft_configs_backing_;
    DeviceBuffer draft_positions_backing_;
    DeviceBuffer draft_drawn_backing_;
    DeviceBuffer draft_sampling_workspace_backing_;
    std::optional<DeviceArena> draft_sampling_workspace_;
    std::vector<std::int32_t> draft_drawn_host_;
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
    // The widest block a round may issue (anchor plus mask rows), planned from the artifact's spec
    // and, for a plain DFlash, the options' draft length. Zero when no draft can run.
    std::int32_t draft_width_           = 0;
};

} // namespace ninfer::models::nemotron_h
