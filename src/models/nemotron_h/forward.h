#pragma once

// One Nemotron-H decoder block, composed from the Ops that already exist, and the pass over them.
//
// The block is the reference's own, `h = h + mixer(norm(h))`, where the norm is a *plain* RMSNorm
// over the whole row (`ops::rmsnorm`) -- not the gated grouped norm inside a Mamba mixer -- and the
// mixer is whatever the config's block list says:
//
//  * Mamba-2: `ops::linear` for the projection and the output projection, `ops::causal_conv1d_silu`
//    for the depthwise convolution, `ops::mamba2_scan` for the selective scan, and
//    `ops::gated_group_rmsnorm` for the output norm. The projection splits into z, x/B/C and dt;
//    the convolution consumes x, B and C together and its activation applies to all three, because
//    the split happens *after* it; the scan adds `D*x` to its readout; the norm gates first and
//    normalises within groups.
//  * LatentMoE: `ops::nemotron_h_moe_router`, `ops::nemotron_h_moe_experts` and two `ops::linear`
//    latent projections, with `ops::relu2` in both the bank and the shared expert. The router runs
//    on the norm's output, the projections bracket the bank, and the shared expert is a relu2 MLP
//    on the *mixer's input* rather than in the latent space, added to the routed result.
//  * GQA attention: `ops::linear` projections and `ops::sliding_causal_attention` as plain causal
//    grouped attention -- no bias, no sinks, no window, and no rotation. Neither the checkpoint's
//    own attention nor the installed reference calls `apply_rotary_pos_emb` anywhere (the
//    decorator is inert and no `position_embeddings` argument reaches the body), and the config has
//    no `rope_theta` at all, so this mixer has no rotation to compose and none is applied.
//
// Scope: this is the *whole* pass -- prefill and decode are the same composition, differing only in
// how many tokens it is handed and in where its state is. A pass is one sequence's tokens over the
// blocks in order, and the state it advances is the engine's: a Mamba mixer's convolution history
// and FP32 recurrent plane and each attention block's K/V, all of them the caller's storage
// (`BlockState`). A pass driven with no state is the same computation from a zero state, which is
// what a prompt's first prefill is, so the two are one function with a branch rather than two
// functions that can drift.
//
// Two op domains the composition answers for, established by reading the Ops rather than this
// family's checkpoint:
//
//  * `ops::mamba2_scan`'s key (state) extent is a constant of its kernel -- 128, the checkpoint's
//    own `ssm_state_size` -- and the Op takes *no scale argument*, because its readout is
//    deliberately unscaled (a factor of `1/sqrt(128)` there would be a silent correctness bug). A
//    config with a narrower state is therefore carried by zero-extending the key axis of B, C and
//    the state, which is exact rather than approximate: a zero key contributes `dt*x*0` to the
//    update and `0*0` to the readout, and `exp(dt*A)*0` stays exactly zero, so the tail never
//    moves. It needs no compensation precisely because there is no scale to compensate. A state
//    wider than 128 has no route and is refused.
//  * `ops::sliding_causal_attention`'s head widths are a finite set -- 32, 64, 128 and 256 -- that
//    the real checkpoint's own 128-wide head sits in, and it takes its `scale` as an argument
//    instead of deriving one from its operand's width. A head the Op does not admit is therefore
//    carried at the narrowest width it does: each head's real dims stay contiguous at its carried
//    head's start with zeros elsewhere, which is exact -- a zero query or key dim adds nothing to a
//    score, and a zero value dim contributes `probability * 0` to every output -- and the Op is
//    given this config's own `1/sqrt(head_dim)`, so the padded dims cannot move the result. There
//    is no rotation carry: this mixer rotates nothing, so the only thing the padding preserves is a
//    width the Op's own domain forces and not a pair structure.
//
// `Model` binds the parameters the composition reads; `forward_block` refuses a layer whose bound
// set does not match its config's kind, and each mixer refuses a parameter whose shape disagrees
// with the config's own values before it hands it to an Op.

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/nemotron_h/attention_state.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/experts.h"
#include "models/nemotron_h/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::nemotron_h {

// One bound parameter: the name the checkpoint uses (for messages and for a verification harness)
// and the resident view the Ops receive. `weight` is engaged for a matrix a projection or a bank Op
// consumes, and stays unset for a norm gain, a bias or an FP32 per-head vector.
struct BoundWeight {
    std::string name;
    Tensor tensor;
    Weight weight;
};

class LoadPlan;

// The parameters a model's config names, bound by id. A model with no artifact loader yet is built
// by binding each parameter by hand, which is also what a verification harness does from a
// reference dump; the loader binds the same ids from the artifact's materialized bytes.
//
// The bound views borrow the model's backing store, so a Model must outlive every view taken from
// it, and `backing_` is declared first so it is destroyed last.
class Model {
public:
    // The hand-bound model: no backing store, so every tensor and weight must be the caller's. Its
    // routed banks carry no residency split -- a hand-bound model binds each bank where the caller
    // put it.
    Model(TextConfig config, ModelWeights weights);

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    // How the loader split the routed banks between device-resident uploads and pinned host memory,
    // which is what the pool stages the non-resident ones against. A hand-bound model reports an
    // empty plan: no bank of it was uploaded by us.
    [[nodiscard]] const ExpertResidencyPlan& expert_residency() const noexcept {
        return residency_;
    }

    // The speculative draft this artifact carries, when it carries one. It travels with the model
    // so the Program can size its buffers and drive it; absent means the artifact has no draft.
    [[nodiscard]] const std::optional<draft::Spec>& draft() const noexcept { return draft_; }

    // Binds one parameter. `weight` is engaged for a matrix the Ops consume; `tensor` is the
    // resident view either way, and a matrix's `tensor` is its logical `[N,K]` row-major view. A
    // quantized matrix has its planes in `weight` and no single-typed tensor, so its `tensor` stays
    // unset (`data == nullptr`) rather than being misrepresented as a payload it is not.
    void bind(WeightId id, std::string name, Tensor tensor, Weight weight = {});

    // The bound parameter. Throws std::invalid_argument when the id was never bound, which is the
    // one failure a config-shaped call must not paper over: a missing parameter is not a default.
    [[nodiscard]] const BoundWeight& bound(WeightId id) const;

    [[nodiscard]] Tensor tensor(WeightId id) const { return bound(id).tensor; }

    [[nodiscard]] const Weight& matrix(WeightId id) const;

    // What the model's own bytes cost, which is where the routed banks' pinned host residency shows
    // up: `pinned_host_bytes` is 37.21 GB of NVFP4 experts that no 32 GB card holds.
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    std::optional<draft::Spec> draft_;
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    // The loader's constructor: the model owns the bytes every bound view borrows, and the split
    // its banks were bound with.
    Model(TextConfig config, ModelWeights weights, artifact::MaterializedArtifact backing,
          ExpertResidencyPlan residency);

    // Destroyed last: every BoundWeight view borrows these bytes.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    ModelWeights weights_;
    ExpertResidencyPlan residency_{};
    std::vector<BoundWeight> parameters_;
    std::vector<bool> bound_;
};

// Uploads the plan's selected bytes and binds every parameter the plan declared, which is why it is
// defined here rather than in the loader: it constructs `Model`, and the loading library links no
// part of the execution side. The plan is consumed; its Reader must outlive this call, and the
// returned model owns the materialized bytes rather than the artifact, so it outlives the Reader.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

// The stages a block produces, reported after each one is written. A stage is named by identity
// rather than by the reference's dotted path, so a consumer maps it to whatever it compares against
// and the composition carries no reference naming. `AttentionQuery`, `AttentionKey` and
// `AttentionValue` are the projections' own outputs, which is what the reference exposes -- before
// any head-width padding; `AttentionProjected` is the attention output before the output
// projection.
enum class BlockStage : std::uint8_t {
    BlockNorm,
    MambaInProj,
    MambaConv,
    MambaNorm,
    MambaOutProj,
    MambaMixer,
    MoeRouterInput,
    MoeRouterLogits,
    MoeRouterIds,
    MoeRouterWeights,
    MoeLatentIn,
    MoeExperts,
    MoeLatentOut,
    MoeShared,
    MoeMixer,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionProjected,
    AttentionOutProj,
    AttentionMixer,
    Block,
    Embedding,
    FinalNorm,
    Logits,
};

// Notified after every stage a call writes, with the block's index and the stage's own storage.
// `tensor` borrows the caller's arena or the composition's workspace and is valid for the duration
// of the callback only; a consumer that needs it longer copies it. A null observer costs nothing.
class BlockObserver {
public:
    virtual ~BlockObserver() = default;
    virtual void on_stage(std::size_t layer, BlockStage stage, const Tensor& tensor) const = 0;
};

// One attention block's state, engaged by a state-driven pass. It is the caller's storage -- the
// family's `AttentionState` -- and this is the binding of one lane's one attention layer into it,
// so the pass appends this block's K/V to it and attends over it without knowing how the state is
// laid out or indexed.
struct AttentionPassState {
    // Null means this block runs without a state, which is a zero-state pass's attention: it
    // attends over its own tokens only and appends nothing.
    AttentionState* state = nullptr;
    std::int32_t lane     = 0;
    // The block's own index among the state's attention layers -- not its block index, which a
    // config whose block kinds interleave (this family's do) maps to another layer.
    std::uint32_t layer = 0;
    // The Op's key-split workspace, or empty for its one-pass route, which computes the same
    // result: `AttentionState::workspace_bytes` sizes it for a pass's token count, and a pass
    // handed less than it would prefer takes the one-pass route instead.
    Tensor workspace;
};

// One block's mutable state, bound by the caller for one lane and one layer. A block engages the
// members its kind has; every tensor is the caller's storage and the pass reads and writes it *in
// place*, so the frontier a pass leaves behind is the state the next pass starts from. A null
// `PassState` is a zero-state pass, and then none of this exists: a Mamba mixer starts from a zero
// history and a zero plane, and an attention block attends over its own tokens.
struct BlockState {
    // Mamba layers: the convolution history, BF16 [conv_channels, conv_kernel - 1], oldest tap
    // first, in `ops::causal_conv1d_silu`'s own state order (tap outermost, channel innermost --
    // which is *not* the reference's own `[conv_dim, kernel]` cached window; the two are a
    // transpose); and the selective scan's FP32 plane [128, value_head_dim, value_heads] in
    // `ops::mamba2_scan`'s own state order (key extent fastest, the key extent being the scan
    // kernel's own constant 128 rather than the config's `state_size`).
    Tensor conv;
    Tensor recurrent;
    // Attention layers: the lane's K/V state for this block's own layer.
    AttentionPassState attention{};
};

// The state one pass advances: one entry per block, in the config's block order. The entries are
// mutable -- a pass reads and writes the state it is handed -- and the span is the caller's.
struct PassState {
    // Empty is a zero-state pass, bit for bit: nothing is read and nothing is written back.
    std::span<BlockState> per_layer{};
    // The absolute position of the pass's first token -- the tokens this lane consumed before it,
    // which is the reference's `past_seen_tokens`. The attention state's own frontier is where the
    // positions actually come from (so the two cannot disagree about what a key's position is);
    // this is the pass's statement of it, and a state-driven pass refuses one that disagrees with
    // the state.
    std::int32_t first_position = 0;
};

// One pass's Mamba state-transition operands, kept so a speculative round can advance its accepted
// prefix without running the model again.
//
// The state transition is driven by two vectors only: the depthwise convolution's input and the
// scan's time steps. Everything else the scan reads -- the convolution's output and the B/C/x the
// scan regroups from it -- is a function of the convolution's input, so a replay recomputes them
// exactly as the pass did and lands on the same state. The entries are per *block* (a block with no
// Mamba mixer leaves its pair empty); `columns` is the width the pass ran, and a replay may consume
// a prefix of it.
struct MambaReplay {
    std::vector<Tensor> conv_in; // [conv_channels, width] per block, Mamba blocks only
    std::vector<Tensor> dt;      // [heads, width] per block
    std::int32_t columns = 0;
};

// The replay's own scratch: the convolution's output and the three operands the scan regroups from
// it, sized for the widest block a round can commit. One set serves every layer, because the replay
// walks the layers one at a time.
struct MambaReplayScratch {
    Tensor conv_out; // [conv_channels, width]
    Tensor scan_x;   // [inner, width]
    Tensor scan_b;   // [128 * groups, width]
    Tensor scan_c;   // [128 * groups, width]
    Tensor scan_out; // [inner, width]: the scan's own readout, written and discarded
};

// Advances `per_layer`'s Mamba state by `columns` of a captured pass: the convolution and the scan,
// nothing else. The caller restores the pre-pass state first (the round's snapshot) and hands the
// pass's own block bindings, so the result is the state the pass itself reached at that column.
void mamba_replay_apply(const Model& model, std::span<BlockState> per_layer,
                        const MambaReplay& capture, MambaReplayScratch& scratch,
                        std::int32_t columns, DeviceExecutionView execution);

// Which columns of a pass the head runs over. A generation step draws the token after the pass's
// last position, and the head costs `vocabulary` outputs per column, so a caller that only draws
// asks for the logits of one column and sizes its buffer for one.
enum class HeadColumns : std::uint8_t {
    Every,    // logits [vocabulary, tokens]: every position of the pass
    LastOnly, // logits [vocabulary, 1]: the pass's last position only
};

// Transient capacity one block needs for `tokens` tokens of one kind, and the whole pass needs for
// the widest block plus the head. Both are the exact peak of the real allocation sequence, taken
// from the same planner the pass runs against a dry-run layout.
[[nodiscard]] std::size_t block_workspace_bytes(const TextConfig& config, std::int32_t tokens);
[[nodiscard]] std::size_t model_workspace_bytes(const TextConfig& config, std::int32_t tokens);

/**
 * Runs one decoder block for `tokens` tokens at once, advancing this block's state.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 `[hidden_size, tokens]` and must not overlap;
 * the block reads the residual stream and writes its updated self. `arena` supplies the block's
 * transient storage and must hold `block_workspace_bytes` for this config and `tokens`.
 *
 * `state` is this layer's entry of the pass's state, or null for a zero-state block. An attention
 * block appends its own K/V to the state *before* its query runs, which is the reference's order,
 * and does not advance the state's frontier -- the pass's caller does that once, after every
 * attention block has appended (`AttentionState::advance`).
 *
 * `expert_pool` places the routed banks: null reads each MoE block's bank where it is bound (over
 * UVA when it is host-pinned, in place when it is device-resident), and a pool stages the banks
 * that are not resident into its device buffers instead. See `RoutedExpertPool`.
 */
void forward_block(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   BlockState* state = nullptr, const BlockObserver* observer = nullptr,
                   RoutedExpertPool* experts = nullptr, MambaReplay* replay = nullptr);

/**
 * Runs a whole text pass: the token embedding, every block in order, the final norm and the head.
 *
 * `token_ids` is contiguous I32 `[tokens]` and every id must be in `[0, vocabulary)`. `logits` is
 * contiguous BF16, `[vocabulary_size, tokens]` for `HeadColumns::Every` and `[vocabulary_size, 1]`
 * for `HeadColumns::LastOnly`, and must not overlap the arena. `arena` must hold
 * `model_workspace_bytes` for this config and `tokens`, which is one block's peak plus the head's.
 *
 * `state` is the lane's state -- one entry per block, the tokens starting at
 * `state->first_position` -- or null for a zero-state pass, whose result is the same computation
 * from a zero state. A state-driven pass leaves every Mamba block's history and plane advanced by
 * its tokens in place and every attention block's K/V appended; it does *not* advance the attention
 * state's own frontier, so a caller that drives one pass after another calls
 * `AttentionState::advance(lane)` once between them.
 *
 * `expert_pool` is as `forward_block`'s.
 */
void forward_model(const Model& model, const Tensor& token_ids, DeviceArena& arena, Tensor& logits,
                   DeviceExecutionView execution, const PassState* state = nullptr,
                   HeadColumns columns           = HeadColumns::Every,
                   const BlockObserver* observer = nullptr,
                   RoutedExpertPool* expert_pool = nullptr, MambaReplay* replay = nullptr);

} // namespace ninfer::models::nemotron_h
