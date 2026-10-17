#pragma once

// The LiLiCorr speculative draft's execution state and its forward: the body that conditions on the
// target's residual taps and the correlation head that turns the draft's hidden state into a block
// of token ids.
//
// The draft is the companion of this tower, not a second model: its input is the target's own
// residual stream at six tapped blocks (`Program::capture_taps`), its context K/V is that same
// fused feature projected through each draft layer's own `key`/`value` weights, and its candidates
// are the target's own LM head applied to the draft's hidden state. Nothing of the draft is
// quantized: the artifact carries its 139 parameters BF16 as the checkpoint stores them, so every
// linear below is a BF16 route and the only rounding is the ops' own BF16 output boundary.
//
// Specification: `docs/maintainer/lilicorr-draft.md` (the "Mathematics", "Storage and state" and
// "Round" sections). The oracle is `tools/verify/lilicorr_reference.py`, whose stage names this
// module reports verbatim -- that is what makes a stage-by-stage diff mechanical, and it is the
// instrument the plan's own post-mortem says is the only one that catches a missing operation.
//
// Three facts shape the implementation, and each is a decision rather than a default:
//
//  * **The context is a ring of the last `window` positions, keyed by absolute position.** Each
//    draft layer's K/V for a context position is a function of the fused target feature at that
//    position, never of the block's own rows, so it is materialized once when the position is
//    committed and read afterwards. The ring is filled in *write* order, and a parallel I32 plane
//    records the absolute position each slot holds; the attention Op compares positions, so slot
//    order does not matter and nothing has to be copied into a linear window. Slots no token has
//    written carry a position far enough in the past to be invisible to every query.
//  * **The block's attention is the tower's Op with two changes that are operands, not kernels.** A
//    per-query-head sink (`attention_sink_bias`) and a symmetric window, the latter expressed
//    through `position_q_high = position_q + window - 1` -- the Op's own bound is one-sided, and
//    opening the upper edge to the window's far side is exactly the reference's
//    `abs(p_key - p_query) < window`. The block's own rows enter as the Op's `k_extra`/`v_extra`,
//    which is the operand pair "a pass's own keys beside a ring that does not hold them yet".
//  * **The head's index arithmetic runs on the host, its algebra on the device.** Selecting the
//    top-8 of a 131072-row vocabulary, the walk over 7 slots, the `[56,56,8]` lattice bias and the
//    `cat(node, anchor, node*anchor)` gather are all `[56, 1024]`-scale or smaller, and none of
//    them has an Op: the only top-k in the tree is pinned to the Qwen head geometry, and the only
//    attention with an additive per-head bias does not exist. So the head runs as device linears
//    and norms with host-side selection between them, and the two crossings a slot costs are a
//    1.8 MiB logits copy and a 0.9 MiB round trip. That is a measured 3% of a verify-shaped block,
//    which is what the paper allows the head.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/draft/draft_spec.h"
#include "models/nemotron_h/forward.h"

#include <memory>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::nemotron_h {

// Notified once per stage the forward writes, with the reference's own stage name and the stage's
// storage. A stage tensor borrows the draft's ring or its forward arena and is valid for the
// duration of the callback only; a consumer that needs it longer copies it. A null observer costs
// nothing.
//
// Layouts are this engine's everywhere -- dimension zero is the feature axis and the token axis is
// the last one -- so a stage that the reference stores as `[B, T, H]` arrives as `[H, T]`, one as
// `[B, heads, T, D]` arrives as `[D, heads, T]`, and a `[B, T, 2, taps, G]` delta arrives as
// `[2*taps*G, T]` in the projection's own row order. The comparison transposes; the values are the
// same elements.
class DraftObserver {
public:
    virtual ~DraftObserver()                                                  = default;
    virtual void on_stage(std::string_view stage, const Tensor& tensor) const = 0;
};

// One proposal round's inputs, in the reference's own terms.
//
// `taps` is the target's captured residual at the draft's six blocks for the positions the context
// covers: BF16 `[taps * hidden, columns]`, the blocks concatenated along the feature axis in
// ascending block order and the columns ascending in absolute position -- exactly the layout
// `PassTaps::features` holds, so a caller that just ran a pass hands it over as it is.
// `block_ids` is the query block, `1 + slots` ids on the host: the already-committed anchor token
// followed by the mask token, and block row `i` predicts the token at its own position.
struct DraftRound {
    Tensor taps;
    // Absolute position of the context's first column; the columns are contiguous, so the last one
    // is the position before `first_block_position`.
    std::int32_t first_context_position = 0;
    // Absolute position of block row zero. The block's rows are contiguous from here.
    std::int32_t first_block_position = 0;
    // Scored slots, which are the mask rows after the anchor: 1 .. block_size - 1.
    std::int32_t slots            = 0;
    const std::int32_t* block_ids = nullptr;
};

// What the draft's state costs, by class. `ring_bytes` is the per-lane cyclic context K/V (five
// layers of BF16 K and V at the draft's two KV heads, `window + block_size` positions deep) and
// `scratch_bytes` is one forward's transient arena, which is planned for the widest block the spec
// admits.
struct DraftMemory {
    std::size_t ring_bytes    = 0;
    std::size_t scratch_bytes = 0;
    // The fixed anchor plane the context writes in place (one `hidden`-wide column).
    std::size_t anchor_bytes = 0;
    // A plain DFlash's proposal: the tower's `[vocabulary, block]` logits, the greedy sampler's
    // per-column operands and its workspace. Zero for a LiLiCorr draft, whose head runs on the host
    // from the same logits it allocates in its own arena.
    std::size_t proposal_bytes = 0;
    std::size_t lanes          = 0;

    [[nodiscard]] std::size_t device_bytes() const noexcept {
        // Two arenas: the shared one (context phase) and the body's own rewound arena, whose
        // allocations must sit at the same offsets every round for the captured body.
        return ring_bytes + 2 * scratch_bytes + anchor_bytes + proposal_bytes;
    }
};

// The draft's mutable state for one Program: the context ring per lane, the weights' geometry, the
// plan-time constants read once from the artifact, and one forward's transient arena.
//
// The state is planned from the spec rather than grown per call, because everything it holds is a
// function of the spec: the ring is `window + block_size` positions deep, and the arena is sized
// for the widest context (`window` columns) and the widest block (`block_size` rows) the spec
// admits. A round that asks for more than that is refused rather than grown into, since a context
// wider than the sliding window would attend over keys the reference never sees.
class DraftContext {
public:
    // `block_width` is the widest block (anchor plus mask rows) a round may issue and
    // `context_window` the tokens of context the ring must hold. A LiLiCorr draft stores both and
    // refuses a caller that overrides them, so its callers pass zero; a plain DFlash stores neither
    // -- its block length is the round's choice and its context is the request's whole prefix -- so
    // its callers must pass them, and a zero for either is refused rather than defaulted to the
    // checkpoint's positional capacity, which would plan a ring nobody can afford.
    DraftContext(const Model& model, const models::draft::Spec& spec, std::int32_t lanes,
                 std::int32_t block_width = 0, std::int32_t context_window = 0);
    ~DraftContext();
    DraftContext(const DraftContext&)            = delete;
    DraftContext& operator=(const DraftContext&) = delete;
    DraftContext(DraftContext&&)                 = delete;
    DraftContext& operator=(DraftContext&&)      = delete;

    // The state these arguments plan, so a caller can check the fit before constructing one. Pure.
    [[nodiscard]] static DraftMemory plan(const Model& model, const models::draft::Spec& spec,
                                          std::int32_t lanes, std::int32_t block_width = 0,
                                          std::int32_t context_window = 0);

    // The same from the two values it actually depends on -- the tower's config and the draft's
    // spec, never a parameter -- so a loader deciding a residency budget can size the draft before
    // any model exists.
    [[nodiscard]] static DraftMemory plan(const TextConfig& config,
                                          const models::draft::Spec& spec, std::int32_t lanes,
                                          std::int32_t block_width = 0,
                                          std::int32_t context_window = 0);

    [[nodiscard]] const DraftMemory& memory() const noexcept { return memory_; }

    [[nodiscard]] const models::draft::Spec& spec() const noexcept { return spec_; }

    [[nodiscard]] std::int32_t lanes() const noexcept { return lanes_; }

    // Positions the context ring holds per lane.
    [[nodiscard]] std::int32_t context_capacity() const noexcept { return context_capacity_; }

    // The token a proposal's mask rows carry.
    [[nodiscard]] std::int32_t mask_token() const noexcept {
        return static_cast<std::int32_t>(spec_.mask_token_id);
    }

    // A new sequence: the lane's ring becomes empty, so a reissued lane never conditions on the
    // previous sequence's target states. Positions no token has written are marked far enough in
    // the past that no query row sees them.
    void reset(std::int32_t lane, DeviceExecutionView execution);

    // One proposal: appends the round's new context positions to the lane's ring, runs the body
    // over the block's rows, applies the head and walks the 7 slots greedily. Returns one token id
    // per slot, in slot order. Synchronizes.
    [[nodiscard]] std::vector<std::int32_t> propose(std::int32_t lane, const DraftRound& round,
                                                    DeviceExecutionView execution,
                                                    const DraftObserver* observer = nullptr);

    // Conditions the lane on a committed pass's residuals: every column of `taps` becomes context,
    // laid at the absolute positions it was captured from, starting at `first_position`. A
    // committed target pass calls this as it completes, which is what lets the ring hold a context
    // assembled from more than one pass -- a prompt wider than the pass its capture came from.
    // `propose` conditions from its own taps too, so a caller that only ever drafts still fills the
    // ring; calling both for the same columns materializes them once. Only the tail the ring can
    // hold is materialized, since a position older than the window is invisible to every query.
    // Does not synchronize.
    void condition(std::int32_t lane, const Tensor& taps, std::int32_t first_position,
                   DeviceExecutionView execution);

    // The last round's walk, so a caller can read it after the fact.
    [[nodiscard]] const std::vector<std::int32_t>& walk(std::int32_t lane) const;

    // The last round's per-slot candidate ids and scores, `[slots][top_k]` and
    // `[slots][top_k][top_k]`, for a caller that wants the drafter's confidence rather than its
    // proposal.
    [[nodiscard]] const std::vector<std::int32_t>& candidates(std::int32_t lane) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    models::draft::Spec spec_{};
    DraftMemory memory_{};
    std::int32_t lanes_            = 0;
    std::int32_t context_capacity_ = 0;
};

} // namespace ninfer::models::nemotron_h
