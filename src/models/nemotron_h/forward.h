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
// Scope: this is the *prefill* pass. A block attends over the tokens it is handed, and a Mamba
// mixer starts from a zero state, so a pass is one sequence's prompt. Mamba's convolution history
// and recurrent plane and the attention K/V are the engine's sequence state, and are not modelled
// here; nothing in this header keeps state between calls.
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
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/experts.h"
#include "models/nemotron_h/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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

// Transient capacity one block needs for `tokens` tokens of one kind, and the whole pass needs for
// the widest block plus the head. Both are the exact peak of the real allocation sequence, taken
// from the same planner the pass runs against a dry-run layout.
[[nodiscard]] std::size_t block_workspace_bytes(const TextConfig& config, std::int32_t tokens);
[[nodiscard]] std::size_t model_workspace_bytes(const TextConfig& config, std::int32_t tokens);

/**
 * Runs one decoder block for `tokens` tokens at once.
 *
 * `hidden_in` and `hidden_out` are contiguous BF16 `[hidden_size, tokens]` and must not overlap;
 * the block reads the residual stream and writes its updated self. `arena` supplies the block's
 * transient storage and must hold `block_workspace_bytes` for this config and `tokens`.
 *
 * `expert_pool` places the routed banks: null reads each MoE block's bank where it is bound (over
 * UVA when it is host-pinned, in place when it is device-resident), and a pool stages the banks
 * that are not resident into its device buffers instead. See `RoutedExpertPool`.
 */
void forward_block(const Model& model, std::size_t layer, const Tensor& hidden_in,
                   DeviceArena& arena, Tensor& hidden_out, DeviceExecutionView execution,
                   const BlockObserver* observer = nullptr, RoutedExpertPool* experts = nullptr);

/**
 * Runs a whole text pass: the token embedding, every block in order, the final norm and the head.
 *
 * `token_ids` is contiguous I32 `[tokens]` and every id must be in `[0, vocabulary)`. `logits` is
 * contiguous BF16 `[vocabulary_size, tokens]` and must not overlap the arena. `arena` must hold
 * `model_workspace_bytes` for this config and `tokens`, which is one block's peak plus the head's.
 * `expert_pool` is as `forward_block`'s.
 */
void forward_model(const Model& model, const Tensor& token_ids, DeviceArena& arena, Tensor& logits,
                   DeviceExecutionView execution, const BlockObserver* observer = nullptr,
                   RoutedExpertPool* expert_pool = nullptr);

} // namespace ninfer::models::nemotron_h
