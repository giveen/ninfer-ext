#pragma once

// Nemotron-H artifact loading: the plan that resolves every parameter the config's block list
// names. The materialization that binds them to the model's own ids is not here -- it constructs
// `Model`, so it lives beside the model on the execution side, and this side reaches it only by
// handing the plan's resolved content over (`LoadPlan::release`).
//
// Two properties of this family shape the split. The block list varies per layer -- a Mamba block
// binds eight parameters, an attention block four, a LatentMoE block eight whose expert count and
// width are that layer's own -- so a name, shape or stored representation the artifact does not
// carry is refused while planning, rather than part way through a 45 GB upload. And the routed
// banks are 37.21 GB of NVFP4 experts, which cannot be device-resident on a 32 GB card, so the plan
// is where their pinned host residency is declared and summed.

#include "artifact/binder.h"       // ParameterReference, Residency
#include "artifact/materializer.h" // MaterializationPlan
#include "models/load_options.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/experts.h" // ExpertResidencyPlan
#include "models/nemotron_h/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::nemotron_h {

// The artifact's text resources, which is where the tokenizer and the chat template live. The views
// borrow bytes the plan read out of the artifact, so they are valid while the plan -- or, once it
// is materialized, the model that took those bytes over -- lives: a consumer that outlives both
// copies them, as the Model holds materialized weights rather than the artifact.
struct TextResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
    std::string_view chat_template_jinja;
};

// One declared parameter, as planning resolved it: the artifact reference it was checked against,
// the name the model binds it under, whether the Ops consume it as a matrix, and the activation
// divisor the artifact declares for the input that reads it. The model binds these in the order the
// plan declared them, so an id addresses the same parameter on both sides of the split.
struct DeclaredParameter {
    artifact::ParameterReference reference;
    std::string name;
    bool matrix              = false;
    float activation_divisor = 0.0F;
};

/**
 * Where the routed experts' device memory comes from, as the caller resolved it.
 *
 * The cache is the elastic term and the K/V is not, so this is spent *after* the caller has
 * resolved contexts, lanes, K/V dtype and chunk: `reserve_bytes` is everything that must stay on the
 * card beside the resident banks -- the family's state for those lanes and windows, one pass's
 * workspace, the staging buffer and its tables, the runtime's own reservation and its headroom --
 * and the resident banks take only what free memory is left over. A cache request that would eat
 * into that reservation is re-derived smaller and reported (`ExpertResidencyPlan::clamped`), never
 * honoured by shrinking the K/V and never over-committed.
 */
struct ExpertResidencyRequest {
    // Automatic takes everything the card has left; otherwise `explicit_bytes` caps the resident
    // banks and is clamped to the same figure.
    bool automatic             = true;
    std::size_t explicit_bytes = 0;
    // The non-expert device bytes, summed as described above. Zero is a valid request for "plan no
    // residency", not for "assume nothing else is resident".
    std::size_t reserve_bytes = 0;
};

// Cold load plan. Every declared parameter is resolved against the artifact while planning, so a
// missing name or a shape or representation the artifact does not store is refused here rather than
// at execution. The plan borrows its Reader until materialization.
class LoadPlan {
public:
    // What a plan hands the execution side when a model is built from it: the config and the weight
    // table the model takes over, the declarations it binds by id, and the materialization that
    // uploads the bytes those declarations name. It is one bundle rather than five accessors
    // because the materializer takes all of it at once.
    struct Content {
        TextConfig config;
        ModelWeights weights;
        std::vector<DeclaredParameter> declarations;
        artifact::MaterializationPlan materialization;
    };

    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const TextConfig& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const TextResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // The routed banks' split, decided while planning because it is a property of the binding: a
    // resident bank is declared device-resident and uploaded once, a non-resident one stays pinned
    // host memory. Reported so a caller can see the residency and the pinned remainder it implies.
    [[nodiscard]] const ExpertResidencyPlan& expert_residency() const;
    // How many parameters the plan declared, so a caller can walk every one of them.
    [[nodiscard]] std::size_t parameter_count() const;
    // The declared reference, including the residency the loader asked for: a routed bank is pinned
    // host memory rather than VRAM, and everything else is device-resident.
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    // Hands the plan's content over and leaves the plan spent, so the 45 GB it named is uploaded
    // once: a second call refuses rather than materializing the same bytes twice. Its caller is the
    // model's materializer, which lives on the execution side beside the `Model` it constructs.
    [[nodiscard]] Content release();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions, ExpertResidencyRequest);
};

// The plan borrows `reader`, so the caller owns the Reader for the plan's lifetime.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {},
                                 ExpertResidencyRequest residency = {});

// One MoE layer's expert count, refused rather than resolved: the bank the artifact stores and the
// router it stores must agree with each other and with the layer's own `n_routed_experts`, and the
// count must serve the router's selection. The two are declared from the same config value, so a
// wrong or stale config would otherwise read a shape-consistent artifact and route to the wrong
// experts -- and a *global* expert count must never reach a layer's bank, because this checkpoint's
// banks differ per layer (240 to 496 experts). `stored_experts` is the artifact's own per-layer
// inventory and `router_experts` the first axis of the router object it stores; the loader reads
// both from the artifact, which is what makes this a check rather than a restatement. Exposed so
// the refusal is testable without a converted artifact.
void verify_expert_count(std::size_t layer, const MoeBankConfig& bank, std::uint64_t stored_experts,
                         std::uint64_t router_experts);

} // namespace ninfer::models::nemotron_h
