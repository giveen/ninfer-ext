#include "models/nemotron_h/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/device.h"       // CUDA_CHECK: the residency budget measures the card
#include "core/weight_view.h" // weight_element_count

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::nemotron_h {
namespace {

using artifact::ArtifactError;
using artifact::ParameterReference;
using artifact::Residency;

// The representations this family requires rather than leaving to the converter: a norm gain, the
// depthwise convolution's payload and its bias are read as BF16 elements by the Ops, and the
// state arithmetic's own parameters -- `A_log`, `D`, `dt_bias` -- plus the router's correction bias
// are consumed in FP32, because that is the precision the reference's own state arithmetic and
// selection use. A BF16 copy would lose exactly what they are for.
constexpr QType kDenseFormat  = QType::BF16;
constexpr QType kScalarFormat = QType::FP32;

class Declarations {
public:
    // `resident` answers, per MoE block, whether that block's bank is device-resident: the split the
    // caller's request resolved. The bank is declared with it, because residency is a property of
    // the binding -- a resident bank is uploaded once and read where it lives, a non-resident one is
    // page-locked host memory a kernel addresses over UVA -- and a parameter demanded both ways
    // would be materialized twice.
    Declarations(artifact::Binder& binder, const ExpertResidencyPlan& residency)
        : binder_(binder), residency_(residency) {}

    // A parameter the Ops consume as a matrix. `input` names the mathematical input that reads it,
    // when the artifact declares one, which is where its activation divisor lives.
    WeightId matrix(std::string name, const artifact::Shape& shape, std::string_view input = {},
                    std::optional<QType> format = {}, Residency residency = Residency::Device) {
        DeclaredParameter pending;
        pending.reference          = binder_.parameter(name, shape, residency, format);
        pending.activation_divisor = activation_divisor(name, input);
        pending.name               = std::move(name);
        pending.matrix             = true;
        return push(std::move(pending));
    }

    // A parameter the Ops read as a plain tensor -- a norm gain, a bias or an FP32 per-head vector.
    // None of them is read by a declared input, so none carries an activation divisor.
    WeightId tensor(std::string name, const artifact::Shape& shape,
                    std::optional<QType> format = {}) {
        DeclaredParameter pending;
        pending.reference = binder_.parameter(name, shape, Residency::Device, format);
        pending.name      = std::move(name);
        return push(std::move(pending));
    }

    // A routed expert bank. The artifact names every expert and gives it a row range of one parent
    // per layer and direction -- the converter's packing group, expert-major -- so the bank an Op
    // consumes is that whole parent rather than any one expert's rows: a row range of a quantized
    // parent cannot re-derive its own scale plane, which is why an expert is not separately
    // bindable. `first_expert` is a name the artifact does declare, whose object is the parent the
    // declared bank shape is checked against.
    WeightId bank(std::string name, std::string first_expert, const artifact::Shape& shape,
                  std::string_view input, std::int32_t pool_layer) {
        const float divisor  = activation_divisor(first_expert, input);
        const auto& bindings = binder_.reader().directory().bindings;
        const auto found     = bindings.find(first_expert);
        if (found == bindings.end()) {
            throw ArtifactError("missing logical parameter " + first_expert);
        }
        if (found->second.parts.empty()) {
            throw ArtifactError(first_expert + ": expert Binding has no parts");
        }
        // The whole-object form is what makes the binder check the parent's own shape against the
        // bank the config derives, so a packing group of another width is refused here.
        artifact::Binding whole;
        whole.whole_object = true;
        whole.elements     = weight_element_count(shape);
        whole.parts.push_back({found->second.parts.front().object, 0, whole.elements});

        const bool device =
            pool_layer >= 0 && pool_layer < static_cast<std::int32_t>(residency_.resident.size())
                ? residency_.resident[static_cast<std::size_t>(pool_layer)]
                : false;
        DeclaredParameter pending;
        pending.reference = binder_.binding(name, whole, shape,
                                            device ? Residency::Device : Residency::HostPinned, {});
        pending.name      = std::move(name);
        pending.matrix             = true;
        pending.activation_divisor = divisor;
        return push(std::move(pending));
    }

    [[nodiscard]] std::vector<DeclaredParameter> take() { return std::move(pending_); }

    // One MoE layer's expert count, from the artifact rather than from the config: the artifact's
    // own per-expert names are its inventory of that layer's bank, and the router's stored first
    // axis is its own count. Both are compared against the layer's `n_routed_experts` by the shared
    // check, so a config that disagrees with the artifact is refused while planning.
    void check_expert_count(std::size_t layer, const std::string& prefix, const MoeBankConfig& bank,
                            WeightId router) {
        const auto& bindings             = binder_.reader().directory().bindings;
        const std::string experts_prefix = prefix + "moe/experts/";
        std::uint64_t stored_experts     = 0;
        while (bindings.contains(experts_prefix + std::to_string(stored_experts) + "/up")) {
            ++stored_experts;
        }
        const auto& reference = pending_.at(router.index).reference;
        if (reference.binding.parts.empty()) {
            throw ArtifactError(reference.name + ": Binding has no parts");
        }
        const auto& shape =
            binder_.reader().directory().tensor(reference.binding.parts.front().object).shape;
        if (shape.empty()) { throw ArtifactError(reference.name + ": object has no shape"); }
        verify_expert_count(layer, bank, stored_experts, shape.front());
    }

private:
    WeightId push(DeclaredParameter pending) {
        pending_.push_back(std::move(pending));
        return WeightId{pending_.size() - 1};
    }

    // The divisor the artifact itself declares for the input that reads `name`, or zero when the
    // parameter declares no input. Zero is what an A16-only input wants: the kernel reads the raw
    // activation and quantizes nothing. An input with no Use at all is a mismatch between this
    // loader's naming and the artifact's, so it is refused rather than defaulted.
    float activation_divisor(std::string_view name, std::string_view input) {
        if (input.empty()) { return 0.0F; }
        const auto& use = binder_.use(name, input);
        float divisor   = 0.0F;
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw ArtifactError(std::string(name) + "@" + std::string(input) +
                                    ": unknown auxiliary " + role);
            }
            const float value = binder_.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw ArtifactError(std::string(name) + "@" + std::string(input) +
                                    ": activation divisor must be positive finite FP32");
            }
            divisor = value;
        }
        return divisor;
    }

    artifact::Binder& binder_;
    const ExpertResidencyPlan& residency_;
    std::vector<DeclaredParameter> pending_;
};

std::string layer_prefix(std::size_t layer) { return "text/layers/" + std::to_string(layer) + "/"; }

MambaWeights declare_mamba(Declarations& declare, const MambaConfig& mamba, std::uint64_t hidden,
                           const std::string& prefix) {
    MambaWeights out;
    // The projection emits z, x/B/C and dt in one matrix, so it stays whole; the convolution is one
    // row per channel with the bias beside it, and its payload is stored tap-major by the converter
    // because that is the order the Op indexes.
    out.in_proj  = declare.matrix(prefix + "mamba/in_projection",
                                  {mamba.in_projection_size(), hidden}, prefix + "mixer_input");
    out.out_proj = declare.matrix(prefix + "mamba/out_projection", {hidden, mamba.inner_size()},
                                  prefix + "mamba/gated_norm_output");
    out.conv     = declare.tensor(prefix + "mamba/convolution",
                                  {mamba.conv_channels(), mamba.conv_kernel}, kDenseFormat);
    out.conv_bias =
        declare.tensor(prefix + "mamba/convolution_bias", {mamba.conv_channels()}, kDenseFormat);
    out.a_log   = declare.tensor(prefix + "mamba/a_log", {mamba.num_heads}, kScalarFormat);
    out.d_skip  = declare.tensor(prefix + "mamba/d_skip", {mamba.num_heads}, kScalarFormat);
    out.dt_bias = declare.tensor(prefix + "mamba/dt_bias", {mamba.num_heads}, kScalarFormat);
    out.norm    = declare.tensor(prefix + "mamba/norm", {mamba.inner_size()}, kDenseFormat);
    return out;
}

AttentionWeights declare_attention(Declarations& declare, const AttentionConfig& attention,
                                   std::uint64_t hidden, const std::string& prefix) {
    // Plain GQA: one query, key and value projection per block, no bias, no sinks and no window,
    // and no rotation anywhere -- so there is nothing else to declare.
    AttentionWeights out;
    out.query  = declare.matrix(prefix + "attention/query", {attention.query_width(), hidden},
                                prefix + "mixer_input");
    out.key    = declare.matrix(prefix + "attention/key", {attention.key_width(), hidden},
                                prefix + "mixer_input");
    out.value  = declare.matrix(prefix + "attention/value", {attention.key_width(), hidden},
                                prefix + "mixer_input");
    out.output = declare.matrix(prefix + "attention/output", {hidden, attention.query_width()},
                                prefix + "attention/attended");
    return out;
}

MoeWeights declare_moe(Declarations& declare, const MoeBankConfig& bank, std::uint64_t hidden,
                       const std::string& prefix, std::size_t layer, std::int32_t pool_layer) {
    const std::uint64_t latent  = bank.expert_input_width(hidden);
    const std::uint64_t experts = bank.num_experts;
    const std::uint64_t width   = bank.intermediate_size;
    const std::uint64_t shared  = bank.shared_intermediate_size;
    const std::string moe       = prefix + "moe/";

    MoeWeights out;
    out.router          = declare.matrix(moe + "router", {experts, hidden}, prefix + "ffn_input");
    out.correction_bias = declare.tensor(moe + "correction_bias", {experts}, kScalarFormat);
    // The two latent projections bracket the bank, and the shared expert is a relu2 MLP at the
    // block's own hidden width, because the reference adds it outside the latent round trip.
    out.fc1_latent  = declare.matrix(moe + "latent_in", {latent, hidden}, prefix + "ffn_input");
    out.fc2_latent  = declare.matrix(moe + "latent_out", {hidden, latent}, moe + "expert_product");
    out.shared_up   = declare.matrix(moe + "shared/up", {shared, hidden}, prefix + "ffn_input");
    out.shared_down = declare.matrix(moe + "shared/down", {hidden, shared}, moe + "shared/product");
    // One bank per direction, expert-major with one divisor for the whole parent: expert `e` is the
    // rows `[e * width, (e + 1) * width)` of `up` and `[e * latent, (e + 1) * latent)` of `down`.
    out.experts_up   = declare.bank(moe + "experts/up", moe + "experts/0/up",
                                    {experts * width, latent}, moe + "expert_input", pool_layer);
    out.experts_down = declare.bank(moe + "experts/down", moe + "experts/0/down",
                                    {experts * latent, width}, moe + "experts/0/product",
                                    pool_layer);
    // The bank's and the router's expert counts come from the artifact, not from the config value
    // both were declared with, and have to agree with it and with each other.
    declare.check_expert_count(layer, prefix, bank, out.router);
    return out;
}

TextWeights bind_text(Declarations& declare, const TextConfig& config) {
    const std::uint64_t hidden     = config.hidden_size;
    const std::uint64_t vocabulary = config.vocab_size;

    TextWeights out;
    // The embedding's and the head's representations are the converter's choice: the gather and the
    // head's linear both dispatch on what the weight actually is.
    out.token_embedding = declare.matrix("text/token_embedding", {vocabulary, hidden});
    out.output_head = declare.matrix("text/output_head", {vocabulary, hidden}, "text/final_hidden");
    out.final_norm  = declare.tensor("text/final_norm", {hidden}, kDenseFormat);

    out.layers.reserve(config.num_hidden_layers());
    for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
        const std::string prefix = layer_prefix(layer);
        LayerWeights weights;
        weights.kind       = config.block_kind(static_cast<std::uint32_t>(layer));
        weights.block_norm = declare.tensor(prefix + "block_norm", {hidden}, kDenseFormat);
        switch (weights.kind) {
        case BlockKind::Mamba:
            if (!config.mamba) {
                throw ArtifactError("block list has a Mamba layer but the artifact has no mamba "
                                    "configuration");
            }
            weights.mamba = declare_mamba(declare, *config.mamba, hidden, prefix);
            break;
        case BlockKind::Attention:
            if (!config.attention) {
                throw ArtifactError("block list has an attention layer but the artifact has no "
                                    "attention configuration");
            }
            weights.attention = declare_attention(declare, *config.attention, hidden, prefix);
            break;
        case BlockKind::Moe:
            weights.moe = declare_moe(declare, config.moe_bank(static_cast<std::uint32_t>(layer)),
                                      hidden, prefix, layer,
                                      expert_pool_layer(config, static_cast<std::uint32_t>(layer)));
            break;
        }
        out.layers.push_back(std::move(weights));
    }
    return out;
}

} // namespace

void verify_expert_count(std::size_t layer, const MoeBankConfig& bank,
                         const std::uint64_t stored_experts, const std::uint64_t router_experts) {
    if (stored_experts != bank.num_experts || router_experts != bank.num_experts) {
        throw ArtifactError("layer " + std::to_string(layer) + ": the artifact's bank holds " +
                            std::to_string(stored_experts) + " experts and its router " +
                            std::to_string(router_experts) + ", where this layer's config says " +
                            std::to_string(bank.num_experts) +
                            "; a global expert count must not reach a layer's bank");
    }
    if (bank.experts_per_token == 0 || bank.experts_per_token > bank.num_experts) {
        throw ArtifactError("layer " + std::to_string(layer) + ": the config routes to " +
                            std::to_string(bank.experts_per_token) + " of " +
                            std::to_string(bank.num_experts) + " experts");
    }
}

struct LoadPlan::Impl {
    TextConfig config;
    ModelWeights weights;
    TextResources resources;
    artifact::Binder binder;
    std::vector<DeclaredParameter> pending;
    artifact::MaterializationPlan plan;
    ExpertResidencyPlan residency;

    explicit Impl(const artifact::Reader& reader) : binder(reader) {}
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const TextConfig& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const TextResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->plan; }

const ExpertResidencyPlan& LoadPlan::expert_residency() const { return impl_->residency; }

std::size_t LoadPlan::parameter_count() const { return impl_->pending.size(); }

const ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

LoadPlan::Content LoadPlan::release() {
    if (!impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data = std::move(impl_);
    return Content{std::move(data->config), std::move(data->weights), std::move(data->pending),
                   std::move(data->plan)};
}

namespace {

// The device bytes the routed banks' *residents* may have: free device memory less the dense side
// and less everything the caller reserved. The dense side is taken as the artifact's payload less
// the banks themselves, which over-states it by the container's framing and the text resources --
// and over-stating it only shrinks the cache, so the direction is safe.
std::size_t expert_residency_budget(const artifact::Reader& reader,
                                    std::span<const ops::ExpertBankShape> banks,
                                    const ExpertResidencyRequest& request) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::uint64_t bank_bytes = 0;
    for (const ops::ExpertBankShape& bank : banks) {
        bank_bytes += static_cast<std::uint64_t>(expert_bank_bytes(bank));
    }
    const std::uint64_t payload = reader.file_bytes();
    const std::uint64_t dense   = payload > bank_bytes ? payload - bank_bytes : 0;
    const std::uint64_t needed  = dense + request.reserve_bytes;
    // Nothing to plan when the card cannot cover the dense weights and the reservation beside them:
    // zero is the honest answer, and *this* is not the place to refuse the load -- the residency
    // planner's subject is the experts, and the side that measured the state, the workspace and the
    // runtime's own reservation is the instance, whose own check names those numbers. Refusing here
    // would also make a request for no device expert bytes -- which asks for nothing at all -- fail
    // for a reason that is not its own.
    if (needed >= free_bytes) { return 0; }
    return static_cast<std::size_t>(free_bytes - needed);
}

} // namespace

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options,
                   ExpertResidencyRequest request) {
    // The family carries the text tower alone: no artifact component or option selects anything
    // else, so every input plans the same set of parameters.
    (void)options;
    auto out    = std::make_unique<LoadPlan::Impl>(reader);
    out->config = parse_config(reader.directory()).text;
    // The routed banks' split is decided before anything is declared, because it *is* a property of
    // the declaration: the same request that a resident bank is uploaded once under is the one a
    // non-resident bank stays page-locked under.
    const std::vector<ops::ExpertBankShape> banks = expert_bank_geometry(out->config);
    out->residency = plan_expert_residency(
        banks, expert_residency_budget(reader, banks, request), !request.automatic,
        request.explicit_bytes);
    // The tokenizer's and the chat template's bytes are artifact resources rather than weights, and
    // this is where a frontend reads them. They borrow the reader's mapped bytes, and the Binder
    // must still be alive to hand them over, so this comes before it is finished.
    const auto resource = [&](std::string_view role) {
        const auto bytes = out->binder.host_object(out->binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    out->resources.tokenizer_json         = resource("tokenizer.json");
    out->resources.tokenizer_config_json  = resource("tokenizer_config.json");
    out->resources.generation_config_json = resource("generation_config.json");
    out->resources.chat_template_jinja    = resource("chat_template.jinja");
    Declarations declarations(out->binder, out->residency);
    out->weights.text = bind_text(declarations, out->config);
    out->pending      = declarations.take();
    // Finishing consumes the Binder, so it must come after every parameter request.
    out->plan = std::move(out->binder).finish();
    return LoadPlan(std::move(out));
}

} // namespace ninfer::models::nemotron_h
