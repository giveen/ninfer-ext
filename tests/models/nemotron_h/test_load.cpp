// The Nemotron-H artifact loader: the plan resolves every name, shape and stored representation
// before any byte is uploaded, and materialization binds every id the config's block list implies.
//
// With NINFER_NEMOTRON_ARTIFACT unset this skips -- the artifact is 45.28 GB, of which 37.21 GB of
// routed experts are pinned in host memory, and a routine run should not pay for that. The refusal
// path needs no artifact, so it always runs.

#include "artifact/binder.h" // ParameterReference, Residency
#include "artifact/reader.h"
#include "core/device.h"
#include "models/nemotron_h/forward.h" // Model, materialize_model
#include "models/nemotron_h/load.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace nemotron = ninfer::models::nemotron_h;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "nemotron_h load: " << message << '\n';
        ++failures;
    }
}

// The per-layer parameter sets the block list implies, counted independently of the loader: a Mamba
// block binds eight parameters, an attention block four, a LatentMoE block eight.
std::size_t expected_parameters(const nemotron::TextConfig& config) {
    std::size_t count = 3; // token embedding, output head, final norm
    for (const auto kind : config.block_kinds) {
        count += 1; // the block norm
        count += kind == nemotron::BlockKind::Mamba       ? 8
                 : kind == nemotron::BlockKind::Attention ? 4
                                                          : 8;
    }
    return count;
}

// What the plan resolved for one id, against the shape the config derives and the residency the
// loader has to ask for: a routed bank is pinned host memory, everything else is VRAM.
void check_reference(const nemotron::LoadPlan& plan, nemotron::WeightId id,
                     const std::vector<std::uint64_t>& shape, artifact::Residency residency,
                     const std::string& what) {
    const auto& reference = plan.parameter(id);
    check(reference.shape == shape, what + ": declared shape");
    check(reference.residency == residency, what + ": residency");
}

// The representation the artifact actually stores. This is what the binder refuses while planning,
// so it is also what makes each declared format below a check rather than a restatement.
void check_stored_format(const artifact::Reader& reader, const nemotron::LoadPlan& plan,
                         nemotron::WeightId id, QType expected, const std::string& what) {
    const auto& reference = plan.parameter(id);
    for (const auto& part : reference.binding.parts) {
        if (reader.geometry(part.object).format != expected) {
            check(false, what + ": stored representation");
            return;
        }
    }
}

// The whole parent object a reference resolves to, which is what a bank's own bytes are.
std::uint64_t parent_bytes(const artifact::Reader& reader, const nemotron::LoadPlan& plan,
                           nemotron::WeightId id) {
    std::uint64_t bytes = 0;
    for (const auto& part : plan.parameter(id).binding.parts) {
        bytes += reader.geometry(part.object).bytes;
    }
    return bytes;
}

void check_matrix(const nemotron::Model& model, nemotron::WeightId id, QType format,
                  std::int32_t rows, std::int32_t columns, const std::string& what) {
    const Weight& weight = model.matrix(id);
    check(weight.qtype == format, what + ": representation");
    check(weight.n == rows && weight.k == columns,
          what + ": geometry is " + std::to_string(weight.n) + "x" + std::to_string(weight.k) +
              " where the config says " + std::to_string(rows) + "x" + std::to_string(columns));
}

void check_tensor(const nemotron::Model& model, nemotron::WeightId id, DType dtype,
                  const std::string& what) {
    check(model.tensor(id).dtype == dtype, what + ": dtype");
}

// Every parameter of one block, read back through the model's own accessors: an id that was never
// bound throws here, and a weight bound at another geometry or representation is reported.
void check_layer(const nemotron::Model& model, std::size_t layer, const std::string& what) {
    const nemotron::TextConfig& config    = model.config();
    const nemotron::LayerWeights& weights = model.weights().text.layers.at(layer);
    const auto index                      = static_cast<std::uint32_t>(layer);
    const auto rows = [](std::uint64_t value) { return static_cast<std::int32_t>(value); };
    const std::int32_t hidden = rows(config.hidden_size);

    check(weights.kind == config.block_kind(index), what + ": bound kind");
    check_tensor(model, weights.block_norm, DType::BF16, what + " block_norm");
    if (weights.kind == nemotron::BlockKind::Mamba) {
        check(config.mamba.has_value() && weights.mamba.has_value(), what + ": mamba parameters");
        if (!config.mamba || !weights.mamba) { return; }
        const nemotron::MambaConfig& mamba = *config.mamba;
        const nemotron::MambaWeights& ids  = *weights.mamba;
        check_matrix(model, ids.in_proj, QType::FP8_E4M3FN_ROW_BF16,
                     rows(mamba.in_projection_size()), hidden, what + " in_projection");
        check_matrix(model, ids.out_proj, QType::FP8_E4M3FN_ROW_BF16, hidden,
                     rows(mamba.inner_size()), what + " out_projection");
        check_tensor(model, ids.conv, DType::BF16, what + " convolution");
        check_tensor(model, ids.conv_bias, DType::BF16, what + " convolution_bias");
        check_tensor(model, ids.a_log, DType::FP32, what + " a_log");
        check_tensor(model, ids.d_skip, DType::FP32, what + " d_skip");
        check_tensor(model, ids.dt_bias, DType::FP32, what + " dt_bias");
        check_tensor(model, ids.norm, DType::BF16, what + " mamba norm");
        return;
    }
    if (weights.kind == nemotron::BlockKind::Attention) {
        check(config.attention.has_value() && weights.attention.has_value(),
              what + ": attention parameters");
        if (!config.attention || !weights.attention) { return; }
        const nemotron::AttentionConfig& attention = *config.attention;
        const nemotron::AttentionWeights& ids      = *weights.attention;
        check_matrix(model, ids.query, QType::FP8_E4M3FN_ROW_BF16, rows(attention.query_width()),
                     hidden, what + " query");
        check_matrix(model, ids.key, QType::FP8_E4M3FN_ROW_BF16, rows(attention.key_width()),
                     hidden, what + " key");
        check_matrix(model, ids.value, QType::FP8_E4M3FN_ROW_BF16, rows(attention.key_width()),
                     hidden, what + " value");
        check_matrix(model, ids.output, QType::FP8_E4M3FN_ROW_BF16, hidden,
                     rows(attention.query_width()), what + " attention output");
        return;
    }
    check(weights.moe.has_value(), what + ": MoE parameters");
    if (!weights.moe) { return; }
    const nemotron::MoeBankConfig& bank = config.moe_bank(index);
    const nemotron::MoeWeights& ids     = *weights.moe;
    const std::int32_t experts          = rows(bank.num_experts);
    const std::int32_t latent           = rows(bank.expert_input_width(config.hidden_size));
    const std::int32_t width            = rows(bank.intermediate_size);
    const std::int32_t shared           = rows(bank.shared_intermediate_size);
    // The bank is one parent per direction, expert-major, and the Op reads expert `e` in place.
    check_matrix(model, ids.router, QType::BF16, experts, hidden, what + " router");
    check_tensor(model, ids.correction_bias, DType::FP32, what + " correction_bias");
    check_matrix(model, ids.fc1_latent, QType::BF16, latent, hidden, what + " latent_in");
    check_matrix(model, ids.fc2_latent, QType::BF16, hidden, latent, what + " latent_out");
    check_matrix(model, ids.shared_up, QType::NVFP4, shared, hidden, what + " shared/up");
    check_matrix(model, ids.shared_down, QType::NVFP4, hidden, shared, what + " shared/down");
    check_matrix(model, ids.experts_up, QType::NVFP4, experts * width, latent,
                 what + " experts/up");
    check_matrix(model, ids.experts_down, QType::NVFP4, experts * latent, width,
                 what + " experts/down");
}

// A path this loader cannot plan against, whatever else is true of it.
bool refused(const std::filesystem::path& path) {
    try {
        artifact::Reader reader(path);
        const auto plan = nemotron::plan_load(reader);
        (void)plan;
    } catch (const std::exception&) { return true; }
    return false;
}

// The per-layer expert count's invariant, on numbers that need no artifact: the artifact's own bank
// and router counts must agree with each other and with the layer's config, and the config's
// selection must fit inside them. This is the check a wrong or stale config would slip past, since
// the router and the bank are both declared from that same config value.
std::string expert_count_refusal(std::size_t layer, std::uint32_t config_experts,
                                 std::uint32_t experts_per_token, std::uint64_t stored_experts,
                                 std::uint64_t router_experts) {
    nemotron::MoeBankConfig bank;
    bank.num_experts       = config_experts;
    bank.experts_per_token = experts_per_token;
    try {
        nemotron::verify_expert_count(layer, bank, stored_experts, router_experts);
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

} // namespace

int main() {
    // The refusal path first, because it needs no artifact: planning resolves every declared
    // parameter against the artifact before anything is allocated, so a path that is not one is
    // refused while planning rather than 45 GB later.
    {
        const std::filesystem::path absent =
            std::filesystem::temp_directory_path() / "ninfer_nemotron_h_load_absent.ninfer";
        const std::filesystem::path garbage =
            std::filesystem::temp_directory_path() / "ninfer_nemotron_h_load_garbage.ninfer";
        std::filesystem::remove(absent);
        {
            std::ofstream out(garbage, std::ios::binary | std::ios::trunc);
            out << "this is not an artifact";
        }
        check(refused(absent), "planning against an absent path should be refused");
        check(refused(garbage),
              "planning against a file that is not an artifact should be refused");
        std::filesystem::remove(garbage);
    }

    // The expert count a layer's bank and router must agree on, with a deliberately wrong bank:
    // this is the mistake a global expert count produces, and the artifact-free case for it.
    {
        const std::string mismatch = expert_count_refusal(7, 240, 9, 304, 240);
        check(!mismatch.empty(), "a bank of another expert count should be refused");
        check(mismatch.find("layer 7") != std::string::npos &&
                  mismatch.find("304") != std::string::npos &&
                  mismatch.find("240") != std::string::npos,
              "the refusal should name the layer and both counts: " + mismatch);
        if (!mismatch.empty()) {
            std::cout << "nemotron_h load: refused a mismatched bank: " << mismatch << '\n';
        }
        check(!expert_count_refusal(3, 240, 9, 240, 512).empty(),
              "a router over another expert count should be refused");
        check(!expert_count_refusal(5, 240, 496, 240, 240).empty(),
              "a selection wider than the bank should be refused");
        check(expert_count_refusal(1, 240, 9, 240, 240).empty(),
              "a bank and router that agree with the config should be accepted");
    }
    if (failures != 0) { return 1; }

    const char* path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }

    try {
        artifact::Reader reader(path);
        const auto plan_started = std::chrono::steady_clock::now();
        auto plan               = nemotron::plan_load(reader);
        const double plan_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_started).count();

        const nemotron::TextConfig& config = plan.config();
        const std::size_t expected         = expected_parameters(config);
        check(config.num_hidden_layers() != 0, "the artifact's block list is empty");
        check(plan.parameter_count() == expected,
              "declared " + std::to_string(plan.parameter_count()) +
                  " parameters, where the config's block list implies " + std::to_string(expected));
        check(plan.weights().text.layers.size() == config.num_hidden_layers(),
              "declared layer count");

        // Copied rather than held by reference: the views point into bytes the materialized model
        // owns, so they are valid for as long as it is -- but the plan itself is consumed below.
        const nemotron::TextResources resources = plan.resources();
        check(!resources.tokenizer_json.empty(), "the artifact should carry tokenizer.json");
        check(!resources.tokenizer_config_json.empty(), "the artifact should carry a tokenizer "
                                                        "config");
        check(!resources.generation_config_json.empty(),
              "the artifact should carry a generation config");
        check(!resources.chat_template_jinja.empty(), "the artifact should carry a chat template");
        check(resources.tokenizer_json.front() == '{', "tokenizer.json should be JSON");

        const nemotron::TextWeights& text = plan.weights().text;
        const std::uint64_t hidden        = config.hidden_size;
        check_reference(plan, text.token_embedding, {config.vocab_size, hidden},
                        artifact::Residency::Device, "token_embedding");
        check_reference(plan, text.output_head, {config.vocab_size, hidden},
                        artifact::Residency::Device, "output_head");
        check_reference(plan, text.final_norm, {hidden}, artifact::Residency::Device, "final_norm");
        check_stored_format(reader, plan, text.token_embedding, QType::BF16, "token_embedding");
        check_stored_format(reader, plan, text.output_head, QType::FP8_E4M3FN_ROW_BF16,
                            "output_head");
        check_stored_format(reader, plan, text.final_norm, QType::BF16, "final_norm");

        // Every layer's own set, with its own bank width and expert count, and the residency the
        // 45 GB artifact forces: the routed banks cannot live in VRAM beside the dense side.
        std::uint64_t bank_bytes = 0;
        for (std::size_t layer = 0; layer < config.num_hidden_layers(); ++layer) {
            const nemotron::LayerWeights& weights = text.layers[layer];
            const auto index                      = static_cast<std::uint32_t>(layer);
            const std::string what                = "layer " + std::to_string(layer);
            check(weights.kind == config.block_kind(index), what + ": declared kind");
            check_reference(plan, weights.block_norm, {hidden}, artifact::Residency::Device,
                            what + " block_norm");
            if (weights.kind == nemotron::BlockKind::Mamba) {
                check(weights.mamba.has_value(), what + ": mamba parameters");
                if (!weights.mamba) { continue; }
                const nemotron::MambaConfig& mamba = *config.mamba;
                const nemotron::MambaWeights& ids  = *weights.mamba;
                check_stored_format(reader, plan, ids.in_proj, QType::FP8_E4M3FN_ROW_BF16,
                                    what + " in_projection");
                check_stored_format(reader, plan, ids.out_proj, QType::FP8_E4M3FN_ROW_BF16,
                                    what + " out_projection");
                check_reference(plan, ids.conv, {mamba.conv_channels(), mamba.conv_kernel},
                                artifact::Residency::Device, what + " convolution");
                check_stored_format(reader, plan, ids.conv, QType::BF16, what + " convolution");
                check_stored_format(reader, plan, ids.a_log, QType::FP32, what + " a_log");
                check_stored_format(reader, plan, ids.d_skip, QType::FP32, what + " d_skip");
                check_stored_format(reader, plan, ids.dt_bias, QType::FP32, what + " dt_bias");
            } else if (weights.kind == nemotron::BlockKind::Attention) {
                check(weights.attention.has_value(), what + ": attention parameters");
                if (!weights.attention) { continue; }
                const nemotron::AttentionConfig& attention = *config.attention;
                const nemotron::AttentionWeights& ids      = *weights.attention;
                check_reference(plan, ids.query, {attention.query_width(), hidden},
                                artifact::Residency::Device, what + " query");
                check_reference(plan, ids.key, {attention.key_width(), hidden},
                                artifact::Residency::Device, what + " key");
                check_reference(plan, ids.value, {attention.key_width(), hidden},
                                artifact::Residency::Device, what + " value");
                check_stored_format(reader, plan, ids.output, QType::FP8_E4M3FN_ROW_BF16,
                                    what + " attention output");
            } else {
                check(weights.moe.has_value(), what + ": MoE parameters");
                if (!weights.moe) { continue; }
                const nemotron::MoeBankConfig& bank = config.moe_bank(index);
                const nemotron::MoeWeights& ids     = *weights.moe;
                const std::uint64_t latent          = bank.expert_input_width(hidden);
                check_reference(plan, ids.router, {bank.num_experts, hidden},
                                artifact::Residency::Device, what + " router");
                check_reference(plan, ids.correction_bias, {bank.num_experts},
                                artifact::Residency::Device, what + " correction_bias");
                check_reference(plan, ids.fc1_latent, {latent, hidden}, artifact::Residency::Device,
                                what + " latent_in");
                check_reference(plan, ids.fc2_latent, {hidden, latent}, artifact::Residency::Device,
                                what + " latent_out");
                check_reference(plan, ids.shared_up, {bank.shared_intermediate_size, hidden},
                                artifact::Residency::Device, what + " shared/up");
                check_reference(plan, ids.shared_down, {hidden, bank.shared_intermediate_size},
                                artifact::Residency::Device, what + " shared/down");
                // The bank is the layer's own expert count and width, and it is pinned host memory.
                check_reference(
                    plan, ids.experts_up,
                    {static_cast<std::uint64_t>(bank.num_experts) * bank.intermediate_size, latent},
                    artifact::Residency::HostPinned, what + " experts/up");
                check_reference(
                    plan, ids.experts_down,
                    {static_cast<std::uint64_t>(bank.num_experts) * latent, bank.intermediate_size},
                    artifact::Residency::HostPinned, what + " experts/down");
                check_stored_format(reader, plan, ids.experts_up, QType::NVFP4,
                                    what + " experts/up");
                check_stored_format(reader, plan, ids.experts_down, QType::NVFP4,
                                    what + " experts/down");
                check_stored_format(reader, plan, ids.shared_up, QType::NVFP4, what + " shared/up");
                bank_bytes += parent_bytes(reader, plan, ids.experts_up) +
                              parent_bytes(reader, plan, ids.experts_down);
            }
        }

        DeviceContext device;
        const std::size_t parameter_count = plan.parameter_count();
        const auto load_started           = std::chrono::steady_clock::now();
        auto model                        = nemotron::materialize_model(std::move(plan), device);
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();

        // The consumed plan's own config is gone with it, so every check below reads the copy the
        // model was loaded with.
        const nemotron::TextConfig& loaded = model->config();
        const std::int32_t vocabulary      = static_cast<std::int32_t>(loaded.vocab_size);
        const std::int32_t wide            = static_cast<std::int32_t>(loaded.hidden_size);

        // Every block, through the model's own accessors: an unbound id throws, a weight of another
        // geometry or representation is reported, and every kind's set is walked.
        for (std::size_t layer = 0; layer < loaded.num_hidden_layers(); ++layer) {
            check_layer(*model, layer, "layer " + std::to_string(layer));
        }
        check_matrix(*model, model->weights().text.token_embedding, QType::BF16, vocabulary, wide,
                     "token_embedding");
        check_matrix(*model, model->weights().text.output_head, QType::FP8_E4M3FN_ROW_BF16,
                     vocabulary, wide, "output_head");
        check_tensor(*model, model->weights().text.final_norm, DType::BF16, "final_norm");

        // The pinned set is exactly the routed banks, and the device side is the rest: that is the
        // residency split the 32 GB card's VRAM forces.
        const auto& stats = model->storage_stats();
        check(stats.pinned_host_bytes == bank_bytes,
              "pinned bytes are " + std::to_string(stats.pinned_host_bytes) +
                  " where the routed banks are " + std::to_string(bank_bytes));
        check(stats.device_capacity_bytes < stats.pinned_host_bytes,
              "the device side should hold less than the pinned banks");
        check(stats.mapped_file_bytes == 0, "no parameter should be file-mapped");
        check(stats.pinned_host_bytes != 0, "the routed banks should be pinned");

        std::cout << "nemotron_h load: resources tokenizer.json " << resources.tokenizer_json.size()
                  << " B, tokenizer_config.json " << resources.tokenizer_config_json.size()
                  << " B, generation_config.json " << resources.generation_config_json.size()
                  << " B, chat_template.jinja " << resources.chat_template_jinja.size() << " B\n";
        std::cout << "nemotron_h load: " << parameter_count << " parameters, "
                  << stats.pinned_host_bytes << " pinned bytes, " << stats.device_capacity_bytes
                  << " device bytes, plan " << plan_seconds << " s, materialize " << load_seconds
                  << " s\n";
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h load: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "nemotron_h load: ok\n";
    return 0;
}
