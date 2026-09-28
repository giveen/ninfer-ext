#include "calibration.h"

#include <algorithm>
#include <filesystem>

#include "hessian_io.h"
#include "options.h"
#include "parameter_reader.h"
#include "trace_reader.h"

#include "artifact/reader.h"
#include "models/qwen3_5/execution/calibration.h"
#include "ninfer/engine.h"
#include "quantize/exl3/hessian.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::quantize::app {
namespace {

using Site = ninfer::models::qwen3_5::execution::CalibrationSite;
using Key  = std::pair<int, int>; // (site, layer)

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Which projection input a logical parameter consumes. Site and Text layer fully determine the
// activation, so every member of a shared-input group maps to the same Hessian.
std::optional<Key> site_for(const std::string& name) {
    const auto layer_prefix = std::string("text/layers/");
    if (name == "text/output_head") {
        return std::make_pair(static_cast<int>(Site::OutputHead), -1);
    }
    // The MTP layer's activations: one layer (sentinel -1), the trunk's final hidden state plus the
    // next-token embedding at the stem, then its own attention and MLP. Its MLP reuses the Mlp sites
    // because ffn() observes them with the same -1 sentinel.
    if (name == "mtp/input_projection") {
        return std::make_pair(static_cast<int>(Site::MtpStem), -1);
    }
    const auto mtp_prefix = std::string("mtp/layers/");
    if (name.starts_with(mtp_prefix)) {
        const std::size_t slash = name.find('/', mtp_prefix.size());
        if (slash == std::string::npos) { return std::nullopt; }
        const std::string tail = name.substr(slash + 1);
        const auto mtp_key     = [](Site site) {
            return std::make_pair(static_cast<int>(site), -1);
        };
        if (tail == "attention/query" || tail == "attention/key" || tail == "attention/gate" ||
            tail == "attention/value") {
            return mtp_key(Site::MtpAttentionInput);
        }
        if (tail == "attention/output") { return mtp_key(Site::MtpAttentionOutput); }
        if (tail == "mlp/gate" || tail == "mlp/up") { return mtp_key(Site::MlpInput); }
        if (tail == "mlp/down") { return mtp_key(Site::MlpActivation); }
        return std::nullopt;
    }
    if (!name.starts_with(layer_prefix)) { return std::nullopt; }
    const std::size_t slash = name.find('/', layer_prefix.size());
    if (slash == std::string::npos) { return std::nullopt; }
    int layer = 0;
    try {
        layer = std::stoi(name.substr(layer_prefix.size(), slash - layer_prefix.size()));
    } catch (const std::exception&) { return std::nullopt; }
    const std::string tail = name.substr(slash + 1);
    const auto key = [&](Site site) { return std::make_pair(static_cast<int>(site), layer); };
    if (tail == "attention/query" || tail == "attention/key" || tail == "attention/gate" ||
        tail == "attention/value") {
        return key(Site::AttentionInput);
    }
    if (tail == "attention/output") { return key(Site::AttentionOutput); }
    if (tail == "gdn/query" || tail == "gdn/key" || tail == "gdn/value" || tail == "gdn/z") {
        return key(Site::GdnInput);
    }
    if (tail == "gdn/a_projection" || tail == "gdn/b_projection") { return key(Site::GdnResidual); }
    if (tail == "gdn/output") { return key(Site::GdnOutput); }
    if (tail == "mlp/gate" || tail == "mlp/up") { return key(Site::MlpInput); }
    if (tail == "mlp/down") { return key(Site::MlpActivation); }
    return std::nullopt;
}

// Device Hessian for the sites of one layer group. Groups are separate passes over the same rows:
// the weights never change, so the observed activations (and the resulting Hessians) are identical
// to a single pass, while peak device memory stays bounded by the widest group.
class HessianGroup {
public:
    HessianGroup(std::filesystem::path hessians, std::map<Key, std::vector<std::string>> names)
        : hessians_(std::move(hessians)), names_(std::move(names)) {}
    ~HessianGroup() { release(); }
    HessianGroup(const HessianGroup&)            = delete;
    HessianGroup& operator=(const HessianGroup&) = delete;

    void observe(Site site, int layer, const ninfer::Tensor& input, cudaStream_t stream) {
        const Key key{static_cast<int>(site), layer};
        if (!names_.contains(key)) { return; }
        if (input.dtype != ninfer::DType::BF16 || input.ne[2] != 1 || input.ne[3] != 1) { return; }
        const std::int64_t k = input.ne[0];
        const std::int64_t tokens = input.ne[1];
        if (k <= 0 || tokens <= 0 || k % 128 != 0) { return; }
        auto& slot = slots_[key];
        if (slot.data == nullptr) {
            slot.k = k;
            check(cudaMalloc(&slot.data, static_cast<std::size_t>(k) * k * sizeof(float)),
                  "cudaMalloc(Hessian)");
            check(cudaMemset(slot.data, 0, static_cast<std::size_t>(k) * k * sizeof(float)),
                  "cudaMemset(Hessian)");
        } else if (slot.k != k) {
            throw std::runtime_error("calibration site changed its input width");
        }
        ninfer::quantize::exl3::accumulate_hessian_kt(
            static_cast<const __nv_bfloat16*>(input.data), tokens, k, slot.data, stream);
    }

    // Copy every observed Hessian to host and write one file per name that shares it.
    std::uint64_t flush() {
        check(cudaDeviceSynchronize(), "Hessian calibration synchronize");
        std::uint64_t written = 0;
        for (auto& [key, slot] : slots_) {
            if (slot.data == nullptr) { continue; }
            const auto names = names_.find(key);
            if (names == names_.end()) { continue; }
            std::vector<float> host(static_cast<std::size_t>(slot.k) * slot.k);
            check(cudaMemcpy(host.data(), slot.data, host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(Hessian)");
            for (const std::string& name : names->second) {
                write_f32_file(hessians_ / (sanitize_parameter_name(name) + ".h.f32"), host);
                ++written;
            }
        }
        release();
        return written;
    }

private:
    struct Slot {
        float* data    = nullptr;
        std::int64_t k = 0;
    };
    void release() noexcept {
        for (auto& [key, slot] : slots_) {
            if (slot.data != nullptr) { (void)cudaFree(slot.data); }
        }
        slots_.clear();
    }
    std::filesystem::path hessians_;
    std::map<Key, std::vector<std::string>> names_;
    std::map<Key, Slot> slots_;
};

class ObserverScope {
public:
    explicit ObserverScope(ninfer::models::qwen3_5::execution::CalibrationObserver observer) {
        ninfer::models::qwen3_5::execution::calibration_observer() = std::move(observer);
    }
    ~ObserverScope() { ninfer::models::qwen3_5::execution::calibration_observer() = nullptr; }
    ObserverScope(const ObserverScope&)            = delete;
    ObserverScope& operator=(const ObserverScope&) = delete;
};

} // namespace

CalibrationResult calibrate(const std::filesystem::path& activation_model,
                            const std::filesystem::path& weights_artifact,
                            const CalibrationTrace& trace,
                            const std::filesystem::path& hessians, int device,
                            std::uint32_t max_rows, std::uint64_t device_budget_bytes) {
    if (trace.rows == 0 || trace.row_tokens == 0) {
        throw std::invalid_argument("calibration trace is empty");
    }
    if (!std::filesystem::is_regular_file(activation_model)) {
        throw std::invalid_argument("activation model artifact does not exist: " +
                                    activation_model.string());
    }
    const std::uint32_t rows =
        max_rows == 0 ? trace.rows : std::min(max_rows, trace.rows);

    // Names per observed site, and the device bytes each layer's Hessians need.
    std::map<Key, std::vector<std::string>> names;
    std::map<Key, std::uint64_t> site_bytes;
    {
        const ninfer::artifact::Reader reader(weights_artifact);
        for (const LinearParameter& parameter : enumerate_linear_parameters(reader)) {
            const auto key = site_for(parameter.name);
            if (!key) { continue; }
            names[*key].push_back(parameter.name);
            site_bytes[*key] =
                std::max(site_bytes[*key],
                         static_cast<std::uint64_t>(parameter.k) * parameter.k * sizeof(float));
        }
    }
    std::map<int, std::uint64_t> layer_bytes;
    for (const auto& [key, bytes] : site_bytes) { layer_bytes[key.second] += bytes; }
    if (names.empty()) {
        throw std::runtime_error("weights artifact has no calibratable Text projections");
    }

    // Partition the observed layers into consecutive groups that fit the device budget. Layers
    // beyond a group's first are only charged when their bytes differ, which they do not: this
    // greedy pass keeps each group's sum at or below the budget.
    std::vector<std::vector<int>> groups;
    {
        std::vector<int> pending;
        std::uint64_t used = 0;
        for (const auto& [layer, bytes] : layer_bytes) {
            const std::uint64_t cost = std::max<std::uint64_t>(bytes, 1);
            if (!pending.empty() && used + cost > device_budget_bytes) {
                groups.push_back(std::move(pending));
                pending.clear();
                used = 0;
            }
            pending.push_back(layer);
            used += cost;
        }
        if (!pending.empty()) { groups.push_back(std::move(pending)); }
    }

    CalibrationResult result;
    result.rows = rows;
    std::filesystem::create_directories(hessians);
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path = activation_model;
    engine_options.purpose       = ninfer::EnginePurpose::CausalScoring;
    engine_options.device        = device;
    engine_options.max_context   = trace.row_tokens;
    // The MTP layer only runs when its backend is selected, so its activations would otherwise go
    // unobserved and its Hessians unwritten. Scoring still returns the trunk's logits; the MTP
    // prefill is extra work, done only for an artifact that carries an MTP component.
    bool has_mtp = false;
    for (const auto& [key, value] : names) {
        for (const auto& name : value) {
            if (name.starts_with("mtp/")) {
                has_mtp = true;
                break;
            }
        }
        if (has_mtp) { break; }
    }
    // The scoring Program rejects a speculative backend by design, so the MTP sites are observed in
    // a separate generation pass below, not here. The engine is held in an optional so it can be
    // released before the generation engine loads: both are ~16 GiB and do not fit together.
    std::optional<ninfer::Engine> engine(std::in_place, std::move(engine_options));

    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        std::map<Key, std::vector<std::string>> group_names;
        for (const int layer : groups[group_index]) {
            for (const auto& [key, value] : names) {
                if (key.second == layer) { group_names.emplace(key, value); }
            }
        }
        HessianGroup accumulator(hessians, std::move(group_names));
        ObserverScope observer([&](Site site, int layer, const ninfer::Tensor& input,
                                   cudaStream_t stream) {
            accumulator.observe(site, layer, input, stream);
        });
        for (std::uint32_t row = 0; row < rows; ++row) {
            const std::int32_t length = trace.lengths[row];
            if (length < 2) { continue; }
            std::vector<ninfer::TokenId> tokens(
                trace.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(row) * trace.row_tokens,
                trace.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(row) * trace.row_tokens + length);
            (void)engine->score_tokens(std::move(tokens), 1);
            if (group_index == 0) { result.tokens += static_cast<std::uint64_t>(length); }
        }
        result.hessians += accumulator.flush();
        std::fprintf(stderr, "calibration group %zu/%zu done (%llu Hessians so far)\n",
                     group_index + 1, groups.size(),
                     static_cast<unsigned long long>(result.hessians));
    }
    // The MTP layer only runs under generation, so its sites are observed in a second pass with a
    // generation engine. Release the scoring engine first: both are ~16 GiB.
    engine.reset();
    if (has_mtp) {
        std::map<Key, std::vector<std::string>> mtp_names;
        for (const auto& [key, value] : names) {
            for (const auto& name : value) {
                if (name.starts_with("mtp/")) { mtp_names[key].push_back(name); }
            }
        }
        if (!mtp_names.empty()) {
            const std::uint32_t mtp_rows = std::min<std::uint32_t>(rows, 8);
            // CUDA graphs are off: the observer allocates and copies, which graph capture forbids.
            // The observer accumulates only the MTP names, so the Text Hessians are undisturbed.
            ninfer::EngineOptions gen_options;
            gen_options.artifact_path  = activation_model;
            gen_options.purpose        = ninfer::EnginePurpose::Generation;
            gen_options.device         = device;
            gen_options.max_context    = trace.row_tokens;
            gen_options.use_cuda_graph = false;
            gen_options.speculative.backend      = SpeculativeBackend::Mtp;
            gen_options.speculative.draft_tokens = 1;
            ninfer::Engine gen(std::move(gen_options));
            HessianGroup mtp_accumulator(hessians, std::move(mtp_names));
            ObserverScope mtp_observer([&](Site site, int layer, const ninfer::Tensor& input,
                                           cudaStream_t stream) {
                mtp_accumulator.observe(site, layer, input, stream);
            });
            for (std::uint32_t row = 0; row < mtp_rows; ++row) {
                const std::int32_t length = trace.lengths[row];
                if (length < 2) { continue; }
                std::vector<ninfer::TokenId> tokens(
                    trace.input_ids.begin() +
                        static_cast<std::ptrdiff_t>(row) * trace.row_tokens,
                    trace.input_ids.begin() +
                        static_cast<std::ptrdiff_t>(row) * trace.row_tokens + length);
                ninfer::RequestOptions request;
                // A few output tokens, not one: the MTP attention output and MLP only run in
                // mtp_forward_tail, which the prefill alone does not reach.
                request.execution.requested_output_tokens = 4;
                request.execution.sampling.temperature    = 0.0F;
                request.stop.include_model_defaults       = false;
                (void)gen.generate(gen.prepare_tokens(std::move(tokens)), request);
            }
            result.hessians += mtp_accumulator.flush();
            std::fprintf(stderr, "MTP calibration over %u rows (%llu Hessians so far)\n", mtp_rows,
                         static_cast<unsigned long long>(result.hessians));
        }
    }
    result.skipped_names = 0;
    return result;
}

} // namespace ninfer::quantize::app
