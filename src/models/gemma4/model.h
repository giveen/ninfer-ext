#pragma once

// The materialized Gemma 4 text model: the weights after the artifact's bytes have been uploaded.
//
// The bound views borrow the Model's backing store, so a Model must outlive every view taken from
// it, and `backing_` is declared first so it is destroyed last.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"
#include "models/gemma4/config.h"
#include "models/gemma4/weights.h"
#include "models/load_options.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {

class LoadPlan;

// One materialized parameter: its logical name and the resident view the Ops receive. The activation
// policies the artifact declares per input are not bound yet.
struct BoundWeight {
    std::string name;
    WeightView view;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    // The three accessor forms the Ops need. A norm weight is a plain tensor, a projection is a
    // native weight, and the layer scalar is a host float because mul_scalar takes a value.
    [[nodiscard]] Tensor tensor(WeightId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] float layer_scalar(WeightId id) const;

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(TextConfig config, ModelWeights weights, std::vector<BoundWeight> bound,
          std::vector<float> layer_scalars, artifact::MaterializedArtifact backing);

    // Destroyed last: every BoundWeight view borrows these bytes.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<float> layer_scalars_;
};

// Uploads the plan's selected bytes and resolves every bound view. The plan is consumed.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

// Plan and materialize in one step. The Reader lives for this call only, which is enough because the
// returned Model holds the materialized bytes rather than the artifact.
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                                DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::gemma4
