#pragma once

#include "artifact/materializer.h"
#include "models/gemma4/config.h"
#include "models/gemma4/model.h"
#include "models/gemma4/weights.h"
#include "models/load_options.h"

#include <filesystem>
#include <memory>
#include <string_view>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::gemma4 {

// Cold load plan. Every declared parameter is resolved against the artifact while planning, so a
// name, shape or stored format the artifact does not carry is refused here rather than at
// execution. The plan borrows its Reader until materialization, which is not part of this class yet.
// The artifact's text resources, which is where the tokenizer lives. The views point into the
// reader's mapped files, so they are valid only while that reader lives: a consumer that outlives the
// load must copy them, as a Model holds materialized weights rather than the artifact.
struct TextResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
    std::string_view chat_template_jinja;
};

class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const TextConfig& config() const;
    [[nodiscard]] const TextResources& resources() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // How many parameters the plan declared, so a caller can walk every one of them.
    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

// The plan borrows `reader`, so the caller owns the Reader for the plan's lifetime.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});

} // namespace ninfer::models::gemma4
