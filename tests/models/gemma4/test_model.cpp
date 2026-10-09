#include "core/device.h"
#include "models/gemma4/load.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <string>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "gemma4 model: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_GEMMA_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT is not set\n";
        return 77;
    }

    try {
        DeviceContext device;
        auto model = gemma::load_model(path, models::LoadOptions{}, device);
        check(model != nullptr, "load_model returned nothing");

        const auto& weights = model->weights();
        check(model->config().num_hidden_layers == 60, "layer count");
        check(model->weight_data().size() == 833,
              "bound weight count is " + std::to_string(model->weight_data().size()));

        // A materialized view is a non-empty region list over the backing store; a view with no parts
        // would mean a parameter resolved to nothing.
        std::size_t resident = 0;
        for (const auto& bound : model->weight_data()) {
            if (bound.view.parts.empty()) {
                std::cerr << "gemma4 model: " << bound.name << " resolved to no region\n";
                ++failures;
                continue;
            }
            ++resident;
        }
        check(resident == 833, "resident weight count is " + std::to_string(resident));

        const auto& embedding = model->weight(weights.text.token_embedding);
        check(embedding.name == "text/token_embedding", "token embedding name");

        // The three accessor forms the Ops need: a norm weight as a plain tensor, a projection as a
        // native weight input, and the layer scalar as a host float.
        const auto norm = model->tensor(weights.text.layers[0].input_norm);
        check(norm.ne[0] == 5376 && norm.dtype == DType::BF16, "input norm tensor shape");
        const auto projection = model->input(weights.text.layers[0].attention.query);
        check(projection.policy == ops::LinearPolicy::A16Only, "query projection policy");

        const float scalar = model->layer_scalar(weights.text.layers[0].layer_scalar);
        check(std::isfinite(scalar), "layer scalar is finite");
        check(scalar != 1.0F, "layer scalar is a learned value, not the neutral one");
        const auto& scaled = model->weight(weights.text.layers[0].layer_scalar);
        check(scaled.name == "text/layers/0/layer_scalar", "layer scalar name");
        const auto& query = model->weight(weights.text.layers[0].attention.query);
        check(query.name == "text/layers/0/attention/query", "sliding query name");
        const auto& global_key = model->weight(weights.text.layers[59].attention.key);
        check(global_key.name == "text/layers/59/attention/key", "global key name");
        check(weights.text.layers[59].attention.value.index ==
                  std::numeric_limits<std::size_t>::max(),
              "global layer binds no value");

        std::cout << "gemma4 model: " << model->weight_data().size() << " weights resident, device "
                  << model->storage_stats().device_capacity_bytes << " bytes, "
                  << model->config().num_hidden_layers << " layers\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 model: " << error.what() << '\n';
        return 1;
    }
}
