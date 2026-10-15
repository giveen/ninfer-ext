#include "core/device.h"
#include "models/gemma4/expected_weights.h"
#include "models/gemma4/load.h"
#include "models/gemma4/measurement.h"

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

        // The serving path's measurement-applicability identity: the same artifact answers the same
        // SHA-256 every time, and an empty one would be refused by the serving harness.
        const std::string signature = gemma::prefill_signature(*model);
        check(signature.size() == 64, "prefill signature is a 64-character SHA-256 digest");
        check(signature.find_first_not_of("0123456789abcdef") == std::string::npos,
              "prefill signature is lowercase hexadecimal");
        check(gemma::prefill_signature(*model) == signature, "prefill signature is deterministic");

        const auto& weights = model->weights();
        const auto& config  =model->config();

        // Every expectation below is one of the config's own members, read from the artifact, so
        // the dense 31B (60 layers, 833 parameters) and the MoE 26B-A4B (30 layers, 12118) are held
        // to the same assertions. The declared layer count is the oracle for the layers the loader
        // binds, and `expected_weights` is the count the config's structure implies.
        const std::uint64_t expected_weights =
            ninfer::test::gemma4_expected_parameter_count(config);
        check(weights.text.layers.size() == config.num_hidden_layers, "layer count");
        check(model->weight_data().size() == expected_weights,
              "bound weight count is " + std::to_string(model->weight_data().size()) +
                  ", the structure has " + std::to_string(expected_weights));

        // A materialized view is a non-empty region list over the backing store; a view with no
        // parts would mean a parameter resolved to nothing.
        std::size_t resident = 0;
        for (const auto& bound : model->weight_data()) {
            if (bound.view.parts.empty()) {
                std::cerr << "gemma4 model: " << bound.name << " resolved to no region\n";
                ++failures;
                continue;
            }
            ++resident;
        }
        check(resident == expected_weights, "resident weight count is " + std::to_string(resident) +
                                                ", the structure has " +
                                                std::to_string(expected_weights));

        const auto& embedding = model->weight(weights.text.token_embedding);
        check(embedding.name == "text/token_embedding", "token embedding name");

        // The three accessor forms the Ops need: a norm weight as a plain tensor, a projection as a
        // native weight input, and the layer scalar as a host float.
        const auto norm = model->tensor(weights.text.layers[0].input_norm);
        check(norm.ne[0] == static_cast<std::int64_t>(config.hidden_size) && norm.dtype == DType::BF16, "input norm tensor shape");
        const auto projection = model->input(weights.text.layers[0].attention.query);
        check(projection.policy == ops::LinearPolicy::A16Only, "query projection policy");

        // The artifact's last global layer is the layer the global geometry checks run on -- the
        // 31B's 59, the 26B-A4B's 29 -- resolved from the config's own kinds rather than assumed.
        std::size_t global_layer = 0;
        for (std::size_t layer = 0; layer < weights.text.layers.size(); ++layer) {
            if (!config.sliding_attention(layer)) { global_layer = layer; }
        }

        // The three attention input projections are slices of one fused FP8 parent. A row slice of
        // a RowScale parent is a whole operand, so each prepares to its own shape: the layer's own
        // geometry gives the query rows and, where the layer stores them, the key/value rows.
        {
            const auto& attention = weights.text.layers[0].attention;
            const auto& geometry  = config.geometry(0);
            const std::uint64_t query_rows =
                static_cast<std::uint64_t>(config.num_attention_heads) * geometry.head_dim;
            const std::uint64_t key_rows =
                static_cast<std::uint64_t>(geometry.num_key_value_heads) * geometry.head_dim;
            const auto query      = ops::prepare_linear_weight(model->input(attention.query));
            const auto key        = ops::prepare_linear_weight(model->input(attention.key));
            const auto value      = ops::prepare_linear_weight(model->input(attention.value));
            check(static_cast<std::uint64_t>(query.weight.n) == query_rows &&
                      static_cast<std::uint64_t>( query.weight.k) == config.hidden_size,
                  "sliced query projection shape");
            check(static_cast<std::uint64_t>(key.weight.n) == key_rows &&
                      static_cast<std::uint64_t>( key.weight.k) == config.hidden_size, "sliced key projection shape");
            check(static_cast<std::uint64_t>(value.weight.n) == key_rows &&
                      static_cast<std::uint64_t>( value.weight.k) == config.hidden_size,
                  "sliced value projection shape");

            // The global layers slice the wider parent, at the global head width.
            const auto& global      = weights.text.layers[global_layer].attention;
            const auto& global_geometry = config.geometry(global_layer);
            const std::uint64_t global_query_rows =
                static_cast<std::uint64_t>(config.num_attention_heads) * global_geometry.head_dim;
            const std::uint64_t global_key_rows =
                static_cast<std::uint64_t>(global_geometry.num_key_value_heads) *
                global_geometry.head_dim;
            const auto global_query = ops::prepare_linear_weight(model->input(global.query));
            const auto global_key   = ops::prepare_linear_weight(model->input(global.key));
            check(static_cast<std::uint64_t>(global_query.weight.n) == global_query_rows, "sliced global query projection shape");
            check(static_cast<std::uint64_t>(global_key.weight.n) == global_key_rows, "sliced global key projection shape");
        }
        const float scalar = model->layer_scalar(weights.text.layers[0].layer_scalar);
        check(std::isfinite(scalar), "layer scalar is finite");
        check(scalar != 1.0F, "layer scalar is a learned value, not the neutral one");
        const auto& scaled = model->weight(weights.text.layers[0].layer_scalar);
        check(scaled.name == "text/layers/0/layer_scalar", "layer scalar name");
        const auto& query = model->weight(weights.text.layers[0].attention.query);
        check(query.name == "text/layers/0/attention/query", "sliding query name");
        const auto& global_key = model->weight(weights.text.layers[global_layer].attention.key);
        check(global_key.name == "text/layers/" + std::to_string(global_layer) + "/attention/key", "global key name");
        check(weights.text.layers[global_layer].attention.value.index ==
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
