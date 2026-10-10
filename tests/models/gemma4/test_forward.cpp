#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/load.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "gemma4 forward: " << message << '\n';
        ++failures;
    }
}

// The row layout the cache uses, so the test can address a slot without duplicating the ring rule.
struct SlotLayout {
    std::int32_t tokens;
    std::size_t row_bytes;
};

SlotLayout layout_of(const gemma::TextConfig& config, std::size_t layer, std::int32_t capacity) {
    const bool global = !config.sliding_attention(layer);
    const std::int32_t head_dim =
        static_cast<std::int32_t>(global ? config.global.shared.head_dim : config.sliding.head_dim);
    const std::int32_t heads = static_cast<std::int32_t>(
        global ? config.global.shared.num_key_value_heads : config.sliding.num_key_value_heads);
    const std::int32_t width =
        global ? head_dim + 2 * static_cast<std::int32_t>(config.global.rope_angles) : head_dim;
    const std::int32_t tokens =
        global ? capacity : static_cast<std::int32_t>(config.sliding_window);
    return {tokens, static_cast<std::size_t>(width) * heads * sizeof(std::uint16_t)};
}

void scribble_slot(void* base, const SlotLayout& layout, std::int32_t slot) {
    CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(base) +
                              static_cast<std::size_t>(slot) * layout.row_bytes,
                          0x7F, layout.row_bytes));
}

float decode_bf16(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_GEMMA_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        DeviceContext device;
        auto model = gemma::load_model(path, models::LoadOptions{}, device);
        const auto& config        = model->config();
        const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
        const std::size_t bytes   = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);

        // The plan's analysis of this checkpoint records the layer scalars as 0.44 to 0.99 with layer
        // 59 at 0.0364. They scale the whole residual stream once per layer, so they are what explains
        // the hidden state's final magnitude; the load path is wrong if they leave that band.
        double scalar_min = 1.0e30;
        double scalar_max = 0.0;
        for (const auto& layer : model->weights().text.layers) {
            const float value = model->layer_scalar(layer.layer_scalar);
            check(std::isfinite(value) && value != 0.0F, "a layer scalar is not a usable value");
            const double magnitude = std::fabs(static_cast<double>(value));
            scalar_min           = std::min(scalar_min, magnitude);
            scalar_max           = std::max(scalar_max, magnitude);
        }
        check(scalar_min > 0.01 && scalar_max < 1.0,
              "layer scalars left the expected band: " + std::to_string(scalar_min) + " to " +
                  std::to_string(scalar_max));

        DeviceArena arena(gemma::layer_workspace_bytes(config, 1));
        void* in  = nullptr;
        void* out = nullptr;
        check(cudaMalloc(&in, bytes) == cudaSuccess && cudaMalloc(&out, bytes) == cudaSuccess,
              "could not allocate the hidden state");

        // BF16 is the upper half of the FP32 pattern, rounded to nearest even. The magnitude is the
        // one the embedding produces, so every weight of every sliding layer participates.
        std::vector<std::uint16_t> host(hidden);
        std::mt19937 generator(4242U);
        std::normal_distribution<float> distribution(0.0F, 2.0F);
        for (auto& value : host) {
            const float sample = distribution(generator) * std::sqrt(static_cast<float>(hidden));
            std::uint32_t bits;
            std::memcpy(&bits, &sample, sizeof(bits));
            bits += 0x7FFFU + ((bits >> 16) & 1U);
            value = static_cast<std::uint16_t>(bits >> 16);
        }
        check(cudaMemcpy(in, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess,
              "could not upload the hidden state");

        std::size_t sliding = 0;
        std::size_t global  = 0;
        for (const auto& layer : model->weights().text.layers) {
            (layer.mixer == gemma::MixerKind::SlidingAttention ? sliding : global) += 1;
        }
        check(sliding == 50 && global == 10,
              "layer kinds: " + std::to_string(sliding) + " sliding, " + std::to_string(global) +
                  " global");

        // Every layer of the stack, in order, one arena, dispatching on each layer's attention kind.
        // Sliding layers hold the window in a ring, so the cache is sized only for the global layers'
        // tokens; the plan's paging is what lifts that bound.
        const std::size_t total          = model->weights().text.layers.size();
        const std::int32_t capacity      = 64;
        const std::int32_t tokens_run    = 8;
        gemma::KvCache cache;
        cache.configure(config, capacity);

        const auto run_token = [&](std::int32_t position) {
            // Each token starts from the embedding's hidden state, so a pass is a function of that
            // state and the cache alone.
            CUDA_CHECK(cudaMemcpy(in, host.data(), bytes, cudaMemcpyHostToDevice));
            for (std::size_t index = 0; index < total; ++index) {
                Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
                Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
                gemma::forward_layer(*model, index, hidden_in, position, 1, cache, arena,
                                     hidden_out, device.execution_view());
                std::swap(in, out);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        };
        const auto read_back = [&] {
            std::vector<std::uint16_t> state(hidden);
            CUDA_CHECK(cudaMemcpy(state.data(), in, bytes, cudaMemcpyDeviceToHost));
            return state;
        };

        for (std::int32_t position = 0; position < tokens_run; ++position) run_token(position);
        check(total == 60, "ran " + std::to_string(total) + " layers");
        const std::vector<std::uint16_t> rehearsed = read_back();

        // The cache makes a pass a function of its state: replaying the last token over the same rows
        // and the same input must reproduce it bit for bit, which is what the two checks below need.
        run_token(tokens_run - 1);
        check(read_back() == rehearsed, "replaying the last token changed its hidden state");

        // Slots no token has written are invisible. Filling every one of them with a large positive
        // pattern must leave the last token's hidden state untouched, for the ring and for the rows.
        for (std::size_t layer = 0; layer < total; ++layer) {
            const SlotLayout layout = layout_of(config, layer, capacity);
            for (std::int32_t slot = tokens_run; slot < layout.tokens; ++slot) {
                scribble_slot(cache.keys(layer).data, layout, slot);
                if (config.sliding_attention(layer)) {
                    scribble_slot(cache.values(layer).data, layout, slot);
                }
            }
        }
        run_token(tokens_run - 1);
        check(read_back() == rehearsed, "an unwritten cache slot changed the last token");

        // The control: slot 3 holds a token the window still sees, so corrupting it must change the
        // result. Without this, the check above would also pass if the cache were ignored entirely.
        for (std::size_t layer = 0; layer < total; ++layer) {
            const SlotLayout layout = layout_of(config, layer, capacity);
            scribble_slot(cache.keys(layer).data, layout, 3);
            if (config.sliding_attention(layer)) {
                scribble_slot(cache.values(layer).data, layout, 3);
            }
        }
        run_token(tokens_run - 1);
        check(read_back() != rehearsed, "a written cache slot was not attended");

        std::vector<std::uint16_t> result(hidden);
        check(cudaMemcpy(result.data(), in, bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
              "could not read the hidden state back");

        double largest  = 0.0;
        double smallest = 1.0e30;
        for (const auto value : result) {
            const float decoded = decode_bf16(value);
            check(std::isfinite(decoded), "non-finite hidden state");
            const double magnitude = std::fabs(static_cast<double>(decoded));
            largest                = std::max(largest, magnitude);
            smallest               = std::min(smallest, magnitude);
        }
        check(largest < 1.0e4, "hidden state magnitude blew up: " + std::to_string(largest));
        check(largest > 0.0, "hidden state is all zeros");

        // The dispatcher rejects a layer index outside the stack.
        bool refused = false;
        try {
            Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
            gemma::forward_layer(*model, total, hidden_in, 0, 1, cache, arena, hidden_in,
                                 device.execution_view());
        } catch (const std::exception&) { refused = true; }
        check(refused, "a layer index outside the stack was accepted");

        std::cout << "gemma4 forward: " << total << " layers (" << sliding << " sliding, " << global
                  << " global) over " << tokens_run << " cached tokens, largest |h| = " << largest
                  << ", smallest |h| = " << smallest << ", layer scalars in [" << scalar_min << ", "
                  << scalar_max << "], layer 59 = "
                  << std::fabs(model->layer_scalar(model->weights().text.layers[59].layer_scalar))
                  << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 forward: " << error.what() << '\n';
        return 1;
    }
}
