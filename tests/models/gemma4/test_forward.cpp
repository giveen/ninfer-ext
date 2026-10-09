#include "core/arena.h"
#include "core/device.h"
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

        DeviceArena arena(gemma::sliding_layer_workspace_bytes(config));
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

        std::size_t ran = 0;
        for (std::size_t layer = 0; layer < model->weights().text.layers.size(); ++layer) {
            if (model->weights().text.layers[layer].mixer != gemma::MixerKind::SlidingAttention) {
                continue;
            }
            Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
            Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
            gemma::forward_sliding_layer(*model, layer, hidden_in, 0, arena, hidden_out,
                                         device.execution_view());
            std::swap(in, out);
            ++ran;
        }
        check(ran == 50, "ran " + std::to_string(ran) + " sliding layers");
        CUDA_CHECK(cudaDeviceSynchronize());

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

        // Routing: a global layer is not a sliding layer.
        bool refused = false;
        try {
            Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
            gemma::forward_sliding_layer(*model, 5, hidden_in, 0, arena, hidden_in,
                                         device.execution_view());
        } catch (const std::exception&) { refused = true; }
        check(refused, "a global layer was accepted by the sliding path");

        std::cout << "gemma4 forward: " << ran << " sliding layers, largest |h| = " << largest
                  << ", smallest |h| = " << smallest << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 forward: " << error.what() << '\n';
        return 1;
    }
}
