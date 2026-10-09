// Dumps one decoder layer's input and output as FP32 so an independent implementation can be checked
// against the engine on exactly the same input. The input is the same synthetic hidden state the
// forward test uses, written back as the FP32 values the BF16 storage holds.

#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/load.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

float decode_bf16(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

bool write_f32(const std::string& path, const std::vector<std::uint16_t>& values) {
    std::ofstream out(path, std::ios::binary);
    for (const std::uint16_t value : values) {
        const float decoded = decode_bf16(value);
        out.write(reinterpret_cast<const char*>(&decoded), sizeof(decoded));
    }
    return static_cast<bool>(out);
}

// One deterministic hidden state per position, so a multi-token run exercises the attention instead of
// attending over a single repeated key.
std::vector<std::uint16_t> hidden_for(std::int32_t hidden, std::int32_t position) {
    std::vector<std::uint16_t> state(static_cast<std::size_t>(hidden));
    std::mt19937 generator(4242U + static_cast<unsigned>(position));
    std::normal_distribution<float> distribution(0.0F, 2.0F);
    for (auto& value : state) {
        const float sample = distribution(generator) * std::sqrt(static_cast<float>(hidden));
        std::uint32_t bits;
        std::memcpy(&bits, &sample, sizeof(bits));
        bits += 0x7FFFU + ((bits >> 16) & 1U);
        value = static_cast<std::uint16_t>(bits >> 16);
    }
    return state;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT is not set\n";
        return 77;
    }
    const char* layer_text  = std::getenv("NINFER_GEMMA_LAYER");
    const char* tokens_text = std::getenv("NINFER_GEMMA_TOKENS");
    const std::size_t layer = layer_text != nullptr ? std::stoul(layer_text) : 0;
    const std::int32_t tokens = tokens_text != nullptr ? std::stoi(tokens_text) : 1;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        DeviceContext device;
        auto model = gemma::load_model(artifact, models::LoadOptions{}, device);
        const auto& config        = model->config();
        const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
        const std::size_t bytes   = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);

        std::vector<std::uint16_t> inputs;
        std::vector<std::uint16_t> outputs;
        void* in  = nullptr;
        void* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, bytes));
        CUDA_CHECK(cudaMalloc(&out, bytes));

        gemma::KvCache cache;
        cache.configure(config, 64);
        DeviceArena arena(gemma::layer_workspace_bytes(config));
        for (std::int32_t position = 0; position < tokens; ++position) {
            const std::vector<std::uint16_t> state = hidden_for(hidden, position);
            inputs.insert(inputs.end(), state.begin(), state.end());
            CUDA_CHECK(cudaMemcpy(in, state.data(), bytes, cudaMemcpyHostToDevice));
            Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
            Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
            gemma::forward_layer(*model, layer, hidden_in, position, cache, arena, hidden_out,
                                 device.execution_view());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<std::uint16_t> produced(hidden);
            CUDA_CHECK(cudaMemcpy(produced.data(), out, bytes, cudaMemcpyDeviceToHost));
            outputs.insert(outputs.end(), produced.begin(), produced.end());
        }

        if (!write_f32("/tmp/gemma_layer.in.f32", inputs) ||
            !write_f32("/tmp/gemma_layer.out.f32", outputs)) {
            std::cerr << "gemma4 layer: could not write the dump\n";
            return 1;
        }
        std::cout << "gemma4 layer: layer " << layer << ", "
                  << (config.sliding_attention(layer) ? "sliding" : "global") << ", " << tokens
                  << " tokens, wrote /tmp/gemma_layer.in.f32 and /tmp/gemma_layer.out.f32\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 layer: " << error.what() << '\n';
        return 1;
    }
}
