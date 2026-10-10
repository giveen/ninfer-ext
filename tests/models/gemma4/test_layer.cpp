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
    const char* head_text   = std::getenv("NINFER_GEMMA_HEAD");
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

        // The reference can supply a realistic input: the embedding of a few tokens, which is what the
        // layer stack sees in service. White noise of the same magnitude drives the logits far past the
        // soft cap, where the head can only be checked by sign rather than by value.
        std::vector<std::uint16_t> provided;
        if (const char* input_path = std::getenv("NINFER_GEMMA_INPUT")) {
            std::ifstream source(input_path, std::ios::binary);
            const std::size_t count = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(hidden);
            std::vector<float> values(count);
            source.read(reinterpret_cast<char*>(values.data()),
                        static_cast<std::streamsize>(count * sizeof(float)));
            if (source.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
                std::cerr << "gemma4 layer: could not read " << input_path << '\n';
                return 1;
            }
            provided.resize(count);
            for (std::size_t index = 0; index < count; ++index) {
                std::uint32_t bits;
                std::memcpy(&bits, &values[index], sizeof(bits));
                bits += 0x7FFFU + ((bits >> 16) & 1U);
                provided[index] = static_cast<std::uint16_t>(bits >> 16);
            }
        }

        std::vector<std::uint16_t> inputs;
        std::vector<std::uint16_t> outputs;
        void* in  = nullptr;
        void* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, bytes));
        CUDA_CHECK(cudaMalloc(&out, bytes));

        gemma::KvCache cache;
        cache.configure(config, 64, 1);
        DeviceArena arena(gemma::layer_workspace_bytes(config, 1));
        for (std::int32_t position = 0; position < tokens; ++position) {
            const std::vector<std::uint16_t> state =
                provided.empty()
                    ? hidden_for(hidden, position)
                    : std::vector<std::uint16_t>(
                          provided.begin() + static_cast<std::ptrdiff_t>(position) * hidden,
                          provided.begin() + static_cast<std::ptrdiff_t>(position + 1) * hidden);
            inputs.insert(inputs.end(), state.begin(), state.end());
            CUDA_CHECK(cudaMemcpy(in, state.data(), bytes, cudaMemcpyHostToDevice));
            Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
            Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
            gemma::forward_layer(*model, layer, hidden_in, position, 1, cache, arena, hidden_out,
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

        // The head runs on the last token's layer output, which is dumped above, so the reference can
        // check it on exactly that input instead of propagating its own layer result into it.
        // Head-only mode: run the head on the provided hidden state and stop, so an outside
        // implementation's own stack output can be fed through this model's head alone.
        if (std::getenv("NINFER_GEMMA_HEAD_ONLY") != nullptr) {
            if (provided.empty()) {
                std::cerr << "gemma4 layer: head-only mode needs NINFER_GEMMA_INPUT\n";
                return 1;
            }
            // A batch needs its own buffers and its own scratch: the ones above are sized for the
            // single-token layer runs.
            const std::size_t vocabulary = config.vocab_size;
            void* head_in                = nullptr;
            void* logits                 = nullptr;
            CUDA_CHECK(cudaMalloc(&head_in, static_cast<std::size_t>(tokens) * bytes));
            CUDA_CHECK(cudaMalloc(&logits, static_cast<std::size_t>(vocabulary) *
                                               static_cast<std::size_t>(tokens) *
                                               sizeof(std::uint16_t)));
            CUDA_CHECK(cudaMemcpy(head_in, provided.data(),
                                  static_cast<std::size_t>(tokens) * bytes, cudaMemcpyHostToDevice));
            DeviceArena head_arena(gemma::layer_workspace_bytes(config, tokens));
            Tensor head_input(static_cast<std::uint8_t*>(head_in), DType::BF16, {hidden, tokens});
            Tensor logit_tensor(static_cast<std::uint8_t*>(logits), DType::BF16,
                                {static_cast<std::int32_t>(vocabulary), tokens});
            gemma::forward_head(*model, head_input, tokens, head_arena, logit_tensor,
                                device.execution_view());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<std::uint16_t> produced(static_cast<std::size_t>(vocabulary) *
                                                static_cast<std::size_t>(tokens));
            CUDA_CHECK(cudaMemcpy(produced.data(), logits,
                                  produced.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            // Written as per-column rows, one target per column, so the comparison is positional.
            std::ofstream dump("/tmp/gemma_head_only.f32", std::ios::binary);
            for (std::size_t column = 0; column < static_cast<std::size_t>(tokens); ++column) {
                for (std::size_t id = 0; id < vocabulary; ++id) {
                    const float value =
                        decode_bf16(produced[id + column * static_cast<std::size_t>(vocabulary)]);
                    dump.write(reinterpret_cast<const char*>(&value), sizeof(value));
                }
            }
            std::cout << "gemma4 layer: head-only over " << tokens << " tokens, wrote "
                      << "/tmp/gemma_head_only.f32\n";
            return 0;
        }

        if (head_text != nullptr && *head_text != '0') {
            const std::size_t vocabulary = config.vocab_size;
            void* logits                 = nullptr;
            CUDA_CHECK(cudaMalloc(&logits, vocabulary * sizeof(std::uint16_t)));
            Tensor head_input(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
            Tensor logit_tensor(static_cast<std::uint8_t*>(logits), DType::BF16,
                                {static_cast<std::int32_t>(vocabulary), 1});
            gemma::forward_head(*model, head_input, 1, arena, logit_tensor,
                                device.execution_view());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<std::uint16_t> produced(vocabulary);
            CUDA_CHECK(cudaMemcpy(produced.data(), logits, vocabulary * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            if (!write_f32("/tmp/gemma_head.out.f32", produced)) {
                std::cerr << "gemma4 layer: could not write the head dump\n";
                return 1;
            }
            CUDA_CHECK(cudaFree(logits));
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
