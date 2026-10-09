// Generates greedily from a prompt given as token ids, so the whole pipeline runs end to end: the
// embedding, sixty cached decoder layers, the head, and an argmax. Tokenizing and detokenizing happen
// outside, so this needs no tokenizer.
//
//   NINFER_GEMMA_ARTIFACT  the artifact
//   NINFER_GEMMA_PROMPT    a file of int32 little-endian token ids
//   NINFER_GEMMA_STEPS     how many tokens to generate (default 8)
//
// Writes the prompt followed by the generated ids to /tmp/gemma_gen.out.i32.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/load.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

std::vector<std::int32_t> read_ids(const char* path) {
    std::ifstream source(path, std::ios::binary | std::ios::ate);
    const std::streamsize bytes = source.tellg();
    source.seekg(0);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
    source.read(reinterpret_cast<char*>(ids.data()), bytes);
    return ids;
}

float decode_bf16(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (artifact == nullptr || *artifact == '\0' || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    const char* steps_text = std::getenv("NINFER_GEMMA_STEPS");
    const std::int32_t steps = steps_text != nullptr ? std::stoi(steps_text) : 8;
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
        const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);

        std::vector<std::int32_t> ids = read_ids(prompt_path);
        if (ids.empty()) {
            std::cerr << "gemma4 generate: the prompt is empty\n";
            return 1;
        }
        const std::int32_t prompt_length = static_cast<std::int32_t>(ids.size());

        gemma::KvCache cache;
        cache.configure(config, prompt_length + steps + 8);
        DeviceArena arena(gemma::layer_workspace_bytes(config));

        void* id_buffer = nullptr;
        void* state_a   = nullptr;
        void* state_b   = nullptr;
        void* logits    = nullptr;
        CUDA_CHECK(cudaMalloc(&id_buffer, sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&state_a, static_cast<std::size_t>(hidden) * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMalloc(&state_b, static_cast<std::size_t>(hidden) * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMalloc(&logits, static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t)));

        const cudaStream_t stream = device.execution_view().stream;
        const Weight table =
            native_weight(model->weight(model->weights().text.token_embedding).view);
        // The embedding scale the converter derives: the square root of the hidden size.
        const float scale = std::sqrt(static_cast<float>(hidden));

        const auto embed = [&](std::int32_t id) {
            CUDA_CHECK(cudaMemcpyAsync(id_buffer, &id, sizeof(id), cudaMemcpyHostToDevice, stream));
            Tensor id_tensor(static_cast<std::uint8_t*>(id_buffer), DType::I32, {1});
            Tensor out(static_cast<std::uint8_t*>(state_a), DType::BF16, {hidden, 1});
            ops::embedding(id_tensor, table, scale, out, stream);
        };
        // Sixty layers, so an even number of swaps leaves the result where it started.
        const auto run_stack = [&](std::int32_t position) {
            void* in  = state_a;
            void* out = state_b;
            for (std::size_t layer = 0; layer < model->weights().text.layers.size(); ++layer) {
                Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
                Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
                gemma::forward_layer(*model, layer, hidden_in, position, cache, arena, hidden_out,
                                     device.execution_view());
                std::swap(in, out);
            }
        };
        const auto sample = [&] {
            Tensor head_in(static_cast<std::uint8_t*>(state_a), DType::BF16, {hidden, 1});
            Tensor logit_tensor(static_cast<std::uint8_t*>(logits), DType::BF16, {vocabulary, 1});
            gemma::forward_head(*model, head_in, arena, logit_tensor, device.execution_view());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary));
            CUDA_CHECK(cudaMemcpy(host.data(), logits,
                                  static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            float best      = -1.0e30F;
            std::int32_t winner = 0;
            for (std::int32_t id = 0; id < vocabulary; ++id) {
                const float value = decode_bf16(host[static_cast<std::size_t>(id)]);
                if (value > best) {
                    best   = value;
                    winner = id;
                }
            }
            std::cout << "  token " << winner << " (greedy logit " << best << ")\n";
            return winner;
        };

        for (std::int32_t index = 0; index < prompt_length; ++index) {
            embed(ids[static_cast<std::size_t>(index)]);
            run_stack(index);
        }
        for (std::int32_t step = 0; step < steps; ++step) {
            const std::int32_t winner = sample();
            ids.push_back(winner);
            embed(winner);
            run_stack(prompt_length + step);
        }

        std::ofstream output("/tmp/gemma_gen.out.i32", std::ios::binary);
        output.write(reinterpret_cast<const char*>(ids.data()),
                     static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
        std::cout << "gemma4 generate: " << prompt_length << " prompt tokens, " << steps
                  << " generated, wrote /tmp/gemma_gen.out.i32\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 generate: " << error.what() << '\n';
        return 1;
    }
}
