// Generates greedily through the model's own Program, so the whole pipeline runs end to end: the
// embedding, sixty cached decoder layers, the head, and an argmax. Tokenizing and detokenizing happen
// outside, so this needs no tokenizer.
//
//   NINFER_GEMMA_ARTIFACT  the artifact
//   NINFER_GEMMA_PROMPT    a file of int32 little-endian token ids
//   NINFER_GEMMA_STEPS     how many tokens to generate (default 8)
//
// Writes the prompt followed by the generated ids to /tmp/gemma_gen.out.i32.

#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"

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
    const char* artifact    = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (artifact == nullptr || *artifact == '\0' || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    const char* steps_text   = std::getenv("NINFER_GEMMA_STEPS");
    const std::int32_t steps = steps_text != nullptr ? std::stoi(steps_text) : 8;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        DeviceContext device;
        auto model = gemma::load_model(artifact, models::LoadOptions{}, device);
        const std::int32_t vocabulary =
            static_cast<std::int32_t>(model->config().vocab_size);

        std::vector<std::int32_t> ids = read_ids(prompt_path);
        if (ids.empty()) {
            std::cerr << "gemma4 generate: the prompt is empty\n";
            return 1;
        }
        const std::int32_t prompt_length = static_cast<std::int32_t>(ids.size());

        gemma::Program program(*model, prompt_length + steps + 8, 1, device);
        program.prefill(0, ids, device.execution_view());

        // Greedy: the model produces logits and sampling is the caller's business.
        const auto sample = [&] {
            std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(host.data(), program.logits().data,
                                  static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            float best          = -1.0e30F;
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

        for (std::int32_t step = 0; step < steps; ++step) {
            const std::int32_t winner = sample();
            ids.push_back(winner);
            program.decode(0, winner, device.execution_view());
        }
        if (program.position(0) != prompt_length + steps) {
            std::cerr << "gemma4 generate: the Program consumed " << program.position(0)
                      << " tokens, expected " << prompt_length + steps << '\n';
            return 1;
        }

        std::ofstream output("/tmp/gemma_gen.out.i32", std::ios::binary);
        output.write(reinterpret_cast<const char*>(ids.data()),
                     static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
        std::cout << "gemma4 generate: " << prompt_length << " prompt tokens, " << steps
                  << " generated through the Program, wrote /tmp/gemma_gen.out.i32\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 generate: " << error.what() << '\n';
        return 1;
    }
}
