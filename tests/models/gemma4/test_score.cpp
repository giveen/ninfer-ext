// Scores a token sequence the way ninfer-perplexity does: the Program returns the natural
// log-probability of every target token, and this reports the mean negative log-likelihood and the
// resulting perplexity. It also writes the per-position log-probabilities so the independent reference
// can be compared entry by entry rather than only through the aggregate.
//
//   NINFER_GEMMA_ARTIFACT  the artifact
//   NINFER_GEMMA_PROMPT    a file of int32 little-endian token ids
//   NINFER_GEMMA_FIRST_TARGET  first scored position (default 1)

#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
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

} // namespace

int main() {
    const char* artifact    = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt_path = std::getenv("NINFER_GEMMA_PROMPT");
    if (artifact == nullptr || *artifact == '\0' || prompt_path == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    const char* first_text   = std::getenv("NINFER_GEMMA_FIRST_TARGET");
    const std::int32_t first = first_text != nullptr ? std::stoi(first_text) : 1;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        DeviceContext device;
        auto model = gemma::load_model(artifact, models::LoadOptions{}, device);
        const std::vector<std::int32_t> ids = read_ids(prompt_path);

        gemma::Program program(*model, static_cast<std::int32_t>(ids.size()) + 8, device);
        const std::vector<float> logprobs =
            program.causal_score(ids, first, device.execution_view());

        double total = 0.0;
        for (const float logprob : logprobs) {
            if (!std::isfinite(logprob)) {
                std::cerr << "gemma4 score: a non-finite log-probability\n";
                return 1;
            }
            total -= static_cast<double>(logprob);
        }
        const double mean_nll = total / static_cast<double>(logprobs.size());
        std::cout << "gemma4 score: " << ids.size() << " tokens, " << logprobs.size()
                  << " scored from " << first << ", mean NLL " << mean_nll << ", perplexity "
                  << std::exp(mean_nll) << '\n';
        std::cout << "  first log-probabilities:";
        for (std::size_t index = 0; index < std::min<std::size_t>(logprobs.size(), 6); ++index) {
            std::cout << ' ' << logprobs[index];
        }
        std::cout << '\n';

        std::ofstream output("/tmp/gemma_score.out.f32", std::ios::binary);
        output.write(reinterpret_cast<const char*>(logprobs.data()),
                     static_cast<std::streamsize>(logprobs.size() * sizeof(float)));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 score: " << error.what() << '\n';
        return 1;
    }
}
