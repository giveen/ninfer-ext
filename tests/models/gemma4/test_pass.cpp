// Do a Program's passes and the same layers run one at a time compute the same thing?
//
// A Program runs its layers through `run_pass`, which hands `run_layers` one `PassSegment` and
// takes every position and ring slot out of the device `indices`. Anything a caller builds by
// calling `forward_layer` per layer has to agree with that, or a hand-rolled reference is measuring
// something else. The sweep over prompt length separates a positional or batching effect from a
// systematic one.
//
// The readbacks are stream-ordered: the ops run on the execution stream, and a synchronous
// `cudaMemcpy` is not ordered against work queued there, so a read issued that way returns whatever
// the buffer held when it happened to run -- which is a race, and it reads as a disagreement at the
// shortest prompt (the fastest pass) and as agreement at the longer ones.
//
// NINFER_GEMMA_ARTIFACT an artifact to run
// NINFER_GEMMA_PROMPT   I32 token ids

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

constexpr std::int32_t kPass = 512;

std::vector<std::int32_t> read_ids(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::int32_t> ids;
    std::int32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof(id))) ids.push_back(id);
    return ids;
}

float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded            = 0.0F;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

} // namespace

int main() {
    const char* path   = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt = std::getenv("NINFER_GEMMA_PROMPT");
    if (path == nullptr || prompt == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_PROMPT must be set\n";
        return 77;
    }
    try {
        DeviceContext device;
        auto model                      = gemma::load_model(path, models::LoadOptions{}, device);
        const gemma::TextConfig& config = model->config();
        const std::int32_t h            = static_cast<std::int32_t>(config.hidden_size);
        const std::int32_t v            = static_cast<std::int32_t>(config.vocab_size);
        const cudaStream_t stream       = device.execution_view().stream;

        std::vector<std::int32_t> ids = read_ids(prompt);
        if (ids.size() > static_cast<std::size_t>(kPass)) ids.resize(kPass);
        if (ids.size() < 2) throw std::runtime_error("the prompt needs at least two tokens");

        const auto bf16 = [](std::int32_t rows, std::int32_t columns) {
            void* pointer = nullptr;
            CUDA_CHECK(cudaMalloc(&pointer, static_cast<std::size_t>(rows) * columns * 2));
            return Tensor(static_cast<std::uint8_t*>(pointer), DType::BF16, {rows, columns});
        };
        const auto read_ordered = [&](void* host, const void* source, std::size_t bytes) {
            CUDA_CHECK(cudaMemcpyAsync(host, source, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        };

        int failures = 0;
        for (const std::int32_t p : {1, 2, 8, 64, static_cast<std::int32_t>(ids.size())}) {
            if (p > static_cast<std::int32_t>(ids.size())) continue;
            const auto tokens = static_cast<std::size_t>(p);

            gemma::Program program(*model, p + 2, 1, device, 0);
            program.prefill(0, std::span(ids.data(), tokens), device.execution_view());
            std::vector<std::uint16_t> from_program(static_cast<std::size_t>(v));
            read_ordered(from_program.data(), program.logits().data,
                         static_cast<std::size_t>(v) * sizeof(std::uint16_t));

            gemma::KvCache cache;
            cache.configure(config, p + 1, 0);
            cache.allocate();
            DeviceArena arena(gemma::layer_workspace_bytes(config, kPass));
            Tensor state[2]         = {bf16(h, kPass), bf16(h, kPass)};
            Tensor manual_logits    = bf16(v, 1);
            std::int32_t* id_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&id_buffer, tokens * sizeof(std::int32_t)));
            CUDA_CHECK(cudaMemcpyAsync(id_buffer, ids.data(), tokens * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            const Tensor id_tensor(id_buffer, DType::I32, {p});
            Tensor embedded(state[0].data, DType::BF16, {h, p});
            ops::embedding(id_tensor,
                           native_weight(model->weight(model->weights().text.token_embedding).view),
                           config.embedding_scale, embedded, stream);
            int in = 0;
            {
                DeviceArena::Scope scope = arena.scope();
                const gemma::PassIndices indices =
                    gemma::write_pass_indices(cache, 0, p, arena, stream);
                for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
                    Tensor hidden_in(state[in].data, DType::BF16, {h, p});
                    Tensor hidden_out(state[1 - in].data, DType::BF16, {h, p});
                    gemma::forward_layer(*model, layer, hidden_in, 0, p, indices, cache, arena,
                                         hidden_out, device.execution_view());
                    in = 1 - in;
                }
            }
            {
                DeviceArena::Scope scope = arena.scope();
                Tensor final_column(static_cast<std::uint8_t*>(state[in].data) +
                                        static_cast<std::size_t>(p - 1) *
                                            static_cast<std::size_t>(h) * 2,
                                    DType::BF16, {h, 1});
                gemma::forward_head(*model, final_column, 1, arena, manual_logits,
                                    device.execution_view(), nullptr);
            }
            std::vector<std::uint16_t> from_manual(static_cast<std::size_t>(v));
            read_ordered(from_manual.data(), manual_logits.data,
                         static_cast<std::size_t>(v) * sizeof(std::uint16_t));

            std::size_t differing      = 0;
            float worst                = 0.0F;
            std::size_t argmax_program = 0, argmax_manual = 0;
            for (std::size_t i = 0; i < from_manual.size(); ++i) {
                if (from_manual[i] != from_program[i]) ++differing;
                worst = std::max(
                    worst, std::fabs(decode_bf16(from_manual[i]) - decode_bf16(from_program[i])));
                if (decode_bf16(from_program[i]) > decode_bf16(from_program[argmax_program])) {
                    argmax_program = i;
                }
                if (decode_bf16(from_manual[i]) > decode_bf16(from_manual[argmax_manual])) {
                    argmax_manual = i;
                }
            }
            std::cout << "  p=" << p << ": " << differing << " of " << from_manual.size()
                      << " logits differ, worst " << worst << ", argmax " << argmax_program
                      << (argmax_program == argmax_manual ? " (same)" : " (DIFFERENT)") << '\n';
            failures += differing != 0 ? 1 : 0;

            CUDA_CHECK(cudaFree(id_buffer));
            CUDA_CHECK(cudaFree(state[0].data));
            CUDA_CHECK(cudaFree(state[1].data));
            CUDA_CHECK(cudaFree(manual_logits.data));
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 pass: " << error.what() << '\n';
        return 1;
    }
}
