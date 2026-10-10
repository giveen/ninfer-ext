// The assistant drafter against its reference.
//
// The target runs a real prompt through the same layer stack the Program uses, then the drafter takes
// a few greedy steps from the target's state. Every input the drafter read (the anchor, the target's
// post-final-norm state, the shared sliding ring and global rows) and every output it wrote (logits
// and projected state per step) is dumped, and tools/verify/gemma4_draft_reference.py runs
// transformers' Gemma4AssistantForCausalLM in FP32 on exactly those inputs to compare.
//
// NINFER_GEMMA_ARTIFACT  an artifact converted with --components text,mtp
// NINFER_GEMMA_PROMPT    I32 token ids
// NINFER_GEMMA_DRAFT_DUMP the directory to write

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/load.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

constexpr std::int32_t kSteps = 4;
constexpr std::int32_t kPass  = 512;

std::vector<std::int32_t> read_ids(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::int32_t> ids;
    std::int32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof(id))) ids.push_back(id);
    return ids;
}

void dump(const std::filesystem::path& file, const void* device, std::size_t bytes) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<char> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    std::ofstream(file, std::ios::binary).write(host.data(), static_cast<std::streamsize>(bytes));
}

float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

} // namespace

int main() {
    const char* path   = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* prompt = std::getenv("NINFER_GEMMA_PROMPT");
    const char* out    = std::getenv("NINFER_GEMMA_DRAFT_DUMP");
    if (path == nullptr || prompt == nullptr || out == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT, NINFER_GEMMA_PROMPT and NINFER_GEMMA_DRAFT_DUMP "
                     "must be set\n";
        return 77;
    }
    try {
        DeviceContext device;
        models::LoadOptions options;
        options.speculative = SpeculativeBackend::Mtp;
        auto model           = gemma::load_model(path, options, device);
        const auto& config   = model->config();
        const auto& draft    = model->draft_config().value();
        const std::int32_t h = static_cast<std::int32_t>(config.hidden_size);
        const std::int32_t v = static_cast<std::int32_t>(config.vocab_size);
        const cudaStream_t stream = device.execution_view().stream;

        const std::vector<std::int32_t> ids = read_ids(prompt);
        const std::int32_t p = static_cast<std::int32_t>(ids.size());
        if (p < 2) throw std::runtime_error("the prompt needs at least two tokens");

        gemma::KvCache cache;
        cache.configure(config, p + kSteps + 1, kSteps);
        cache.allocate();
        DeviceArena arena(gemma::layer_workspace_bytes(config, kPass));
        const auto bf16 = [](std::int32_t rows, std::int32_t columns) {
            void* pointer = nullptr;
            CUDA_CHECK(cudaMalloc(&pointer, static_cast<std::size_t>(rows) * columns * 2));
            return Tensor(static_cast<std::uint8_t*>(pointer), DType::BF16, {rows, columns});
        };
        Tensor state[2] = {bf16(h, kPass), bf16(h, kPass)};
        Tensor normed   = bf16(h, 1);
        Tensor logits   = bf16(v, 1);
        Tensor input    = bf16(2 * h, 1);
        Tensor projected = bf16(h, 1);
        std::int32_t* id_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&id_buffer, kPass * sizeof(std::int32_t)));
        const Weight table =
            native_weight(model->weight(model->weights().text.token_embedding).view);

        // The target's prefill, in the Program's passes.
        std::int32_t last = 0;
        for (std::int32_t first = 0; first < p; first += kPass) {
            const std::int32_t tokens = std::min(kPass, p - first);
            CUDA_CHECK(cudaMemcpy(id_buffer, ids.data() + first, tokens * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice));
            const Tensor id_tensor(id_buffer, DType::I32, {tokens});
            Tensor embedded(state[0].data, DType::BF16, {h, tokens});
            ops::embedding(id_tensor, table, config.embedding_scale, embedded, stream);
            int in = 0;
            for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
                Tensor hidden_in(state[in].data, DType::BF16, {h, tokens});
                Tensor hidden_out(state[1 - in].data, DType::BF16, {h, tokens});
                gemma::forward_layer(*model, layer, hidden_in, first, tokens, cache, arena,
                                     hidden_out, device.execution_view());
                in = 1 - in;
            }
            last = tokens;
            if (first + tokens == p) {
                const Tensor final_column(static_cast<std::uint8_t*>(state[in].data) +
                                              static_cast<std::size_t>(tokens - 1) * h * 2,
                                          DType::BF16, {h, 1});
                gemma::forward_head(*model, final_column, 1, arena, logits, device.execution_view(),
                                    &normed);
            }
        }
        (void)last;

        // The anchor is the target's greedy token at p, which no pass has consumed.
        Tensor chosen(id_buffer, DType::I32, {1});
        ops::argmax(logits, chosen, v, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::int32_t> tokens(1);
        CUDA_CHECK(cudaMemcpy(tokens.data(), id_buffer, sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        {
            std::vector<std::uint16_t> host(static_cast<std::size_t>(v));
            CUDA_CHECK(cudaMemcpy(host.data(), logits.data, host.size() * 2, cudaMemcpyDeviceToHost));
            float largest = -1.0e30F;
            std::size_t non_finite = 0;
            for (const auto value : host) {
                const float decoded = decode_bf16(value);
                if (!std::isfinite(decoded)) ++non_finite;
                else largest = std::max(largest, decoded);
            }
            if (non_finite != 0 || !(largest > -1.0e30F)) {
                throw std::runtime_error("the target's logits are not finite (" +
                                         std::to_string(non_finite) + " non-finite)");
            }
        }

        const std::filesystem::path dir(out);
        std::filesystem::create_directories(dir);
        dump(dir / "hidden0.bf16", normed.data, static_cast<std::size_t>(h) * 2);

        Tensor embedded(input.data, DType::BF16, {h, 1});
        Tensor carried(static_cast<std::uint8_t*>(input.data) + static_cast<std::size_t>(h) * 2,
                       DType::BF16, {h, 1});
        int failures = 0;
        for (std::int32_t step = 0; step < kSteps; ++step) {
            const Tensor token(id_buffer, DType::I32, {1});
            ops::embedding(token, table, config.embedding_scale, embedded, stream);
            CUDA_CHECK(cudaMemcpyAsync(carried.data, step == 0 ? normed.data : projected.data,
                                       static_cast<std::size_t>(h) * 2, cudaMemcpyDeviceToDevice,
                                       stream));
            gemma::forward_draft(*model, input, p, cache, arena, logits, projected,
                                 device.execution_view());
            dump(dir / ("step" + std::to_string(step) + "_logits.bf16"), logits.data,
                 static_cast<std::size_t>(v) * 2);
            dump(dir / ("step" + std::to_string(step) + "_hidden.bf16"), projected.data,
                 static_cast<std::size_t>(h) * 2);
            ops::argmax(logits, chosen, v, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::int32_t next = 0;
            CUDA_CHECK(cudaMemcpy(&next, id_buffer, sizeof(next), cudaMemcpyDeviceToHost));
            if (next < 0 || next >= v) ++failures;
            tokens.push_back(next);
            std::vector<std::uint16_t> host(static_cast<std::size_t>(v));
            CUDA_CHECK(cudaMemcpy(host.data(), logits.data, host.size() * 2, cudaMemcpyDeviceToHost));
            for (const auto value : host) {
                if (!std::isfinite(decode_bf16(value))) {
                    ++failures;
                    break;
                }
            }
        }

        // The caches the drafter read: the target's last sliding ring and its last global rows.
        const std::size_t sliding = draft.target_sliding_layer;
        const std::size_t global  = draft.target_global_layer;
        const std::int32_t ring   = cache.ring_tokens();
        const std::size_t d       = config.sliding.head_dim;
        const std::size_t hkv     = config.sliding.num_key_value_heads;
        const std::size_t width   = config.global.shared.head_dim + 2 * config.global.rope_angles;
        const std::size_t ghkv    = config.global.shared.num_key_value_heads;
        const gemma::KvCache& reader = cache;
        dump(dir / "ring_keys.bf16", reader.keys(sliding).data, d * hkv * ring * 2);
        dump(dir / "ring_values.bf16", reader.values(sliding).data, d * hkv * ring * 2);
        dump(dir / "ring_positions.i32", reader.positions(sliding).data, ring * 4u);
        dump(dir / "global_rows.bf16", reader.keys(global).data, width * ghkv * p * 2);

        std::ofstream meta(dir / "meta.txt");
        meta << "position " << p << "\nring " << ring << "\nsliding_layer " << sliding
             << "\nglobal_layer " << global << "\ntokens";
        for (const auto token : tokens) meta << ' ' << token;
        meta << '\n';
        std::cout << "gemma4 draft: p " << p << ", anchor " << tokens[0] << ", drafts";
        for (std::size_t i = 1; i < tokens.size(); ++i) std::cout << ' ' << tokens[i];
        std::cout << "; dumped to " << dir << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gemma4 draft: " << error.what() << '\n';
        return 1;
    }
}
