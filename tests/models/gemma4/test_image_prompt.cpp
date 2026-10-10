// An image prompt end to end: the frontend renders a user turn holding images and a question,
// preprocesses the images and expands their blocks; the Program encodes them, runs the prompt with
// each image attending bidirectionally within itself, and generates greedily. The prompt ids and every
// step's logits are written for tools/verify/gemma4_image_prompt_reference.py, which runs transformers'
// Gemma4ForConditionalGeneration on the same images and question and compares.
//
// Those outputs barely depend on whether an image attends bidirectionally, so the test also runs the
// prompt up to the end of its first image through the layers directly, once with the image's block
// bound and once causal, and writes the head's logits at a sample of the image's own positions, where
// the mask decides what each token sees (image_columns.i32, image_logits_{bidirectional,causal}.bf16).
//
// NINFER_GEMMA_ARTIFACT      an artifact converted with --components text,vision
// NINFER_GEMMA_IMAGES        image paths separated by ':' (lossless, so both sides decode the same pixels)
// NINFER_GEMMA_IMAGE_DUMP    the directory to write: ids.i32, steps.i32, logits.bf16

#include "artifact/reader.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/gemma4/cache.h"
#include "models/gemma4/forward.h"
#include "models/gemma4/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program.h"
#include "models/gemma4/vision.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

constexpr std::int32_t kSteps = 24;
constexpr const char* kQuestion = "Describe each image in one sentence.";

float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

template <class T> void write(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
}

} // namespace

int main() {
    const char* path   = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* images = std::getenv("NINFER_GEMMA_IMAGES");
    const char* dump   = std::getenv("NINFER_GEMMA_IMAGE_DUMP");
    if (path == nullptr || images == nullptr || dump == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT, NINFER_GEMMA_IMAGES and NINFER_GEMMA_IMAGE_DUMP "
                     "must be set\n";
        return 77;
    }
    try {
        DeviceContext device;
        models::LoadOptions options;
        options.vision = true;
        artifact::Reader reader(path);
        auto plan = gemma::plan_load(reader, options);
        const gemma::FrontendResources resources{
            .tokenizer_json         = std::string(plan.resources().tokenizer_json),
            .tokenizer_config_json  = std::string(plan.resources().tokenizer_config_json),
            .generation_config_json = std::string(plan.resources().generation_config_json),
            .chat_template_jinja    = std::string(plan.resources().chat_template_jinja),
        };
        auto model = gemma::materialize_model(std::move(plan), device);
        const gemma::Frontend frontend(resources, 4096);

        ChatMessage message;
        message.role = ChatRole::User;
        std::stringstream list(images);
        for (std::string image; std::getline(list, image, ':');) {
            MessagePart part;
            part.kind        = MessagePartKind::Media;
            part.media.kind  = MediaKind::Image;
            part.media.bytes = read_bytes(image);
            message.parts.push_back(std::move(part));
        }
        MessagePart question;
        question.text = kQuestion;
        message.parts.push_back(std::move(question));
        PromptInput input;
        input.messages.push_back(std::move(message));
        const gemma::PreparedPrompt prompt = frontend.prepare(std::move(input));

        std::vector<gemma::PromptImage> prompt_images;
        for (const auto& image : prompt.images) {
            prompt_images.push_back({.begin   = static_cast<std::int32_t>(image.begin),
                                     .patches = {.grid_width  = image.image.grid_width,
                                                 .grid_height = image.image.grid_height,
                                                 .pixels      = *image.image.pixels}});
            std::cout << "image at " << image.begin << ": " << image.image.grid_width << "x"
                      << image.image.grid_height << " patches, " << image.image.soft_tokens()
                      << " soft tokens\n";
        }
        const std::int32_t length = static_cast<std::int32_t>(prompt.ids.size());
        gemma::Program program(*model, length + kSteps + 8, 1, device);
        const DeviceExecutionView execution = device.execution_view();
        program.prefill(0, prompt.ids, execution, prompt_images);

        const std::int32_t vocabulary = static_cast<std::int32_t>(model->config().vocab_size);
        std::vector<std::uint16_t> logits;
        std::vector<std::int32_t> steps;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary));
        for (std::int32_t step = 0; step < kSteps; ++step) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(host.data(), program.logits().data, host.size() * 2,
                                  cudaMemcpyDeviceToHost));
            logits.insert(logits.end(), host.begin(), host.end());
            std::int32_t winner = 0;
            for (std::int32_t id = 1; id < vocabulary; ++id) {
                if (decode_bf16(host[static_cast<std::size_t>(id)]) >
                    decode_bf16(host[static_cast<std::size_t>(winner)])) {
                    winner = id;
                }
            }
            steps.push_back(winner);
            program.decode(0, winner, execution);
        }
        std::vector<std::int32_t> ids(prompt.ids.begin(), prompt.ids.end());
        const std::filesystem::path out(dump);
        std::filesystem::create_directories(out);
        write(out / "ids.i32", ids);
        write(out / "steps.i32", steps);
        write(out / "logits.bf16", logits);
        std::ofstream(out / "question.txt") << kQuestion;

        // The first image's own positions, with and without the block bound.
        const auto& config  = model->config();
        const auto& first   = prompt_images.front();
        const std::int32_t h      = static_cast<std::int32_t>(config.hidden_size);
        const std::int32_t prefix = first.begin + first.tokens();
        std::vector<std::int32_t> columns;
        for (std::int32_t c = first.begin; c < prefix; c += 8) columns.push_back(c);
        if (columns.back() != prefix - 1) columns.push_back(prefix - 1);
        write(out / "image_columns.i32", columns);
        DeviceArena arena(std::max(gemma::layer_workspace_bytes(config, prefix),
                                   gemma::vision_workspace_bytes(*model->vision_config(),
                                                                 first.patches.patches())));
        const auto bf16 = [](std::int32_t rows, std::int32_t count) {
            void* pointer = nullptr;
            CUDA_CHECK(cudaMalloc(&pointer, static_cast<std::size_t>(rows) * count * 2));
            return Tensor(static_cast<std::uint8_t*>(pointer), DType::BF16, {rows, count});
        };
        Tensor state[2]    = {bf16(h, prefix), bf16(h, prefix)};
        Tensor head_logits = bf16(vocabulary, 1);
        void* id_buffer    = nullptr;
        void* high_buffer  = nullptr;
        CUDA_CHECK(cudaMalloc(&id_buffer, static_cast<std::size_t>(prefix) * 4));
        CUDA_CHECK(cudaMalloc(&high_buffer, static_cast<std::size_t>(prefix) * 4));
        CUDA_CHECK(cudaMemcpy(id_buffer, prompt.ids.data(), static_cast<std::size_t>(prefix) * 4,
                              cudaMemcpyHostToDevice));
        std::vector<std::int32_t> high(static_cast<std::size_t>(prefix));
        for (std::int32_t t = 0; t < prefix; ++t) {
            high[static_cast<std::size_t>(t)] = t >= first.begin ? prefix - 1 : t;
        }
        CUDA_CHECK(cudaMemcpy(high_buffer, high.data(), high.size() * 4, cudaMemcpyHostToDevice));
        const Weight table = native_weight(model->weight(model->weights().text.token_embedding).view);
        for (const bool bidirectional : {true, false}) {
            gemma::KvCache cache;
            cache.configure(config, prefix + 8);
            cache.allocate();
            Tensor embedded(state[0].data, DType::BF16, {h, prefix});
            ops::embedding(Tensor(id_buffer, DType::I32, {prefix}), table, config.embedding_scale,
                           embedded, execution.stream);
            Tensor features(static_cast<std::uint8_t*>(state[0].data) +
                                static_cast<std::size_t>(first.begin) * h * 2,
                            DType::BF16, {h, first.tokens()});
            gemma::encode_image(*model, first.patches, arena, features, execution);
            const Tensor bound = bidirectional ? Tensor(high_buffer, DType::I32, {prefix, 1}) : Tensor{};
            int in = 0;
            for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
                Tensor hidden_in(state[in].data, DType::BF16, {h, prefix});
                Tensor hidden_out(state[1 - in].data, DType::BF16, {h, prefix});
                gemma::forward_layer(*model, layer, hidden_in, 0, prefix, cache, arena, hidden_out,
                                     execution, bound);
                in = 1 - in;
            }
            std::vector<std::uint16_t> column_logits;
            for (const std::int32_t c : columns) {
                const Tensor column(static_cast<std::uint8_t*>(state[in].data) +
                                        static_cast<std::size_t>(c) * h * 2,
                                    DType::BF16, {h, 1});
                gemma::forward_head(*model, column, 1, arena, head_logits, execution);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(host.data(), head_logits.data, host.size() * 2,
                                      cudaMemcpyDeviceToHost));
                column_logits.insert(column_logits.end(), host.begin(), host.end());
            }
            write(out / (bidirectional ? "image_logits_bidirectional.bf16"
                                       : "image_logits_causal.bf16"),
                  column_logits);
        }
        for (Tensor* tensor : {&state[0], &state[1], &head_logits}) CUDA_CHECK(cudaFree(tensor->data));
        CUDA_CHECK(cudaFree(id_buffer));
        CUDA_CHECK(cudaFree(high_buffer));
        std::cout << length << " prompt tokens, " << prompt.images.size() << " images, " << kSteps
                  << " greedy steps written to " << dump << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
