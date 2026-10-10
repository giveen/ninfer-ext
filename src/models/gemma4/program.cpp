#include "models/gemma4/program.h"

#include "core/weight_view.h"
#include "models/gemma4/forward.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::models::gemma4 {
namespace {

// Widening BF16 to FP32 is the upper half of the pattern, so it is exact.
float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

} // namespace

Program::Program(const Model& model, std::int32_t capacity, DeviceContext& device)
    : model_(&model), cache_(), arena_(layer_workspace_bytes(model.config())),
      capacity_(capacity) {
    if (capacity < 1) throw std::invalid_argument("gemma4 Program: capacity must be positive");
    const std::int32_t hidden     = static_cast<std::int32_t>(model.config().hidden_size);
    const std::int32_t vocabulary = static_cast<std::int32_t>(model.config().vocab_size);
    cache_.configure(model.config(), capacity);
    CUDA_CHECK(cudaMalloc(&id_buffer_, sizeof(std::int32_t)));
    for (void*& state : state_) {
        CUDA_CHECK(cudaMalloc(&state, static_cast<std::size_t>(hidden) * sizeof(std::uint16_t)));
    }
    // The head's output lives outside the layer arena because it outlives a call: it is what the
    // caller reads.
    void* logits = nullptr;
    CUDA_CHECK(cudaMalloc(&logits, static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t)));
    logits_ = Tensor(static_cast<std::uint8_t*>(logits), DType::BF16, {vocabulary, 1});
    (void)device;
}

Program::~Program() {
    if (id_buffer_ != nullptr) (void)cudaFree(id_buffer_);
    for (void* state : state_) {
        if (state != nullptr) (void)cudaFree(state);
    }
    if (logits_.data != nullptr) (void)cudaFree(logits_.data);
}

void Program::reset() {
    cache_.reset();
    position_ = 0;
}

void Program::run_token(std::int32_t id, std::int32_t position, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model_->config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);

    CUDA_CHECK(cudaMemcpyAsync(id_buffer_, &id, sizeof(id), cudaMemcpyHostToDevice, stream));
    Tensor id_tensor(static_cast<std::uint8_t*>(id_buffer_), DType::I32, {1});
    Tensor embedded(static_cast<std::uint8_t*>(state_[0]), DType::BF16, {hidden, 1});
    // The embedding scale the converter derives: the square root of the hidden size.
    const Weight table =
        native_weight(model_->weight(model_->weights().text.token_embedding).view);
    ops::embedding(id_tensor, table, std::sqrt(static_cast<float>(hidden)), embedded, stream);

    // Sixty layers, so an even number of swaps leaves the result where it started.
    void* in  = state_[0];
    void* out = state_[1];
    for (std::size_t layer = 0; layer < model_->weights().text.layers.size(); ++layer) {
        Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, 1});
        Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, 1});
        forward_layer(*model_, layer, hidden_in, position, cache_, arena_, hidden_out, execution);
        std::swap(in, out);
    }

    Tensor head_in(static_cast<std::uint8_t*>(state_[0]), DType::BF16, {hidden, 1});
    forward_head(*model_, head_in, arena_, logits_, execution);
}

void Program::prefill(std::span<const std::int32_t> ids, DeviceExecutionView execution) {
    for (const std::int32_t id : ids) {
        run_token(id, position_, execution);
        ++position_;
    }
}

std::vector<float> Program::causal_score(std::span<const std::int32_t> ids,
                                         std::int32_t first_target, DeviceExecutionView execution) {
    if (ids.size() < 2) throw std::invalid_argument("gemma4 Program: scoring needs two tokens");
    if (first_target < 1 || first_target >= static_cast<std::int32_t>(ids.size())) {
        throw std::invalid_argument("gemma4 Program: first_target is outside the sequence");
    }
    if (ids.size() > static_cast<std::size_t>(capacity_)) {
        throw std::invalid_argument("gemma4 Program: the sequence is longer than the capacity");
    }
    reset();
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    logits_host_.resize(static_cast<std::size_t>(vocabulary));

    std::vector<float> scores;
    scores.reserve(ids.size() - static_cast<std::size_t>(first_target));
    for (std::int32_t index = 0; index < static_cast<std::int32_t>(ids.size()); ++index) {
        if (index >= first_target) {
            // The logits in hand were produced by the preceding position, so they are this token's
            // distribution.
            CUDA_CHECK(cudaMemcpyAsync(logits_host_.data(), logits_.data,
                                       static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, execution.stream));
            CUDA_CHECK(cudaStreamSynchronize(execution.stream));
            double largest = -std::numeric_limits<double>::infinity();
            for (std::int32_t id = 0; id < vocabulary; ++id) {
                largest = std::max(largest, static_cast<double>(
                                                decode_bf16(logits_host_[static_cast<std::size_t>(id)])));
            }
            double total = 0.0;
            for (std::int32_t id = 0; id < vocabulary; ++id) {
                total += std::exp(static_cast<double>(
                                      decode_bf16(logits_host_[static_cast<std::size_t>(id)])) -
                                  largest);
            }
            const double chosen = static_cast<double>(
                decode_bf16(logits_host_[static_cast<std::size_t>(ids[static_cast<std::size_t>(index)])]));
            scores.push_back(static_cast<float>(chosen - largest - std::log(total)));
        }
        decode(ids[static_cast<std::size_t>(index)], execution);
    }
    return scores;
}

void Program::decode(std::int32_t id, DeviceExecutionView execution) {
    run_token(id, position_, execution);
    ++position_;
}

} // namespace ninfer::models::gemma4
