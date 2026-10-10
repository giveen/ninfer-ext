#include "models/gemma4/program.h"

#include "core/weight_view.h"
#include "models/gemma4/forward.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
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
    : model_(&model), cache_(), arena_(layer_workspace_bytes(model.config(), kBatch)),
      capacity_(capacity) {
    if (capacity < 1) throw std::invalid_argument("gemma4 Program: capacity must be positive");
    const std::int32_t hidden     = static_cast<std::int32_t>(model.config().hidden_size);
    const std::int32_t vocabulary = static_cast<std::int32_t>(model.config().vocab_size);
    cache_.configure(model.config(), capacity);
    CUDA_CHECK(cudaMalloc(&id_buffer_, static_cast<std::size_t>(kBatch) * sizeof(std::int32_t)));
    for (void*& state : state_) {
        CUDA_CHECK(cudaMalloc(&state, static_cast<std::size_t>(hidden) *
                                          static_cast<std::size_t>(kBatch) *
                                          sizeof(std::uint16_t)));
    }
    // The head's output lives outside the layer arena because it outlives a call: it is what the
    // caller reads.
    void* logits = nullptr;
    CUDA_CHECK(cudaMalloc(&logits, static_cast<std::size_t>(vocabulary) *
                                       static_cast<std::size_t>(kBatch) * sizeof(std::uint16_t)));
    logits_ = Tensor(static_cast<std::uint8_t*>(logits), DType::BF16, {vocabulary, kBatch});
    id_host_.resize(static_cast<std::size_t>(kBatch));
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

void Program::run_batch(const std::int32_t* ids, std::int32_t tokens, std::int32_t first_position,
                        DeviceExecutionView execution) {
    if (tokens < 1 || tokens > kBatch) {
        throw std::invalid_argument("gemma4 Program: a batch must be one to kBatch tokens");
    }
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model_->config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);

    std::memcpy(id_host_.data(), ids, static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(id_buffer_, id_host_.data(),
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    Tensor id_tensor(static_cast<std::uint8_t*>(id_buffer_), DType::I32, {tokens});
    Tensor embedded(static_cast<std::uint8_t*>(state_[0]), DType::BF16, {hidden, tokens});
    // The embedding scale the converter derives: the square root of the hidden size.
    const Weight table =
        native_weight(model_->weight(model_->weights().text.token_embedding).view);
    ops::embedding(id_tensor, table, std::sqrt(static_cast<float>(hidden)), embedded, stream);

    // Sixty layers, so an even number of swaps leaves the result where it started.
    void* in  = state_[0];
    void* out = state_[1];
    for (std::size_t layer = 0; layer < model_->weights().text.layers.size(); ++layer) {
        Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, tokens});
        Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, tokens});
        forward_layer(*model_, layer, hidden_in, first_position, tokens, cache_, arena_, hidden_out,
                      execution);
        std::swap(in, out);
    }

    Tensor head_in(static_cast<std::uint8_t*>(state_[0]), DType::BF16, {hidden, tokens});
    // The head's linear expects an output of exactly [N,T], so a batch narrower than the buffer writes
    // into its first `tokens` columns, which are contiguous.
    Tensor logits_batch(static_cast<std::uint8_t*>(logits_.data), DType::BF16,
                        {static_cast<std::int32_t>(logits_.ne[0]), tokens});
    forward_head(*model_, head_in, tokens, arena_, logits_batch, execution);
    last_batch_ = tokens;
}

void Program::prefill(std::span<const std::int32_t> ids, DeviceExecutionView execution) {
    if (position_ + static_cast<std::int32_t>(ids.size()) > capacity_) {
        throw std::invalid_argument("gemma4 Program: the sequence exceeds the cache capacity");
    }
    std::size_t consumed = 0;
    while (consumed < ids.size()) {
        const std::int32_t batch =
            static_cast<std::int32_t>(std::min<std::size_t>(kBatch, ids.size() - consumed));
        run_batch(ids.data() + consumed, batch, position_, execution);
        position_ += batch;
        consumed += static_cast<std::size_t>(batch);
    }
}

std::vector<float> Program::causal_score(std::span<const std::int32_t> ids,
                                         std::int32_t first_target, const LogitsSink& sink_in,
                                         DeviceExecutionView execution) {
    if (ids.size() < 2) throw std::invalid_argument("gemma4 Program: scoring needs two tokens");
    if (first_target < 1 || first_target >= static_cast<std::int32_t>(ids.size())) {
        throw std::invalid_argument("gemma4 Program: first_target is outside the sequence");
    }
    if (ids.size() > static_cast<std::size_t>(capacity_)) {
        throw std::invalid_argument("gemma4 Program: the sequence is longer than the capacity");
    }
    reset();
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    logits_host_.resize(static_cast<std::size_t>(vocabulary) * static_cast<std::size_t>(kBatch));

    std::vector<float> scores;
    scores.reserve(ids.size() - static_cast<std::size_t>(first_target));
    // The sink sees the soft-capped logits the distribution is taken from, one row per scored target,
    // side by side and each `vocabulary` wide, which is the layout the KLD route reads. It is handed a
    // view of the staging buffer, so it must return before the next batch overwrites it.
    LogitsSink sink = std::move(sink_in);
    const std::int32_t count = static_cast<std::int32_t>(ids.size());
    std::int32_t position    = 0;
    while (position < count) {
        const std::int32_t batch = std::min(kBatch, count - position);
        run_batch(ids.data() + position, batch, position, execution);

        // A column of the head predicts the token after its own position, so this batch covers the
        // targets from position+1 to position+batch, clipped to the sequence and to first_target.
        const std::int32_t first = std::max(position + 1, first_target);
        const std::int32_t last  = std::min(position + batch, count - 1);
        if (first <= last) {
            CUDA_CHECK(cudaMemcpyAsync(logits_host_.data(), logits_.data,
                                       static_cast<std::size_t>(vocabulary) *
                                           static_cast<std::size_t>(batch) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, execution.stream));
            CUDA_CHECK(cudaStreamSynchronize(execution.stream));
            for (std::int32_t target = first; target <= last; ++target) {
                const std::uint16_t* column =
                    logits_host_.data() +
                    static_cast<std::size_t>(target - position - 1) *
                        static_cast<std::size_t>(vocabulary);
                double largest = -std::numeric_limits<double>::infinity();
                for (std::int32_t id = 0; id < vocabulary; ++id) {
                    largest = std::max(largest, static_cast<double>(decode_bf16(column[id])));
                }
                double total = 0.0;
                for (std::int32_t id = 0; id < vocabulary; ++id) {
                    total += std::exp(static_cast<double>(decode_bf16(column[id])) - largest);
                }
                const double chosen =
                    static_cast<double>(decode_bf16(column[ids[static_cast<std::size_t>(target)]]));
                scores.push_back(static_cast<float>(chosen - largest - std::log(total)));
            }
            if (sink) {
                const std::size_t offset = static_cast<std::size_t>(first - position - 1) *
                                           static_cast<std::size_t>(vocabulary);
                const std::size_t columns = static_cast<std::size_t>(last - first + 1);
                ScoredLogits view;
                view.values       = std::span<const std::uint16_t>(logits_host_.data() + offset,
                                                                   columns *
                                                                       static_cast<std::size_t>(vocabulary));
                view.vocab_size   = static_cast<std::uint32_t>(vocabulary);
                view.columns      = static_cast<std::uint32_t>(columns);
                view.first_target = static_cast<std::uint32_t>(first);
                sink(view);
            }
        }
        position += batch;
    }
    return scores;
}

void Program::decode(std::int32_t id, DeviceExecutionView execution) {
    run_batch(&id, 1, position_, execution);
    ++position_;
}

} // namespace ninfer::models::gemma4
