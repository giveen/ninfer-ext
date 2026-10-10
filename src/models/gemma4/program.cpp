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

void* device_alloc(std::size_t bytes, std::size_t& total) {
    void* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, bytes));
    total += bytes;
    return pointer;
}

std::size_t token_count_bytes(const TextConfig& config) {
    return static_cast<std::size_t>(config.vocab_size) * sizeof(std::int32_t);
}

std::size_t shared_bytes(const TextConfig& config) {
    const std::size_t pass       = static_cast<std::size_t>(Program::kPass);
    const std::size_t hidden     = config.hidden_size;
    const std::size_t vocabulary = config.vocab_size;
    return layer_workspace_bytes(config, Program::kPass) +
           ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(vocabulary), 1, 1) +
           pass * sizeof(std::int32_t) + 2 * hidden * pass * sizeof(std::uint16_t) +
           vocabulary * static_cast<std::size_t>(Program::kScorePass) * sizeof(std::uint16_t);
}

} // namespace

std::size_t Program::device_bytes(const TextConfig& config, std::int32_t capacity,
                                  std::int32_t lanes) {
    const std::size_t lane =
        KvCache::device_bytes(config, capacity) + token_count_bytes(config);
    return shared_bytes(config) + static_cast<std::size_t>(lanes) * lane;
}

Program::Program(const Model& model, std::int32_t capacity, std::int32_t lanes,
                 DeviceContext& device)
    : model_(&model), arena_(layer_workspace_bytes(model.config(), kPass)),
      sampling_workspace_(std::max<std::size_t>(
          1, ops::sampling_workspace_capacity_bytes(
                 static_cast<std::int32_t>(model.config().vocab_size), 1, 1))),
      capacity_(capacity) {
    if (capacity < 1) throw std::invalid_argument("gemma4 Program: capacity must be positive");
    if (lanes < 1) throw std::invalid_argument("gemma4 Program: lanes must be positive");
    const TextConfig& config      = model.config();
    const std::int32_t hidden     = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    allocated_bytes_              = arena_.capacity() + sampling_workspace_.capacity();

    lanes_.resize(static_cast<std::size_t>(lanes));
    for (Lane& lane : lanes_) {
        lane.cache.configure(config, capacity);
        lane.cache.allocate();
        allocated_bytes_ += lane.cache.bytes();
        lane.token_counts = device_alloc(token_count_bytes(config), allocated_bytes_);
        CUDA_CHECK(cudaMemset(lane.token_counts, 0, token_count_bytes(config)));
    }
    id_buffer_ =
        device_alloc(static_cast<std::size_t>(kPass) * sizeof(std::int32_t), allocated_bytes_);
    for (void*& state : state_) {
        state = device_alloc(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(kPass) *
                                 sizeof(std::uint16_t),
                             allocated_bytes_);
    }
    sampling_[0] = device_alloc(static_cast<std::size_t>(lanes) * sizeof(ops::SamplingConfig),
                                allocated_bytes_);
    sampling_[1] = device_alloc(sizeof(std::int32_t), allocated_bytes_);
    sampling_[2] = device_alloc(sizeof(std::int32_t), allocated_bytes_);
    // The head's output lives outside the layer arena because it outlives a call: it is what the
    // caller reads.
    void* logits = device_alloc(static_cast<std::size_t>(vocabulary) *
                                    static_cast<std::size_t>(kScorePass) * sizeof(std::uint16_t),
                                allocated_bytes_);
    logits_ = Tensor(static_cast<std::uint8_t*>(logits), DType::BF16, {vocabulary, kScorePass});
    id_host_.resize(static_cast<std::size_t>(kPass));
    for (std::int32_t lane = 0; lane < lanes; ++lane) reset(lane);
    (void)device;
}

Program::~Program() {
    for (Lane& lane : lanes_) {
        if (lane.token_counts != nullptr) (void)cudaFree(lane.token_counts);
    }
    if (id_buffer_ != nullptr) (void)cudaFree(id_buffer_);
    for (void* state : state_) {
        if (state != nullptr) (void)cudaFree(state);
    }
    for (void* buffer : sampling_) {
        if (buffer != nullptr) (void)cudaFree(buffer);
    }
    if (logits_.data != nullptr) (void)cudaFree(logits_.data);
}

Program::Lane& Program::lane_at(std::int32_t lane) {
    if (lane < 0 || lane >= lanes()) throw std::out_of_range("gemma4 Program: lane out of range");
    return lanes_[static_cast<std::size_t>(lane)];
}

const Program::Lane& Program::lane_at(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes()) throw std::out_of_range("gemma4 Program: lane out of range");
    return lanes_[static_cast<std::size_t>(lane)];
}

void Program::reset(std::int32_t lane_index, const ops::SamplingConfig& sampling) {
    Lane& lane = lane_at(lane_index);
    lane.cache.reset();
    lane.position = 0;
    CUDA_CHECK(cudaMemset(lane.token_counts, 0, token_count_bytes(model_->config())));
    ops::SamplingConfig installed = sampling;
    installed.token_counts        = static_cast<std::int32_t*>(lane.token_counts);
    installed.mask                = {};
    CUDA_CHECK(cudaMemcpy(static_cast<ops::SamplingConfig*>(sampling_[0]) + lane_index, &installed,
                          sizeof(installed), cudaMemcpyHostToDevice));
}

void Program::run_batch(KvCache& cache, const std::int32_t* ids, std::int32_t tokens,
                        std::int32_t first_position, bool every_column,
                        DeviceExecutionView execution) {
    if (tokens < 1 || tokens > (every_column ? kScorePass : kPass)) {
        throw std::invalid_argument("gemma4 Program: a pass is wider than its buffers");
    }
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model_->config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);

    // The host staging is reused by the next pass, so the copy must have left it before this returns;
    // a pageable source makes cudaMemcpyAsync stage it before returning.
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
        forward_layer(*model_, layer, hidden_in, first_position, tokens, cache, arena_, hidden_out,
                      execution);
        std::swap(in, out);
    }

    // The head's linear expects an output of exactly [N,T], so a pass narrower than the buffer writes
    // into its first columns, which are contiguous. Without every column, only the last token's hidden
    // state reaches the head.
    const std::int32_t columns = every_column ? tokens : 1;
    Tensor head_in(static_cast<std::uint8_t*>(in) + static_cast<std::size_t>(tokens - columns) *
                                                        static_cast<std::size_t>(hidden) *
                                                        sizeof(std::uint16_t),
                   DType::BF16, {hidden, columns});
    Tensor logits_batch(static_cast<std::uint8_t*>(logits_.data), DType::BF16,
                        {static_cast<std::int32_t>(logits_.ne[0]), columns});
    forward_head(*model_, head_in, columns, arena_, logits_batch, execution);
    last_batch_ = columns;
}

void Program::prefill(std::int32_t lane_index, std::span<const std::int32_t> ids,
                      DeviceExecutionView execution) {
    Lane& lane = lane_at(lane_index);
    if (ids.empty()) throw std::invalid_argument("gemma4 Program: prefill needs tokens");
    if (static_cast<std::size_t>(lane.position) + ids.size() >
        static_cast<std::size_t>(capacity_)) {
        throw std::invalid_argument("gemma4 Program: the sequence exceeds the cache capacity");
    }
    std::size_t consumed = 0;
    while (consumed < ids.size()) {
        const std::int32_t batch =
            static_cast<std::int32_t>(std::min<std::size_t>(kPass, ids.size() - consumed));
        run_batch(lane.cache, ids.data() + consumed, batch, lane.position, false, execution);
        lane.position += batch;
        consumed += static_cast<std::size_t>(batch);
    }
}

void Program::decode(std::int32_t lane_index, std::int32_t id, DeviceExecutionView execution) {
    Lane& lane = lane_at(lane_index);
    if (lane.position >= capacity_) {
        throw std::invalid_argument("gemma4 Program: the sequence exceeds the cache capacity");
    }
    run_batch(lane.cache, &id, 1, lane.position, false, execution);
    ++lane.position;
}

std::int32_t Program::sample(std::int32_t lane_index, DeviceExecutionView execution) {
    const Lane& lane          = lane_at(lane_index);
    const cudaStream_t stream = execution.stream;
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    const std::int32_t position   = lane.position;
    CUDA_CHECK(cudaMemcpyAsync(sampling_[2], &position, sizeof(position), cudaMemcpyHostToDevice,
                               stream));
    Tensor out(static_cast<std::uint8_t*>(sampling_[1]), DType::I32, {1});
    Tensor positions(static_cast<std::uint8_t*>(sampling_[2]), DType::I32, {1});
    sampling_workspace_.reset();
    ops::sample(logits(), out, vocabulary,
                static_cast<const ops::SamplingConfig*>(sampling_[0]) + lane_index, positions,
                ops::kSamplePurposeDecode, sampling_workspace_, stream);
    std::int32_t token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&token, sampling_[1], sizeof(token), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return token;
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
    reset(0);
    KvCache& cache                = lanes_.front().cache;
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    logits_host_.resize(static_cast<std::size_t>(vocabulary) * static_cast<std::size_t>(kScorePass));

    std::vector<float> scores;
    scores.reserve(ids.size() - static_cast<std::size_t>(first_target));
    // The sink sees the soft-capped logits the distribution is taken from, one row per scored target,
    // side by side and each `vocabulary` wide, which is the layout the KLD route reads. It is handed a
    // view of the staging buffer, so it must return before the next batch overwrites it.
    LogitsSink sink = std::move(sink_in);
    const std::int32_t count = static_cast<std::int32_t>(ids.size());
    std::int32_t position    = 0;
    while (position < count) {
        const std::int32_t batch = std::min(kScorePass, count - position);
        run_batch(cache, ids.data() + position, batch, position, true, execution);

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
    lanes_.front().position = count;
    return scores;
}

} // namespace ninfer::models::gemma4
