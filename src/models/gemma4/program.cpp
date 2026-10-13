#include "models/gemma4/program.h"

#include "core/weight_view.h"
#include "models/gemma4/forward.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/speculative_round.h"

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

// The round's I32 block: drafts [K], target argmax [K+1], licensed [K+1], then anchor, length,
// extent, licensed count and accepted count.
constexpr std::size_t kRoundInts = 3 * (Program::kMaximumDraftTokens + 1) + 5;
// Per lane, with its column of the step tokens (see speculate()).
constexpr std::size_t kRoundLaneInts = kRoundInts + Program::kMaximumDraftTokens + 1;

std::size_t sampling_bytes(const TextConfig& config, std::int32_t draft_tokens) {
    const auto vocabulary = static_cast<std::int32_t>(config.vocab_size);
    std::size_t bytes     = ops::sampling_workspace_capacity_bytes(vocabulary, 1, 1);
    if (draft_tokens > 0) {
        bytes = std::max(bytes, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                    vocabulary, 1, draft_tokens, 1, 1));
    }
    return std::max<std::size_t>(1, bytes);
}

// The pass scratch: one layer of a full pass, or one image's encoder when the tower is bound.
std::size_t arena_bytes(const TextConfig& config, const VisionConfig* vision,
                        KvCacheStorage storage, std::int32_t ring_slack) {
    std::size_t bytes = layer_workspace_bytes(config, Program::kPass, storage, ring_slack);
    if (vision != nullptr) {
        bytes = std::max(bytes, vision_workspace_bytes(
                                    *vision, static_cast<std::int32_t>(vision->max_patches())));
    }
    return bytes;
}

std::size_t shared_bytes(const TextConfig& config, std::int32_t draft_tokens,
                         const VisionConfig* vision, KvCacheStorage storage) {
    const std::size_t pass       = static_cast<std::size_t>(Program::kPass);
    const std::size_t hidden     = config.hidden_size;
    const std::size_t vocabulary = config.vocab_size;
    const std::size_t score      = static_cast<std::size_t>(Program::kScorePass);
    std::size_t bytes =
        arena_bytes(config, vision, storage, draft_tokens) + sampling_bytes(config, draft_tokens) +
        pass * sizeof(std::int32_t) + 2 * hidden * pass * sizeof(std::uint16_t) +
        vocabulary * score * sizeof(std::uint16_t) + hidden * score * sizeof(std::uint16_t);
    if (vision != nullptr) bytes += pass * sizeof(std::int32_t);
    return bytes;
}

// What speculation adds per lane: the drafter's input, state and logits columns, the round block,
// the lane's hidden state and the verify pass's states it keeps for the commit.
std::size_t draft_lane_bytes(const TextConfig& config, std::int32_t draft_tokens) {
    if (draft_tokens <= 0) return 0;
    const std::size_t hidden     = config.hidden_size;
    const std::size_t vocabulary = config.vocab_size;
    return (3 * hidden + vocabulary + hidden +
            (static_cast<std::size_t>(draft_tokens) + 1) * hidden) *
               sizeof(std::uint16_t) +
           kRoundLaneInts * sizeof(std::int32_t);
}

} // namespace

std::size_t Program::device_bytes(const TextConfig& config, std::int32_t capacity,
                                  std::int32_t lanes, std::int32_t draft_tokens,
                                  const VisionConfig* vision, KvCacheStorage storage) {
    const std::size_t lane = KvCache::device_bytes(config, capacity, draft_tokens, storage) +
                             token_count_bytes(config) + draft_lane_bytes(config, draft_tokens);
    return shared_bytes(config, draft_tokens, vision, storage) +
           static_cast<std::size_t>(lanes) * lane;
}

Program::Program(const Model& model, std::int32_t capacity, std::int32_t lanes,
                 DeviceContext& device, std::int32_t draft_tokens, bool cuda_graphs,
                 KvCacheStorage storage)
    : model_(&model),
      arena_(arena_bytes(model.config(), model.vision_config() ? &*model.vision_config() : nullptr,
                         storage, draft_tokens)),
      sampling_workspace_(sampling_bytes(model.config(), draft_tokens)),
      draft_tokens_(draft_tokens), capacity_(capacity), cuda_graphs_(cuda_graphs),
      kv_storage_(storage) {
    if (capacity < 1) throw std::invalid_argument("gemma4 Program: capacity must be positive");
    // A decode round names its lanes by a bit each (decode_graphs_).
    if (lanes < 1 || lanes > 64)
        throw std::invalid_argument("gemma4 Program: lanes must be in [1, 64]");
    if (draft_tokens < 0 || draft_tokens > kMaximumDraftTokens) {
        throw std::invalid_argument("gemma4 Program: draft tokens must be in [0, 7]");
    }
    if (draft_tokens > 0 && !model.draft_config()) {
        throw std::invalid_argument("gemma4 Program: speculation needs the drafter loaded");
    }
    const TextConfig& config      = model.config();
    const std::int32_t hidden     = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    allocated_bytes_              = arena_.capacity() + sampling_workspace_.capacity();

    lanes_.resize(static_cast<std::size_t>(lanes));
    for (Lane& lane : lanes_) {
        lane.cache.configure(config, capacity, draft_tokens, storage);
        lane.cache.allocate();
        allocated_bytes_ += lane.cache.bytes();
        lane.token_counts = device_alloc(token_count_bytes(config), allocated_bytes_);
        CUDA_CHECK(cudaMemset(lane.token_counts, 0, token_count_bytes(config)));
        if (draft_tokens > 0) {
            lane.hidden = device_alloc(static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                                       allocated_bytes_);
            lane.round_hidden =
                device_alloc(static_cast<std::size_t>(draft_tokens + 1) *
                                 static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                             allocated_bytes_);
        }
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
    sampling_[1] =
        device_alloc(static_cast<std::size_t>(lanes) * sizeof(std::int32_t), allocated_bytes_);
    sampling_[2] =
        device_alloc(static_cast<std::size_t>(lanes) * sizeof(std::int32_t), allocated_bytes_);
    sampled_host_.resize(static_cast<std::size_t>(lanes));
    // The head's output lives outside the layer arena because it outlives a call: it is what the
    // caller reads.
    void* logits = device_alloc(static_cast<std::size_t>(vocabulary) *
                                    static_cast<std::size_t>(kScorePass) * sizeof(std::uint16_t),
                                allocated_bytes_);
    logits_ = Tensor(static_cast<std::uint8_t*>(logits), DType::BF16, {vocabulary, kScorePass});
    void* head_hidden = device_alloc(
        static_cast<std::size_t>(hidden) * kScorePass * sizeof(std::uint16_t), allocated_bytes_);
    head_hidden_ =
        Tensor(static_cast<std::uint8_t*>(head_hidden), DType::BF16, {hidden, kScorePass});
    if (draft_tokens > 0) {
        const auto bf16 = [&](std::int32_t rows) {
            return Tensor(static_cast<std::uint8_t*>(device_alloc(
                              static_cast<std::size_t>(rows) * static_cast<std::size_t>(lanes) *
                                  sizeof(std::uint16_t),
                              allocated_bytes_)),
                          DType::BF16, {rows, lanes});
        };
        draft_input_  = bf16(2 * hidden);
        draft_hidden_ = bf16(hidden);
        draft_logits_ = bf16(vocabulary);
        round_ints_ =
            device_alloc(kRoundLaneInts * static_cast<std::size_t>(lanes) * sizeof(std::int32_t),
                         allocated_bytes_);
        round_host_.resize(kRoundLaneInts * static_cast<std::size_t>(lanes));
    }
    id_host_.resize(static_cast<std::size_t>(kPass));
    if (model.vision_config()) {
        high_buffer_ =
            device_alloc(static_cast<std::size_t>(kPass) * sizeof(std::int32_t), allocated_bytes_);
        high_host_.resize(static_cast<std::size_t>(kPass));
    }
    for (std::int32_t lane = 0; lane < lanes; ++lane) reset(lane);
    (void)device;
}

Program::~Program() {
    for (Lane& lane : lanes_) {
        if (lane.token_counts != nullptr) (void)cudaFree(lane.token_counts);
    }
    if (id_buffer_ != nullptr) (void)cudaFree(id_buffer_);
    if (high_buffer_ != nullptr) (void)cudaFree(high_buffer_);
    for (const auto& [start, stop] : vision_events_) {
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
    }
    for (void* state : state_) {
        if (state != nullptr) (void)cudaFree(state);
    }
    for (void* buffer : sampling_) {
        if (buffer != nullptr) (void)cudaFree(buffer);
    }
    for (Lane& lane : lanes_) {
        if (lane.hidden != nullptr) (void)cudaFree(lane.hidden);
        if (lane.round_hidden != nullptr) (void)cudaFree(lane.round_hidden);
    }
    for (const Tensor* tensor :
         {&logits_, &head_hidden_, &draft_input_, &draft_hidden_, &draft_logits_}) {
        if (tensor->data != nullptr) (void)cudaFree(tensor->data);
    }
    if (round_ints_ != nullptr) (void)cudaFree(round_ints_);
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
    lane.position       = 0;
    lane.round_position = -1;
    CUDA_CHECK(cudaMemset(lane.token_counts, 0, token_count_bytes(model_->config())));
    ops::SamplingConfig installed = sampling;
    installed.token_counts        = static_cast<std::int32_t*>(lane.token_counts);
    installed.mask                = {};
    CUDA_CHECK(cudaMemcpy(static_cast<ops::SamplingConfig*>(sampling_[0]) + lane_index, &installed,
                          sizeof(installed), cudaMemcpyHostToDevice));
}

void Program::run_batch(KvCache& cache, const std::int32_t* ids, std::int32_t tokens,
                        std::int32_t first_position, bool every_column,
                        DeviceExecutionView execution, std::span<const PromptImage> images) {
    if (tokens < 1 || tokens > (every_column ? kScorePass : kPass)) {
        throw std::invalid_argument("gemma4 Program: a pass is wider than its buffers");
    }
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model_->config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);

    // The host staging is reused by the next pass, so the copy must have left it before this
    // returns; a pageable source makes cudaMemcpyAsync stage it before returning.
    std::memcpy(id_host_.data(), ids, static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(id_buffer_, id_host_.data(),
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    run_pass(cache, tokens, first_position, every_column, execution, images);
}

void Program::embed(std::int32_t tokens, cudaStream_t stream) {
    const std::int32_t hidden = static_cast<std::int32_t>(model_->config().hidden_size);
    Tensor id_tensor(static_cast<std::uint8_t*>(id_buffer_), DType::I32, {tokens});
    Tensor embedded(static_cast<std::uint8_t*>(state_[0]), DType::BF16, {hidden, tokens});
    // The embedding scale the converter derives: the square root of the hidden size.
    const Weight table = native_weight(model_->weight(model_->weights().text.token_embedding).view);
    ops::embedding(id_tensor, table, std::sqrt(static_cast<float>(hidden)), embedded, stream);
}

void Program::run_pass(KvCache& cache, std::int32_t tokens, std::int32_t first_position,
                       bool every_column, DeviceExecutionView execution,
                       std::span<const PromptImage> images) {
    const cudaStream_t stream = execution.stream;
    const TextConfig& config  = model_->config();
    const std::int32_t hidden = static_cast<std::int32_t>(config.hidden_size);
    embed(tokens, stream);

    // An image's features replace its tokens' embeddings as they are, with no embedding scale, and
    // its tokens' sliding upper bound is the image's last position. The encoder's scratch is the
    // layer arena, which no layer holds yet.
    Tensor position_high;
    if (!images.empty()) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            high_host_[static_cast<std::size_t>(token)] = first_position + token;
        }
        if (vision_events_used_ == vision_events_.size()) {
            cudaEvent_t start = nullptr, stop = nullptr;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            vision_events_.emplace_back(start, stop);
        }
        const auto [vision_start, vision_stop] = vision_events_[vision_events_used_++];
        CUDA_CHECK(cudaEventRecord(vision_start, stream));
        for (const PromptImage& image : images) {
            Tensor features(static_cast<std::uint8_t*>(state_[0]) +
                                static_cast<std::size_t>(image.begin) * hidden *
                                    sizeof(std::uint16_t),
                            DType::BF16, {hidden, image.tokens()});
            encode_image(*model_, image.patches, arena_, features, execution);
            const std::int32_t last = first_position + image.begin + image.tokens() - 1;
            for (std::int32_t token = image.begin; token < image.begin + image.tokens(); ++token) {
                high_host_[static_cast<std::size_t>(token)] = last;
            }
        }
        CUDA_CHECK(cudaEventRecord(vision_stop, stream));
        // A pageable source is staged before cudaMemcpyAsync returns, so the next pass may reuse
        // it.
        CUDA_CHECK(cudaMemcpyAsync(high_buffer_, high_host_.data(),
                                   static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        position_high = Tensor(static_cast<std::uint8_t*>(high_buffer_), DType::I32, {tokens, 1});
    }

    DeviceArena::Scope scope = arena_.scope();
    const PassSegment segment{&cache, first_position, 0, tokens};
    const PassIndices indices = write_pass_indices(std::span(&segment, 1), arena_, stream);
    run_layers(std::span(&segment, 1), indices, every_column, execution, position_high);
}

void Program::run_layers(std::span<const PassSegment> segments, const PassIndices& indices,
                         bool every_column, DeviceExecutionView execution,
                         const Tensor& position_high) {
    const std::int32_t hidden = static_cast<std::int32_t>(model_->config().hidden_size);
    const std::int32_t tokens = static_cast<std::int32_t>(indices.positions.ne[0]);
    // Sixty layers, so an even number of swaps leaves the result where it started.
    void* in  = state_[0];
    void* out = state_[1];
    for (std::size_t layer = 0; layer < model_->weights().text.layers.size(); ++layer) {
        Tensor hidden_in(static_cast<std::uint8_t*>(in), DType::BF16, {hidden, tokens});
        Tensor hidden_out(static_cast<std::uint8_t*>(out), DType::BF16, {hidden, tokens});
        forward_layer(*model_, layer, hidden_in, segments, indices, arena_, hidden_out, execution,
                      position_high);
        std::swap(in, out);
    }

    // The head's linear expects an output of exactly [N,T], so a pass narrower than the buffer
    // writes into its first columns, which are contiguous. Without every column, only the last
    // token's hidden state reaches the head.
    const std::int32_t columns = every_column ? tokens : 1;
    Tensor head_in(static_cast<std::uint8_t*>(in) + static_cast<std::size_t>(tokens - columns) *
                                                        static_cast<std::size_t>(hidden) *
                                                        sizeof(std::uint16_t),
                   DType::BF16, {hidden, columns});
    Tensor logits_batch(static_cast<std::uint8_t*>(logits_.data), DType::BF16,
                        {static_cast<std::int32_t>(logits_.ne[0]), columns});
    forward_head(*model_, head_in, columns, arena_, logits_batch, execution, &head_hidden_);
    last_batch_ = columns;
}

void Program::keep_hidden(Lane& lane, std::int32_t column, cudaStream_t stream) {
    if (lane.hidden == nullptr) return;
    const std::size_t bytes =
        static_cast<std::size_t>(model_->config().hidden_size) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpyAsync(lane.hidden,
                               static_cast<std::uint8_t*>(head_hidden_.data) +
                                   static_cast<std::size_t>(column) * bytes,
                               bytes, cudaMemcpyDeviceToDevice, stream));
}

void Program::prefill(std::int32_t lane_index, std::span<const std::int32_t> ids,
                      DeviceExecutionView execution, std::span<const PromptImage> images) {
    Lane& lane = lane_at(lane_index);
    if (ids.empty()) throw std::invalid_argument("gemma4 Program: prefill needs tokens");
    if (static_cast<std::size_t>(lane.position) + ids.size() >
        static_cast<std::size_t>(capacity_)) {
        throw std::invalid_argument("gemma4 Program: the sequence exceeds the cache capacity");
    }
    if (!images.empty() && !model_->vision_config()) {
        throw std::invalid_argument("gemma4 Program: images need the vision tower loaded");
    }
    std::int32_t previous_end = 0;
    for (const PromptImage& image : images) {
        const std::int32_t end = image.begin + image.tokens();
        if (image.tokens() < 1 || image.begin < previous_end ||
            end > static_cast<std::int32_t>(ids.size()) || image.tokens() > kPass) {
            throw std::invalid_argument(
                "gemma4 Program: images must be ascending, disjoint and inside the call");
        }
        for (std::int32_t token = image.begin; token < end; ++token) {
            if (ids[static_cast<std::size_t>(token)] != model_->vision_config()->image_token_id) {
                throw std::invalid_argument(
                    "gemma4 Program: an image's tokens must be image tokens");
            }
        }
        previous_end = end;
    }
    std::size_t consumed = 0;
    std::size_t next     = 0; // the first image not consumed yet
    std::vector<PromptImage> pass_images;
    while (consumed < ids.size()) {
        std::int32_t batch =
            static_cast<std::int32_t>(std::min<std::size_t>(kPass, ids.size() - consumed));
        const std::int32_t first = static_cast<std::int32_t>(consumed);
        // A pass that would end inside an image ends where the image begins instead.
        pass_images.clear();
        for (std::size_t index = next; index < images.size(); ++index) {
            const PromptImage& image = images[index];
            if (image.begin >= first + batch) break;
            if (image.begin + image.tokens() > first + batch) {
                batch = image.begin - first;
                break;
            }
            PromptImage local = image;
            local.begin -= first;
            pass_images.push_back(local);
        }
        next += pass_images.size();
        run_batch(lane.cache, ids.data() + consumed, batch, lane.position, false, execution,
                  pass_images);
        lane.position += batch;
        consumed += static_cast<std::size_t>(batch);
    }
    keep_hidden(lane, 0, execution.stream);
    lane.round_position = -1;
    lane.column         = 0;
}

double Program::take_vision_seconds() {
    double seconds = 0.0;
    for (std::size_t pair = 0; pair < vision_events_used_; ++pair) {
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventSynchronize(vision_events_[pair].second));
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, vision_events_[pair].first,
                                        vision_events_[pair].second));
        seconds += milliseconds / 1000.0;
    }
    vision_events_used_ = 0;
    return seconds;
}

void Program::decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution) {
    decode(std::span(&lane, 1), std::span(&id, 1), execution);
}

void Program::decode(std::span<const std::int32_t> lanes, std::span<const std::int32_t> ids,
                     DeviceExecutionView execution) {
    const std::int32_t count = static_cast<std::int32_t>(lanes.size());
    if (count < 1 || ids.size() != lanes.size() || count > kScorePass) {
        throw std::invalid_argument("gemma4 Program: a decode round needs one id per lane");
    }
    // The columns follow the lanes in ascending order, so a set of lanes always lays its pass out
    // the same way and its captured graph matches whatever order the caller named them in.
    std::vector<std::int32_t> order(lanes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int32_t>(i);
    std::sort(order.begin(), order.end(),
              [&](std::int32_t a, std::int32_t b) { return lanes[a] < lanes[b]; });
    std::uint64_t set = 0;
    std::vector<PassSegment> segments;
    segments.reserve(lanes.size());
    for (std::int32_t column = 0; column < count; ++column) {
        const std::int32_t index = order[static_cast<std::size_t>(column)];
        Lane& lane               = lane_at(lanes[index]);
        const std::uint64_t bit  = std::uint64_t{1} << lanes[index];
        if ((set & bit) != 0)
            throw std::invalid_argument("gemma4 Program: a lane decodes once a round");
        set |= bit;
        if (lane.position >= capacity_) {
            throw std::invalid_argument("gemma4 Program: the sequence exceeds the cache capacity");
        }
        segments.push_back({&lane.cache, lane.position, column, 1});
        id_host_[static_cast<std::size_t>(column)] = ids[index];
    }
    const cudaStream_t stream = execution.stream;
    // The tokens and the pass indices are written before the pass, outside any graph: they are the
    // only inputs a decode round takes from the host.
    CUDA_CHECK(cudaMemcpyAsync(id_buffer_, id_host_.data(),
                               static_cast<std::size_t>(count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    DeviceArena::Scope scope  = arena_.scope();
    const PassIndices indices = write_pass_indices(segments, arena_, stream);
    const auto pass           = [&] {
        embed(count, stream);
        run_layers(segments, indices, true, execution, {});
    };
    // A round of one-token segments issues the same launches at every position (forward.h), so the
    // graph a set of lanes captures replays every later round of that set. The first decode of the
    // Program runs eagerly, so every launcher's one-time setup has happened before anything is
    // captured; the legacy default stream cannot be captured at all.
    if (cuda_graphs_ && decode_warm_ && stream != nullptr) {
        auto found = decode_graphs_.find(set);
        if (found == decode_graphs_.end()) {
            if (decode_graphs_.size() >= kDecodeGraphs) {
                decode_graphs_.erase(std::min_element(
                    decode_graphs_.begin(), decode_graphs_.end(), [](const auto& a, const auto& b) {
                        return a.second.last_used < b.second.last_used;
                    }));
            }
            DecodeGraphDefinition definition;
            definition.capture(stream, pass);
            found = decode_graphs_.try_emplace(set).first;
            found->second.executable.instantiate(definition);
        }
        found->second.last_used = ++decode_rounds_;
        found->second.executable.launch(stream);
        last_batch_ = count;
    } else {
        pass();
        decode_warm_ = true;
    }
    for (std::int32_t column = 0; column < count; ++column) {
        Lane& lane = lane_at(lanes[order[static_cast<std::size_t>(column)]]);
        ++lane.position;
        lane.column = column;
        keep_hidden(lane, column, stream);
        lane.round_position = -1;
    }
}

Program::Round Program::speculate(std::int32_t lane_index, std::int32_t anchor, std::int32_t drafts,
                                  DeviceExecutionView execution) {
    return std::move(speculate(std::span(&lane_index, 1), std::span(&anchor, 1),
                               std::span(&drafts, 1), execution)
                         .front());
}

std::vector<Program::Round> Program::speculate(std::span<const std::int32_t> lanes,
                                               std::span<const std::int32_t> anchors,
                                               std::span<const std::int32_t> drafts,
                                               DeviceExecutionView execution) {
    const std::int32_t count = static_cast<std::int32_t>(lanes.size());
    if (count < 1 || anchors.size() != lanes.size() || drafts.size() != lanes.size() ||
        count > this->lanes()) {
        throw std::invalid_argument(
            "gemma4 Program: a round needs one anchor and draft count per lane");
    }
    // The lanes run in descending draft count, so the lanes still drafting at any step are a prefix
    // of the batch and every step's columns stay contiguous.
    std::vector<std::int32_t> order(lanes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int32_t>(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int32_t a, std::int32_t b) { return drafts[a] > drafts[b]; });
    std::uint64_t set    = 0;
    std::int32_t columns = 0;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const Lane& lane        = lane_at(lanes[i]);
        const std::uint64_t bit = std::uint64_t{1} << lanes[i];
        if ((set & bit) != 0)
            throw std::invalid_argument("gemma4 Program: a lane drafts once a round");
        set |= bit;
        if (drafts[i] < 1 || drafts[i] > draft_tokens_) {
            throw std::invalid_argument(
                "gemma4 Program: a round drafts one to draft_tokens() tokens");
        }
        if (lane.position < 1 || lane.position + drafts[i] + 1 > capacity_) {
            throw std::invalid_argument("gemma4 Program: the round does not fit the lane");
        }
        columns += drafts[i] + 1;
    }
    if (columns > kScorePass) {
        throw std::invalid_argument("gemma4 Program: the round's verify pass exceeds its buffers");
    }
    const cudaStream_t stream      = execution.stream;
    const TextConfig& config       = model_->config();
    const std::int32_t hidden      = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t vocabulary  = static_cast<std::int32_t>(config.vocab_size);
    const std::size_t hidden_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    const std::size_t i32          = sizeof(std::int32_t);
    const std::int32_t width       = kMaximumDraftTokens + 1;

    // The I32 area: one kRoundInts block per batch slot, laid out as kRoundInts says, then the step
    // tokens [kMaximumDraftTokens + 1, count]: row 0 the anchors, row s the drafts of step s - 1.
    auto* ints       = static_cast<std::int32_t*>(round_ints_);
    const auto block = [&](std::int32_t slot) {
        return ints + static_cast<std::size_t>(slot) * kRoundInts;
    };
    std::int32_t* steps = ints + static_cast<std::size_t>(this->lanes()) * kRoundInts;
    std::fill(round_host_.begin(), round_host_.end(), 0);
    std::int32_t* step_host =
        round_host_.data() + static_cast<std::size_t>(this->lanes()) * kRoundInts;
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const std::int32_t i = order[static_cast<std::size_t>(slot)];
        std::int32_t* scalars =
            round_host_.data() + static_cast<std::size_t>(slot) * kRoundInts + 3 * width;
        scalars[0] = anchors[i];
        scalars[1] = lane_at(lanes[i]).position + 1; // the position the first licensed token takes
        scalars[2] = drafts[i];
        step_host[slot] = anchors[i];
    }
    // Pageable sources are staged before cudaMemcpyAsync returns.
    CUDA_CHECK(cudaMemcpyAsync(ints, round_host_.data(),
                               static_cast<std::size_t>(count) * kRoundInts * i32,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(steps, step_host, static_cast<std::size_t>(count) * i32,
                               cudaMemcpyHostToDevice, stream));

    // Draft: each step reads [embedding(token); state] at the anchor's position, and its argmax is
    // the next step's token. The drafter's input interleaves the two halves per column, so the
    // embedding goes through scratch and both halves are placed with pitched copies.
    DeviceArena::Scope scope = arena_.scope();
    const Weight table = native_weight(model_->weight(model_->weights().text.token_embedding).view);
    Tensor embedded    = arena_.alloc(DType::BF16, {hidden, count});
    std::vector<DraftLane> draft_lanes(lanes.size());
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const Lane& lane  = lane_at(lanes[order[static_cast<std::size_t>(slot)]]);
        draft_lanes[slot] = {&lane.cache, lane.position};
    }
    auto* input                     = static_cast<std::uint8_t*>(draft_input_.data);
    const std::int32_t steps_needed = drafts[order.front()];
    for (std::int32_t step = 0; step < steps_needed; ++step) {
        std::int32_t active = 0;
        while (active < count && drafts[order[static_cast<std::size_t>(active)]] > step) ++active;
        const Tensor token(static_cast<void*>(steps + static_cast<std::size_t>(step) * count),
                           DType::I32, {active});
        Tensor embedded_active(embedded.data, DType::BF16, {hidden, active});
        ops::embedding(token, table, config.embedding_scale, embedded_active, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(input, 2 * hidden_bytes, embedded.data, hidden_bytes,
                                     hidden_bytes, static_cast<std::size_t>(active),
                                     cudaMemcpyDeviceToDevice, stream));
        if (step == 0) {
            for (std::int32_t slot = 0; slot < active; ++slot) {
                CUDA_CHECK(
                    cudaMemcpyAsync(input + (2 * static_cast<std::size_t>(slot) + 1) * hidden_bytes,
                                    lane_at(lanes[order[static_cast<std::size_t>(slot)]]).hidden,
                                    hidden_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                input + hidden_bytes, 2 * hidden_bytes, draft_hidden_.data, hidden_bytes,
                hidden_bytes, static_cast<std::size_t>(active), cudaMemcpyDeviceToDevice, stream));
        }
        const Tensor input_active(input, DType::BF16, {2 * hidden, active});
        Tensor logits_active(draft_logits_.data, DType::BF16, {vocabulary, active});
        Tensor hidden_active(draft_hidden_.data, DType::BF16, {hidden, active});
        forward_draft(*model_, input_active, std::span(draft_lanes.data(), active), arena_,
                      logits_active, hidden_active, execution);
        Tensor chosen(static_cast<void*>(steps + static_cast<std::size_t>(step + 1) * count),
                      DType::I32, {active});
        ops::argmax(logits_active, chosen, vocabulary, stream);
    }

    // Verify: one target pass runs every lane's [anchor, drafts] at p..p+drafts as its own segment,
    // with every column's logits.
    std::vector<PassSegment> segments;
    segments.reserve(lanes.size());
    std::int32_t column = 0;
    auto* ids           = static_cast<std::int32_t*>(id_buffer_);
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const std::int32_t i = order[static_cast<std::size_t>(slot)];
        Lane& lane           = lane_at(lanes[i]);
        // Column `slot` of the step rows, down the anchor and drafts.
        CUDA_CHECK(cudaMemcpy2DAsync(
            ids + column, i32, steps + slot, static_cast<std::size_t>(count) * i32, i32,
            static_cast<std::size_t>(drafts[i]) + 1, cudaMemcpyDeviceToDevice, stream));
        segments.push_back({&lane.cache, lane.position, column, drafts[i] + 1});
        column += drafts[i] + 1;
    }
    embed(columns, stream);
    {
        DeviceArena::Scope pass_scope = arena_.scope();
        const PassIndices indices     = write_pass_indices(segments, arena_, stream);
        run_layers(segments, indices, true, execution, {});
    }

    // Accept by each lane's sampler, against its own columns.
    const std::size_t logit_bytes = static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t);
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const std::int32_t i       = order[static_cast<std::size_t>(slot)];
        const std::int32_t n       = drafts[i];
        const PassSegment& segment = segments[static_cast<std::size_t>(slot)];
        std::int32_t* lane_ints    = block(slot);
        std::int32_t* argmaxes     = lane_ints + width;
        std::int32_t* licensed     = lane_ints + 2 * width;
        std::int32_t* scalars = lane_ints + 3 * width; // anchor, length, extent, count, accepted
        auto* first_logit     = static_cast<std::uint8_t*>(logits_.data) +
                                static_cast<std::size_t>(segment.column) * logit_bytes;
        const Tensor lane_logits(first_logit, DType::BF16, {vocabulary, n + 1, 1});
        Tensor target_tokens(static_cast<void*>(argmaxes), DType::I32, {n + 1, 1});
        ops::argmax(Tensor(first_logit, DType::BF16, {vocabulary, n + 1}), target_tokens,
                    vocabulary, stream);
        const Tensor draft_tensor(static_cast<void*>(ids + segment.column + 1), DType::I32, {n, 1});
        const Tensor extents(static_cast<void*>(scalars + 2), DType::I32, {1});
        Tensor lengths(static_cast<void*>(scalars + 1), DType::I32, {1});
        Tensor anchor(static_cast<void*>(scalars), DType::I32, {1});
        Tensor licensed_tokens(static_cast<void*>(licensed), DType::I32, {n + 1, 1});
        Tensor licensed_count(static_cast<void*>(scalars + 3), DType::I32, {1});
        Tensor accepted(static_cast<void*>(scalars + 4), DType::I32, {1});
        // The accepts run in stream order, so each may reuse the workspace the previous one used.
        sampling_workspace_.reset();
        ops::speculative_accept_greedy_drafts(
            target_tokens, lane_logits, draft_tensor, extents, lengths, anchor, licensed_tokens,
            licensed_count, accepted, vocabulary,
            static_cast<const ops::SamplingConfig*>(sampling_[0]) + lanes[i], sampling_workspace_,
            stream);
        // The commit picks one of these states after later passes have reused the head's buffer.
        Lane& lane = lane_at(lanes[i]);
        CUDA_CHECK(cudaMemcpyAsync(lane.round_hidden,
                                   static_cast<std::uint8_t*>(head_hidden_.data) +
                                       static_cast<std::size_t>(segment.column) * hidden_bytes,
                                   static_cast<std::size_t>(n + 1) * hidden_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(round_host_.data(), ints,
                               static_cast<std::size_t>(count) * kRoundInts * i32,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Round> rounds(lanes.size());
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const std::int32_t i     = order[static_cast<std::size_t>(slot)];
        const std::int32_t* host = round_host_.data() + static_cast<std::size_t>(slot) * kRoundInts;
        const std::int32_t licensed_count = host[3 * width + 3];
        Round& round                      = rounds[static_cast<std::size_t>(i)];
        round.accepted_drafts             = host[3 * width + 4];
        if (licensed_count < 1 || licensed_count > drafts[i] + 1 ||
            round.accepted_drafts != licensed_count - 1) {
            throw std::runtime_error("gemma4 Program: the accept Op returned an invalid round");
        }
        round.tokens.assign(host + 2 * width, host + 2 * width + licensed_count);
        Lane& lane          = lane_at(lanes[i]);
        lane.round_position = lane.position;
        lane.round_drafts   = drafts[i];
    }
    return rounds;
}

void Program::commit_round(std::int32_t lane_index, std::int32_t kept,
                           DeviceExecutionView execution) {
    Lane& lane = lane_at(lane_index);
    if (lane.round_position < 0 || kept < 1 || kept > lane.round_drafts + 1) {
        throw std::logic_error("gemma4 Program: no round to commit, or an invalid kept count");
    }
    // The verify pass consumed p..p+drafts; the anchor and kept-1 tokens stay, and column kept-1 is
    // the state that produced the last kept token, the next anchor.
    lane.position = lane.round_position + kept;
    const std::size_t bytes =
        static_cast<std::size_t>(model_->config().hidden_size) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpyAsync(lane.hidden,
                               static_cast<std::uint8_t*>(lane.round_hidden) +
                                   static_cast<std::size_t>(kept - 1) * bytes,
                               bytes, cudaMemcpyDeviceToDevice, execution.stream));
    lane.cache.mark_stale(lane.round_position + lane.round_drafts + 1);
    lane.round_position = -1;
}

std::int32_t Program::sample(std::int32_t lane, DeviceExecutionView execution) {
    return sample(std::span(&lane, 1), execution).front();
}

std::vector<std::int32_t> Program::sample(std::span<const std::int32_t> lanes,
                                          DeviceExecutionView execution) {
    if (lanes.empty() || lanes.size() > lanes_.size()) {
        throw std::invalid_argument("gemma4 Program: sample takes one to lanes() lanes");
    }
    const cudaStream_t stream      = execution.stream;
    const std::int32_t vocabulary  = static_cast<std::int32_t>(model_->config().vocab_size);
    const std::size_t column_bytes = static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t);
    auto* drawn                    = static_cast<std::int32_t*>(sampling_[1]);
    auto* positions                = static_cast<std::int32_t*>(sampling_[2]);
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const Lane& lane = lane_at(lanes[i]);
        // A pageable source is staged before cudaMemcpyAsync returns.
        CUDA_CHECK(cudaMemcpyAsync(positions + i, &lane.position, sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        const Tensor column(static_cast<std::uint8_t*>(logits_.data) +
                                static_cast<std::size_t>(lane.column) * column_bytes,
                            DType::BF16, {vocabulary, 1});
        Tensor out(static_cast<void*>(drawn + i), DType::I32, {1});
        Tensor position(static_cast<void*>(positions + i), DType::I32, {1});
        // The draws run in stream order, so each may reuse the workspace the previous one used.
        sampling_workspace_.reset();
        ops::sample(column, out, vocabulary,
                    static_cast<const ops::SamplingConfig*>(sampling_[0]) + lanes[i], position,
                    ops::kSamplePurposeDecode, sampling_workspace_, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), drawn, lanes.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {sampled_host_.begin(),
            sampled_host_.begin() + static_cast<std::ptrdiff_t>(lanes.size())};
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
    logits_host_.resize(static_cast<std::size_t>(vocabulary) *
                        static_cast<std::size_t>(kScorePass));

    std::vector<float> scores;
    scores.reserve(ids.size() - static_cast<std::size_t>(first_target));
    // The sink sees the soft-capped logits the distribution is taken from, one row per scored
    // target, side by side and each `vocabulary` wide, which is the layout the KLD route reads. It
    // is handed a view of the staging buffer, so it must return before the next batch overwrites
    // it.
    LogitsSink sink          = std::move(sink_in);
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
                    logits_host_.data() + static_cast<std::size_t>(target - position - 1) *
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
                const std::size_t offset  = static_cast<std::size_t>(first - position - 1) *
                                            static_cast<std::size_t>(vocabulary);
                const std::size_t columns = static_cast<std::size_t>(last - first + 1);
                ScoredLogits view;
                view.values = std::span<const std::uint16_t>(
                    logits_host_.data() + offset, columns * static_cast<std::size_t>(vocabulary));
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
