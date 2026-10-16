#include "models/nemotron_h/attention_state.h"

#include "core/checked_math.h"
#include "core/layout.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::nemotron_h {
namespace {

std::int32_t to_int32(std::uint64_t value, std::string_view what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("nemotron_h attention state: ") + std::string(what) +
                                  " exceeds the int32 extent a tensor is declared with");
    }
    return static_cast<std::int32_t>(value);
}

std::int32_t positive(std::uint64_t value, std::string_view what) {
    const std::int32_t narrowed = to_int32(value, what);
    if (narrowed <= 0) {
        throw std::invalid_argument(std::string("nemotron_h attention state: ") +
                                    std::string(what) + " must be positive");
    }
    return narrowed;
}

// A count that arrives already signed: a negative one is a caller's mistake rather than an extent
// that overflowed, so it gets its own message.
std::int32_t positive_count(std::int32_t value, std::string_view what) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("nemotron_h attention state: ") +
                                    std::string(what) + " must be positive");
    }
    return value;
}

// The head widths the Op's launcher admits.
bool admitted_head_width(std::int32_t head_dim) {
    return head_dim == 32 || head_dim == 64 || head_dim == 128 || head_dim == 256;
}

} // namespace

std::int32_t carried_attention_head(const AttentionConfig& attention) {
    const std::int32_t head = static_cast<std::int32_t>(attention.head_dim);
    if (head <= 0) {
        throw std::invalid_argument("nemotron_h attention state: the config's attention head is " +
                                    std::to_string(head) + " wide, which is not a head");
    }
    if (admitted_head_width(head)) { return head; }
    // The narrowest width the Op admits that holds the head's real dims, so the pad is the smallest
    // the domain allows rather than a fixed 256.
    if (head < 32) { return 32; }
    if (head < 64) { return 64; }
    if (head < 128) { return 128; }
    if (head < 256) { return 256; }
    throw std::invalid_argument(
        "nemotron_h attention state: no landed route for a " + std::to_string(head) +
        "-wide attention head: ops::sliding_causal_attention admits 32, 64, 128 and 256");
}

namespace {

// The three regions of one allocation, planned once so `device_bytes` and the constructor cannot
// disagree.
struct AttentionStatePlan {
    TensorRegion keys;
    TensorRegion values;
    TensorRegion positions;
    std::size_t bytes = 0;
};

AttentionStatePlan plan_state(std::int32_t head_dim, std::int32_t kv_heads, std::int32_t capacity,
                              std::int32_t lanes, std::uint32_t layers) {
    if (layers == 0) {
        throw std::invalid_argument("nemotron_h attention state: layers must be nonzero");
    }
    const std::int32_t planes = positive(
        checked_mul<std::uint64_t>(layers, static_cast<std::uint64_t>(lanes), "the state's planes"),
        "the state's plane count");
    const std::int32_t rows =
        positive(checked_mul<std::uint64_t>(capacity, static_cast<std::uint64_t>(lanes),
                                            "the state's position rows"),
                 "the state's position rows");

    LayoutBuilder builder;
    AttentionStatePlan plan;
    plan.keys = builder.add_tensor(
        DType::BF16, {head_dim, kv_heads, capacity, planes}, 256,
        "nemotron_h attention keys [head_dim, kv_heads, capacity, layer * lane]");
    plan.values = builder.add_tensor(
        DType::BF16, {head_dim, kv_heads, capacity, planes}, 256,
        "nemotron_h attention values [head_dim, kv_heads, capacity, layer * lane]");
    plan.positions = builder.add_tensor(DType::I32, {rows}, 256,
                                        "nemotron_h attention positions [lane][capacity]");
    plan.bytes     = builder.finish(256, "nemotron_h attention state");
    return plan;
}

void require_kv_operand(const Tensor& tensor, const char* role, std::int32_t head_dim,
                        std::int32_t kv_heads, std::int32_t tokens) {
    if (tensor.data == nullptr || tensor.dtype != DType::BF16 || !tensor.is_contiguous() ||
        tensor.ne[0] != head_dim || tensor.ne[1] != kv_heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("nemotron_h attention state: ") + role +
                                    " must be contiguous BF16 [" + std::to_string(head_dim) + ", " +
                                    std::to_string(kv_heads) + ", " + std::to_string(tokens) +
                                    ", 1]");
    }
}

void require_query_operand(const Tensor& tensor, const char* role, std::int32_t head_dim,
                           std::int32_t query_heads, std::int32_t tokens) {
    if (tensor.data == nullptr || tensor.dtype != DType::BF16 || !tensor.is_contiguous() ||
        tensor.ne[0] != head_dim || tensor.ne[1] != query_heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("nemotron_h attention state: ") + role +
                                    " must be contiguous BF16 [" + std::to_string(head_dim) + ", " +
                                    std::to_string(query_heads) + ", " + std::to_string(tokens) +
                                    ", 1]");
    }
}

} // namespace

std::size_t AttentionState::device_bytes(const AttentionConfig& attention, std::uint32_t layers,
                                         std::int32_t capacity, std::int32_t lanes) {
    const auto carried  = carried_attention_head(attention);
    const auto kv_heads = positive(attention.num_key_value_heads, "the attention KV head count");
    (void)positive_count(lanes, "the lane count");
    const AttentionStatePlan plan =
        plan_state(carried, kv_heads, positive_count(capacity, "the capacity"), lanes, layers);
    return plan.bytes;
}

AttentionState::AttentionState(const AttentionConfig& attention, std::uint32_t layers,
                               std::int32_t capacity, std::int32_t lanes)
    : head_dim_(positive(attention.head_dim, "the attention head width")),
      carried_head_(carried_attention_head(attention)),
      kv_heads_(positive(attention.num_key_value_heads, "the attention KV head count")),
      query_heads_(positive(attention.num_attention_heads, "the attention query head count")),
      capacity_(positive_count(capacity, "the capacity")), layers_(layers) {
    if (layers_ == 0) {
        throw std::invalid_argument("nemotron_h attention state: layers must be nonzero");
    }
    if (!ops::valid_attention_head_geometry(
            ops::AttentionHeadGeometry{carried_head_, query_heads_, kv_heads_})) {
        throw std::invalid_argument(
            "nemotron_h attention state: the query head count must be a multiple of the KV head "
            "count");
    }
    positive_count(lanes, "the lane count");

    const AttentionStatePlan plan = plan_state(carried_head_, kv_heads_, capacity_, lanes, layers_);
    state_                        = DeviceBuffer(plan.bytes);
    const DeviceSpan backing{state_.p, state_.bytes};
    keys_      = plan.keys.bind(backing);
    values_    = plan.values.bind(backing);
    positions_ = plan.positions.bind(backing);
    lanes_state_.resize(static_cast<std::size_t>(lanes));
    for (Lane& lane : lanes_state_) { lane.appended.assign(layers_, false); }
}

// The one device allocation is an RAII member and the views borrow it, so destruction is the
// release; the views must not outlive this object.
AttentionState::~AttentionState() = default;

std::size_t AttentionState::workspace_bytes(std::int32_t tokens) const {
    const ops::AttentionHeadGeometry geometry = route_geometry();
    return ops::sliding_causal_attention_workspace_bytes(
        geometry, positive_count(tokens, "the query token count"), capacity_, 1);
}

ops::AttentionHeadGeometry AttentionState::route_geometry() const {
    // The carried width is what the Op is handed; `carried_attention_head` already refused a head
    // with no route, so the width is one the launcher admits.
    return ops::AttentionHeadGeometry{carried_head_, query_heads_, kv_heads_};
}

void AttentionState::reset_all(cudaStream_t stream) {
    for (std::int32_t lane = 0; lane < lanes(); ++lane) { reset(lane, stream); }
}

void AttentionState::reset(std::int32_t lane, cudaStream_t stream) {
    Lane& state = lane_at(lane);
    if (state.pending != 0) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    " has an unadvanced step, so resetting it would drop the "
                                    "layers that already appended it");
    }
    state.filled  = 0;
    state.pending = 0;
    std::fill(state.appended.begin(), state.appended.end(), false);
    // Every slot unwritten: the rows are the state's only index, so writing them is what makes the
    // slots invisible.
    positions_host_.assign(static_cast<std::size_t>(capacity_), kUnwrittenPosition);
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::uint8_t*>(positions_.data) +
            static_cast<std::size_t>(lane) * capacity_ * sizeof(std::int32_t),
        positions_host_.data(), static_cast<std::size_t>(capacity_) * sizeof(std::int32_t),
        cudaMemcpyHostToDevice, stream));
}

std::int32_t AttentionState::filled(std::int32_t lane) const { return lane_at(lane).filled; }

std::int32_t AttentionState::pending(std::int32_t lane) const { return lane_at(lane).pending; }

AttentionStateView AttentionState::layer(std::int32_t lane, std::uint32_t layer) const {
    const Lane& state = lane_at(lane);
    if (layer >= layers_) {
        throw std::out_of_range("nemotron_h attention state: layer " + std::to_string(layer) +
                                " is not one of this state's " + std::to_string(layers_) +
                                " attention layers");
    }
    const std::size_t index       = layer_index(lane, layer);
    const std::size_t plane_bytes = plane_elements() * sizeof(std::uint16_t);
    return AttentionStateView{
        .keys   = Tensor(static_cast<std::uint8_t*>(keys_.data) + index * plane_bytes, DType::BF16,
                         {carried_head_, kv_heads_, capacity_, 1}),
        .values = Tensor(static_cast<std::uint8_t*>(values_.data) + index * plane_bytes,
                         DType::BF16, {carried_head_, kv_heads_, capacity_, 1}),
        .positions = Tensor(static_cast<std::uint8_t*>(positions_.data) +
                                static_cast<std::size_t>(lane) * capacity_ * sizeof(std::int32_t),
                            DType::I32, {capacity_, 1}),
        .filled    = state.filled,
    };
}

void AttentionState::append(std::int32_t lane, std::uint32_t layer, const Tensor& keys,
                            const Tensor& values, std::int32_t tokens, cudaStream_t stream) {
    Lane& state = lane_at(lane);
    if (layer >= layers_) {
        throw std::out_of_range("nemotron_h attention state: layer " + std::to_string(layer) +
                                " is not one of this state's " + std::to_string(layers_) +
                                " attention layers");
    }
    const std::int32_t count = positive_count(tokens, "the appended token count");
    if (state.pending != 0 && state.pending != count) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    "'s step appended " + std::to_string(state.pending) +
                                    " tokens to another layer, and layer " + std::to_string(layer) +
                                    " now appends " + std::to_string(count) +
                                    ": every attention layer of one step writes the same tokens");
    }
    if (state.appended[layer]) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    "'s layer " + std::to_string(layer) +
                                    " already appended this step");
    }
    if (state.filled + count > capacity_) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    " holds " + std::to_string(capacity_) +
                                    " tokens, and appending " + std::to_string(count) + " to its " +
                                    std::to_string(state.filled) + " exceeds that");
    }
    require_kv_operand(keys, "keys", carried_head_, kv_heads_, count);
    require_kv_operand(values, "values", carried_head_, kv_heads_, count);

    // The source is the projection's own output for this layer, already at the stored width, in the
    // Op's element order, so the tokens are one contiguous run in both the source and this lane's
    // plane.
    const std::size_t block_bytes =
        static_cast<std::size_t>(count) * carried_head_ * kv_heads_ * sizeof(std::uint16_t);
    const std::size_t offset =
        static_cast<std::size_t>(state.filled) * carried_head_ * kv_heads_ * sizeof(std::uint16_t);
    const std::size_t index       = layer_index(lane, layer);
    const std::size_t plane_bytes = plane_elements() * sizeof(std::uint16_t);
    CUDA_CHECK(
        cudaMemcpyAsync(static_cast<std::uint8_t*>(keys_.data) + index * plane_bytes + offset,
                        keys.data, block_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(static_cast<std::uint8_t*>(values_.data) + index * plane_bytes + offset,
                        values.data, block_bytes, cudaMemcpyDeviceToDevice, stream));

    // The positions are the sequence's own: slot `filled + i` holds the position `filled + i`,
    // which is what makes the append's keys visible to this step's query and nothing beyond it.
    positions_host_.resize(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        positions_host_[static_cast<std::size_t>(i)] = state.filled + i;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::uint8_t*>(positions_.data) +
            (static_cast<std::size_t>(lane) * capacity_ + state.filled) * sizeof(std::int32_t),
        positions_host_.data(), static_cast<std::size_t>(count) * sizeof(std::int32_t),
        cudaMemcpyHostToDevice, stream));

    state.appended[layer] = true;
    state.pending         = count;
}

void AttentionState::advance(std::int32_t lane) {
    Lane& state = lane_at(lane);
    if (state.pending == 0) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    " has no appended step to advance");
    }
    const auto missing = std::find(state.appended.begin(), state.appended.end(), false);
    if (missing != state.appended.end()) {
        throw std::invalid_argument(
            "nemotron_h attention state: lane " + std::to_string(lane) + "'s attention layer " +
            std::to_string(static_cast<std::uint32_t>(missing - state.appended.begin())) +
            " did not append this step, so advancing would hide " + std::to_string(state.pending) +
            " tokens from it");
    }
    state.filled += state.pending;
    state.pending = 0;
    std::fill(state.appended.begin(), state.appended.end(), false);
}

void AttentionState::attend(std::int32_t lane, std::uint32_t layer, const Tensor& query,
                            Tensor& out, const Tensor& workspace,
                            DeviceExecutionView execution) const {
    const Lane& state = lane_at(lane);
    if (layer >= layers_) {
        throw std::out_of_range("nemotron_h attention state: layer " + std::to_string(layer) +
                                " is not one of this state's " + std::to_string(layers_) +
                                " attention layers");
    }
    if (state.pending == 0) {
        throw std::invalid_argument("nemotron_h attention state: lane " + std::to_string(lane) +
                                    " has nothing appended to attend over");
    }
    if (!state.appended[layer]) {
        throw std::invalid_argument(
            "nemotron_h attention state: lane " + std::to_string(lane) + "'s layer " +
            std::to_string(layer) +
            " has not appended this step, and the reference appends a token's key and value before "
            "its query runs");
    }
    require_query_operand(query, "query", carried_head_, query_heads_, state.pending);
    require_query_operand(out, "out", carried_head_, query_heads_, state.pending);

    const AttentionStateView view = this->layer(lane, layer);
    // The query's own positions are the appended ones; the keys' are the lane's whole row, whose
    // unwritten slots hold kUnwrittenPosition and are therefore invisible.
    const Tensor position_q(static_cast<std::uint8_t*>(view.positions.data) +
                                static_cast<std::size_t>(state.filled) * sizeof(std::int32_t),
                            DType::I32, {state.pending, 1});
    const Tensor nothing;
    // The config's own head width, not the stored one: the carry is exact only because the pad dims
    // are zero, and the scale is what keeps a padded query or key from moving a score by anything
    // other than the zeros it contributes.
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    ops::sliding_causal_attention(query, view.keys, view.values, position_q, nothing,
                                  view.positions, nothing, nothing, nothing, route_geometry(),
                                  kUnboundedWindow, scale, out, workspace, execution);
}

AttentionState::Lane& AttentionState::lane_at(std::int32_t lane) {
    if (lane < 0 || lane >= lanes()) {
        throw std::out_of_range("nemotron_h attention state: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes()) +
                                " lanes");
    }
    return lanes_state_[static_cast<std::size_t>(lane)];
}

const AttentionState::Lane& AttentionState::lane_at(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes()) {
        throw std::out_of_range("nemotron_h attention state: lane " + std::to_string(lane) +
                                " is not one of this state's " + std::to_string(lanes()) +
                                " lanes");
    }
    return lanes_state_[static_cast<std::size_t>(lane)];
}

std::size_t AttentionState::layer_index(std::int32_t lane, std::uint32_t layer) const {
    return static_cast<std::size_t>(lane) * layers_ + layer;
}

std::size_t AttentionState::plane_elements() const {
    return static_cast<std::size_t>(carried_head_) * kv_heads_ * capacity_;
}

} // namespace ninfer::models::nemotron_h
