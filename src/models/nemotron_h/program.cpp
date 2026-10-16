#include "models/nemotron_h/program.h"

#include "core/layout.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace ninfer::models::nemotron_h {
namespace {

// The one narrowing every derived extent goes through, so a config whose numbers do not fit the
// pools' int32 extents is refused instead of wrapped into a smaller state.
std::int32_t narrow(std::uint64_t value, std::string_view what) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("nemotron_h Program: ") + std::string(what) +
                                  " exceeds the int32 extent the state pools are declared with");
    }
    return static_cast<std::int32_t>(value);
}

std::int32_t positive(std::uint64_t value, std::string_view what) {
    const std::int32_t narrowed = narrow(value, what);
    if (narrowed <= 0) {
        throw std::invalid_argument(std::string("nemotron_h Program: ") + std::string(what) +
                                    " must be positive");
    }
    return narrowed;
}

std::uint32_t layer_count(const TextConfig& config, BlockKind kind) {
    return static_cast<std::uint32_t>(
        std::count(config.block_kinds.begin(), config.block_kinds.end(), kind));
}

// The convolution's history: one sample per tap but the current one. A kernel of fewer than two
// taps has no history to keep, which is a config the state cannot represent rather than a
// zero-width plane.
std::uint64_t conv_width(const MambaConfig& mamba) {
    if (mamba.conv_kernel < 2) {
        throw std::invalid_argument("nemotron_h Program: the Mamba convolution's kernel must have "
                                    "at least two taps for its "
                                    "history to have a width");
    }
    return mamba.conv_kernel - 1;
}

// Everything the state layouts are planned from, validated once: both the constructor and the pure
// `device_bytes` report see only a geometry the state can hold exactly.
struct StateGeometry {
    std::uint32_t mamba_layers     = 0;
    std::uint32_t attention_layers = 0;
    LinearAttentionStatePoolSpec mamba{};
    std::int32_t capacity         = 0;
    std::int32_t lanes            = 0;
    std::int32_t workspace_tokens = 0;
};

StateGeometry state_geometry(const TextConfig& config, const EngineOptions& options) {
    StateGeometry geometry;
    geometry.capacity = positive(options.max_context, "EngineOptions::max_context");
    geometry.lanes    = positive(options.max_concurrency, "EngineOptions::max_concurrency");
    geometry.workspace_tokens =
        positive(options.prefill_chunk.value_or(kDefaultPrefillChunk), "the prefill chunk");

    // No quantized KV is wired for this family: the attention state's planes are BF16 because that
    // is the dtype `ops::sliding_causal_attention` declares its operands with, and a codec would
    // have to be decoded into that scratch per call.
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument(
            "nemotron_h Program: the attention K/V is BF16 because that is the dtype "
            "ops::sliding_causal_attention declares its operands with, and no quantized K/V codec "
            "is wired for this family's one K/V route");
    }

    geometry.mamba_layers = layer_count(config, BlockKind::Mamba);
    if (geometry.mamba_layers != 0) {
        if (!config.mamba) {
            throw std::invalid_argument(
                "nemotron_h Program: a Mamba block requires the mamba shape in the config");
        }
        const MambaConfig& mamba = *config.mamba;
        // The recurrent plane is the state `ops::mamba2_scan` reads and writes, whose key extent is
        // a constant of its kernel. A narrower configured state is carried by zero-extending the
        // key axis, which is exact; a wider one has no route, and shrinking it here would silently
        // drop state the model needs.
        if (mamba.state_size > static_cast<std::uint32_t>(kMambaScanStateExtent)) {
            throw std::invalid_argument(
                "nemotron_h Program: the config's Mamba state size is wider than the " +
                std::to_string(kMambaScanStateExtent) +
                " key extent ops::mamba2_scan's state plane is pinned to, and the scan has no "
                "route "
                "for a wider state");
        }
        geometry.mamba = LinearAttentionStatePoolSpec{
            .layers = geometry.mamba_layers,
            .conv_channels =
                positive(mamba.conv_channels(), "the Mamba convolution's channel count"),
            .conv_width     = positive(conv_width(mamba), "the Mamba convolution's history width"),
            .value_heads    = positive(mamba.num_heads, "the Mamba head count"),
            .value_head_dim = positive(mamba.head_dim, "the Mamba head width"),
            .key_head_dim   = kMambaScanStateExtent,
            .slot_count     = geometry.lanes,
            .conv_dtype     = DType::BF16,
        };
    }

    geometry.attention_layers = layer_count(config, BlockKind::Attention);
    if (geometry.attention_layers != 0) {
        if (!config.attention) {
            throw std::invalid_argument("nemotron_h Program: an attention block requires the "
                                        "attention shape in the config");
        }
        const AttentionConfig& attention = *config.attention;
        // The state itself is the config's own geometry -- head width, KV head count, one plane per
        // layer per lane -- and `AttentionState` validates it, including the divisibility of the
        // query head count. The three extents are checked here only so that a degenerate shape is
        // refused before any allocation.
        (void)positive(attention.head_dim, "the attention head width");
        (void)positive(attention.num_key_value_heads, "the attention KV head count");
        (void)positive(attention.num_attention_heads, "the attention query head count");
    }
    return geometry;
}

// The planned Mamba state: its layout, so the reported bytes are the planner's own.
struct StatePlan {
    LinearAttentionStatePoolLayout mamba{};
    std::size_t allocation_bytes = 0;
    bool mamba_present           = false;
};

StatePlan plan_state(const StateGeometry& geometry, LayoutBuilder& builder) {
    StatePlan plan;
    if (geometry.mamba_layers != 0) {
        plan.mamba         = plan_linear_attention_state_pool(builder, geometry.mamba);
        plan.mamba_present = true;
    }
    plan.allocation_bytes = builder.finish(256, "nemotron_h state");
    return plan;
}

// The refusal every computing entry point makes. It names the entry point, so a log or a test can
// tell which one was driven, and the missing piece in the same words kMissingStatePersistence
// documents.
[[noreturn]] void refuse_missing_state(std::string_view entry) {
    throw std::logic_error(std::string("nemotron_h Program: ") + std::string(entry) +
                           " cannot run: " + std::string(kMissingStatePersistence));
}

} // namespace

std::size_t Program::device_bytes(const TextConfig& config, const EngineOptions& options) {
    const StateGeometry geometry = state_geometry(config, options);
    LayoutBuilder builder;
    const StatePlan plan = plan_state(geometry, builder);
    return plan.allocation_bytes +
           (geometry.attention_layers != 0
                ? AttentionState::device_bytes(*config.attention, geometry.attention_layers,
                                               geometry.capacity, geometry.lanes)
                : 0) +
           model_workspace_bytes(config, geometry.workspace_tokens);
}

Program::Program(const Model& model, const EngineOptions& options, DeviceContext& device)
    : model_(&model) {
    const StateGeometry geometry = state_geometry(model.config(), options);
    capacity_                    = geometry.capacity;
    lanes_                       = geometry.lanes;
    workspace_bytes_             = model_workspace_bytes(model.config(), geometry.workspace_tokens);

    LayoutBuilder builder;
    StatePlan plan = plan_state(geometry, builder);
    if (plan.allocation_bytes != 0) {
        mamba_backing_ = DeviceBuffer(plan.allocation_bytes);
        const DeviceSpan backing{mamba_backing_.p, mamba_backing_.bytes};
        if (plan.mamba_present) { mamba_.emplace(backing, plan.mamba); }
    }
    mamba_layout_ = std::move(plan.mamba);
    if (geometry.attention_layers != 0) {
        attention_.emplace(*model.config().attention, geometry.attention_layers, capacity_, lanes_);
    }
    state_allocation_bytes_ =
        plan.allocation_bytes + (attention_ ? attention_->allocated_bytes() : 0);
    lanes_state_.resize(static_cast<std::size_t>(lanes_));

    // The empty state a new sequence starts from, written once so a Program that never created a
    // request still holds a state it documents rather than uninitialized memory: the zero frontier
    // the landed forward's prefill begins from, and positions no token has written.
    if (mamba_) { mamba_->zero_all(device.stream); }
    if (attention_) { attention_->reset_all(device.stream); }
}

// The two device allocations are RAII members and the pools borrow them, so destruction is the
// release: there is no partially built object to unwind and no raw pointer to free by hand.
Program::~Program() = default;

StateMemorySummary Program::memory_summary() const noexcept {
    StateMemorySummary summary;
    for (const LayoutRegion& region : mamba_layout_.conv) {
        summary.mamba_conv_bytes += region.bytes;
    }
    for (const LayoutRegion& region : mamba_layout_.recurrent) {
        summary.mamba_recurrent_bytes += region.bytes;
    }
    if (attention_) {
        summary.attention_kv_bytes =
            attention_->keys_bytes() + attention_->values_bytes() + attention_->positions_bytes();
    }
    summary.workspace_bytes        = workspace_bytes_;
    summary.state_allocation_bytes = state_allocation_bytes_;
    return summary;
}

void Program::create(std::int32_t lane, DeviceExecutionView execution) {
    Lane& state = lane_at(lane);
    if (state.in_use) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) +
                                    " already holds a request, and two sequences cannot share one "
                                    "lane's state");
    }
    // The state of a new sequence is empty, so a lane reissued to another request never inherits
    // the previous one's convolution history, recurrent plane or K/V.
    if (mamba_) { mamba_->zero_slot(lane, execution.stream); }
    if (attention_) { attention_->reset(lane, execution.stream); }
    state.in_use = true;
}

void Program::finish(std::int32_t lane) { release_lane(lane); }

void Program::abort(std::int32_t lane) { release_lane(lane); }

void Program::prefill(std::int32_t lane, std::span<const std::int32_t> ids,
                      DeviceExecutionView execution) {
    (void)execution;
    (void)occupied_lane(lane);
    if (ids.empty()) {
        throw std::invalid_argument("nemotron_h Program: a prefill pass needs at least one token");
    }
    refuse_missing_state("prefill");
}

void Program::decode(std::span<const std::int32_t> lanes, std::span<const std::int32_t> ids,
                     DeviceExecutionView execution) {
    (void)execution;
    if (lanes.size() != ids.size()) {
        throw std::invalid_argument("nemotron_h Program: a decode pass needs one id per lane");
    }
    for (const std::int32_t lane : lanes) { (void)occupied_lane(lane); }
    refuse_missing_state("decode");
}

void Program::decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution) {
    (void)id;
    (void)execution;
    (void)occupied_lane(lane);
    refuse_missing_state("decode");
}

void Program::commit(std::int32_t lane, std::int32_t kept, DeviceExecutionView execution) {
    (void)kept;
    (void)execution;
    (void)occupied_lane(lane);
    refuse_missing_state("commit");
}

void Program::discard(std::int32_t lane) {
    (void)occupied_lane(lane);
    refuse_missing_state("discard");
}

Program::Lane& Program::lane_at(std::int32_t lane) {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h Program: lane " + std::to_string(lane) +
                                " is not one of this Program's " + std::to_string(lanes_) +
                                " lanes");
    }
    return lanes_state_[static_cast<std::size_t>(lane)];
}

const Program::Lane& Program::lane_at(std::int32_t lane) const {
    if (lane < 0 || lane >= lanes_) {
        throw std::out_of_range("nemotron_h Program: lane " + std::to_string(lane) +
                                " is not one of this Program's " + std::to_string(lanes_) +
                                " lanes");
    }
    return lanes_state_[static_cast<std::size_t>(lane)];
}

Program::Lane& Program::occupied_lane(std::int32_t lane) {
    Lane& state = lane_at(lane);
    if (!state.in_use) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) +
                                    " holds no request, so there is no sequence to drive");
    }
    return state;
}

void Program::release_lane(std::int32_t lane) {
    Lane& state = lane_at(lane);
    if (!state.in_use) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) +
                                    " holds no request, so there is nothing to release");
    }
    state.in_use = false;
}

} // namespace ninfer::models::nemotron_h
