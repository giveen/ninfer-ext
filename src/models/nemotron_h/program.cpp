#include "models/nemotron_h/program.h"

#include "core/layout.h"
#include "ninfer/ops/sliding_causal_attention.h"

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

// The staged expert pool's device bytes under the caller's cache policy: the two staged banks,
// the slot-cache slots the policy asks for beside them, and the pager's tables. It is what a
// Program holds for the routed experts, and it is the *elastic* half of a load -- the part a cache
// request sizes. The pool is not, however, reserved beside the residency budget: it is inside it,
// because the plan that decides the resident banks spends the same figure on both halves. A caller
// deriving that budget from `device_bytes` therefore takes this term out again, through
// `expert_pool_device_bytes`, or it reserves the pool twice and hands the resident banks one pool's
// worth less than the card can afford.
std::size_t staged_expert_bytes(const TextConfig& config, const EngineOptions& options) {
    const std::size_t pool = expert_pool_bytes(config, options.expert_cache.cache_slots);
    if (pool == 0) { return 0; }
    if (options.expert_cache.mode == ExpertCacheMode::Automatic) { return pool; }
    return options.expert_cache.explicit_bytes >= pool
               ? pool
               : 0;
}

// The Op's own key-split workspace, sized for a one-token step: that is the pass whose visible span
// makes the split worth taking (a wider pass's own shape gives the Op one split anyway), and an
// Op handed less than it would prefer takes its one-pass route instead, which is the same result.
std::size_t attention_workspace_bytes(const TextConfig& config, const StateGeometry& geometry) {
    if (geometry.attention_layers == 0) { return 0; }
    const AttentionConfig& attention = *config.attention;
    const ops::AttentionHeadGeometry heads{
        carried_attention_head(attention), static_cast<std::int32_t>(attention.num_attention_heads),
        static_cast<std::int32_t>(attention.num_key_value_heads)};
    return ops::sliding_causal_attention_workspace_bytes(heads, 1, geometry.capacity, 1);
}

// One pass's own operands and output: the ids it uploads, the logits its head leaves (one column
// per lane, which `HeadColumns::LastOnly` is what makes enough) and the attention route's
// workspace.
std::size_t pass_buffer_bytes(const TextConfig& config, const StateGeometry& geometry) {
    return static_cast<std::size_t>(geometry.workspace_tokens) * sizeof(std::int32_t) +
           static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(geometry.lanes) *
               sizeof(std::uint16_t) +
           attention_workspace_bytes(config, geometry);
}

// The sampler's storage: one device config per lane, the `[vocabulary]` occurrence counts its
// penalties read (one array per lane, so two lanes' counts never alias), one round's drawn tokens
// and logical positions, and the Op's own workspace for every parallel lane count in [1, lanes].
std::size_t sampling_buffer_bytes(const TextConfig& config, const StateGeometry& geometry) {
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    const std::size_t lanes       = static_cast<std::size_t>(geometry.lanes);
    return lanes * sizeof(ops::SamplingConfig) +
           lanes * static_cast<std::size_t>(vocabulary) * sizeof(std::int32_t) +
           2 * lanes * sizeof(std::int32_t) +
           ops::sampling_workspace_capacity_bytes(vocabulary, 1, geometry.lanes);
}

// The sizes the constructor's two arenas are built with. They are computed from the options alone
// rather than from the geometry the constructor body derives, because a member arena is constructed
// in the initializer list.
std::size_t pass_arena_bytes(const TextConfig& config, const EngineOptions& options) {
    return model_workspace_bytes(config, state_geometry(config, options).workspace_tokens);
}

std::size_t sampling_arena_bytes(const TextConfig& config, const EngineOptions& options) {
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    const StateGeometry geometry  = state_geometry(config, options);
    // The Op takes its workspace by reference even when it reports it needs none (a token domain a
    // single block covers), and a `DeviceArena` cannot hold zero bytes: one byte is the smallest
    // allocation there is, and the Op allocates nothing from it on that route.
    return std::max<std::size_t>(
        ops::sampling_workspace_capacity_bytes(vocabulary, 1, geometry.lanes), 1);
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
           model_workspace_bytes(config, geometry.workspace_tokens) +
           pass_buffer_bytes(config, geometry) + sampling_buffer_bytes(config, geometry) +
           staged_expert_bytes(config, options);
}

std::size_t Program::expert_pool_device_bytes(const TextConfig& config,
                                              const EngineOptions& options) {
    return
           staged_expert_bytes(config, options);
}

Program::Program(const Model& model, const EngineOptions& options, DeviceContext& device)
    : model_(&model), arena_(pass_arena_bytes(model.config(), options)),
      sampling_workspace_(sampling_arena_bytes(model.config(), options)) {
    const StateGeometry geometry  = state_geometry(model.config(), options);
    const TextConfig& config      = model.config();
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    capacity_                     = geometry.capacity;
    lanes_                        = geometry.lanes;
    chunk_                        = geometry.workspace_tokens;
    workspace_bytes_              = pass_arena_bytes(config, options);
    pass_bytes_                   = pass_buffer_bytes(config, geometry);
    sampling_bytes_               = sampling_buffer_bytes(config, geometry);

    LayoutBuilder builder;
    StatePlan plan = plan_state(geometry, builder);
    if (plan.allocation_bytes != 0) {
        mamba_backing_ = DeviceBuffer(plan.allocation_bytes);
        const DeviceSpan backing{mamba_backing_.p, mamba_backing_.bytes};
        if (plan.mamba_present) { mamba_.emplace(backing, plan.mamba); }
    }
    mamba_layout_ = std::move(plan.mamba);
    if (geometry.attention_layers != 0) {
        attention_.emplace(*config.attention, geometry.attention_layers, capacity_, lanes_);
    }
    state_allocation_bytes_ =
        plan.allocation_bytes + (attention_ ? attention_->allocated_bytes() : 0);
    lanes_state_.resize(static_cast<std::size_t>(lanes_));

    // The pass's own storage: the arena every pass runs against, the ids it uploads and the logits
    // the head leaves, one column per lane.
    ids_backing_    = DeviceBuffer(static_cast<std::size_t>(chunk_) * sizeof(std::int32_t));
    ids_            = Tensor(ids_backing_.p, DType::I32, {chunk_});
    logits_backing_ = DeviceBuffer(static_cast<std::size_t>(vocabulary) *
                                   static_cast<std::size_t>(lanes_) * sizeof(std::uint16_t));
    logits_         = Tensor(logits_backing_.p, DType::BF16, {vocabulary, lanes_});
    // The attention route's workspace, when the Op's own query reports one for a one-token step. An
    // unset tensor means the Op takes its one-pass route.
    const std::size_t attention_workspace = attention_workspace_bytes(config, geometry);
    if (attention_workspace != 0) {
        attention_workspace_backing_ = DeviceBuffer(attention_workspace);
        attention_workspace_         = Tensor(attention_workspace_backing_.p, DType::U8,
                                              {static_cast<std::int32_t>(attention_workspace)});
    }

    // The sampler. The counts are one array per lane because the Op's contract forbids two active
    // requests sharing one; the workspace is constructed with the object.
    sampling_backing_ =
        DeviceBuffer(static_cast<std::size_t>(lanes_) * sizeof(ops::SamplingConfig));
    counts_backing_    = DeviceBuffer(static_cast<std::size_t>(lanes_) *
                                      static_cast<std::size_t>(vocabulary) * sizeof(std::int32_t));
    drawn_backing_     = DeviceBuffer(static_cast<std::size_t>(lanes_) * sizeof(std::int32_t));
    positions_backing_ = DeviceBuffer(static_cast<std::size_t>(lanes_) * sizeof(std::int32_t));
    sampled_host_.resize(static_cast<std::size_t>(lanes_));

    // The empty state a new sequence starts from, written once so a Program that never created a
    // request still holds a state it documents rather than uninitialized memory: the zero frontier
    // a sequence's first pass begins from, and positions no token has written.
    if (mamba_) { mamba_->zero_all(device.stream); }
    if (attention_) { attention_->reset_all(device.stream); }

    // The staged banks. They are allocated here rather than at bind time because the split that
    // decided the resident banks did not: what a *resident* bank costs was uploaded with the
    // weights, and what staging costs is this pool. The plan decides both whether there is a pool
    // and how big it is -- a card that could not afford the staged banks planned none, and one that
    // could not afford the slot cache it was asked for planned a smaller one -- so the pool follows
    // the model's own residency rather than re-deriving it from the options.
    if(model.expert_residency().staging_bytes != 0) {
        experts_.emplace(model, expert_top_k(model.config()));
    }
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
    summary.pass_bytes             = pass_bytes_;
    summary.sampling_bytes         = sampling_bytes_;
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
    // the previous one's convolution history, recurrent plane or K/V, and its cursor returns to the
    // sequence's first position.
    if (mamba_) { mamba_->zero_slot(lane, execution.stream); }
    if (attention_) { attention_->reset(lane, execution.stream); }
    state.position = 0;
    state.in_use   = true;
    // A greedy sampler until the request installs its own: sampling is the caller's choice and a
    // lane that never configures one draws the argmax rather than refusing.
    configure(lane, ops::SamplingConfig{}, execution);
}

void Program::configure(std::int32_t lane, const ops::SamplingConfig& sampling,
                        DeviceExecutionView execution) {
    (void)occupied_lane(lane);
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    std::int32_t* const counts =
        static_cast<std::int32_t*>(counts_backing_.p) +
        static_cast<std::size_t>(lane) * static_cast<std::size_t>(vocabulary);
    CUDA_CHECK(cudaMemsetAsync(
        counts, 0, static_cast<std::size_t>(vocabulary) * sizeof(std::int32_t), execution.stream));
    ops::SamplingConfig installed = sampling;
    // The lane's own count array, and no mask: neither is the caller's to place.
    installed.token_counts = counts;
    installed.mask         = {};
    CUDA_CHECK(cudaMemcpyAsync(static_cast<ops::SamplingConfig*>(sampling_backing_.p) + lane,
                               &installed, sizeof(installed), cudaMemcpyHostToDevice,
                               execution.stream));
}

void Program::finish(std::int32_t lane) { release_lane(lane); }

void Program::abort(std::int32_t lane) { release_lane(lane); }

void Program::bind_state(std::int32_t lane, std::vector<BlockState>& blocks) {
    const TextConfig& config = model_->config();
    blocks.assign(config.num_hidden_layers(), BlockState{});
    // The pool's layers are the Mamba *blocks* in their own order, and the attention state's are
    // the attention blocks': a block index handed to either pool unchanged would bind another
    // block's state on a config whose block kinds interleave, which this one's do.
    std::uint32_t mamba_layer     = 0;
    std::uint32_t attention_layer = 0;
    for (std::size_t layer = 0; layer < blocks.size(); ++layer) {
        switch (config.block_kind(static_cast<std::uint32_t>(layer))) {
        case BlockKind::Mamba:
            if (!mamba_) {
                throw std::logic_error("nemotron_h Program: a Mamba block with no Mamba state "
                                       "pool, which the constructor refuses");
            }
            blocks[layer].conv      = mamba_->conv_slot(mamba_layer, lane);
            blocks[layer].recurrent = mamba_->recurrent_slot(mamba_layer, lane);
            ++mamba_layer;
            break;
        case BlockKind::Attention:
            if (!attention_) {
                throw std::logic_error("nemotron_h Program: an attention block with no attention "
                                       "state, which the constructor refuses");
            }
            blocks[layer].attention.state     = &*attention_;
            blocks[layer].attention.lane      = lane;
            blocks[layer].attention.layer     = attention_layer++;
            blocks[layer].attention.workspace = attention_workspace();
            break;
        case BlockKind::Moe:
            break;
        }
    }
}

void Program::run_pass(std::int32_t lane, const std::int32_t* ids, std::int32_t tokens,
                       DeviceExecutionView execution) {
    Lane& state               = occupied_lane(lane);
    const cudaStream_t stream = execution.stream;
    // The ids are uploaded per pass rather than kept resident from the caller's span: the pass is
    // enqueued and the caller's memory may go away before it runs. The buffer is sized for the
    // widest pass this Program issues, so the pass's own width is declared rather than inherited
    // from the allocation -- `forward_model` takes its token count from the tensor it is handed.
    CUDA_CHECK(cudaMemcpyAsync(ids_.data, ids,
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    const Tensor pass_ids(ids_.data, DType::I32, {tokens});
    bind_state(lane, pass_blocks_);
    const PassState pass{.per_layer = pass_blocks_, .first_position = state.position};
    // The head runs over the pass's last position only: that is the logits a draw reads, and the
    // other columns are the model's own values for positions no caller asked about.
    Tensor column = logits_column(lane);
    forward_model(*model_, pass_ids, arena_, column, execution, &pass, HeadColumns::LastOnly,
                  nullptr, experts_ ? &*experts_ : nullptr);
    if (attention_) { attention_->advance(lane); }
    state.position += tokens;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Program::prefill(std::int32_t lane, std::span<const std::int32_t> ids,
                      DeviceExecutionView execution) {
    const Lane& state = occupied_lane(lane);
    if (ids.empty()) {
        throw std::invalid_argument("nemotron_h Program: a prefill pass needs at least one token");
    }
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    for (const std::int32_t id : ids) {
        if (id < 0 || id >= vocabulary) {
            throw std::invalid_argument("nemotron_h Program: token " + std::to_string(id) +
                                        " is outside the model's vocabulary of " +
                                        std::to_string(vocabulary));
        }
    }
    if (static_cast<std::int64_t>(state.position) + static_cast<std::int64_t>(ids.size()) >
        static_cast<std::int64_t>(capacity_)) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) + " holds " +
                                    std::to_string(capacity_) +
                                    " tokens and the pass would take it"
                                    " past that");
    }
    // A prompt wider than one pass is the same computation in more calls: the state carries the
    // boundary, which is exactly what the passes' first positions state.
    for (std::int32_t done = 0; done < static_cast<std::int32_t>(ids.size());) {
        const std::int32_t step = std::min(chunk_, static_cast<std::int32_t>(ids.size()) - done);
        run_pass(lane, ids.data() + done, step, execution);
        done += step;
    }
}

void Program::decode(std::span<const std::int32_t> lanes, std::span<const std::int32_t> ids,
                     DeviceExecutionView execution) {
    if (lanes.empty() || lanes.size() != ids.size()) {
        throw std::invalid_argument("nemotron_h Program: a decode pass needs one id per lane");
    }
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const Lane& state = occupied_lane(lanes[row]);
        for (std::size_t before = 0; before < row; ++before) {
            if (lanes[before] == lanes[row]) {
                throw std::invalid_argument(
                    "nemotron_h Program: a decode pass names each lane once, "
                    "and lane " +
                    std::to_string(lanes[row]) + " is named twice");
            }
        }
        if (ids[row] < 0 || ids[row] >= vocabulary) {
            throw std::invalid_argument("nemotron_h Program: token " + std::to_string(ids[row]) +
                                        " is outside the model's vocabulary of " +
                                        std::to_string(vocabulary));
        }
        if (state.position + 1 > capacity_) {
            throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lanes[row]) +
                                        " holds " + std::to_string(capacity_) +
                                        " tokens, so it cannot consume another");
        }
    }
    // One lane per pass: `ops::mamba2_scan` has no batch axis, so the lanes share the weights'
    // reads but not one launch.
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        run_pass(lanes[row], ids.data() + row, 1, execution);
    }
}

void Program::decode(std::int32_t lane, std::int32_t id, DeviceExecutionView execution) {
    decode(std::span(&lane, 1), std::span(&id, 1), execution);
}

std::vector<std::int32_t> Program::sample(std::span<const std::int32_t> lanes,
                                          DeviceExecutionView execution) {
    if (lanes.empty() || lanes.size() > static_cast<std::size_t>(lanes_)) {
        throw std::invalid_argument("nemotron_h Program: sample takes one to lanes() lanes");
    }
    const cudaStream_t stream     = execution.stream;
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    auto* drawn                   = static_cast<std::int32_t*>(drawn_backing_.p);
    auto* positions               = static_cast<std::int32_t*>(positions_backing_.p);
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const Lane& lane = lane_at(lanes[row]);
        for (std::size_t before = 0; before < row; ++before) {
            if (lanes[before] == lanes[row]) {
                throw std::invalid_argument(
                    "nemotron_h Program: a lane draws once a round, and lane " +
                    std::to_string(lanes[row]) + " is named twice");
            }
        }
        // The position the draw is keyed by: the token this draw produced occupies it, which is
        // what makes the same sequence and seed reproduce the same tokens.
        const std::int32_t position = lane.position;
        CUDA_CHECK(cudaMemcpyAsync(positions + row, &position, sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        Tensor out(static_cast<void*>(drawn + row), DType::I32, {1});
        Tensor logical(static_cast<void*>(positions + row), DType::I32, {1});
        sampling_workspace_.reset();
        ops::sample(logits(lanes[row]), out, vocabulary,
                    static_cast<const ops::SamplingConfig*>(sampling_backing_.p) + lanes[row],
                    logical, ops::kSamplePurposeDecode, sampling_workspace_, stream);
        // The token counts the penalties read, which the draw itself increments: the next draw of
        // the lane sees this one.
    }
    CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), drawn, lanes.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {sampled_host_.begin(),
            sampled_host_.begin() + static_cast<std::ptrdiff_t>(lanes.size())};
}

std::int32_t Program::sample(std::int32_t lane, DeviceExecutionView execution) {
    return sample(std::span(&lane, 1), execution).front();
}

std::int32_t Program::position(std::int32_t lane) const { return lane_at(lane).position; }

Tensor Program::logits(std::int32_t lane) const { return logits_column(lane); }

Tensor Program::logits_column(std::int32_t lane) const {
    (void)lane_at(lane);
    return Tensor(static_cast<std::uint8_t*>(logits_.data) +
                      static_cast<std::size_t>(lane) * static_cast<std::size_t>(logits_.ne[0]) *
                          sizeof(std::uint16_t),
                  DType::BF16, {logits_.ne[0], 1});
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
