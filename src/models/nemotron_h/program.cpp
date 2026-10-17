#include "models/nemotron_h/program.h"

#include "core/layout.h"
#include "ninfer/ops/sliding_causal_attention.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <utility>

namespace ninfer::models::nemotron_h {
namespace {

// `ops::mamba2_scan`'s own key extent -- a constant of its kernel, not of this config; the replay's
// grouped B/C planes are zero-extended to it exactly as a pass's are.
constexpr std::int32_t kScanKeyExtent = 128;

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

// Copies a pass's tap blocks out while it runs. The composition reports each block's residual as
// `BlockStage::Block`, and that tensor borrows the pass's own storage -- so the copy has to happen
// inside the callback, on the pass's stream, which is why this is an observer rather than a read a
// caller does afterwards.
class TapObserver : public BlockObserver {
public:
    TapObserver(PassTaps& taps, cudaStream_t stream) : taps_(&taps), stream_(stream) {}

    void on_stage(std::size_t layer, BlockStage stage, const Tensor& tensor) const override {
        if (stage != BlockStage::Block) { return; }
        const auto found = std::find(taps_->layers.begin(), taps_->layers.end(), layer);
        if (found == taps_->layers.end()) { return; }
        const std::size_t index   = static_cast<std::size_t>(found - taps_->layers.begin());
        const std::size_t columns = static_cast<std::size_t>(taps_->features.ne[0]);
        if (tensor.dtype != DType::BF16 || tensor.ne[1] != taps_->tokens ||
            static_cast<std::size_t>(tensor.ne[0]) * taps_->layers.size() != columns) {
            throw std::logic_error("nemotron_h Program: block " + std::to_string(layer) +
                                   "'s residual is not the tap the capture was armed for");
        }
        const std::size_t width = static_cast<std::size_t>(tensor.ne[0]) * dtype_size(DType::BF16);
        auto* const destination = static_cast<std::byte*>(taps_->features.data) + index * width;
        CUDA_CHECK(cudaMemcpy2DAsync(destination, static_cast<std::size_t>(taps_->features.nb[1]),
                                     tensor.data, static_cast<std::size_t>(tensor.nb[1]), width,
                                     static_cast<std::size_t>(taps_->tokens),
                                     cudaMemcpyDeviceToDevice, stream_));
        captured_ |= 1U << index;
    }

    [[nodiscard]] std::uint32_t captured() const noexcept { return captured_; }

    [[nodiscard]] std::uint32_t expected() const noexcept {
        return taps_->layers.size() == 32
                   ? ~0U
                   : static_cast<std::uint32_t>((1U << taps_->layers.size()) - 1U);
    }

private:
    PassTaps* const taps_;
    cudaStream_t stream_;
    mutable std::uint32_t captured_ = 0;
};

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
    return options.expert_cache.explicit_bytes >= pool ? pool : 0;
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

// The Mamba pool's bytes for one lane: every layer's convolution history and FP32 recurrent plane.
// It is the image a speculative round snapshots before its verify pass and restores after, because
// neither the convolution history nor the recurrent plane can be un-advanced.
std::size_t lane_mamba_bytes(const LinearAttentionStatePoolSpec& pool) {
    const std::size_t conv      = static_cast<std::size_t>(pool.conv_channels) *
                                  static_cast<std::size_t>(pool.conv_width) *
                                  dtype_size(pool.conv_dtype);
    const std::size_t recurrent = static_cast<std::size_t>(pool.key_head_dim) *
                                  static_cast<std::size_t>(pool.value_head_dim) *
                                  static_cast<std::size_t>(pool.value_heads) * sizeof(float);
    return static_cast<std::size_t>(pool.layers) * (conv + recurrent);
}

// The draft's own device storage, when a round will run: the context ring and one forward's
// scratch (`DraftContext::plan`), one lane's Mamba image per lane, the verify block's
// `[vocabulary, block_size]` logits, and the greedy sampler's per-column operands and workspace.
// Pure, and zero for a model with no draft.
// The block (anchor plus mask rows) a round may issue. A LiLiCorr draft stores it; a plain DFlash's
// is the request's `draft_tokens + 1`, which is zero when the caller has not asked for speculation.
std::int32_t draft_block_width(const models::draft::Spec& spec, const EngineOptions& options) {
    if (spec.flavor == models::draft::Spec::Flavor::Lilicorr) {
        return positive(spec.block_size, "the draft's block size");
    }
    if (options.speculative.backend == SpeculativeBackend::None ||
        options.speculative.draft_tokens == 0) {
        return 0;
    }
    return static_cast<std::int32_t>(options.speculative.draft_tokens) + 1;
}

std::size_t draft_device_bytes(const TextConfig& config,
                               const std::optional<models::draft::Spec>& spec,
                               const EngineOptions& options, const StateGeometry& geometry) {
    if (!spec) { return 0; }
    const std::int32_t width = draft_block_width(*spec, options);
    if (width == 0) { return 0; }
    const std::int32_t vocabulary = static_cast<std::int32_t>(config.vocab_size);
    // A plain DFlash's ring holds the request's context; a LiLiCorr draft's holds its window and
    // ignores the capacity handed here.
    std::size_t bytes = DraftContext::plan(config, *spec, geometry.lanes, width, options.max_context)
                            .device_bytes();
    if (geometry.mamba_layers != 0) {
        bytes += lane_mamba_bytes(geometry.mamba) * static_cast<std::size_t>(geometry.lanes);
    }
    bytes += static_cast<std::size_t>(vocabulary) * static_cast<std::size_t>(width) *
             dtype_size(DType::BF16);
    bytes +=
        static_cast<std::size_t>(width) * (sizeof(ops::SamplingConfig) + 2 * sizeof(std::int32_t));
    bytes += std::max<std::size_t>(
        ops::sampling_workspace_capacity_bytes(vocabulary, 1, std::max(geometry.lanes, width)), 1);
    return bytes;
}

double millis_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
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

std::size_t Program::device_bytes(const Config& config, const EngineOptions& options) {
    const StateGeometry geometry = state_geometry(config.text, options);
    return device_bytes(config.text, options) +
           draft_device_bytes(config.text, config.draft, options, geometry);
}

std::size_t Program::device_bytes(const Model& model, const EngineOptions& options) {
    const StateGeometry geometry = state_geometry(model.config(), options);
    return device_bytes(model.config(), options) +
           draft_device_bytes(model.config(), model.draft(), options, geometry);
}

std::size_t Program::expert_pool_device_bytes(const TextConfig& config,
                                              const EngineOptions& options) {
    return staged_expert_bytes(config, options);
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
    draft_width_ = model.draft() ? draft_block_width(*model.draft(), options) : 0;
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
    if (model.expert_residency().staging_bytes != 0) {
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
    // A new sequence's draft ring is empty for the same reason: the previous request's target
    // residuals are not this one's context.
    if (draft_) { draft_->reset(lane, execution); }
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

void Program::capture_taps(std::span<const std::uint32_t> layers) {
    if (layers.empty()) {
        stop_capturing_taps();
        return;
    }
    std::vector<std::uint32_t> armed(layers.begin(), layers.end());
    if (!std::is_sorted(armed.begin(), armed.end()) ||
        std::adjacent_find(armed.begin(), armed.end()) != armed.end()) {
        throw std::invalid_argument("nemotron_h Program: the draft's tap layers must be ascending "
                                    "and distinct");
    }
    const std::uint32_t blocks = static_cast<std::uint32_t>(model_->config().num_hidden_layers());
    for (const std::uint32_t layer : armed) {
        if (layer >= blocks) {
            throw std::invalid_argument("nemotron_h Program: tap block " + std::to_string(layer) +
                                        " is not one of the model's " + std::to_string(blocks) +
                                        " blocks");
        }
    }
    if (armed.size() > 32) {
        throw std::invalid_argument("nemotron_h Program: a capture holds at most 32 taps");
    }
    taps_.layers         = std::move(armed);
    taps_.tokens         = 0;
    taps_.first_position = 0;
    // The buffer follows the arms rather than the pass, so re-arming for another request keeps the
    // bytes it already has. It is only viewed here when it was allocated for exactly this many
    // layers: a different arm changes the row count, and a view of the old bytes would then read
    // past them -- `run_pass` allocates instead.
    const std::int32_t rows =
        static_cast<std::int32_t>(taps_.layers.size() * model_->config().hidden_size);
    taps_.features = taps_capacity_ > 0 && taps_rows_ == rows && taps_backing_.p != nullptr
                         ? Tensor(taps_backing_.p, DType::BF16, {rows, taps_capacity_})
                         : Tensor{};
}

void Program::stop_capturing_taps() noexcept {
    taps_.layers.clear();
    taps_.features       = Tensor{};
    taps_.tokens         = 0;
    taps_.first_position = 0;
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
    // The draft's conditioning is captured inside the pass, on the pass's stream: the residual
    // tensors it reads are the pass's own storage and are gone once it returns.
    if (taps_.armed()) {
        const std::int32_t rows =
            static_cast<std::int32_t>(taps_.layers.size() * model_->config().hidden_size);
        if (taps_capacity_ < tokens || taps_rows_ != rows) {
            taps_backing_ =
                DeviceBuffer(static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) *
                             dtype_size(DType::BF16));
            taps_capacity_ = tokens;
            taps_rows_     = rows;
        }
        taps_.features       = Tensor(taps_backing_.p, DType::BF16, {rows, tokens});
        taps_.tokens         = tokens;
        taps_.first_position = state.position;
    }
    const TapObserver tap_observer(taps_, stream);
    const BlockObserver* const observer = taps_.armed() ? &tap_observer : nullptr;
    forward_model(*model_, pass_ids, arena_, column, execution, &pass, HeadColumns::LastOnly,
                  observer, experts_ ? &*experts_ : nullptr);
    if (taps_.armed() && tap_observer.captured() != tap_observer.expected()) {
        throw std::logic_error("nemotron_h Program: the pass did not publish every tap block's "
                               "residual, so the capture is incomplete");
    }
    // An engaged draft takes this pass's columns as context: a committed pass is exactly the set of
    // positions the drafter conditions on, so the ring is filled by the passes themselves rather
    // than by one capture that would have to hold the whole prompt. The verify pass is not this
    // path -- it bypasses `run_pass` -- so a discarded block never reaches the ring.
    if (draft_ && taps_.armed()) {
        draft_->condition(lane, taps_.features, taps_.first_position, execution);
    }
    if (attention_) { attention_->advance(lane); }
    state.position += tokens;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Program::verify_pass(std::int32_t lane, const std::int32_t* ids, std::int32_t tokens,
                          DeviceExecutionView execution) {
    Lane& state               = occupied_lane(lane);
    const cudaStream_t stream = execution.stream;
    CUDA_CHECK(cudaMemcpyAsync(ids_.data, ids,
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    const Tensor pass_ids(ids_.data, DType::I32, {tokens});
    bind_state(lane, pass_blocks_);
    const PassState pass{.per_layer = pass_blocks_, .first_position = state.position};
    // Every column's logits: the acceptance test is each row's argmax. The buffer is sized for the
    // widest block, so a shorter round views its first `tokens` columns -- the layout is the same
    // (`[vocabulary, columns]`, vocabulary fastest), only the column count differs.
    Tensor logits(draft_logits_backing_.p, DType::BF16,
                  {static_cast<std::int32_t>(model_->config().vocab_size), tokens});
    // The block's residual taps are this round's conditioning when the commit is a replay: the
    // committed prefix is the first `accepted + 1` columns of what this pass captures, so the
    // capture is armed here exactly as a committed pass arms it.
    if (taps_.armed()) {
        const std::int32_t rows =
            static_cast<std::int32_t>(taps_.layers.size() * model_->config().hidden_size);
        if (taps_capacity_ < tokens || taps_rows_ != rows) {
            taps_backing_ = DeviceBuffer(static_cast<std::size_t>(rows) *
                                         static_cast<std::size_t>(tokens) * dtype_size(DType::BF16));
            taps_capacity_ = tokens;
            taps_rows_     = rows;
        }
        taps_.features       = Tensor(taps_backing_.p, DType::BF16, {rows, tokens});
        taps_.tokens         = tokens;
        taps_.first_position = state.position;
    }
    const TapObserver tap_observer(taps_, stream);
    const BlockObserver* const observer = taps_.armed() ? &tap_observer : nullptr;
    // The pass with a state replay is one fixed sequence of kernel launches over fixed buffers --
    // the ids and the logits are this Program's own -- so it is captured once per lane and launched
    // as a graph. The tap copies are part of the capture; their destinations are the capture's own
    // plane. A staged expert pool branches on the host, so a pass that uses one stays eager.
    if (replay_ready_) { replay_capture_.columns = tokens; }
    const auto forward = [&] {
        forward_model(*model_, pass_ids, arena_, logits, execution, &pass, HeadColumns::Every,
                      observer, experts_ ? &*experts_ : nullptr,
                      replay_ready_ ? &replay_capture_ : nullptr);
    };
    if (!experts_.has_value()) {
        DecodeGraphDefinition& definition = verify_graphs_[static_cast<std::size_t>(lane)];
        DecodeGraphExecutable& executable = verify_execs_[static_cast<std::size_t>(lane)];
        if (!executable.ready()) {
            definition.capture(stream, forward);
            executable.instantiate(definition);
            executable.upload(stream);
            if (taps_.armed() && tap_observer.captured() != tap_observer.expected()) {
                throw std::logic_error("nemotron_h Program: the verify pass did not publish every "
                                       "tap block's residual, so the capture is incomplete");
            }
        }
        executable.launch(stream);
    } else {
        forward();
        if (taps_.armed() && tap_observer.captured() != tap_observer.expected()) {
            throw std::logic_error("nemotron_h Program: the verify pass did not publish every tap "
                                   "block's residual, so the capture is incomplete");
        }
    }
    if (attention_) { attention_->advance(lane); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Program::replay_commit(std::int32_t lane, std::int32_t accepted,
                            DeviceExecutionView execution) {
    Lane& state               = occupied_lane(lane);
    const cudaStream_t stream = execution.stream;
    const std::int32_t columns = accepted + 1;
    // The Mamba state: the accepted prefix of the verify pass's own transition, from the state the
    // round restored.
    mamba_replay_apply(*model_, pass_blocks_, replay_capture_, replay_scratch_, columns,
                       execution);
    // Attention: the rows the verify appended are the committed ones, so the frontier moves forward
    // to `columns` and the tail the round rejected is left past it.
    if (attention_) { attention_->rewind(lane, state.position + columns); }
    // The head's logits at the last accepted column, which is the token a plain decode of this
    // prefix would have left for the next draw.
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    CUDA_CHECK(cudaMemcpy2DAsync(logits_column(lane).data, static_cast<std::size_t>(vocabulary) *
                                                              sizeof(std::uint16_t),
                                 static_cast<std::uint8_t*>(draft_logits_.data) +
                                     static_cast<std::size_t>(accepted) *
                                         static_cast<std::size_t>(vocabulary) *
                                         sizeof(std::uint16_t),
                                 static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t),
                                 static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t), 1,
                                 cudaMemcpyDeviceToDevice, stream));
    // The conditioning: the pass's capture holds the block's columns in order, and the committed
    // prefix is its first `columns`. The view narrows to them, because `taps_.features` is what the
    // next round's proposal reads as its context width.
    taps_.features = Tensor(taps_.features.data, DType::BF16,
                            {taps_.features.ne[0], columns});
    taps_.tokens = columns;
    if (draft_ && taps_.armed()) {
        draft_->condition(lane, taps_.features, taps_.first_position, execution);
    }
    state.position += columns;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Program::snapshot_state(std::int32_t lane, cudaStream_t stream) {
    if (!mamba_) { return; }
    auto* const destination = static_cast<std::uint8_t*>(draft_rollback_.p);
    std::size_t offset      = 0;
    for (std::uint32_t layer = 0; layer < mamba_->layer_count(); ++layer) {
        for (std::int32_t side = 0; side < 2; ++side) {
            const Tensor plane =
                side == 0 ? mamba_->conv_slot(layer, lane) : mamba_->recurrent_slot(layer, lane);
            if (offset + plane.bytes() > draft_rollback_.bytes) {
                throw std::logic_error("nemotron_h Program: the Mamba snapshot is smaller than the "
                                       "state it has to hold");
            }
            CUDA_CHECK(cudaMemcpyAsync(destination + offset, plane.data, plane.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            offset += plane.bytes();
        }
    }
    if (offset != draft_rollback_.bytes) {
        throw std::logic_error("nemotron_h Program: the Mamba snapshot holds " +
                               std::to_string(draft_rollback_.bytes) + " bytes for a state of " +
                               std::to_string(offset));
    }
}

void Program::restore_state(std::int32_t lane, cudaStream_t stream) {
    if (!mamba_) { return; }
    const auto* const source = static_cast<const std::uint8_t*>(draft_rollback_.p);
    std::size_t offset       = 0;
    for (std::uint32_t layer = 0; layer < mamba_->layer_count(); ++layer) {
        for (std::int32_t side = 0; side < 2; ++side) {
            const Tensor plane =
                side == 0 ? mamba_->conv_slot(layer, lane) : mamba_->recurrent_slot(layer, lane);
            CUDA_CHECK(cudaMemcpyAsync(plane.data, source + offset, plane.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            offset += plane.bytes();
        }
    }
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

// ---- The draft --------------------------------------------------------------------------------

void Program::engage_draft(DeviceExecutionView execution) {
    if (draft_) { return; }
    // The caller's queued passes touch the state this is about to allocate beside, so they finish
    // before it does; the draft's own construction path writes on the default stream, and the
    // device barrier after it is what makes those writes visible to every later stream.
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    const std::optional<models::draft::Spec>& spec = model_->draft();
    if (!spec) {
        throw std::invalid_argument("nemotron_h Program: this artifact carries no draft, so there "
                                    "is nothing to engage");
    }
    if (draft_width_ < 2) {
        throw std::invalid_argument(
            "nemotron_h Program: the draft has no block width; a plain DFlash's is the request's "
            "draft_tokens + 1, so start the Program with speculation enabled");
    }
    const std::int32_t width      = draft_width_;
    const std::int32_t vocabulary = static_cast<std::int32_t>(model_->config().vocab_size);
    if (mamba_) { draft_rollback_ = DeviceBuffer(lane_mamba_bytes(mamba_layout_.spec)); }
    draft_logits_backing_ = DeviceBuffer(static_cast<std::size_t>(vocabulary) *
                                         static_cast<std::size_t>(width) * dtype_size(DType::BF16));
    draft_logits_         = Tensor(draft_logits_backing_.p, DType::BF16, {vocabulary, width});
    draft_configs_backing_ =
        DeviceBuffer(static_cast<std::size_t>(width) * sizeof(ops::SamplingConfig));
    draft_positions_backing_ = DeviceBuffer(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    draft_drawn_backing_     = DeviceBuffer(static_cast<std::size_t>(width) * sizeof(std::int32_t));
    draft_drawn_host_.resize(static_cast<std::size_t>(width));
    // Every column verifies greedily, and no occurrence-count array is named: a verify draw must
    // not move the request's own counts, and the acceptance contract is the target's argmax.
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(width));
    draft_configs_backing_.copy_from_host(configs.data(),
                                          configs.size() * sizeof(ops::SamplingConfig));
    const std::size_t workspace_bytes = std::max<std::size_t>(
        ops::sampling_workspace_capacity_bytes(vocabulary, 1, std::max(lanes_, width)), 1);
    draft_sampling_workspace_backing_ = DeviceBuffer(workspace_bytes);
    draft_sampling_workspace_.emplace(
        DeviceSpan{draft_sampling_workspace_backing_.p, draft_sampling_workspace_backing_.bytes});
    draft_.emplace(*model_, *spec, lanes_, draft_width_, capacity_);
    verify_graphs_.resize(static_cast<std::size_t>(lanes_));
    verify_execs_.resize(static_cast<std::size_t>(lanes_));
    // The state replay's operands and one layer's scratch, planned for the widest block a round can
    // issue. Only this family's Mamba blocks take a capture; the rest keep a null pair.
    if (mamba_) {
        const MambaConfig& mamba   = *model_->config().mamba;
        const std::int32_t channels = static_cast<std::int32_t>(mamba.conv_channels());
        const std::int32_t heads    = static_cast<std::int32_t>(mamba.num_heads);
        const std::int32_t inner    = static_cast<std::int32_t>(mamba.inner_size());
        const std::int32_t groups   = static_cast<std::int32_t>(mamba.n_groups);
        const std::size_t columns   = static_cast<std::size_t>(draft_width_);
        const std::size_t per_block =
            (static_cast<std::size_t>(channels) + heads) * columns * sizeof(std::uint16_t);
        replay_capture_backing_ = DeviceBuffer(per_block * mamba_layout_.spec.layers);
        replay_capture_.conv_in.resize(model_->config().num_hidden_layers());
        replay_capture_.dt.resize(model_->config().num_hidden_layers());
        auto* base = static_cast<std::uint8_t*>(replay_capture_backing_.p);
        std::size_t offset = 0;
        for (std::uint32_t block = 0; block < model_->config().num_hidden_layers(); ++block) {
            if (model_->config().block_kind(block) != BlockKind::Mamba) { continue; }
            replay_capture_.conv_in[block] = Tensor(base + offset, DType::BF16,
                                                    {channels, draft_width_});
            offset += static_cast<std::size_t>(channels) * columns * sizeof(std::uint16_t);
            replay_capture_.dt[block] = Tensor(base + offset, DType::BF16, {heads, draft_width_});
            offset += static_cast<std::size_t>(heads) * columns * sizeof(std::uint16_t);
        }
        if (offset != replay_capture_backing_.bytes) {
            throw std::logic_error("nemotron_h Program: the replay capture holds " +
                                   std::to_string(replay_capture_backing_.bytes) +
                                   " bytes for a state of " + std::to_string(offset));
        }
        const std::size_t scratch_floats =
            (static_cast<std::size_t>(channels) + 2 * static_cast<std::size_t>(inner) +
             static_cast<std::size_t>(kScanKeyExtent) * groups * 2) *
            columns * sizeof(std::uint16_t);
        replay_scratch_backing_ = DeviceBuffer(scratch_floats);
        auto* scratch_base = static_cast<std::uint8_t*>(replay_scratch_backing_.p);
        std::size_t scratch_offset = 0;
        const auto carve = [&](std::int32_t rows) {
            Tensor view(scratch_base + scratch_offset, DType::BF16, {rows, draft_width_});
            scratch_offset += static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t);
            return view;
        };
        replay_scratch_.conv_out = carve(channels);
        replay_scratch_.scan_x   = carve(inner);
        replay_scratch_.scan_b   = carve(kScanKeyExtent * groups);
        replay_scratch_.scan_c   = carve(kScanKeyExtent * groups);
        replay_scratch_.scan_out = carve(inner);
        if (scratch_offset != replay_scratch_backing_.bytes) {
            throw std::logic_error("nemotron_h Program: the replay scratch holds " +
                                   std::to_string(replay_scratch_backing_.bytes) +
                                   " bytes for a state of " + std::to_string(scratch_offset));
        }
        replay_ready_ = true;
    }
    // The conditioning is the draft's own taps and nobody else's: arming them here is what keeps a
    // caller from capturing another block set for it. The plane is sized for the widest pass this
    // Program can issue before it is armed: a later pass that grew it would leave the verify's
    // captured pass writing a plane that had been replaced.
    {
        const std::int32_t rows =
            static_cast<std::int32_t>(spec->target_layer_ids.size() *
                                      model_->config().hidden_size);
        if (taps_capacity_ < chunk_ || taps_rows_ != rows) {
            taps_backing_ = DeviceBuffer(static_cast<std::size_t>(rows) *
                                         static_cast<std::size_t>(chunk_) *
                                         dtype_size(DType::BF16));
            taps_capacity_ = chunk_;
            taps_rows_     = rows;
        }
        taps_.features = Tensor(taps_backing_.p, DType::BF16, {rows, chunk_});
    }
    capture_taps(spec->target_layer_ids);
    // The ring's empty state and the head's constant tables are written on the construction path's
    // default stream, which this orders against the caller's.
    CUDA_CHECK(cudaDeviceSynchronize());
}

DraftRoundReport Program::draft_round(std::int32_t lane, std::int32_t anchor,
                                      DeviceExecutionView execution,
                                      const DraftObserver* observer, bool single_row_commit,
                                      std::int32_t requested_slots) {
    Lane& state = occupied_lane(lane);
    if (!draft_) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) +
                                    " holds no draft state; engage_draft() before the request's "
                                    "first pass");
    }
    const models::draft::Spec& spec = *model_->draft();
    // The block is one anchor row plus the mask rows scored this round. The head takes the slot
    // count per round and its per-slot tables are indexed 0..slots-1, so a shorter block is a
    // parameter rather than an artifact: `requested_slots` is a measurement arm's width.
    const std::int32_t maximum =
        draft_width_ > 1 ? draft_width_ - 1 : static_cast<std::int32_t>(spec.block_size) - 1;
    const std::int32_t slots = requested_slots > 0 ? requested_slots : maximum;
    const std::int32_t width        = slots + 1;
    const std::int32_t vocabulary   = static_cast<std::int32_t>(model_->config().vocab_size);
    if (slots < 1 || slots > maximum) {
        throw std::logic_error("nemotron_h Program: the draft's block size leaves no mask slot");
    }
    if (!taps_.armed()) {
        throw std::invalid_argument("nemotron_h Program: the draft's tap capture is not armed, so "
                                    "the drafter has no conditioning");
    }
    if (taps_.layers != spec.target_layer_ids) {
        throw std::invalid_argument("nemotron_h Program: the tap capture names blocks other than "
                                    "the draft's own, so its context would be another model's "
                                    "residual stream");
    }
    if (taps_.tokens <= 0) {
        throw std::invalid_argument("nemotron_h Program: no pass has run since the capture was "
                                    "armed, so the drafter has no context");
    }
    if (taps_.first_position + taps_.tokens != state.position) {
        throw std::invalid_argument(
            "nemotron_h Program: the tap capture ends at position " +
            std::to_string(taps_.first_position + taps_.tokens) + " and the lane's frontier is " +
            std::to_string(state.position) +
            "; a round conditions on the committed prefix and nothing else");
    }
    if (state.position + width > capacity_) {
        throw std::invalid_argument("nemotron_h Program: lane " + std::to_string(lane) + " holds " +
                                    std::to_string(capacity_) + " tokens and a verify block of " +
                                    std::to_string(width) + " would take it past that");
    }
    if (anchor < 0 || anchor >= vocabulary) {
        throw std::invalid_argument("nemotron_h Program: the round's anchor token " +
                                    std::to_string(anchor) + " is outside the vocabulary");
    }

    DraftRoundReport report;
    report.anchor          = anchor;
    report.anchor_position = state.position;
    // The drafter's own block is the anchor followed by mask rows: its head proposes one token per
    // mask slot, each from the same query, which is what makes a block drafter one forward per
    // round rather than one per token.
    std::vector<std::int32_t> mask_block(static_cast<std::size_t>(width),
                                         static_cast<std::int32_t>(spec.mask_token_id));
    mask_block[0] = anchor;

    // ---- 1. The proposal: the draft's body over the context and the head's walk -----------------
    auto began = std::chrono::steady_clock::now();
    const DraftRound round{.taps                   = taps_.features,
                           .first_context_position = taps_.first_position,
                           .first_block_position   = state.position,
                           .slots                  = slots,
                           .block_ids              = mask_block.data()};
    report.proposed = draft_->propose(lane, round, execution, observer);
    report.draft_ms = millis_since(began);
    if (static_cast<std::int32_t>(report.proposed.size()) != slots) {
        throw std::logic_error("nemotron_h draft: the walk returned " +
                               std::to_string(report.proposed.size()) + " ids for a block of " +
                               std::to_string(width) + " rows");
    }

    // ---- 2. The image the verify pass will overwrite --------------------------------------------
    // The convolution history and the FP32 recurrent plane cannot be un-advanced, so the round
    // restores them and re-runs the accepted prefix; the attention K/V rolls back by frontier
    // alone.
    began = std::chrono::steady_clock::now();
    snapshot_state(lane, execution.stream);
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    report.restore_ms = millis_since(began);

    // ---- 3. The target over the block it is being asked to verify ------------------------------
    // The target consumes the anchor followed by the *proposed* tokens, never the mask rows: row
    // `r` then holds a real token at a real position, so its argmax is the target's own prediction
    // for the position draft slot `r` filled. Feeding mask rows instead makes every row past the
    // anchor predict from a token no sequence contains, which measures zero acceptance and reads
    // exactly like a bad drafter.
    std::vector<std::int32_t> block_ids(static_cast<std::size_t>(width), anchor);
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        block_ids[static_cast<std::size_t>(slot) + 1] =
            report.proposed[static_cast<std::size_t>(slot)];
    }
    began = std::chrono::steady_clock::now();
    verify_pass(lane, block_ids.data(), width, execution);
    report.verify_ms = millis_since(began);

    // ---- 4. The target's own argmax at every column: the acceptance decision --------------------
    began = std::chrono::steady_clock::now();
    std::vector<std::int32_t> block_positions(static_cast<std::size_t>(width));
    for (std::int32_t column = 0; column < width; ++column) {
        block_positions[static_cast<std::size_t>(column)] = state.position + column;
    }
    draft_positions_backing_.copy_from_host(block_positions.data(),
                                            block_positions.size() * sizeof(std::int32_t));
    draft_sampling_workspace_->reset();
    Tensor drawn(draft_drawn_backing_.p, DType::I32, {width});
    Tensor logical(draft_positions_backing_.p, DType::I32, {width});
    Tensor verify_logits(draft_logits_backing_.p, DType::BF16, {vocabulary, width});
    ops::sample(verify_logits, drawn, vocabulary,
                static_cast<const ops::SamplingConfig*>(draft_configs_backing_.p), logical,
                ops::kSamplePurposeSpeculativeAccept, *draft_sampling_workspace_, execution.stream);
    CUDA_CHECK(cudaMemcpyAsync(draft_drawn_host_.data(), draft_drawn_backing_.p,
                               block_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, execution.stream));
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    report.accept_ms = millis_since(began);

    std::int32_t accepted = 0;
    while (accepted < slots && report.proposed[static_cast<std::size_t>(accepted)] ==
                                   draft_drawn_host_[static_cast<std::size_t>(accepted)]) {
        ++accepted;
    }
    report.accepted = accepted;
    report.bonus    = draft_drawn_host_[static_cast<std::size_t>(accepted)];
    report.verified.assign(draft_drawn_host_.begin(),
                           draft_drawn_host_.begin() + static_cast<std::ptrdiff_t>(width));

    // ---- 5. Back to the frontier the round started from -----------------------------------------
    began = std::chrono::steady_clock::now();
    restore_state(lane, execution.stream);
    // With a replay the attention frontier stays where the verify left it and the replay moves it
    // to the committed column; a rewind is only ever backwards, so rewinding twice would be wrong.
    const bool replay_commit =
        replay_ready_ && !single_row_commit && replay_capture_.columns >= accepted + 1;
    if (attention_ && !replay_commit) { attention_->rewind(lane, state.position); }
    CUDA_CHECK(cudaStreamSynchronize(execution.stream));
    report.restore_ms += millis_since(began);

    // ---- 6. The committed prefix, as a pass of its own ------------------------------------------
    // This is the pass whose state and taps are real: `accepted + 1` tokens -- the anchor and the
    // accepted drafts -- so the next round conditions on committed positions only, and the lane's
    // cursor moves exactly where a plain decode would have left it.
    report.committed.assign(static_cast<std::size_t>(accepted) + 1, anchor);
    for (std::int32_t index = 0; index < accepted; ++index) {
        report.committed[static_cast<std::size_t>(index) + 1] =
            report.proposed[static_cast<std::size_t>(index)];
    }
    // A measurement arm, never the shipping path: committing one row at a time makes the taps the
    // next round conditions on come from width-1 passes, which is the numerics a plain decode
    // produces. It costs one pass per token instead of one per round.
    began = std::chrono::steady_clock::now();
    if (single_row_commit) {
        for (std::int32_t index = 0; index < accepted + 1; ++index) {
            run_pass(lane, report.committed.data() + index, 1, execution);
        }
    } else if (replay_commit) {
        // The verify pass already computed the accepted prefix: its Mamba transition operands and
        // its attention rows are this round's committed state, so the commit needs no pass.
        this->replay_commit(lane, accepted, execution);
    } else {
        run_pass(lane, report.committed.data(), accepted + 1, execution);
    }
    report.commit_ms = millis_since(began);
    return report;
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
