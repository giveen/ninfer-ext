#include "runtime/engine/nemotron_instance.h"

#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/nemotron_h/load.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::runtime {
namespace nemotron_contract {
namespace {

// The refusals the merged cache machinery asks for. This model runs root-only with the context
// cache switched off, so none of these is reached; each names itself rather than pretending the
// feature exists.
[[noreturn]] void unsupported(const char* what) {
    throw std::logic_error(std::string("Nemotron-H runs without the context cache; ") + what +
                           " is not implemented for it");
}

// The refusal for execution entry points the family Program owns but a lane that never completed a
// pass cannot reach: the tokens a decode or a commit would carry do not exist yet, so there is
// nothing to hand the family Program and no second refusal to invent. The missing piece is quoted
// in the family's own words.
[[noreturn]] void refuse_execution(const char* entry) {
    throw std::logic_error(std::string("nemotron_h Engine Program: ") + entry + " cannot run: " +
                           std::string(models::nemotron_h::kMissingStatePersistence));
}

} // namespace

// ---- refusals -----------------------------------------------------------------------------------

qwen::PressureTargetHandle PressurePlanningSession::identity_target(PlanningCandidateId) const {
    unsupported("pressure planning");
}

qwen::PressureTargetHandle PressurePlanningSession::root_maximal_target(PlanningCandidateId) {
    unsupported("pressure planning");
}

qwen::PressureTargetHandle PressurePlanningSession::maximal_target(PlanningCandidateId) {
    unsupported("pressure planning");
}

qwen::PressureTargetHandle
PressurePlanningSession::recency_maximal_target(PlanningCandidateId, std::uint32_t,
                                                std::span<const std::uint32_t>, bool) {
    unsupported("pressure planning");
}

std::uint32_t PressurePlanningSession::ranked_owner_count() const {
    unsupported("pressure planning");
}

void PressurePlanningSession::set_eviction_licence(std::uint32_t, std::span<const std::uint32_t>) {
    unsupported("pressure planning");
}

qwen::PressureConstructionCursor
PressurePlanningSession::begin_construction(qwen::PressureTargetHandle, bool) {
    unsupported("pressure planning");
}

PressureConstructionStep
PressurePlanningSession::next_construction_option(qwen::PressureConstructionCursor&) {
    unsupported("pressure planning");
}

void PressurePlanningSession::choose_construction(qwen::PressureConstructionCursor&,
                                                  PressureConstructionOptionId) {
    unsupported("pressure planning");
}

std::optional<qwen::PressureTargetHandle>
PressurePlanningSession::construction_target(const qwen::PressureConstructionCursor&) {
    unsupported("pressure planning");
}

PressureTargetGuidance PressurePlanningSession::guidance(qwen::PressureTargetHandle) {
    unsupported("pressure planning");
}

qwen::AssessedPressureTarget PressurePlanningSession::assess(qwen::PressureTargetHandle) {
    unsupported("pressure planning");
}

qwen::PreparedPressureExpansion
PressurePlanningSession::prepare_expansion(qwen::PressureTargetHandle, std::uint32_t) {
    unsupported("pressure planning");
}

qwen::PressureExpansionView
PressurePlanningSession::commit_expansion(qwen::PreparedPressureExpansion&&) {
    unsupported("pressure planning");
}

PrefillWork
PressurePlanningSession::shared_capture_split_prefill_work(const qwen::AssessedPressureTarget&,
                                                           const PreparedPrompt&,
                                                           std::span<const std::uint32_t>) const {
    unsupported("pressure planning");
}

std::optional<ResourcePlan> PressurePlanningSession::seal(qwen::AssessedPressureTarget&&,
                                                          const PreparedPrompt&,
                                                          FinalScheduleIntent) {
    unsupported("pressure planning");
}

std::optional<qwen::CapturePressurePlan>
PressurePlanningSession::seal_capture(qwen::AssessedPressureTarget&&) {
    unsupported("pressure planning");
}

PressurePlanningSession NemotronProgram::begin_pressure_planning(
    std::span<const AdmissionCandidate* const>, std::span<const PlanningCandidateId>,
    std::span<const qwen::ContinuationHandle* const>, std::span<const PlanningOwnerId>,
    std::span<const qwen::SharedPrefixHandle* const>, std::span<const PlanningOwnerId>,
    std::span<const PlanningOwnerId>) {
    unsupported("pressure planning");
}

qwen::CaptureAssessment NemotronProgram::inspect_capture(const qwen::CaptureOffer&,
                                                         const qwen::SharedPrefixHandle*,
                                                         const qwen::SharedPrefixHandle*,
                                                         std::optional<CheckpointRef>, bool) const {
    unsupported("prefix capture");
}

std::vector<CheckpointRecoveryAlternativeWork>
NemotronProgram::checkpoint_recovery_work(const qwen::ContinuationHandle&, CheckpointRef) const {
    unsupported("checkpoint recovery");
}

std::vector<CheckpointRecoveryAlternativeWork>
NemotronProgram::checkpoint_recovery_work(const qwen::SharedPrefixHandle&, CheckpointRef) const {
    unsupported("checkpoint recovery");
}

qwen::CapturePressurePlanningSession NemotronProgram::begin_capture_pressure_planning(
    const qwen::CaptureAssessment&, std::span<const qwen::ContinuationHandle* const>,
    std::span<const PlanningOwnerId>, std::span<const qwen::SharedPrefixHandle* const>,
    std::span<const PlanningOwnerId>, std::span<const PlanningOwnerId>) {
    unsupported("prefix capture");
}

bool NemotronProgram::shared_capture_matches(const qwen::CaptureOffer&,
                                             const qwen::SharedPrefixHandle&) const {
    unsupported("prefix capture");
}

void NemotronProgram::skip_capture(qwen::CaptureOffer&&) { unsupported("prefix capture"); }

ContextTransactionReserveStatus
NemotronProgram::reserve_active_capture(qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*,
                                        const qwen::SharedPrefixHandle*,
                                        std::optional<CheckpointRef>, bool, CancellationFlagView) {
    unsupported("prefix capture");
}

ContextTransactionReserveStatus NemotronProgram::reserve_active_capture_with_pressure(
    qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*, const qwen::SharedPrefixHandle*,
    std::optional<CheckpointRef>, bool, qwen::CapturePressurePlan&&, CancellationFlagView) {
    unsupported("prefix capture");
}

// ---- Program ------------------------------------------------------------------------------------

NemotronProgram::NemotronProgram(const models::nemotron_h::Model& model,
                                 const EngineOptions& options, DeviceContext& device)
    : model_(model), device_(device), execution_(device.execution_view()),
      program_(model, options, device),
      lanes_(static_cast<std::size_t>(options.max_concurrency)),
      prefill_chunk_(options.prefill_chunk.value_or(models::nemotron_h::kDefaultPrefillChunk)) {
    if (prefill_chunk_ == 0) {
        throw std::invalid_argument("nemotron_h Engine Program: the prefill chunk must be positive");
    }
}

std::vector<float> NemotronProgram::causal_score(PreparedPrompt&&, std::uint32_t,
                                                 const LogitsSink&) {
    // A scoring pass starts from the zero state the landed forward already has, so this is not the
    // state-persistence gap: the family's Program simply exposes no scoring entry point yet.
    throw std::logic_error("nemotron_h Engine Program: causal scoring is not implemented: the "
                           "family's Program has no scoring entry point");
}

MemorySummary NemotronProgram::memory_summary() const {
    const models::nemotron_h::StateMemorySummary state = program_.memory_summary();
    const artifact::MaterializationStats& storage    = model_.storage_stats();
    MemorySummary summary{};
    summary.device                   = device_.device;
    summary.max_context              = static_cast<std::uint32_t>(program_.capacity());
    summary.kv_capacity              = static_cast<std::uint32_t>(program_.capacity());
    summary.kv_cache                 = KvCacheStorage::BFloat16;
    summary.sequence.capacity_bytes  = state.state_allocation_bytes;
    summary.sequence.used_bytes      = state.state_allocation_bytes;
    summary.sequence.peak_used_bytes = state.state_allocation_bytes;
    summary.weights.capacity_bytes   = storage.h2d_bytes;
    summary.weights.used_bytes       = storage.h2d_bytes;
    summary.weights.peak_used_bytes  = storage.h2d_bytes;
    summary.prefill_chunk            = prefill_chunk_;
    summary.idle_prefill_chunk       = prefill_chunk_;
    // The pass's transient arena is not allocated here: the landed forward takes one from its
    // caller, and this Program allocates none. What a request needs beside the state is reported by
    // the instance's KvCapacityResolution.runtime_reservation_bytes, which is exactly the state
    // plus one pass at this chunk width.
    return summary;
}

RequestBasePlan NemotronProgram::plan_request(const PreparedPrompt& prompt,
                                              const ResolvedExecutionOptions& options) {
    if (prompt.ids.empty()) throw std::invalid_argument("prompt must contain tokens");
    const std::uint32_t capacity = static_cast<std::uint32_t>(program_.capacity());
    if (prompt.ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    const std::int64_t vocabulary = static_cast<std::int64_t>(model_.config().vocab_size);
    for (const TokenId id : prompt.ids) {
        if (id < 0 || id >= vocabulary) {
            throw std::invalid_argument("prompt contains token outside the public token domain");
        }
    }
    const ResolvedSamplingParameters& sampling = options.sampling;
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty) || sampling.top_p < 0.0F ||
        sampling.top_p > 1.0F || sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("sampling parameters are out of range");
    }

    RequestBasePlan base;
    RequestPlanSummary& summary     = base.summary_;
    summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.ids.size());
    summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_out = capacity - summary.prompt_tokens + 1U;
    summary.effective_output_tokens  = std::min(options.requested_output_tokens, capacity_out);
    summary.effective_limit_reason   = options.requested_output_tokens <= capacity_out
                                           ? FinishReason::OutputLimit
                                           : FinishReason::ContextCapacity;
    summary.prefix_reuse_path        = PrefixReusePath::Root;
    summary.publish_continuation     = false;
    // This family has no vision component, so the prefill work is text alone; the image markers a
    // prompt could carry are refused by the frontend before a request is planned.
    base.prefill_work_ = make_prefill_work(0, summary.prompt_tokens, 0, 0, prefill_chunk_);
    const std::uint64_t prefill_units =
        1ULL + (static_cast<std::uint64_t>(summary.prompt_tokens) - 1ULL) / prefill_chunk_;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    summary.service_work_quanta = prefill_units + decode_units;
    base.sampling_              = sampling;
    return base;
}

bool NemotronProgram::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return base.summary().prompt_tokens >= 1 &&
           base.summary().prompt_tokens <= static_cast<std::uint32_t>(program_.capacity());
}

std::optional<AdmissionCandidate>
NemotronProgram::inspect_admission(const PreparedPrompt&, const RequestBasePlan& base,
                                   LaneId destination, const qwen::ContinuationHandle* source,
                                   const qwen::SharedPrefixHandle* shared_source,
                                   std::optional<CheckpointRef> checkpoint, bool) {
    // Only the root exists: nothing is ever catalogued, so there is no source to resume from.
    if (source != nullptr || shared_source != nullptr || checkpoint) return std::nullopt;
    if (destination.value >= lanes_.size()) return std::nullopt;
    AdmissionCandidate candidate;
    candidate.summary_                          = base.summary();
    candidate.sampling_                         = base.sampling_;
    candidate.lane_                             = destination.value;
    IdentityMaterializationAssessment& identity = candidate.identity_;
    identity.physical_status                    = lanes_[destination.value].phase == Phase::Free
                                                      ? MaterializationPhysicalStatus::Feasible
                                                      : MaterializationPhysicalStatus::Infeasible;
    // A root has no source to retain; the controller reads Retain as "keep a source" and refuses it.
    identity.source_mode                         = PrivateSourceMode::ConsumeToActive;
    identity.machine_work.remaining_prefill_work = base.prefill_work_;
    identity.machine_work.reused_prompt_tokens   = 0;
    identity.pressure_may_change_machine_work    = false;
    identity.expandable                          = false;
    identity.projection_work                     = 1;
    identity.assessment_digest                   = revision_;
    return candidate;
}

std::optional<ResourcePlan> NemotronProgram::seal_identity(const AdmissionCandidate& candidate,
                                                           const PreparedPrompt&,
                                                           FinalScheduleIntent) {
    if (candidate.identity_.physical_status != MaterializationPhysicalStatus::Feasible ||
        lanes_[candidate.lane_].phase != Phase::Free) {
        return std::nullopt;
    }
    AdmissionCandidate copy;
    copy.summary_  = candidate.summary_;
    copy.identity_ = candidate.identity_;
    copy.sampling_ = candidate.sampling_;
    copy.lane_     = candidate.lane_;
    return ResourcePlan(std::move(copy), resource_revision());
}

PrefillWork NemotronProgram::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                                               const PreparedPrompt&,
                                                               std::span<const std::uint32_t>) {
    return candidate.identity_.machine_work.remaining_prefill_work;
}

ContextTransactionReserveStatus
NemotronProgram::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                            CancellationFlagView) {
    if (transaction_ || plan.revision_ != resource_revision() ||
        lanes_[plan.admission_.lane_].phase != Phase::Free) {
        return ContextTransactionReserveStatus::Aborted;
    }
    transaction_.emplace(Transaction{.lane     = plan.admission_.lane_,
                                     .prompt   = std::move(prompt),
                                     .sampling = plan.admission_.sampling_});
    return ContextTransactionReserveStatus::Reserved;
}

ContextTransactionProgress NemotronProgram::progress_context_transaction(CancellationFlagView) {
    if (!transaction_) throw std::logic_error("no context transaction is open");
    Transaction& transaction = *transaction_;
    Lane& lane               = lanes_[transaction.lane];
    // A root materialization is an empty lane. The family Program owns what "empty" means for this
    // model -- the Mamba frontier and the recurrent plane of its slot zeroed, and the attention
    // positions of its K/V unwritten -- so the lane is handed to it rather than zeroed here.
    program_.create(static_cast<std::int32_t>(transaction.lane), execution_);
    lane.phase       = Phase::Prefill;
    lane.epoch       = next_epoch_++;
    lane.prompt      = std::move(transaction.prompt.ids);
    lane.timings     = {};
    lane.speculative = {};
    bump_revision();
    MaterializationResult result;
    result.status    = ContextTransactionStatus::Published;
    result.published = StartResult{SequenceHandle(this, transaction.lane, lane.epoch)};
    return result;
}

void NemotronProgram::finalize_context_transaction() noexcept { transaction_.reset(); }

PrefillProgress NemotronProgram::advance_prefill(SequenceHandle sequence, PrefillPace,
                                                 ExecutionTiming*, TokenMaskProvider*) {
    Lane* lane = valid_lane(sequence);
    if (lane == nullptr || lane->phase != Phase::Prefill) {
        throw std::logic_error("prefill advanced on a sequence that is not prefilling");
    }
    // The lane's whole prompt goes to the family Program in one call: chunking exists to bound a
    // pass's width, and the pass never runs, so no cursor is advanced here. `prefill` refuses
    // because the landed forward keeps no state -- a lane driven through it would compute a
    // zero-state model -- and that refusal is the one the caller must see, in its own words.
    const std::int32_t lane_index = static_cast<std::int32_t>(sequence.lane_);
    program_.prefill(lane_index, std::span<const std::int32_t>(lane->prompt), execution_);
    // Unreachable while the forward keeps no state; present so this entry point's contract is
    // complete on the day it does.
    throw std::logic_error(
        "nemotron_h Engine Program: the family Program returned from a prefill pass, which the "
        "landed zero-state forward does not do");
}

PendingBatch NemotronProgram::decode(std::span<const SequenceHandle>,
                                     std::span<const RoundBudget>, ExecutionTiming*,
                                     TokenMaskProvider*) {
    refuse_execution("decode");
}

ExecutionTiming NemotronProgram::append_forced_tokens(std::span<const SequenceHandle>,
                                                      std::span<const TokenId>, std::uint32_t,
                                                      std::span<const std::optional<std::uint32_t>>,
                                                      ExecutionTiming*) {
    // The forced span is the thinking budget's channel close, which this checkpoint has no target
    // control token for; nothing can reach a forced span, and no token could be appended to a lane
    // whose prefill refuses.
    refuse_execution("append_forced_tokens");
}

qwen::CommitResult NemotronProgram::commit(PendingBatch&&, std::span<const CommitDecision>,
                                           CommitObservation, ExecutionTiming*) {
    refuse_execution("commit");
}

qwen::DiscardResult NemotronProgram::abort_pending(PendingBatch&& pending) noexcept {
    // No round can be open: a decode round refuses, so this Program never handed out a pending
    // batch and has no lane state to roll back. The row count is reported so the caller sees which
    // batch was dropped.
    qwen::DiscardResult out;
    out.row_count = pending.row_count();
    return out;
}

qwen::FinishResult NemotronProgram::finish(SequenceHandle) noexcept {
    // A lane that never committed a token has no finished sequence to release. The Engine's
    // terminal settlement only reaches a sequence this Program reported Finishable, and it reports
    // none: the family's forward refuses before a round can complete.
    return {};
}

qwen::AbortResult NemotronProgram::abort(SequenceHandle sequence) noexcept {
    qwen::AbortResult out;
    Lane* lane = valid_lane(sequence);
    if (lane == nullptr) return out;
    out.timings     = lane->timings;
    out.speculative = lane->speculative;
    out.status      = ConsumeStatus::Consumed;
    release(sequence.lane_);
    return out;
}

void NemotronProgram::fail_all_cleanup() noexcept {
    transaction_.reset();
    for (std::uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].phase != Phase::Free) release(lane);
    }
    bump_revision();
}

qwen::PhysicalUsageSnapshot NemotronProgram::physical_usage() const noexcept {
    qwen::PhysicalUsageSnapshot snapshot;
    snapshot.resource_revision = resource_revision();
    // One lane holds one request, and a lane's state is one slot of the Mamba pool and one K/V
    // plane per attention layer, so the occupied lane count is the occupied state-slot count.
    std::uint32_t occupied = 0;
    for (const Lane& lane : lanes_) {
        if (lane.phase != Phase::Free) ++occupied;
    }
    snapshot.device_state_slots = occupied;
    return snapshot;
}

NemotronProgram::Lane* NemotronProgram::valid_lane(SequenceHandle sequence) noexcept {
    if (sequence.owner_ != this || sequence.lane_ >= lanes_.size()) return nullptr;
    Lane& lane = lanes_[sequence.lane_];
    if (lane.phase == Phase::Free || lane.epoch != sequence.epoch_) return nullptr;
    return &lane;
}

void NemotronProgram::release(std::uint32_t lane) noexcept {
    Lane& entry = lanes_[lane];
    if (entry.phase != Phase::Free) {
        // The family Program owns lane occupancy, so it is released there rather than forgotten
        // here. This runs from noexcept entry points (finish/abort/fail_all_cleanup), and the one
        // way the call can throw is the family Program disagreeing about occupancy -- that is, the
        // lane it was handed is already free, which is exactly the state this call establishes, so
        // the disagreement converges instead of terminating the process.
        try {
            program_.finish(static_cast<std::int32_t>(lane));
        } catch (...) {}
        entry.phase = Phase::Free;
    }
    entry.prompt = {};
    bump_revision();
}

} // namespace nemotron_contract

// ---- Instance -----------------------------------------------------------------------------------

EngineOptions nemotron_engine_options(EngineOptions options) {
    // What this model cannot do is refused by name rather than run as if it had been asked for.
    const auto refuse = [](const char* what) {
        throw std::invalid_argument(std::string("Nemotron-H does not support ") + what + " yet");
    };
    // No quantized K/V is wired for this family: the attention state's planes are BF16 because that
    // is the dtype `ops::sliding_causal_attention` declares its operands with, and a codec would
    // have to be decoded into that scratch per call. Streaming has the same answer: there is no
    // host KV arena this family's route could read from.
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        refuse("a KV cache format other than BF16 (the attention state's planes are BF16 because "
               "that is the dtype its Op declares)");
    }
    if (options.kv_stream) refuse("KV streaming");
    // Every lane holds max_context tokens of attention K/V, so an explicit capacity below one
    // request's window cannot be honoured; a larger one is simply not needed.
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        refuse("a KV capacity below max_context (each lane reserves a full window)");
    }
    // Switched off rather than refused, because each is on by default for some product and Engine
    // reports what it runs with: the context cache, speculation (this family has no drafter),
    // Vision (no vision component) and CUDA graphs (the Program runs eager passes). A request
    // cannot reach any of them.
    options.context_cache  = ContextCacheOptions{.enabled = false};
    options.speculative    = SpeculativeOptions{};
    options.enable_vision  = false;
    options.use_cuda_graph = false;
    options.kv_capacity    = KvCapacityPolicy::explicit_capacity(options.max_context);
    // The routed experts are host-resident, so the documented default for such a model is the
    // widest prefill chunk; it is also the width the Program's reported workspace is planned for.
    if (!options.prefill_chunk) options.prefill_chunk = models::nemotron_h::kDefaultPrefillChunk;
    if (!options.idle_prefill_chunk) options.idle_prefill_chunk = *options.prefill_chunk;
    return normalize_engine_options(std::move(options));
}

NemotronInstance::NemotronInstance(std::unique_ptr<models::nemotron_h::Model> source,
                                   const models::nemotron_h::FrontendResources& resources,
                                   const EngineOptions& options, DeviceContext& device)
    : model(std::move(source)), frontend(resources, options.max_context), program(nullptr),
      capacity(options.max_context) {
    const std::size_t needed = models::nemotron_h::Program::device_bytes(model->config(), options);
    std::size_t free_bytes   = 0;
    std::size_t total_bytes  = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // The lane states, the attention K/V and the pass's workspace are allocated up front, so a
    // configuration that does not fit is refused here with the numbers rather than by an allocation
    // failure mid-request.
    if (needed > free_bytes) {
        throw std::invalid_argument(
            "Nemotron-H needs " + std::to_string(needed >> 20) + " MiB for " +
            std::to_string(options.max_concurrency) + " lane(s) of " + std::to_string(capacity) +
            " tokens, but only " + std::to_string(free_bytes >> 20) +
            " MiB of device memory is free after the weights; lower --max-ctx or "
            "--max-concurrency");
    }
    program = std::make_unique<nemotron_contract::NemotronProgram>(*model, options, device);
    std::size_t after_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&after_bytes, &total_bytes));
    kv_capacity_resolution.mode                          = KvCapacityMode::Explicit;
    kv_capacity_resolution.main_page_groups              = 1;
    kv_capacity_resolution.maximum_main_page_groups      = 1;
    kv_capacity_resolution.resolved_tokens               = capacity;
    kv_capacity_resolution.runtime_reservation_bytes     = needed;
    kv_capacity_resolution.available_after_weights_bytes = free_bytes;
    kv_capacity_resolution.available_after_startup_bytes = after_bytes;
}

NemotronInstance::~NemotronInstance() = default;

std::unique_ptr<NemotronInstance> load_nemotron_instance(const EngineOptions& options,
                                                        DeviceContext& device) {
    artifact::Reader reader(options.artifact_path);
    auto plan = models::nemotron_h::plan_load(reader, models::load_options(options));
    // The resources are views into the reader's mapping, which does not outlive this load.
    const models::nemotron_h::TextResources& views = plan.resources();
    models::nemotron_h::FrontendResources resources{
        .tokenizer_json         = std::string(views.tokenizer_json),
        .tokenizer_config_json  = std::string(views.tokenizer_config_json),
        .generation_config_json = std::string(views.generation_config_json),
        .chat_template_jinja    = std::string(views.chat_template_jinja),
    };
    if (!options.chat_template_path.empty()) {
        std::ifstream file(options.chat_template_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("cannot read chat template: " +
                                        options.chat_template_path.string());
        }
        resources.chat_template_jinja.assign(std::istreambuf_iterator<char>(file), {});
    }
    auto model =
        models::nemotron_h::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    return std::make_unique<NemotronInstance>(std::move(model), resources, options, device);
}

bool artifact_is_nemotron(const std::string& path) {
    try {
        artifact::Reader reader(path);
        const artifact::Json& config = reader.directory().component("text").config;
        // The per-layer block list, with the Mamba and router shapes beside it, is what this
        // family's converter emits and no other family has. The config parser validates the exact
        // key set, so this is a scan for a distinctive subset rather than a restatement of it.
        return config.contains("block_configs") && config.contains("mamba") &&
               config.contains("router");
    } catch (...) {
        // A config this probe cannot read is not this family's, and this runs before the Qwen
        // fallback: returning false leaves the error, and its message, to the family that owns it.
        return false;
    }
}

} // namespace ninfer::runtime
