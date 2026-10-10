#include "runtime/engine/gemma_instance.h"

#include "artifact/reader.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace gemma_contract {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void unsupported(const char* what) {
    throw std::logic_error(std::string("Gemma runs without the context cache; ") + what +
                           " is not implemented for it");
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

ops::SamplingConfig translate(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    return out;
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
std::uint32_t PressurePlanningSession::ranked_owner_count() const { unsupported("pressure planning"); }
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
PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const qwen::AssessedPressureTarget&, const PreparedPrompt&, std::span<const std::uint32_t>) const {
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

PressurePlanningSession GemmaProgram::begin_pressure_planning(
    std::span<const AdmissionCandidate* const>, std::span<const PlanningCandidateId>,
    std::span<const qwen::ContinuationHandle* const>, std::span<const PlanningOwnerId>,
    std::span<const qwen::SharedPrefixHandle* const>, std::span<const PlanningOwnerId>,
    std::span<const PlanningOwnerId>) {
    unsupported("pressure planning");
}
qwen::CaptureAssessment GemmaProgram::inspect_capture(const qwen::CaptureOffer&,
                                                      const qwen::SharedPrefixHandle*,
                                                      const qwen::SharedPrefixHandle*,
                                                      std::optional<CheckpointRef>, bool) const {
    unsupported("prefix capture");
}
std::vector<CheckpointRecoveryAlternativeWork>
GemmaProgram::checkpoint_recovery_work(const qwen::ContinuationHandle&, CheckpointRef) const {
    unsupported("checkpoint recovery");
}
std::vector<CheckpointRecoveryAlternativeWork>
GemmaProgram::checkpoint_recovery_work(const qwen::SharedPrefixHandle&, CheckpointRef) const {
    unsupported("checkpoint recovery");
}
qwen::CapturePressurePlanningSession GemmaProgram::begin_capture_pressure_planning(
    const qwen::CaptureAssessment&, std::span<const qwen::ContinuationHandle* const>,
    std::span<const PlanningOwnerId>, std::span<const qwen::SharedPrefixHandle* const>,
    std::span<const PlanningOwnerId>, std::span<const PlanningOwnerId>) {
    unsupported("prefix capture");
}
bool GemmaProgram::shared_capture_matches(const qwen::CaptureOffer&,
                                          const qwen::SharedPrefixHandle&) const {
    unsupported("prefix capture");
}
void GemmaProgram::skip_capture(qwen::CaptureOffer&&) { unsupported("prefix capture"); }
ContextTransactionReserveStatus
GemmaProgram::reserve_active_capture(qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*,
                                     const qwen::SharedPrefixHandle*, std::optional<CheckpointRef>,
                                     bool, CancellationFlagView) {
    unsupported("prefix capture");
}
ContextTransactionReserveStatus GemmaProgram::reserve_active_capture_with_pressure(
    qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*, const qwen::SharedPrefixHandle*,
    std::optional<CheckpointRef>, bool, qwen::CapturePressurePlan&&, CancellationFlagView) {
    unsupported("prefix capture");
}
ExecutionTiming GemmaProgram::append_forced_tokens(std::span<const SequenceHandle>,
                                                   std::span<const TokenId>, std::uint32_t,
                                                   std::span<const std::optional<std::uint32_t>>,
                                                   ExecutionTiming*) {
    throw std::logic_error("Gemma output has no forced control tokens");
}

// ---- Program ------------------------------------------------------------------------------------

GemmaProgram::GemmaProgram(const models::gemma4::Model& model, std::int32_t capacity,
                           std::int32_t lanes, std::uint32_t prefill_chunk,
                           std::uint32_t idle_prefill_chunk, DeviceContext& device)
    : model_(model), device_(device), execution_(device.execution_view()),
      program_(model, capacity, lanes, device), lanes_(static_cast<std::size_t>(lanes)),
      prefill_chunk_(prefill_chunk), idle_chunk_(idle_prefill_chunk) {
    if (prefill_chunk_ == 0 || idle_chunk_ == 0) {
        throw std::invalid_argument("gemma4 Program: prefill chunks must be positive");
    }
}

std::vector<float> GemmaProgram::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                              const LogitsSink& logits) {
    return program_.causal_score(prompt.ids, static_cast<std::int32_t>(first_target), logits,
                                 execution_);
}

MemorySummary GemmaProgram::memory_summary() const {
    MemorySummary summary{};
    summary.device                     = device_.device;
    summary.max_context                = static_cast<std::uint32_t>(program_.capacity());
    summary.kv_capacity                = static_cast<std::uint32_t>(program_.capacity());
    summary.kv_cache                   = KvCacheStorage::BFloat16;
    summary.sequence.capacity_bytes    = program_.allocated_bytes();
    summary.sequence.used_bytes        = program_.allocated_bytes();
    summary.sequence.peak_used_bytes   = program_.allocated_bytes();
    summary.weights.capacity_bytes     = model_.storage_stats().h2d_bytes;
    summary.weights.used_bytes         = model_.storage_stats().h2d_bytes;
    summary.weights.peak_used_bytes    = model_.storage_stats().h2d_bytes;
    summary.prefill_chunk              = prefill_chunk_;
    summary.idle_prefill_chunk         = idle_chunk_;
    return summary;
}

GemmaProgram::Lane* GemmaProgram::valid_lane(SequenceHandle sequence) noexcept {
    if (sequence.owner_ != this || sequence.lane_ >= lanes_.size()) return nullptr;
    Lane& lane = lanes_[sequence.lane_];
    if (lane.phase == Phase::Free || lane.epoch != sequence.epoch_) return nullptr;
    return &lane;
}

void GemmaProgram::release(std::uint32_t lane) noexcept {
    Lane& entry = lanes_[lane];
    entry.phase      = Phase::Free;
    entry.prompt     = {};
    entry.cursor     = 0;
    entry.next_input = -1;
    entry.pending    = false;
    bump_revision();
}

RequestBasePlan GemmaProgram::plan_request(const PreparedPrompt& prompt,
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
    RequestPlanSummary& summary      = base.summary_;
    summary.prompt_tokens            = static_cast<std::uint32_t>(prompt.ids.size());
    summary.requested_output_tokens  = options.requested_output_tokens;
    const std::uint32_t capacity_out = capacity - summary.prompt_tokens + 1U;
    summary.effective_output_tokens  = std::min(options.requested_output_tokens, capacity_out);
    summary.effective_limit_reason   = options.requested_output_tokens <= capacity_out
                                           ? FinishReason::OutputLimit
                                           : FinishReason::ContextCapacity;
    summary.prefix_reuse_path        = PrefixReusePath::Root;
    summary.publish_continuation     = false;
    base.prefill_work_ = make_prefill_work(0, summary.prompt_tokens, 0, 0, idle_chunk_);
    const std::uint64_t prefill_units =
        1ULL + (static_cast<std::uint64_t>(summary.prompt_tokens) - 1ULL) / prefill_chunk_;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    summary.service_work_quanta = prefill_units + decode_units;
    base.sampling_              = sampling;
    return base;
}

bool GemmaProgram::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return base.summary().prompt_tokens >= 1 &&
           base.summary().prompt_tokens <= static_cast<std::uint32_t>(program_.capacity());
}

std::optional<AdmissionCandidate>
GemmaProgram::inspect_admission(const PreparedPrompt&, const RequestBasePlan& base,
                                LaneId destination, const qwen::ContinuationHandle* source,
                                const qwen::SharedPrefixHandle* shared_source,
                                std::optional<CheckpointRef> checkpoint, bool) {
    // Only the root exists: nothing is ever catalogued, so there is no source to resume from.
    if (source != nullptr || shared_source != nullptr || checkpoint) return std::nullopt;
    if (destination.value >= lanes_.size()) return std::nullopt;
    AdmissionCandidate candidate;
    candidate.summary_  = base.summary();
    candidate.sampling_ = base.sampling_;
    candidate.lane_     = destination.value;
    IdentityMaterializationAssessment& identity = candidate.identity_;
    identity.physical_status = lanes_[destination.value].phase == Phase::Free
                                   ? MaterializationPhysicalStatus::Feasible
                                   : MaterializationPhysicalStatus::Infeasible;
    // A root has no source to retain; the controller reads Retain as "keep a source" and refuses it.
    identity.source_mode                     = PrivateSourceMode::ConsumeToActive;
    identity.machine_work.remaining_prefill_work = base.prefill_work_;
    identity.machine_work.reused_prompt_tokens   = 0;
    identity.pressure_may_change_machine_work    = false;
    identity.expandable                          = false;
    identity.projection_work                     = 1;
    identity.assessment_digest                   = revision_;
    return candidate;
}

std::optional<ResourcePlan> GemmaProgram::seal_identity(const AdmissionCandidate& candidate,
                                                        const PreparedPrompt&, FinalScheduleIntent) {
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

PrefillWork GemmaProgram::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                                            const PreparedPrompt&,
                                                            std::span<const std::uint32_t>) {
    return candidate.identity_.machine_work.remaining_prefill_work;
}

ContextTransactionReserveStatus
GemmaProgram::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
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

ContextTransactionProgress GemmaProgram::progress_context_transaction(CancellationFlagView) {
    if (!transaction_) throw std::logic_error("no context transaction is open");
    Transaction& transaction = *transaction_;
    Lane& lane               = lanes_[transaction.lane];
    // A root materialization is an empty lane: the cache forgets its rows and the sampler starts
    // over, which is all the state this model has.
    program_.reset(static_cast<std::int32_t>(transaction.lane), translate(transaction.sampling));
    lane.phase      = Phase::Prefill;
    lane.epoch      = next_epoch_++;
    lane.prompt     = std::move(transaction.prompt.ids);
    lane.cursor     = 0;
    lane.next_input = -1;
    lane.pending    = false;
    lane.timings    = {};
    bump_revision();
    MaterializationResult result;
    result.status    = ContextTransactionStatus::Published;
    result.published = StartResult{SequenceHandle(this, transaction.lane, lane.epoch)};
    return result;
}

void GemmaProgram::finalize_context_transaction() noexcept { transaction_.reset(); }

PrefillProgress GemmaProgram::advance_prefill(SequenceHandle sequence, PrefillPace pace,
                                              ExecutionTiming* failed_timing, TokenMaskProvider*) {
    Lane* lane = valid_lane(sequence);
    if (lane == nullptr || lane->phase != Phase::Prefill || lane->pending) {
        throw std::logic_error("prefill advanced on a sequence that is not prefilling");
    }
    ExecutionTimingRecorder timing(ExecutionTimingPhase::Submit, failed_timing);
    const auto started            = Clock::now();
    const std::uint32_t total     = static_cast<std::uint32_t>(lane->prompt.size());
    const std::uint32_t width     = pace == PrefillPace::BesideDecode ? prefill_chunk_ : idle_chunk_;
    const std::uint32_t step      = std::min(width, total - lane->cursor);
    const std::int32_t lane_index = static_cast<std::int32_t>(sequence.lane_);
    program_.prefill(lane_index,
                     std::span<const TokenId>(lane->prompt.data() + lane->cursor, step), execution_);
    lane->cursor += step;

    PrefillProgress progress;
    progress.summary = BeginSummary{.prompt_tokens        = total,
                                    .reused_prompt_tokens = 0,
                                    .prefix_reuse_path    = PrefixReusePath::Root};
    progress.processed_prompt_tokens = step;
    progress.service_units = static_cast<std::uint32_t>(1U + (step - 1U) / prefill_chunk_);
    progress.complete      = lane->cursor == total;
    if (progress.complete) {
        timing.begin_wait();
        const TokenId token = program_.sample(lane_index, execution_);
        timing.end_wait();
        PendingBatch pending;
        pending.transaction_ = next_pending_++;
        pending.rows_[0]     = sequence;
        pending.tokens_[0]   = token;
        pending.counts_[0]   = 1;
        pending.row_count_   = 1;
        lane->pending        = true;
        open_pending_        = pending.transaction_;
        lane->timings.first_token_seconds = lane->timings.prefill_seconds + seconds_since(started);
        progress.pending.emplace(std::move(pending));
    } else {
        timing.begin_wait();
        device_.synchronize();
        timing.end_wait();
    }
    lane->timings.prefill_seconds += seconds_since(started);
    progress.timing = timing.finish();
    if (progress.pending) progress.pending->timing_ = progress.timing;
    return progress;
}

PendingBatch GemmaProgram::decode(std::span<const SequenceHandle> sequences,
                                  std::span<const RoundBudget> budgets,
                                  ExecutionTiming* failed_timing, TokenMaskProvider*) {
    if (sequences.empty() || sequences.size() > kMaximumConcurrency ||
        budgets.size() != sequences.size()) {
        throw std::invalid_argument("decode round shape is invalid");
    }
    ExecutionTimingRecorder timing(ExecutionTimingPhase::Submit, failed_timing);
    PendingBatch pending;
    pending.transaction_ = next_pending_++;
    // Each lane runs its own pass: the rows share the weights' reads only through the cache, which is
    // the measured cost of not batching lanes into one pass yet.
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        Lane* lane = valid_lane(sequences[row]);
        if (lane == nullptr || lane->phase != Phase::Active || lane->pending ||
            lane->next_input < 0 || budgets[row].generated_tokens_remaining == 0) {
            throw std::logic_error("decode round row is not an active sequence");
        }
        const auto started            = Clock::now();
        const std::int32_t lane_index = static_cast<std::int32_t>(sequences[row].lane_);
        program_.decode(lane_index, lane->next_input, execution_);
        timing.begin_wait();
        pending.tokens_[row] = program_.sample(lane_index, execution_);
        timing.end_wait();
        timing.resume_submit();
        pending.rows_[row]   = sequences[row];
        pending.counts_[row] = 1;
        lane->pending        = true;
        lane->timings.decode_seconds += seconds_since(started);
    }
    pending.row_count_ = sequences.size();
    open_pending_      = pending.transaction_;
    pending.timing_    = timing.finish();
    return pending;
}

qwen::CommitResult GemmaProgram::commit(PendingBatch&& pending,
                                        std::span<const CommitDecision> decisions,
                                        CommitObservation observation,
                                        ExecutionTiming* failed_timing) {
    ExecutionTimingRecorder timing(ExecutionTimingPhase::Post, failed_timing);
    if (pending.transaction_ == 0 || pending.transaction_ != open_pending_ ||
        decisions.size() != pending.row_count_) {
        throw std::logic_error("pending transaction capability or decision shape is invalid");
    }
    qwen::CommitResult out;
    out.row_count = pending.row_count_;
    for (std::size_t row = 0; row < pending.row_count_; ++row) {
        const CommitDecision& decision = decisions[row];
        Lane* lane                     = valid_lane(pending.rows_[row]);
        if (lane == nullptr || !lane->pending ||
            (decision.cancelled && (decision.accepted_tokens != 0 || !decision.terminal)) ||
            (!decision.cancelled && !decision.failed &&
             (decision.accepted_tokens == 0 || decision.accepted_tokens > 1))) {
            throw std::logic_error("pending transaction decision is invalid");
        }
        lane->pending = false;
        const std::uint32_t lane_index = pending.rows_[row].lane_;
        if (decision.cancelled || decision.failed) {
            out.rows[row].disposition = decision.cancelled ? CommitDisposition::CancelledReleased
                                                           : CommitDisposition::FailedReleased;
            out.rows[row].timings     = lane->timings;
            release(lane_index);
            continue;
        }
        lane->next_input = pending.tokens_[row];
        lane->phase      = decision.terminal ? Phase::Finishable : Phase::Active;
        out.rows[row].disposition =
            decision.terminal ? CommitDisposition::Finishable : CommitDisposition::Active;
        if (observation == CommitObservation::AllRows) out.rows[row].timings = lane->timings;
    }
    open_pending_ = 0;
    out.timing    = timing.finish();
    return out;
}

qwen::DiscardResult GemmaProgram::abort_pending(PendingBatch&& pending) noexcept {
    qwen::DiscardResult out;
    out.row_count = pending.row_count_;
    if (pending.transaction_ == 0 || pending.transaction_ != open_pending_) return out;
    // A discarded round is an execution failure for its rows: their lanes are released, so the
    // Engine fails those requests rather than resuming a sequence whose round never committed.
    for (std::size_t row = 0; row < pending.row_count_; ++row) {
        if (Lane* lane = valid_lane(pending.rows_[row])) {
            (void)lane;
            release(pending.rows_[row].lane_);
        }
    }
    open_pending_ = 0;
    out.status    = ConsumeStatus::Consumed;
    return out;
}

qwen::FinishResult GemmaProgram::finish(SequenceHandle sequence) noexcept {
    qwen::FinishResult out;
    Lane* lane = valid_lane(sequence);
    if (transaction_ || lane == nullptr || lane->phase != Phase::Finishable) return out;
    out.timings     = lane->timings;
    out.disposition = FinishDisposition::Released;
    out.status      = ConsumeStatus::Consumed;
    release(sequence.lane_);
    return out;
}

qwen::AbortResult GemmaProgram::abort(SequenceHandle sequence) noexcept {
    qwen::AbortResult out;
    Lane* lane = valid_lane(sequence);
    if (lane == nullptr || lane->pending) return out;
    out.timings = lane->timings;
    out.status  = ConsumeStatus::Consumed;
    release(sequence.lane_);
    return out;
}

void GemmaProgram::fail_all_cleanup() noexcept {
    transaction_.reset();
    open_pending_ = 0;
    for (std::uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].phase != Phase::Free) release(lane);
    }
    bump_revision();
}

qwen::PhysicalUsageSnapshot GemmaProgram::physical_usage() const noexcept {
    qwen::PhysicalUsageSnapshot snapshot;
    snapshot.resource_revision = resource_revision();
    return snapshot;
}

} // namespace gemma_contract

// ---- Instance -----------------------------------------------------------------------------------

namespace {

constexpr std::uint32_t kDefaultPrefillChunk = 1024;

} // namespace

EngineOptions gemma_engine_options(EngineOptions options) {
    // What this model cannot do is refused by name rather than run as if it had been asked for.
    const auto refuse = [](const char* what) {
        throw std::invalid_argument(std::string("Gemma 4 does not support ") + what + " yet");
    };
    if (options.speculative.backend != SpeculativeBackend::None) refuse("speculative decoding");
    if (options.enable_vision) refuse("Vision input");
    if (options.kv_cache != KvCacheStorage::BFloat16) refuse("a KV cache format other than BF16");
    if (options.kv_stream) refuse("KV streaming");
    // Every lane holds max_context tokens of global KV, so an explicit capacity below one request's
    // window cannot be honoured; a larger one is simply not needed.
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        refuse("a KV capacity below max_context (each lane reserves a full window)");
    }
    // The context cache is on by default for every product, so it is switched off rather than
    // refused; Engine::options() reports it disabled.
    options.context_cache = ContextCacheOptions{.enabled = false};
    options.kv_capacity   = KvCapacityPolicy::explicit_capacity(options.max_context);
    if (!options.prefill_chunk) options.prefill_chunk = kDefaultPrefillChunk;
    if (!options.idle_prefill_chunk) options.idle_prefill_chunk = *options.prefill_chunk;
    return normalize_engine_options(std::move(options));
}

GemmaInstance::GemmaInstance(std::unique_ptr<models::gemma4::Model> source,
                             const models::gemma4::FrontendResources& resources,
                             const EngineOptions& options, DeviceContext& device)
    : model(std::move(source)), frontend(resources, options.max_context), program(nullptr),
      capacity(options.max_context) {
    const std::int32_t lanes = static_cast<std::int32_t>(options.max_concurrency);
    const std::size_t needed =
        models::gemma4::Program::device_bytes(model->config(), static_cast<std::int32_t>(capacity),
                                              lanes);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // The KV cache, the rings and the scratch are allocated up front, so a configuration that does not
    // fit is refused here with the numbers rather than by an allocation failure mid-request.
    if (needed > free_bytes) {
        throw std::invalid_argument(
            "Gemma needs " + std::to_string(needed >> 20) + " MiB for " + std::to_string(lanes) +
            " lane(s) of " + std::to_string(capacity) + " tokens, but only " +
            std::to_string(free_bytes >> 20) +
            " MiB of device memory is free after the weights; lower --max-ctx or --max-concurrency");
    }
    program = std::make_unique<gemma_contract::GemmaProgram>(
        *model, static_cast<std::int32_t>(capacity), lanes, options.prefill_chunk.value(),
        options.idle_prefill_chunk.value(), device);
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

GemmaInstance::~GemmaInstance() = default;

std::unique_ptr<GemmaInstance> load_gemma_instance(const EngineOptions& options,
                                                   DeviceContext& device) {
    artifact::Reader reader(options.artifact_path);
    auto plan = models::gemma4::plan_load(reader, models::load_options(options));
    // The resources are views into the reader's mapping, which does not outlive this load.
    const models::gemma4::TextResources& views = plan.resources();
    models::gemma4::FrontendResources resources{
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
    auto model = models::gemma4::materialize_model(std::move(plan), device,
                                                   &options.startup_observer);
    device.synchronize();
    return std::make_unique<GemmaInstance>(std::move(model), resources, options, device);
}

bool artifact_is_gemma(const std::string& path) {
    artifact::Reader reader(path);
    const artifact::Json& config = reader.directory().component("text").config;
    if (!config.contains("architectures")) { return false; }
    const auto& architectures = config.at("architectures");
    return architectures.is_array() && !architectures.empty() &&
           architectures.at(0) == "Gemma4ForCausalLM";
}

} // namespace ninfer::runtime
