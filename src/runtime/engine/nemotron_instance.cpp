#include "runtime/engine/nemotron_instance.h"

#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/nemotron_h/load.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <chrono>
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

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// The request's sampling parameters as the Op's config. The occurrence counts the penalties read
// and the mask are the family Program's own to place per lane, so this leaves both at the caller's
// defaults; this family has no token mask to pass.
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
    // The routed-expert split: the resident banks in experts and bytes. The pinned remainder beside
    // them moves in the opposite direction and is reported by the model's storage stats, which is
    // where the 37.21 GB side of the trade lives.
    summary.expert_cache_slots = static_cast<std::uint32_t>(program_.expert_residency().resident_experts);
    summary.expert_cache_bytes = program_.expert_residency().resident_bytes;
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
    // positions of its K/V unwritten -- so the lane is handed to it rather than zeroed here, and
    // the request's sampler is installed on top of the greedy one `create` leaves.
    program_.create(static_cast<std::int32_t>(transaction.lane), execution_);
    program_.configure(static_cast<std::int32_t>(transaction.lane), translate(transaction.sampling),
                       execution_);
    lane.phase       = Phase::Prefill;
    lane.epoch       = next_epoch_++;
    lane.prompt      = std::move(transaction.prompt.ids);
    lane.next_input  = -1;
    lane.pending     = false;
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
                                                 ExecutionTiming* failed_timing,
                                                 TokenMaskProvider*) {
    Lane* lane = valid_lane(sequence);
    if (lane == nullptr || lane->phase != Phase::Prefill || lane->pending) {
        throw std::logic_error("prefill advanced on a sequence that is not prefilling");
    }
    ExecutionTimingRecorder timing(ExecutionTimingPhase::Submit, failed_timing);
    const auto started            = Clock::now();
    const std::uint32_t total     = static_cast<std::uint32_t>(lane->prompt.size());
    const std::int32_t lane_index = static_cast<std::int32_t>(sequence.lane_);
    // The lane's whole prompt goes to the family Program in one call: it chunks the prompt into
    // passes of its own configured width, so no cursor lives here and the state at the end of the
    // call is the state after the prompt rather than after a slice of it. The call synchronizes.
    program_.prefill(lane_index, std::span<const std::int32_t>(lane->prompt), execution_);

    PrefillProgress progress;
    progress.summary                 = BeginSummary{.prompt_tokens        = total,
                                                    .reused_prompt_tokens = 0,
                                                    .prefix_reuse_path    = PrefixReusePath::Root};
    progress.processed_prompt_tokens = total;
    progress.service_units = static_cast<std::uint32_t>(1U + (total - 1U) / prefill_chunk_);
    progress.complete      = true;
    // The prefill leaves the last token's logits, so the response's first token is drawn here and
    // committed by the round that follows, exactly as every later token is.
    timing.begin_wait();
    const TokenId token = program_.sample(lane_index, execution_);
    timing.end_wait();
    PendingBatch pending;
    pending.transaction_ = next_pending_++;
    pending.rows_[0]     = sequence;
    pending.tokens_[0]   = token;
    pending.counts_[0]   = 1;
    pending.row_count_   = 1;
    pending.row_stride_  = 1;
    lane->phase          = Phase::Active;
    lane->next_input     = token;
    lane->pending        = true;
    open_pending_        = pending.transaction_;
    // Everything the request spent prefilling, this call included, is its latency to first token.
    lane->timings.first_token_seconds = lane->timings.prefill_seconds + seconds_since(started);
    lane->timings.prefill_seconds += seconds_since(started);
    progress.pending.emplace(std::move(pending));
    progress.timing           = timing.finish();
    progress.pending->timing_ = progress.timing;
    return progress;
}

PendingBatch NemotronProgram::decode(std::span<const SequenceHandle> sequences,
                                     std::span<const RoundBudget> budgets,
                                     ExecutionTiming* failed_timing, TokenMaskProvider*) {
    if (sequences.empty() || sequences.size() > kMaximumConcurrency ||
        budgets.size() != sequences.size()) {
        throw std::invalid_argument("decode round shape is invalid");
    }
    ExecutionTimingRecorder timing(ExecutionTimingPhase::Submit, failed_timing);
    PendingBatch pending;
    pending.transaction_ = next_pending_++;
    // This family has no drafter, so every row is a plain one-token row.
    pending.row_stride_ = 1;
    std::vector<std::int32_t> plain_lanes;
    std::vector<std::int32_t> plain_ids;
    // A row that fails validation must not leave the rows before it marked pending with no
    // PendingBatch handed out (nothing would then match them in commit/abort_pending), so every row
    // is validated before any lane is mutated.
    const auto require_active_row = [&](std::size_t row) -> Lane* {
        Lane* lane = valid_lane(sequences[row]);
        if (lane == nullptr || lane->phase != Phase::Active || lane->next_input < 0 ||
            budgets[row].generated_tokens_remaining == 0) {
            throw std::logic_error("decode round row is not an active sequence");
        }
        return lane;
    };
    for (std::size_t row = 0; row < sequences.size(); ++row) { (void)require_active_row(row); }
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        Lane* lane = require_active_row(row);
        // Two rows naming one sequence would mark the same lane pending twice, and the family
        // Program refuses a pass that names a lane twice for the same reason.
        if (lane->pending) { throw std::logic_error("decode round row is not an active sequence"); }
        plain_lanes.push_back(static_cast<std::int32_t>(sequences[row].lane_));
        plain_ids.push_back(lane->next_input);
        pending.rows_[row]   = sequences[row];
        pending.counts_[row] = 1;
        lane->pending        = true;
    }
    // The round is one batched pass: the lanes share the weights' reads while each attends to its
    // own state. Both calls synchronize, and the draw reads the logits the pass left.
    const auto started = Clock::now();
    program_.decode(plain_lanes, plain_ids, execution_);
    timing.begin_wait();
    const std::vector<std::int32_t> drawn = program_.sample(plain_lanes, execution_);
    timing.end_wait();
    timing.resume_submit();
    const double seconds = seconds_since(started);
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        pending.tokens_[row * pending.row_stride_] = drawn[row];
        valid_lane(sequences[row])->timings.decode_seconds += seconds;
    }
    pending.row_count_ = sequences.size();
    open_pending_      = pending.transaction_;
    pending.timing_    = timing.finish();
    return pending;
}

ExecutionTiming NemotronProgram::append_forced_tokens(std::span<const SequenceHandle>,
                                                      std::span<const TokenId>, std::uint32_t,
                                                      std::span<const std::optional<std::uint32_t>>,
                                                      ExecutionTiming*) {
    // The forced span is the thinking budget's channel close, which this checkpoint has no target
    // control token for: its frontend emits no forced span, so nothing reaches this entry point.
    throw std::logic_error("nemotron_h Engine Program: append_forced_tokens is not implemented: "
                           "this family has no forced control span");
}

qwen::CommitResult NemotronProgram::commit(PendingBatch&& pending,
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
             (decision.accepted_tokens == 0 ||
              decision.accepted_tokens > static_cast<std::uint32_t>(pending.counts_[row])))) {
            throw std::logic_error("pending transaction decision is invalid");
        }
        // The split marks a prefix-identity boundary for the context cache, which this model does
        // not keep, and this family's frontend never sets one.
        if (decision.prefix_execution_split_after) {
            throw std::logic_error("Nemotron-H keeps no prefix identity to split");
        }
        lane->pending = false;
        if (decision.cancelled || decision.failed) {
            out.rows[row].disposition = decision.cancelled ? CommitDisposition::CancelledReleased
                                                           : CommitDisposition::FailedReleased;
            out.rows[row].timings     = lane->timings;
            out.rows[row].speculative = lane->speculative;
            release(pending.rows_[row].lane_);
            continue;
        }
        // A row licensed exactly one token and the round's pass has consumed it, so the lane's
        // state already sits at the frontier after it and the next round feeds that same token.
        const auto kept  = static_cast<std::int32_t>(decision.accepted_tokens);
        lane->next_input = pending.tokens_[row * pending.row_stride_ + (kept - 1)];
        lane->phase      = decision.terminal ? Phase::Finishable : Phase::Active;
        out.rows[row].disposition =
            decision.terminal ? CommitDisposition::Finishable : CommitDisposition::Active;
        if (observation == CommitObservation::AllRows) out.rows[row].timings = lane->timings;
    }
    open_pending_ = 0;
    out.timing    = timing.finish();
    return out;
}

qwen::DiscardResult NemotronProgram::abort_pending(PendingBatch&& pending) noexcept {
    qwen::DiscardResult out;
    out.row_count = pending.row_count();
    if (pending.transaction_ == 0 || pending.transaction_ != open_pending_) return out;
    // A discarded round is an execution failure for its rows: their lanes are released, so the
    // Engine fails those requests rather than resuming a sequence whose round never committed.
    for (std::size_t row = 0; row < pending.row_count_; ++row) {
        if (valid_lane(pending.rows_[row]) != nullptr) { release(pending.rows_[row].lane_); }
    }
    open_pending_ = 0;
    out.status    = ConsumeStatus::Consumed;
    return out;
}

qwen::FinishResult NemotronProgram::finish(SequenceHandle sequence) noexcept {
    qwen::FinishResult out;
    Lane* lane = valid_lane(sequence);
    if (transaction_ || lane == nullptr || lane->phase != Phase::Finishable) return out;
    out.timings     = lane->timings;
    out.speculative = lane->speculative;
    out.disposition = FinishDisposition::Released;
    out.status      = ConsumeStatus::Consumed;
    release(sequence.lane_);
    return out;
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
    // plane per attention layer. That state is taken at `create` and given back at `release`, so a
    // lane counts here whenever it is not free: prefilling, decoding or waiting to be finished.
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
    entry.prompt     = {};
    entry.next_input = -1;
    entry.pending    = false;
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
      capacity(options.max_context), expert_residency(model->expert_residency()) {
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
    // What the expert cache must leave free, resolved from the *engine's own* plan rather than from
    // a constant: the family's state for the configured lanes and windows -- which carries one
    // pass's workspace and the pass's own operands and logits, since the family Program owns and
    // allocates them -- the staging buffer and its tables, the runtime's reservation and the
    // capacity headroom. The K/V is part of that state and never gives way to the cache -- a cache
    // request that would eat into it is re-derived smaller and reported. The pass's output is not
    // added again beside `device_bytes`: a head writing one column per lane
    // (`HeadColumns::LastOnly`) is already counted there, and the `[vocabulary, chunk]` logits a
    // second term would reserve are not an allocation the family makes.
    const models::nemotron_h::Config artifact_config =
        models::nemotron_h::parse_config(reader.directory());
    models::nemotron_h::ExpertResidencyRequest residency;
    residency.automatic = options.expert_cache.mode == ExpertCacheMode::Automatic;
    residency.explicit_bytes = options.expert_cache.explicit_bytes;
    residency.reserve_bytes =
        models::nemotron_h::Program::device_bytes(artifact_config.text, options) +
        models::nemotron_h::kRuntimeReservationBytes +
        (options.kv_capacity.mode == KvCapacityMode::Automatic
             ? options.kv_capacity.automatic_headroom_bytes
             : 0);
    auto plan = models::nemotron_h::plan_load(reader, models::load_options(options), residency);
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
