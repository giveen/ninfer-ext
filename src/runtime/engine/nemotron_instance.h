#pragma once

// The Nemotron-H side of the Engine's instance boundary.
//
// EngineCore, CausalScoreCore and the ResourceManager under them are templated on an instance and
// its ModelContract. runtime::ModelInstance is the Qwen implementation and runtime::GemmaInstance
// the second one; this is the third, for models::nemotron_h. Everything the common machinery can
// reach is mirrored from the Gemma side, with one property stated up front because it decides what
// this file does and does not implement:
//
//  - The family's forward is a *zero-state prefill*: a Mamba mixer starts from a zero state and the
//    pass writes nothing back, and no attention K/V is appended (models/nemotron_h/forward.h). A
//    lane driven through it would therefore compute the output of a zero-state model, which is a
//    wrong model rather than a missing feature, so `models::nemotron_h::Program` refuses every
//    entry point that would consume or commit tokens and names the missing piece in the same words
//    (kMissingStatePersistence). This wrapper implements the request lifecycle -- planning,
//    admission, the lane, the state transaction -- over that Program and lets its refusal surface
//    where the tokens would be consumed, at `advance_prefill`: a generation submitted to this
//    Engine reaches the family's own refusal rather than a crash or a silently wrong answer.
//  - Admission is root-only. The Engine runs this model with the context cache disabled, so the
//    identity candidate is always feasible, is never expandable, and seals directly. No prefix is
//    reused and no capture is ever offered, so the prefix-cache machinery the ResourceManager can
//    reach in principle refuses by name and is never reached.
//  - The family has no speculation and no vision component, so no token masks, drafts or images
//    reach the Program; `nemotron_engine_options` switches those off rather than carrying settings
//    the Engine would silently ignore.
//
// Contract types this model never constructs (continuations, shared prefixes, capture offers and
// their assessments, the result aggregates of commit/finish/abort) are the Qwen contract's own
// pure-data types, exactly as on the Gemma side.

#include "core/device.h"
#include "models/nemotron_h/forward.h"
#include "models/nemotron_h/frontend.h"
#include "models/nemotron_h/program.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/kv_capacity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::runtime {

namespace nemotron_contract {

namespace qwen = models::qwen3_5;

using PreparedPrompt = models::nemotron_h::PreparedPrompt;

class NemotronProgram;

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

    [[nodiscard]] LaneId lane() const noexcept { return LaneId{lane_}; }

private:
    SequenceHandle(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch) {}

    const void* owner_   = nullptr;
    std::uint32_t lane_  = 0;
    std::uint64_t epoch_ = 0;

    friend class NemotronProgram;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept            = default;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept = default;
    RequestBasePlan(const RequestBasePlan&)                = delete;
    RequestBasePlan& operator=(const RequestBasePlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return summary_; }

    [[nodiscard]] const qwen::PreparedContextCache& context_cache() const noexcept {
        return context_cache_;
    }

    [[nodiscard]] std::optional<qwen::PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t) const noexcept {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t) const noexcept {
        return std::nullopt;
    }

private:
    RequestBasePlan() = default;

    RequestPlanSummary summary_;
    qwen::PreparedContextCache context_cache_;
    ResolvedSamplingParameters sampling_;
    PrefillWork prefill_work_;

    friend class NemotronProgram;
};

class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept            = default;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept = default;
    AdmissionCandidate(const AdmissionCandidate&)                = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return summary_; }

    [[nodiscard]] const IdentityMaterializationAssessment& identity_assessment() const noexcept {
        return identity_;
    }

private:
    AdmissionCandidate() = default;

    RequestPlanSummary summary_;
    IdentityMaterializationAssessment identity_;
    ResolvedSamplingParameters sampling_;
    std::uint32_t lane_ = 0;

    friend class NemotronProgram;
};

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ResourcePlan(const ResourcePlan&)                = delete;
    ResourcePlan& operator=(const ResourcePlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return false; }

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision_; }

private:
    ResourcePlan(AdmissionCandidate&& admission, ProgramResourceRevision revision) noexcept
        : admission_(std::move(admission)), revision_(revision) {}

    AdmissionCandidate admission_;
    ProgramResourceRevision revision_;

    friend class NemotronProgram;
    friend class PressurePlanningSession;
};

// Pressure planning searches eviction targets among cached prefixes. With the cache disabled the
// root candidate is always feasible and never expandable, so the planner seals it directly and
// never opens a session; every method refuses by name.
class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept            = default;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept = default;
    PressurePlanningSession(const PressurePlanningSession&)                = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&)     = delete;

    [[noreturn]] qwen::PressureTargetHandle identity_target(PlanningCandidateId) const;
    [[noreturn]] qwen::PressureTargetHandle root_maximal_target(PlanningCandidateId);
    [[noreturn]] qwen::PressureTargetHandle maximal_target(PlanningCandidateId);
    [[noreturn]] qwen::PressureTargetHandle
    recency_maximal_target(PlanningCandidateId, std::uint32_t, std::span<const std::uint32_t> = {},
                           bool = true);
    [[noreturn]] std::uint32_t ranked_owner_count() const;
    [[noreturn]] void set_eviction_licence(std::uint32_t, std::span<const std::uint32_t> = {});

    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept { return 0; }

    [[noreturn]] qwen::PressureConstructionCursor begin_construction(qwen::PressureTargetHandle,
                                                                     bool = false);
    [[noreturn]] PressureConstructionStep
    next_construction_option(qwen::PressureConstructionCursor&);
    [[noreturn]] void choose_construction(qwen::PressureConstructionCursor&,
                                          PressureConstructionOptionId);
    [[noreturn]] std::optional<qwen::PressureTargetHandle>
    construction_target(const qwen::PressureConstructionCursor&);
    [[noreturn]] PressureTargetGuidance guidance(qwen::PressureTargetHandle);
    [[noreturn]] qwen::AssessedPressureTarget assess(qwen::PressureTargetHandle);
    [[noreturn]] qwen::PreparedPressureExpansion
    prepare_expansion(qwen::PressureTargetHandle,
                      std::uint32_t = std::numeric_limits<std::uint32_t>::max());
    [[noreturn]] qwen::PressureExpansionView commit_expansion(qwen::PreparedPressureExpansion&&);

    void discard_expansion(qwen::PreparedPressureExpansion&&) noexcept {}

    [[noreturn]] PrefillWork
    shared_capture_split_prefill_work(const qwen::AssessedPressureTarget&, const PreparedPrompt&,
                                      std::span<const std::uint32_t>) const;
    [[noreturn]] std::optional<ResourcePlan> seal(qwen::AssessedPressureTarget&&,
                                                  const PreparedPrompt&, FinalScheduleIntent);
    [[noreturn]] std::optional<qwen::CapturePressurePlan>
    seal_capture(qwen::AssessedPressureTarget&&);

    [[nodiscard]] bool try_claim_seal_window() noexcept { return false; }

    void release_seal_window() noexcept {}

private:
    PressurePlanningSession() = default;
    friend class NemotronProgram;
};

class PendingBatch {
public:
    PendingBatch() noexcept                          = default;
    PendingBatch(PendingBatch&&) noexcept            = default;
    PendingBatch& operator=(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)                = delete;
    PendingBatch& operator=(const PendingBatch&)     = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept {
        return std::span<const TokenId>(tokens_.data(), row_count_ * row_stride_);
    }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept {
        return std::span<const std::int32_t>(counts_.data(), row_count_);
    }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const {
        if (row >= row_count_) throw std::out_of_range("pending row out of range");
        return false;
    }

    [[nodiscard]] ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    // This family has no draft tokens and no speculation, so one row is one committed token.
    std::array<TokenId, kMaximumConcurrency> tokens_{};
    std::array<std::int32_t, kMaximumConcurrency> counts_{};
    std::size_t row_count_    = 0;
    std::uint32_t row_stride_ = 1;
    ExecutionTiming timing_;

    friend class NemotronProgram;
};

struct PrefillProgress {
    BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint32_t service_units           = 1;
    bool complete                         = false;
    ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<qwen::CaptureOffer> capture;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationResult {
    ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<qwen::MaterializationSourceResult> source;
    std::optional<qwen::MaterializationSharedSourceResult> shared_source;
    std::vector<qwen::MaterializationVictimResult> victims;
    std::vector<qwen::MaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<ContextTransactionInProgress, MaterializationResult, qwen::ActiveCaptureResult>;

// The Engine-facing Program: lanes, admission transactions and the one refusal the family's forward
// forces. The lane bookkeeping is real -- a lane holds one request, `create` empties its state
// through the family Program, and `abort`/`fail_all_cleanup` release it -- while every entry point
// that would consume or commit tokens is the family Program's own refusal. The prefill step hands
// the lane's whole prompt to `models::nemotron_h::Program::prefill`, which refuses: that is the
// boundary the next milestone replaces when the composition persists the Mamba and attention state.
// A sequence can therefore never reach a terminal state, which is why `finish` has nothing to
// report and `decode`/`commit` refuse.
class NemotronProgram {
public:
    NemotronProgram(const models::nemotron_h::Model& model, const EngineOptions& options,
                    DeviceContext& device);

    // Scoring.
    [[noreturn]] std::vector<float>
    causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const LogitsSink& logits);
    [[nodiscard]] MemorySummary memory_summary() const;

    void reset_memory_peaks() noexcept {}

    // Generation.
    [[nodiscard]] bool consumes_token_masks() const noexcept { return false; }

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base, LaneId destination,
                      const qwen::ContinuationHandle* source,
                      const qwen::SharedPrefixHandle* shared_source,
                      std::optional<CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] std::optional<ResourcePlan> seal_identity(const AdmissionCandidate& candidate,
                                                            const PreparedPrompt& prompt,
                                                            FinalScheduleIntent intent);
    [[noreturn]] PressurePlanningSession begin_pressure_planning(
        std::span<const AdmissionCandidate* const>, std::span<const PlanningCandidateId>,
        std::span<const qwen::ContinuationHandle* const>, std::span<const PlanningOwnerId>,
        std::span<const qwen::SharedPrefixHandle* const>, std::span<const PlanningOwnerId>,
        std::span<const PlanningOwnerId>);
    [[nodiscard]] PrefillWork shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                                                const PreparedPrompt& prompt,
                                                                std::span<const std::uint32_t>);
    [[nodiscard]] ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               CancellationFlagView cancellation);

    [[nodiscard]] std::optional<qwen::PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan&, const ResourcePlan&,
                              std::span<const SequenceHandle>) const {
        return std::nullopt;
    }

    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;

    [[nodiscard]] bool has_context_transaction() const noexcept { return transaction_.has_value(); }

    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  PrefillPace pace = PrefillPace::Idle,
                                                  ExecutionTiming* failed_timing = nullptr,
                                                  TokenMaskProvider* masks       = nullptr);
    [[noreturn]] qwen::CaptureAssessment inspect_capture(const qwen::CaptureOffer&,
                                                         const qwen::SharedPrefixHandle*,
                                                         const qwen::SharedPrefixHandle*,
                                                         std::optional<CheckpointRef>, bool) const;
    [[noreturn]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const qwen::ContinuationHandle&, CheckpointRef) const;
    [[noreturn]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const qwen::SharedPrefixHandle&, CheckpointRef) const;
    [[noreturn]] qwen::CapturePressurePlanningSession begin_capture_pressure_planning(
        const qwen::CaptureAssessment&, std::span<const qwen::ContinuationHandle* const>,
        std::span<const PlanningOwnerId>, std::span<const qwen::SharedPrefixHandle* const>,
        std::span<const PlanningOwnerId>, std::span<const PlanningOwnerId>);
    [[noreturn]] bool shared_capture_matches(const qwen::CaptureOffer&,
                                             const qwen::SharedPrefixHandle&) const;
    [[noreturn]] void skip_capture(qwen::CaptureOffer&&);
    [[noreturn]] ContextTransactionReserveStatus
    reserve_active_capture(qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*,
                           const qwen::SharedPrefixHandle*, std::optional<CheckpointRef>, bool,
                           CancellationFlagView);
    [[noreturn]] ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        qwen::CaptureOffer&&, const qwen::SharedPrefixHandle*, const qwen::SharedPrefixHandle*,
        std::optional<CheckpointRef>, bool, qwen::CapturePressurePlan&&, CancellationFlagView);
    [[noreturn]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                     std::span<const RoundBudget> budgets,
                                     ExecutionTiming* failed_timing = nullptr,
                                     TokenMaskProvider* masks       = nullptr);
    [[noreturn]] ExecutionTiming append_forced_tokens(std::span<const SequenceHandle>,
                                                      std::span<const TokenId>, std::uint32_t,
                                                      std::span<const std::optional<std::uint32_t>>,
                                                      ExecutionTiming* = nullptr);
    [[noreturn]] qwen::CommitResult
    commit(PendingBatch&& pending, std::span<const CommitDecision> decisions,
           CommitObservation observation  = CommitObservation::AllRows,
           ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] qwen::DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] qwen::FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] qwen::AbortResult abort(SequenceHandle sequence) noexcept;

    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle, std::uint32_t) const noexcept {
        return std::nullopt;
    }

    [[nodiscard]] qwen::ReleaseResult release_continuation(qwen::ContinuationHandle&&) noexcept {
        return {};
    }

    [[nodiscard]] qwen::ReleaseResult release_shared_prefix(qwen::SharedPrefixHandle&&) noexcept {
        return {};
    }

    void fail_all_cleanup() noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
        return ProgramResourceRevision{revision_};
    }

    [[nodiscard]] qwen::PhysicalUsageSnapshot physical_usage() const noexcept;

private:
    // A lane is free, or holds one request being prefilled. There is no third state: a lane becomes
    // active only when a round commits, and no round can complete while the forward keeps no state.
    enum class Phase : std::uint8_t { Free, Prefill };

    struct Lane {
        Phase phase         = Phase::Free;
        std::uint64_t epoch = 0;
        std::vector<TokenId> prompt;
        GenerationTimings timings;
        SpeculativeStats speculative;
    };

    struct Transaction {
        std::uint32_t lane = 0;
        PreparedPrompt prompt;
        ResolvedSamplingParameters sampling;
    };

    [[nodiscard]] Lane* valid_lane(SequenceHandle sequence) noexcept;
    void release(std::uint32_t lane) noexcept;

    void bump_revision() noexcept { ++revision_; }

    const models::nemotron_h::Model& model_;
    DeviceContext& device_;
    const DeviceExecutionView execution_;
    models::nemotron_h::Program program_;
    std::vector<Lane> lanes_;
    std::optional<Transaction> transaction_;
    std::uint64_t revision_      = 1;
    std::uint64_t next_epoch_    = 1;
    std::uint32_t prefill_chunk_ = 0;
};

} // namespace nemotron_contract

struct NemotronRuntimeTypes {
    using Frontend                   = models::nemotron_h::Frontend;
    using PreparedPrompt             = models::nemotron_h::PreparedPrompt;
    using OutputSession              = models::nemotron_h::OutputSession;
    using PublishedOutput            = models::nemotron_h::PublishedOutput;
    using RequestBasePlan            = nemotron_contract::RequestBasePlan;
    using AdmissionCandidate         = nemotron_contract::AdmissionCandidate;
    using ResourcePlan               = nemotron_contract::ResourcePlan;
    using PersistentBackfillProof    = models::qwen3_5::PersistentBackfillProof;
    using SequenceHandle             = nemotron_contract::SequenceHandle;
    using ContinuationHandle         = models::qwen3_5::ContinuationHandle;
    using SharedPrefixHandle         = models::qwen3_5::SharedPrefixHandle;
    using CaptureOffer               = models::qwen3_5::CaptureOffer;
    using CacheSessionKey            = models::qwen3_5::PreparedSessionKey;
    using ContinuationSummary        = models::qwen3_5::ContinuationSummary;
    using SharedPrefixSummary        = models::qwen3_5::SharedPrefixSummary;
    using PressurePlanningSession    = nemotron_contract::PressurePlanningSession;
    using PressureTargetHandle       = models::qwen3_5::PressureTargetHandle;
    using AssessedPressureTarget     = models::qwen3_5::AssessedPressureTarget;
    using CapturePressurePlan        = models::qwen3_5::CapturePressurePlan;
    using MaterializationResult      = nemotron_contract::MaterializationResult;
    using ContextTransactionProgress = nemotron_contract::ContextTransactionProgress;
    using CaptureAssessment          = models::qwen3_5::CaptureAssessment;
    using ActiveCaptureResult        = models::qwen3_5::ActiveCaptureResult;
    using PendingBatch               = nemotron_contract::PendingBatch;
    using StartResult                = nemotron_contract::StartResult;
    using PrefillProgress            = nemotron_contract::PrefillProgress;
    using CommitResult               = models::qwen3_5::CommitResult;
    using DiscardResult              = models::qwen3_5::DiscardResult;
    using FinishResult               = models::qwen3_5::FinishResult;
    using AbortResult                = models::qwen3_5::AbortResult;
    using ReleaseResult              = models::qwen3_5::ReleaseResult;
    using PhysicalUsageSnapshot      = models::qwen3_5::PhysicalUsageSnapshot;
    using Program                    = nemotron_contract::NemotronProgram;
};

using NemotronPreparedPrompt = models::nemotron_h::PreparedPrompt;

struct NemotronInstance {
    using ModelContract = NemotronRuntimeTypes;
    std::unique_ptr<models::nemotron_h::Model> model;
    models::nemotron_h::Frontend frontend;
    std::unique_ptr<nemotron_contract::NemotronProgram> program;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;

    NemotronInstance(std::unique_ptr<models::nemotron_h::Model> model,
                     const models::nemotron_h::FrontendResources& resources,
                     const EngineOptions& options, DeviceContext& device);
    ~NemotronInstance();
    NemotronInstance(const NemotronInstance&)            = delete;
    NemotronInstance& operator=(const NemotronInstance&) = delete;
};

// Whether an artifact declares this model. This family is detected by its own config: the
// per-layer block list with the Mamba and router shapes beside it, which no other family emits.
// An artifact this family cannot read is not this family's, so the probe answers false rather than
// throwing: it runs before the Qwen fallback, whose own validation owns the error message.
[[nodiscard]] bool artifact_is_nemotron(const std::string& path);

// The options a Nemotron-H Engine actually runs with: the context cache, speculation, Vision and
// CUDA graphs are not implemented for this model, so they are switched off here rather than
// accepted and ignored; a K/V storage or a streaming mode the family has no route for is refused by
// name.
[[nodiscard]] EngineOptions nemotron_engine_options(EngineOptions options);

// Loads an artifact and wraps it the way the Engine's cores expect.
[[nodiscard]] std::unique_ptr<NemotronInstance> load_nemotron_instance(const EngineOptions& options,
                                                                       DeviceContext& device);

} // namespace ninfer::runtime
