#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;
// Qwen4Exp calls with at least this many columns stream whole routed-expert layers.
inline constexpr std::int32_t kQwen4StagedColumns = 256;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

// Qwen4Exp Program resources: the routed-expert cache, the prefill layer bank and PLE buffers.
struct Qwen4PersistentLayout {
    TensorLayout slot_of;  // I32 [cache_layers * experts]
    TensorLayout owner;    // I32 [slots]
    TensorLayout stamp;    // I64 [slots]
    TensorLayout counters; // I64 [3]: clock, hits, misses
    LayoutRegion pool;     // slots * kExpertSlotBytes
    std::optional<LayoutRegion> staged_bank;
    std::optional<TensorLayout> ple_record; // BF16 [residual, draft_window + 1, max_concurrency]
    std::optional<TensorLayout> ple_input;  // BF16 [ple_width, (draft_window + 1) * max_concurrency]
    std::int32_t slots        = 0;
    std::int32_t cache_layers = 0;
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<Qwen4PersistentLayout> qwen4;
    // Pinned Host cost of one complete Main Text KV page group, taken from the same planned
    // geometry the Device pool binds. The Host RAM budget compares one StateImage against the Main
    // pages a re-prefill of the gap it covers would pin, and that comparison is denominated in
    // Main Text pages: the stride's plane inventory (and therefore its group-scale term) follows
    // --kv-dtype, so it is carried rather than re-derived from configuration.
    std::size_t host_kv_text_page_stride = 0;
    std::size_t bytes                    = 0;
    std::size_t kv_payload_bytes         = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    bool adaptive_draft                     = false;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    std::uint32_t expert_cache_slots = 0; // Qwen4Exp only
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    bool adaptive_draft                     = false;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    std::uint32_t expert_cache_slots = 0;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {


[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

// Resolves the single Host RAM budget into the plan's context-cache shape: one StateImage per
// position it retains and Host KV for everything else. StateImages are the fixed per-position cost
// of a checkpoint and Host KV the per-token cost, so the budget first covers the inventory the
// capture path creates — 2 + A images per private owner plus one per shared entry — then spends
// the remaining StateImage headroom under the half-budget cap on extra long anchors per owner, and
// gives Host KV the remainder. Growing A without re-sizing the pool in the same pass would leave
// the Host StateImage pool undersized for the anchors the capture path then creates, so both are
// resolved together.
//
// A never drops below the configured count (or the default when unset): the budget adds anchors,
// it does not remove them. One extra anchor pays for itself only while the gap it covers exceeds
// the Main KV pages one StateImage is worth, which bounds the useful count by the logical page
// space; when even the mandatory inventory exceeds half the budget, the configured count is kept
// so the caller's rejection reports the real shortfall.
//
// `state_image_bytes` is one complete Host StateImage, `host_kv_group_bytes` one Main Text Host KV
// page group, and the capacities are the already-normalized private/shared continuation catalogs.
void resolve_host_cache_budget(ContextCacheOptions& cache, std::uint32_t private_capacity,
                               std::uint32_t shared_capacity, std::uint32_t capacity,
                               std::uint64_t state_image_bytes, std::uint64_t host_kv_group_bytes);

} // namespace ninfer::models::qwen3_5::detail
