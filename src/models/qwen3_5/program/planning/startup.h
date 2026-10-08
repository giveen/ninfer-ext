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
#include "ninfer/ops/offload_moe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;
// Qwen4Exp calls with at least this many columns stream whole routed-expert layers; narrower ones
// fetch their misses into the expert cache. Past 102 columns a cache-route call must split (its
// job sort holds 1024 assignments) while routing already touches most of each layer's experts:
// cold on the RTX 5090, 60 tokens take 1.2 s cached vs 1.6 s staged, 111 take 2.0 s vs 1.6 s.
inline constexpr std::int32_t kQwen4StagedColumns = 103;
// Expert-cache slots that also hold the two staged layer banks (two whole routed layers).
inline constexpr std::int32_t kQwen4StagedBankSlots = 2 * ops::kOffloadMoeExperts;
// Default text-prefill chunks when EngineOptions leaves them open. A prefill step runs the idle
// width when no decode work is waiting and the ordinary chunk beside it: on the RTX 5090 a 4096
// chunk lifts prefill 2-7 % on dense NVFP4 and 19-21 % on the MoE over 1024, while a concurrent
// decode round waiting behind one chunk stalls ~3.4x longer. Every Qwen4Exp chunk of at least
// kQwen4StagedColumns streams each host-resident expert layer once, so Qwen4Exp keeps a wide chunk
// that divides that traffic.
inline constexpr std::uint32_t kDefaultPrefillChunk     = 1024;
inline constexpr std::uint32_t kDefaultIdlePrefillChunk = 4096;
inline constexpr std::uint32_t kQwen4PrefillChunk       = 4096;
// Host-resident routed experts keep that chunk beside decode but let the idle step widen with the
// prompt: a 30,627-token prompt read at 4096 streams each expert layer eight times and at 16384
// twice, measured on the RTX 5090 at 2,095 vs 3,146 tok/s. The wide step's arena is borrowed from the
// expert cache's top slots and returned between wide steps, so widening costs almost no decode
// residency. 16384 rather than 32768: the wider step adds only 1-5 % prefill but widens the lend
// enough to cost the cache ~3 % (K7 C8 decode -7 %). --max-context caps it as well.
inline constexpr std::uint32_t kQwen4MaxIdlePrefillChunk = 16384;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

// EAGLE3 prefill: the three concatenated target layers captured over one prefill chunk and their
// positions, consumed in the same chunk to build the draft KV.
struct Eagle3PrefillLayout {
    TensorLayout features;
    TensorLayout positions;
};

// Qwen4Exp Program resources: the routed-expert cache, the prefill layer bank and PLE buffers.
struct Qwen4PersistentLayout {
    TensorLayout slot_of;  // I32 [cache_layers * experts]
    TensorLayout owner;    // I32 [slots]
    TensorLayout stamp;    // I64 [slots]
    TensorLayout counters; // I64 [3]: clock, hits, misses
    // slots * the expert layout's slot bytes, plus the tails of two staged layers. The last
    // kQwen4StagedBankSlots slots double as the two staged layer banks: staged forwards reclaim
    // them, cache-route calls use them as ordinary slots in between.
    LayoutRegion pool;
    std::optional<TensorLayout> ple_record; // BF16 [residual, draft_window + 1, max_concurrency]
    std::optional<TensorLayout> ple_input;  // BF16 [ple_width, (draft_window + 1) * max_concurrency]
    std::int32_t slots        = 0;
    std::int32_t cache_layers = 0;
    // P1 lend: when the idle prefill width exceeds the ordinary chunk, the top `slots - base_slots`
    // cache slots double as the wide prefill workspace. They are cache during decode and a borrowed
    // arena during a wide idle step. `wide_offset`/`wide_bytes` locate that arena in the pool;
    // `bank_slot` is the first staged-bank slot (past the cache and the lend region).
    std::int32_t base_slots  = 0; // cache slots while a wide prefill borrows the lend region
    std::int32_t bank_slot   = 0;
    std::size_t  wide_offset = 0;
    std::size_t  wide_bytes  = 0;
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    std::optional<Eagle3PrefillLayout> eagle3;
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
    // P1 lend: `capacity` sizes the widest arena (a wide idle prefill); `permanent_capacity` is the
    // arena the Program always owns (decode rounds and the ordinary narrow prefill). For Qwen4Exp
    // the difference is carried by the cached lend slots rather than a second allocation.
    std::size_t permanent_capacity = 0;
    std::size_t lend_bytes         = 0;
    std::int32_t lend_slots        = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0; // service unit; the width beside decode work
    std::uint32_t prefill_width             = 0; // widest chunk, a multiple of prefill_chunk
    std::uint32_t draft_window              = 0;
    // EAGLE3 static draft-tree root width (1 is a chain).
    std::uint32_t tree_width                = 1;
    bool adaptive_draft                     = false;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    LookupDraftMode lookup_drafts           = LookupDraftMode::Off;
    std::uint32_t lookup_min_match          = 8;
    // A draft window wider than the MTP layer can draft; every round is then a lookup round and the
    // MTP draft phases are skipped.
    bool lookup_only                        = false;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    std::uint32_t expert_cache_slots = 0;     // Qwen4Exp only
    bool ngram_stream                = false; // Qwen4Exp only: resolved n-gram residency
    bool kv_stream                   = false;
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
    std::uint32_t prefill_width             = 0;
    std::uint32_t draft_window              = 0;
    // EAGLE3 static draft-tree root width (1 is a chain).
    std::uint32_t tree_width                = 1;
    bool adaptive_draft                     = false;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    LookupDraftMode lookup_drafts           = LookupDraftMode::Off;
    std::uint32_t lookup_min_match          = 8;
    // A draft window wider than the MTP layer can draft; every round is then a lookup round and the
    // MTP draft phases are skipped.
    bool lookup_only                        = false;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    std::uint32_t expert_cache_slots = 0;
    bool ngram_stream                = false;
    bool kv_stream                   = false;
    // MTP rounds of several requests run as ordinary rounds plus an MTP KV append.
    bool plain_mtp_batches = false;
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
[[nodiscard]] std::vector<std::uint32_t>
idle_prefill_chunk_candidates_impl(const execution::Parameters& parameters,
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
