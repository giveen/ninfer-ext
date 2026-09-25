#pragma once

// ninfer::ops::detail - private launch prototypes for the offloaded MoE.

#include "ninfer/ops/offload_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One token chunk of moe_experts with its workspace carved out by the wrapper.
struct MoeChunk {
    const __nv_bfloat16* x;
    const std::int32_t* expert_ids;
    const std::int32_t* slot_ids;
    const float* weights;
    const float* shared_gate;
    const __nv_bfloat16* shared;
    __nv_bfloat16* y;
    std::int32_t columns;
    std::int32_t slots;
    ExpertWeights source;
    std::int32_t* counts;
    std::int32_t* sorted_assign;
    std::int32_t* sorted_slot;
    std::int32_t* jobs;
    std::int32_t* job_count;
    __nv_bfloat16* act;
    float* partial;
};

// One A4 job is one 64-column tile of one expert's grouped assignments.
inline constexpr std::int32_t kA4JobColumns = 64;

// Jobs of `columns` tokens: whole tiles of every assignment, plus one partial tile per expert.
[[nodiscard]] constexpr std::int32_t a4_max_jobs(std::int32_t columns) {
    return columns * kOffloadMoeTopK / kA4JobColumns + kOffloadMoeExperts;
}

// One token chunk of moe_experts_a4 with its workspace carved out by the wrapper.
struct MoeA4Chunk {
    const __nv_bfloat16* x;
    const std::int32_t* expert_ids;
    const float* weights;
    const float* shared_gate;
    const __nv_bfloat16* shared;
    __nv_bfloat16* y;
    std::int32_t columns;
    ExpertWeights source;
    std::uint8_t* input_codes;  // [columns, H/2]
    std::uint8_t* input_scales; // [columns, H/16]
    std::int32_t* counts;       // [E]
    std::int32_t* offsets;      // [E+1]
    std::int32_t* rank;         // [columns*K]
    std::int32_t* packed_token; // [columns*K]
    std::int32_t* packed_index; // [columns*K]
    std::int32_t* job_experts;  // [a4_max_jobs]
    std::int32_t* job_columns;  // [a4_max_jobs]
    std::int32_t* job_count;    // [1]
    std::uint8_t* middle_codes; // [columns*K, I/2]
    std::uint8_t* middle_scales;
    __nv_bfloat16* grouped; // [columns*K, H]
};

void moe_experts_a4_launch(const MoeA4Chunk& chunk, cudaStream_t stream);

void moe_route_launch(const Tensor& x, const Tensor& router, float* logits, Tensor& ids,
                      Tensor& weights, Tensor& shared_gate, cudaStream_t stream);
void expert_cache_resolve_launch(const Tensor& ids, std::int32_t layer,
                                 const ExpertCacheState& cache, Tensor& slot_ids, Tensor& misses,
                                 cudaStream_t stream);
void expert_cache_stage_launch(const ExpertCacheState& cache, std::int32_t layer,
                               const std::int32_t* resident, const ExpertWeights& bank,
                               const ExpertWeights& staged, cudaStream_t stream);

void expert_cache_fetch_launch(const ExpertWeights& bank, const Tensor& misses,
                               std::int32_t max_misses, const ExpertCacheState& cache,
                               cudaStream_t stream);
void moe_experts_chunk_launch(const MoeChunk& chunk, cudaStream_t stream);

} // namespace ninfer::ops::detail
