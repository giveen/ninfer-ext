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

void moe_route_launch(const Tensor& x, const Tensor& router, Tensor& ids, Tensor& weights,
                      Tensor& shared_gate, cudaStream_t stream);
void expert_cache_resolve_launch(const Tensor& ids, std::int32_t layer,
                                 const ExpertCacheState& cache, Tensor& slot_ids, Tensor& misses,
                                 cudaStream_t stream);
void expert_cache_fetch_launch(const ExpertWeights& bank, const Tensor& misses,
                               std::int32_t max_misses, const ExpertCacheState& cache,
                               cudaStream_t stream);
void moe_experts_chunk_launch(const MoeChunk& chunk, cudaStream_t stream);

} // namespace ninfer::ops::detail
