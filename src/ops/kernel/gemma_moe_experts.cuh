// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// One CTA per token: the token's activation row, the selected expert's gate and up vectors, their
// product and the running output row all live in shared memory, and the two banks are read in place
// at their stored expert-major row spans. This route is correctness-first: it streams each selected
// expert's rows exactly once and keeps no tiling or cross-token work.
#pragma once

#include "ops/kernel/gelu.cuh" // gelu_one

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

__global__ void gemma_moe_experts_kernel(const __nv_bfloat16* __restrict__ x,
                                         const std::int32_t* __restrict__ expert_ids,
                                         const float* __restrict__ expert_weights,
                                         const __nv_bfloat16* __restrict__ gate_up,
                                         const __nv_bfloat16* __restrict__ down, int hidden,
                                         int intermediate, int top_k, int tokens,
                                         __nv_bfloat16* __restrict__ out) {
    extern __shared__ float shared[];
    float* activation  = shared;                 // [hidden]
    float* gate        = activation + hidden;    // [intermediate]
    float* up          = gate + intermediate;    // [intermediate]
    float* product     = up + intermediate;      // [intermediate]
    float* accumulator = product + intermediate; // [hidden]

    const int token = blockIdx.x;
    if (token >= tokens) { return; }
    const __nv_bfloat16* row = x + static_cast<std::size_t>(token) * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        activation[h] = __bfloat162float(row[h]);
    }
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) { accumulator[h] = 0.0F; }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        const int expert   = expert_ids[static_cast<std::size_t>(rank) * tokens + token];
        const float weight = expert_weights[static_cast<std::size_t>(rank) * tokens + token];

        // Expert e's gate rows are [e*2I, e*2I+I) and its up rows follow immediately.
        const __nv_bfloat16* gate_rows =
            gate_up + static_cast<std::size_t>(expert) * 2 * intermediate * hidden;
        const __nv_bfloat16* up_rows = gate_rows + static_cast<std::size_t>(intermediate) * hidden;
        for (int i = threadIdx.x; i < intermediate; i += blockDim.x) {
            const __nv_bfloat16* gate_row = gate_rows + static_cast<std::size_t>(i) * hidden;
            const __nv_bfloat16* up_row   = up_rows + static_cast<std::size_t>(i) * hidden;
            float g                       = 0.0F;
            float u                       = 0.0F;
            for (int h = 0; h < hidden; ++h) {
                const float value = activation[h];
                g += __bfloat162float(gate_row[h]) * value;
                u += __bfloat162float(up_row[h]) * value;
            }
            gate[i] = g;
            up[i]   = u;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < intermediate; i += blockDim.x) {
            product[i] = gelu_one<true>(gate[i]) * up[i];
        }
        __syncthreads();

        // Expert e's down rows are [e*H, (e+1)*H), each of the stored intermediate width.
        const __nv_bfloat16* down_rows =
            down + static_cast<std::size_t>(expert) * hidden * intermediate;
        for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
            const __nv_bfloat16* down_row = down_rows + static_cast<std::size_t>(h) * intermediate;
            float acc                     = 0.0F;
            for (int i = 0; i < intermediate; ++i) {
                acc += __bfloat162float(down_row[i]) * product[i];
            }
            accumulator[h] += weight * acc;
        }
        __syncthreads();
    }

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[static_cast<std::size_t>(token) * hidden + h] = __float2bfloat16(accumulator[h]);
    }
}

} // namespace ninfer::ops::detail
