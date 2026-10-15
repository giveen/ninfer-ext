// Implements: include/ninfer/ops/nemotron_h_moe_experts.h
//
// One route: one CTA per token. The token's activation row, one selected expert's up result and
// the running output row all live in shared memory, and both banks are read in place at their
// stored expert-major row spans. Every geometry and format takes it, and it keeps no tiling,
// no cross-token work and no workspace -- see the launcher for why that is the right first route
// for this Op and where a streamed one belongs.
//
// The bank format is a reader: a BF16 bank hands back its stored element and a low-bit bank decodes
// one (codes, block scale, divisor), and the kernel body is the same for both. The two banks of a
// layer share one format, so the launcher instantiates the two readers together.
#pragma once

#include "ops/kernel/relu2.cuh" // relu2_one

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// A BF16 bank, expert-major: the stored element is the value. The same reader serves both banks of
// a layer because both are one stacked row-major plane; only the row count and the column count
// differ, and the kernel supplies both.
struct Bf16BankReader {
    const __nv_bfloat16* values;
    int columns;

    __device__ __forceinline__ float operator()(int row, int column) const {
        return __bfloat162float(values[static_cast<std::size_t>(row) * columns + column]);
    }
};

template <class UpReader, class DownReader>
__global__ void nemotron_h_moe_experts_kernel(UpReader up, DownReader down, int latent,
                                              int intermediate, int top_k, int tokens,
                                              const std::int32_t* __restrict__ expert_ids,
                                              const float* __restrict__ expert_weights,
                                              const __nv_bfloat16* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out) {
    extern __shared__ float shared[];
    float* activation  = shared;                 // [latent]
    float* product     = activation + latent;    // [intermediate]
    float* accumulator = product + intermediate; // [latent]

    const int token = blockIdx.x;
    if (token >= tokens) { return; }
    const __nv_bfloat16* row = x + static_cast<std::size_t>(token) * latent;
    for (int h = threadIdx.x; h < latent; h += blockDim.x) {
        activation[h] = __bfloat162float(row[h]);
    }
    for (int h = threadIdx.x; h < latent; h += blockDim.x) { accumulator[h] = 0.0F; }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        const std::size_t slot = static_cast<std::size_t>(rank) * tokens + token;
        const int expert       = expert_ids[slot];
        const float weight     = expert_weights[slot];

        // Expert e's up rows are [e*I, (e+1)*I). The activation is already in shared memory, so
        // the four partial sums per row are what keeps four scattered bank loads in flight; the
        // chain is latency-bound rather than arithmetic-bound.
        const int up_base = expert * intermediate;
        for (int i = threadIdx.x; i < intermediate; i += blockDim.x) {
            const int up_row = up_base + i;
            float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
            int h = 0;
            for (; h + 4 <= latent; h += 4) {
                a0 += up(up_row, h) * activation[h];
                a1 += up(up_row, h + 1) * activation[h + 1];
                a2 += up(up_row, h + 2) * activation[h + 2];
                a3 += up(up_row, h + 3) * activation[h + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; h < latent; ++h) { acc += up(up_row, h) * activation[h]; }
            // No gate: the activation is the squared ReLU of this projection's own result.
            product[i] = relu2_one(acc);
        }
        __syncthreads();

        // Expert e's down rows are [e*H, (e+1)*H), each of the intermediate width.
        const int down_base = expert * latent;
        for (int h = threadIdx.x; h < latent; h += blockDim.x) {
            const int down_row = down_base + h;
            float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
            int i = 0;
            for (; i + 4 <= intermediate; i += 4) {
                a0 += down(down_row, i) * product[i];
                a1 += down(down_row, i + 1) * product[i + 1];
                a2 += down(down_row, i + 2) * product[i + 2];
                a3 += down(down_row, i + 3) * product[i + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; i < intermediate; ++i) { acc += down(down_row, i) * product[i]; }
            accumulator[h] += weight * acc;
        }
        __syncthreads();
    }

    for (int h = threadIdx.x; h < latent; h += blockDim.x) {
        out[static_cast<std::size_t>(token) * latent + h] = __float2bfloat16_rn(accumulator[h]);
    }
}

} // namespace ninfer::ops::detail
