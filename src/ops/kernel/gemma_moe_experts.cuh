// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// One CTA per token: the token's activation row, the selected expert's gate and up vectors, their
// product and the running output row all live in shared memory, and the two banks are read in place
// at their stored expert-major row spans. This route is correctness-first: it streams each selected
// expert's rows exactly once and keeps no tiling or cross-token work.
//
// The bank format is a reader: a BF16 bank hands back its stored element, an NVFP4 bank decodes one
// (codes, block scale, divisor), and the kernel body is the same for both. The two banks of a layer
// share one format, so the launcher instantiates the two readers together.
#pragma once

#include "ops/kernel/gelu.cuh"              // gelu_one
#include "ops/linear/nvfp4/nvfp4_codec.cuh" // decode_nvfp4_e2m1x2, decode_nvfp4_e4m3

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// A BF16 bank: the stored element is the value.
struct Bf16BankReader {
    const __nv_bfloat16* values;
    int columns;

    __device__ __forceinline__ float operator()(int row, int column) const {
        return __bfloat162float(values[static_cast<std::size_t>(row) * columns + column]);
    }
};

// An NVFP4 bank: 4-bit e2m1 codes two per byte along the columns, one e4m3 scale per 16-element
// block in the K16M128x4 tiling, and the weight divisor of the source matrix the row belongs to.
// The divisor is applied exactly as the offload MoE route applies its stacked sources':
// `value * (1 / divisors[row / divisor_rows])`.
struct Nvfp4BankReader {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    const float* divisors;
    int divisor_rows;
    int columns;

    __device__ __forceinline__ float operator()(int row, int column) const {
        const std::size_t code_offset =
            static_cast<std::size_t>(row) * (columns / 2) + static_cast<std::size_t>(column / 2);
        const float2 pair        = decode_nvfp4_e2m1x2(codes[code_offset]);
        const float code         = (column & 1) == 0 ? pair.x : pair.y;
        const std::uint8_t scale = scales[nvfp4_scale_byte_runtime(row, column / 16, columns / 64)];
        return code * decode_nvfp4_e4m3(scale) * __frcp_rn(divisors[row / divisor_rows]);
    }
};

template <class GateUpReader, class DownReader>
__global__ void
gemma_moe_experts_kernel(GateUpReader gate_up, DownReader down, int hidden, int intermediate,
                         int top_k, int tokens, const std::int32_t* __restrict__ expert_ids,
                         const float* __restrict__ expert_weights,
                         const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {
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
        const int gate_base = expert * 2 * intermediate;
        const int up_base   = gate_base + intermediate;
        for (int i = threadIdx.x; i < intermediate; i += blockDim.x) {
            const int gate_row = gate_base + i;
            const int up_row   = up_base + i;
            // Four independent partial sums per matrix. This loop is latency-bound on scattered
            // bank loads -- the serial chain would be a few cycles per element, and the measurement
            // is tens
            // -- so what fills the memory pipeline is having four loads in flight, not the
            // arithmetic.
            float g0 = 0.0F, g1 = 0.0F, g2 = 0.0F, g3 = 0.0F;
            float u0 = 0.0F, u1 = 0.0F, u2 = 0.0F, u3 = 0.0F;
            int h = 0;
            for (; h + 4 <= hidden; h += 4) {
                const float v0 = activation[h];
                const float v1 = activation[h + 1];
                const float v2 = activation[h + 2];
                const float v3 = activation[h + 3];
                g0 += gate_up(gate_row, h) * v0;
                g1 += gate_up(gate_row, h + 1) * v1;
                g2 += gate_up(gate_row, h + 2) * v2;
                g3 += gate_up(gate_row, h + 3) * v3;
                u0 += gate_up(up_row, h) * v0;
                u1 += gate_up(up_row, h + 1) * v1;
                u2 += gate_up(up_row, h + 2) * v2;
                u3 += gate_up(up_row, h + 3) * v3;
            }
            float g = (g0 + g1) + (g2 + g3);
            float u = (u0 + u1) + (u2 + u3);
            for (; h < hidden; ++h) {
                const float value = activation[h];
                g += gate_up(gate_row, h) * value;
                u += gate_up(up_row, h) * value;
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
        const int down_base = expert * hidden;
        for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
            const int row = down_base + h;
            float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
            int i = 0;
            for (; i + 4 <= intermediate; i += 4) {
                const float p0 = product[i];
                const float p1 = product[i + 1];
                const float p2 = product[i + 2];
                const float p3 = product[i + 3];
                a0 += down(row, i) * p0;
                a1 += down(row, i + 1) * p1;
                a2 += down(row, i + 2) * p2;
                a3 += down(row, i + 3) * p3;
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; i < intermediate; ++i) { acc += down(row, i) * product[i]; }
            accumulator[h] += weight * acc;
        }
        __syncthreads();
    }

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[static_cast<std::size_t>(token) * hidden + h] = __float2bfloat16(accumulator[h]);
    }
}

} // namespace ninfer::ops::detail
