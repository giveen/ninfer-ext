// Implements: include/ninfer/ops/nemotron_h_moe_experts.h
//
// One route: one CTA per token. The token's activation row, one selected expert's up result and
// the running output row all live in shared memory, and both banks are read in place at their
// stored expert-major row spans. Every geometry and format takes it, and it keeps no tiling,
// no cross-token work and no workspace -- see the launcher for why that is the right first route
// for this Op and where a streamed one belongs.
//
// The bank format is a reader: a BF16 bank hands back its stored element and a low-bit bank decodes
// one (E2M1 code, E4M3 block scale, weight divisor), and the kernel body is the same for both. The
// two banks of a layer share one format, so the launcher instantiates the two readers together.
//
// Two addressing forms exist for a low-bit bank and both are the same reader: a *stored* bank,
// whose planes are the artifact's own (plane p of expert e at `base[p] + e*stride[p]` with the
// per-expert plane size as the stride), and a *staged* bank, where a whole layer's planes were
// copied into one device buffer in exactly that layout. Where the bytes live -- the artifact's
// pinned host mapping read over UVA, a device-resident upload, or a staging buffer -- changes only
// the addresses the reader is handed, which is what lets one route serve the in-place and the
// staged cases.
#pragma once

#include "ops/kernel/relu2.cuh"             // relu2_one
#include "ops/linear/nvfp4/nvfp4_codec.cuh" // decode_nvfp4_e2m1x2, decode_nvfp4_e4m3, nvfp4_scale_byte_runtime

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The E2M1 code table: index 0..7 are the positive values, 8..15 the negatives of the same
// magnitudes. The same table as the offloaded route's, so the two decode a stored weight alike.
__device__ __constant__ float kNemotronE2m1[16] = {0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,
                                                   4.0F,  6.0F,  -0.0F, -0.5F, -1.0F, -1.5F,
                                                   -2.0F, -3.0F, -4.0F, -6.0F};

// A BF16 bank, expert-major: the stored element is the value. The same reader serves both banks of
// a layer because both are one stacked row-major plane; only the row count and the column count
// differ, and the kernel supplies both. `inverse\` is the bank's weight divisor and is always one
// here, so the interface matches the low-bit reader's and costs nothing.
struct Bf16BankReader {
    const __nv_bfloat16* values;
    int columns;

    __device__ __forceinline__ float inverse_divisor(int) const { return 1.0F; }

    __device__ __forceinline__ float operator()(int row, int column, float inverse) const {
        return inverse * __bfloat162float(values[static_cast<std::size_t>(row) * columns + column]);
    }
};

// A block-scaled NVFP4 bank: E2M1 codes (two per byte, low nibble even), one E4M3 scale per 16
// columns in the K16M128x4 swizzle, and one weight divisor per `divisor_rows` rows of the bank.
// Expert e of the bank holds rows `[e*rows_per_expert, (e+1)*rows_per_expert)` of its matrix, and
// its two planes sit at `codes + e*code_stride` and `scales + e*scale_stride`: that is the stored
// bank's layout and, byte for byte, the staged bank's, so the same reader reads both.
//
// Both planes are addressed by the *expert-local* row: expert e owns one run of `rows_per_expert`
// rows and one run of scale blocks inside the bank, and its planes are contiguous at
// `base[p] + e*stride[p]`. That is what makes an expert's planes a plain base-plus-stride, and it is
// exactly why `rows_per_expert` must be a multiple of 128 -- the K16M128x4 tiling covers 128 rows per
// block, so a run that did not start and end on a 128-row boundary would share scale blocks with its
// neighbours and its planes would no longer be contiguous. The Op requires it (see the wrapper)
// rather than computing a second, bank-global address that would only work for some banks.
struct Nvfp4BankReader {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* scales = nullptr;
    const float* divisors      = nullptr;
    std::int64_t code_stride   = 0;
    std::int64_t scale_stride  = 0;
    int rows_per_expert        = 0;
    int experts                = 0;
    int divisor_rows           = 0;
    int columns                = 0;

    // The bank's weight divisor for `row`. A bank whose divisor covers the whole bank -- one word
    // per matrix, which is how this checkpoint stores its routed experts -- resolves to element
    // zero for every row, and the reciprocal is then loop-invariant. The general form is the
    // per-expert one, where a row's divisor is `row / divisor_rows`.
    __device__ __forceinline__ float inverse_divisor(int row) const {
        const int index = divisor_rows >= rows_per_expert * experts ? 0 : row / divisor_rows;
        return __frcp_rn(divisors[index]);
    }

    __device__ __forceinline__ float operator()(int row, int column, float inverse) const {
        const int expert      = row / rows_per_expert;
        const int local       = row - expert * rows_per_expert;
        const std::int64_t at = static_cast<std::int64_t>(expert) * code_stride +
                                static_cast<std::int64_t>(local) * (columns / 2) + (column >> 1);
        const std::uint8_t packed = codes[at];
        const std::uint8_t code   = (column & 1) != 0 ? static_cast<std::uint8_t>(packed >> 4)
                                                      : static_cast<std::uint8_t>(packed & 0x0FU);
        const std::uint8_t scale = scales[static_cast<std::int64_t>(expert) * scale_stride +
                                          nvfp4_scale_byte_runtime(local, column >> 4, columns / 64)];
        return kNemotronE2m1[code] * decode_nvfp4_e4m3(scale) * inverse;
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
        // chain is latency-bound rather than arithmetic-bound. The row's weight divisor is hoisted
        // out of the column loop: it is one reciprocal per row, not one per weight.
        const int up_base = expert * intermediate;
        for (int i = threadIdx.x; i < intermediate; i += blockDim.x) {
            const int up_row    = up_base + i;
            const float inverse = up.inverse_divisor(up_row);
            float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
            int h = 0;
            for (; h + 4 <= latent; h += 4) {
                a0 += up(up_row, h, inverse) * activation[h];
                a1 += up(up_row, h + 1, inverse) * activation[h + 1];
                a2 += up(up_row, h + 2, inverse) * activation[h + 2];
                a3 += up(up_row, h + 3, inverse) * activation[h + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; h < latent; ++h) { acc += up(up_row, h, inverse) * activation[h]; }
            // No gate: the activation is the squared ReLU of this projection's own result.
            product[i] = relu2_one(acc);
        }
        __syncthreads();

        // Expert e's down rows are [e*H, (e+1)*H), each of the intermediate width.
        const int down_base = expert * latent;
        for (int h = threadIdx.x; h < latent; h += blockDim.x) {
            const int down_row  = down_base + h;
            const float inverse = down.inverse_divisor(down_row);
            float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
            int i = 0;
            for (; i + 4 <= intermediate; i += 4) {
                a0 += down(down_row, i, inverse) * product[i];
                a1 += down(down_row, i + 1, inverse) * product[i + 1];
                a2 += down(down_row, i + 2, inverse) * product[i + 2];
                a3 += down(down_row, i + 3, inverse) * product[i + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; i < intermediate; ++i) { acc += down(down_row, i, inverse) * product[i]; }
            accumulator[h] += weight * acc;
        }
        __syncthreads();
    }

    for (int h = threadIdx.x; h < latent; h += blockDim.x) {
        out[static_cast<std::size_t>(token) * latent + h] = __float2bfloat16_rn(accumulator[h]);
    }
}

} // namespace ninfer::ops::detail
