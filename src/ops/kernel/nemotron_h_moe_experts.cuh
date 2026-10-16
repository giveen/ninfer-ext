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
// `base[p] + e*stride[p]`. That is what makes an expert's planes a plain base-plus-stride, and it
// is exactly why `rows_per_expert` must be a multiple of 128 -- the K16M128x4 tiling covers 128
// rows per block, so a run that did not start and end on a 128-row boundary would share scale
// blocks with its neighbours and its planes would no longer be contiguous. The Op requires it (see
// the wrapper) rather than computing a second, bank-global address that would only work for some
// banks.
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
        const std::uint8_t scale =
            scales[static_cast<std::int64_t>(expert) * scale_stride +
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

// ---------------------------------------------------------------------------------------------
// The row-tiled route: one block per (token, rank, output-row tile), the token's ranks summed
// afterwards in ascending order, and the call's token axis walked in tiles so that what it needs
// between the two projections is one tile's storage rather than the whole call's.
//
// The per-token route above is one CTA per token walking the K selected experts serially, so a
// single-token call occupies one SM: at this family's geometry that is ~37.7M MACs over ~21.6 MB of
// bank bytes on one 128-thread block, latency-bound on a one-element-at-a-time read of a 4-bit
// bank. The row-tiled route hands the same arithmetic to one block per (rank, row tile) instead,
// which is legal because no output row depends on another: an expert's `intermediate` up rows and
// its `latent` down rows are each independent, so splitting on rows needs no cross-block reduction.
//
// What it does need is somewhere for a rank's activation to live between the two projections: every
// row tile of a (token, rank) down projection reads that rank's whole up result. That buffer and
// each rank's raw down partial are the caller's workspace, both reported by the query -- and they
// are why the token axis is tiled. One tile's `top_k` products over `intermediate` and partials
// over `latent` is a few MiB at this family's geometry, while the same buffers for the whole
// 4096-token prefill chunk are ~530 MB, which would come out of the residency the offload path
// spends on device-resident banks. The launcher walks the tile sequence on the caller's stream, so
// every tile's three kernels reuse one tile's buffers and no launch depends on the call's width.
//
// A kernel launched for a tile is handed `x`, `expert_ids`, `expert_weights` and `out` already
// offset to the tile's first token, while the routing arrays keep the *call's* token count as their
// rank stride (`tokens`). The workspace therefore indexes tile-locally -- `rank * gridDim.y +
// token-in-tile` -- which is what makes it independent of how many tiles the call has.
//
// The numerics are the per-token route's own, exactly -- not to a tolerance:
//
//  * a row's dot product is four accumulator chains, chain `c` taking the columns congruent to `c`
//    modulo 4 in ascending order, combined as `(a0+a1)+(a2+a3)`, and each element is
//    `((code * scale) * inverse) * activation` -- the same association and the same per-element
//    expression the per-token kernel writes, so the two agree bit for bit;
//  * the down partial is the raw FP32 result and the routing weight is applied by the summing
//    kernel, because the per-token kernel's expression is `accumulator += weight * acc` inside its
//    rank loop: writing the partial scaled would introduce a rounding the wide route does not have;
//  * the summing kernel adds a token's K partials in ascending rank order in FP32 with one
//    BF16 rounding at the output, which is the per-token kernel's own order.
//
// Both banks are read in bulk rather than element by element: one iteration covers a 64-column
// chunk, whose codes are two 16-byte loads and whose four 16-column E4M3 scale bytes are one 32-bit
// load, so the loop is bound by the bytes it streams rather than by the latency of one accessor
// call per weight.
inline constexpr int kRowTiledChunkColumns = 64;

// One expert row of an NVFP4 bank against one activation row, in the per-token kernel's own order.
// `bank_row` is the row's index in the whole bank (so the weight divisor and the expert it belongs
// to both resolve the way the reader resolves them), `columns` its own column count -- `latent` for
// an up row, `intermediate` for a down one -- and `activation(pair)` the two represented activation
// values of columns `2*pair`, `2*pair+1`. `columns` must be a multiple of the chunk width, which
// the plane-addressed route's own multiple-of-128 requirement gives.
template <class Activation>
__device__ __forceinline__ float row_tiled_bank_row(const Nvfp4BankReader& bank, int bank_row,
                                                    const Activation& activation, int columns) {
    const int expert          = bank_row / bank.rows_per_expert;
    const int local           = bank_row - expert * bank.rows_per_expert;
    const float inverse       = bank.inverse_divisor(bank_row);
    const std::uint8_t* codes = bank.codes + static_cast<std::int64_t>(expert) * bank.code_stride +
                                static_cast<std::int64_t>(local) * (columns / 2);
    const std::uint8_t* scales =
        bank.scales + static_cast<std::int64_t>(expert) * bank.scale_stride;
    const int scale_tiles = columns / 64;

    float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
    for (int chunk = 0; chunk < columns; chunk += kRowTiledChunkColumns) {
        const uint4 lo = load_vec<uint4>(codes + (chunk >> 1));
        const uint4 hi = load_vec<uint4>(codes + (chunk >> 1) + 16);
        // One 32-bit load covers the chunk's four 16-column scale groups: the K16M128x4 tiling puts
        // a row's four consecutive groups in one aligned word, and a 64-column chunk starts on a
        // group boundary four groups wide.
        const std::uint32_t packed_scales = load_vec<std::uint32_t>(
            scales + nvfp4_scale_byte_runtime(local, chunk / 16, scale_tiles));
        const std::uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int byte = 0; byte < 32; ++byte) {
            const std::uint8_t packed =
                static_cast<std::uint8_t>(byte_of_word(words[byte >> 2], byte & 3U));
            const float scale = decode_nvfp4_e4m3(
                static_cast<std::uint8_t>((packed_scales >> (8 * (byte >> 3))) & 0xFFU));
            const float2 code            = decode_nvfp4_e2m1x2(packed);
            const float2 activation_pair = activation((chunk >> 1) + byte);
            // `byte` carries columns `chunk+2*byte` and `chunk+2*byte+1`: an even byte's pair is
            // congruent to 0 and 1 modulo 4 and an odd byte's to 2 and 3, which is the wide
            // kernel's chain assignment, and the bytes of a chunk are visited in ascending order,
            // which is that kernel's ascending order inside every chain.
            const float low  = (code.x * scale) * inverse;
            const float high = (code.y * scale) * inverse;
            if ((byte & 1) == 0) {
                a0 += low * activation_pair.x;
                a1 += high * activation_pair.y;
            } else {
                a2 += low * activation_pair.x;
                a3 += high * activation_pair.y;
            }
        }
    }
    return (a0 + a1) + (a2 + a3);
}

// The token's activation row, staged for the block's rows to share: `latent` BF16 values, 8 per
// 16-byte load, which the plane-addressed route's multiple-of-128 width makes exact.
template <int kThreads>
__device__ __forceinline__ void row_tiled_stage_activation(const __nv_bfloat16* __restrict__ row,
                                                           int latent,
                                                           __nv_bfloat16* __restrict__ stage) {
    for (int h = static_cast<int>(threadIdx.x) * 8; h < latent; h += kThreads * 8) {
        store_vec(stage + h, load_vec<uint4>(row + h));
    }
    __syncthreads();
}

// One rank's up projection: `intermediate` rows, each a `latent`-wide dot product with the token's
// activation row, squared and rectified into the workspace's product buffer.
//
// Grid: x the up-row tile, y the token within the launch's tile, z the rank. Shared: `latent` BF16,
// the staged activation.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void nemotron_h_moe_experts_row_tiled_up_kernel(
    Nvfp4BankReader up, int latent, int intermediate, int tokens, int top_k,
    const std::int32_t* __restrict__ expert_ids, const __nv_bfloat16* __restrict__ x,
    float* __restrict__ product) {
    extern __shared__ __nv_bfloat16 stage_activation[]; // [latent]
    const int token = static_cast<int>(blockIdx.y);
    row_tiled_stage_activation<kThreads>(x + static_cast<std::size_t>(token) * latent, latent,
                                         stage_activation);

    const int local = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (local >= intermediate) { return; }
    // `slot` addresses the routing arrays at the call's rank stride; `tile_slot` addresses the
    // workspace, which holds one tile -- see the route's note above.
    const std::size_t slot      = static_cast<std::size_t>(blockIdx.z) * tokens + token;
    const std::size_t tile_slot = static_cast<std::size_t>(blockIdx.z) * gridDim.y + token;
    const int expert            = expert_ids[slot];
    const auto* const pairs     = reinterpret_cast<const std::uint32_t*>(stage_activation);
    const auto activation       = [pairs](int pair) { return bf16x2_bits_to_float2(pairs[pair]); };
    const float acc = row_tiled_bank_row(up, expert * intermediate + local, activation, latent);
    product[tile_slot * intermediate + local] = relu2_one(acc);
}

// One rank's down projection: `latent` rows over the rank's staged activation, each written to the
// workspace's partial buffer unscaled (see the numeric note above).
//
// Grid: x the latent-row tile, y the token within the launch's tile, z the rank. Shared:
// `intermediate` FP32, the rank's activation, which every row of the tile reads.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void nemotron_h_moe_experts_row_tiled_down_kernel(
    Nvfp4BankReader down, int latent, int intermediate, int tokens, int top_k,
    const std::int32_t* __restrict__ expert_ids, const float* __restrict__ product,
    float* __restrict__ partial) {
    extern __shared__ float stage_product[]; // [intermediate]
    const int token             = static_cast<int>(blockIdx.y);
    const std::size_t slot      = static_cast<std::size_t>(blockIdx.z) * tokens + token;
    const std::size_t tile_slot = static_cast<std::size_t>(blockIdx.z) * gridDim.y + token;
    const float* source         = product + tile_slot * intermediate;
    for (int i = static_cast<int>(threadIdx.x) * 4; i < intermediate; i += kThreads * 4) {
        store_vec(stage_product + i, load_vec<float4>(source + i));
    }
    __syncthreads();

    const int local = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (local >= latent) { return; }
    const int expert                  = expert_ids[slot];
    const float* const activation_row = stage_product;
    const auto activation             = [activation_row](int pair) {
        return load_vec<float2>(activation_row + 2 * pair);
    };
    partial[tile_slot * latent + local] =
        row_tiled_bank_row(down, expert * latent + local, activation, intermediate);
}

// A token's K partials summed in ascending rank order in FP32, with the routing weight applied
// where the per-token kernel applies it, and the Op's one BF16 rounding.
//
// Grid: x the latent-column tile, y the token within the launch's tile.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void nemotron_h_moe_experts_row_tiled_rank_sum_kernel(
    const float* __restrict__ partial, const float* __restrict__ expert_weights, int latent,
    int top_k, int tokens, __nv_bfloat16* __restrict__ out) {
    const int token = static_cast<int>(blockIdx.y);
    const int h     = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (h >= latent) { return; }
    float accumulator = 0.0F;
    for (int rank = 0; rank < top_k; ++rank) {
        const std::size_t slot      = static_cast<std::size_t>(rank) * tokens + token;
        const std::size_t tile_slot = static_cast<std::size_t>(rank) * gridDim.y + token;
        accumulator += expert_weights[slot] * partial[tile_slot * latent + h];
    }
    out[static_cast<std::size_t>(token) * latent + h] = __float2bfloat16_rn(accumulator);
}

} // namespace ninfer::ops::detail
