// Implements: include/ninfer/ops/ple.h
// Column-parallel SIMT kernels. Hashing is exact integer arithmetic; every floating-point map
// accumulates in FP32 and rounds once to BF16.
#include "ops/ple/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock          = 256;
constexpr int kMaximumHistory = 32;

// Kernel-side copy of NgramHashTable (std::array is host-only).
struct NgramKernelTable {
    std::int32_t ngram_size;
    std::int32_t heads_per_ngram;
    std::int32_t eos_token_id;
    std::int64_t multipliers[kNgramMaximumSize];
    std::int64_t moduli[kNgramMaximumHeads];
    std::int64_t offsets[kNgramMaximumHeads];
};

NgramKernelTable kernel_table(const NgramHashTable& table) {
    NgramKernelTable out{table.ngram_size, table.heads_per_ngram, table.eos_token_id, {}, {}, {}};
    for (int i = 0; i < kNgramMaximumSize; ++i) { out.multipliers[i] = table.multipliers[i]; }
    for (int i = 0; i < kNgramMaximumHeads; ++i) {
        out.moduli[i]  = table.moduli[i];
        out.offsets[i] = table.offsets[i];
    }
    return out;
}

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 1 << 20));
}

__device__ __forceinline__ std::int32_t combined_token(const std::int32_t* tokens,
                                                       const std::int32_t* history,
                                                       std::int32_t context, std::int32_t slot,
                                                       std::int32_t slots, std::int32_t width,
                                                       std::int32_t lane, std::int32_t index) {
    if (index < context) { return history[static_cast<std::int64_t>(slot) * context + index]; }
    return tokens[static_cast<std::int64_t>(lane) * width + (index - context)];
}

__global__ void ngram_hash_rows_kernel(const std::int32_t* __restrict__ tokens,
                                       const std::int32_t* __restrict__ history,
                                       const std::int32_t* __restrict__ source_slots,
                                       NgramKernelTable table, std::int32_t* __restrict__ rows,
                                       std::int32_t width, std::int32_t lanes, std::int32_t slots) {
    const std::int64_t item = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (item >= static_cast<std::int64_t>(width) * lanes) { return; }
    const std::int32_t w       = static_cast<std::int32_t>(item % width);
    const std::int32_t lane    = static_cast<std::int32_t>(item / width);
    const std::int32_t context = table.ngram_size - 1;
    const std::int32_t slot    = source_slots[lane];
    const std::int32_t current = context + w;
    std::int64_t shifted[kNgramMaximumSize];
    bool closed = false; // an EOS at distance < k closes every longer context
    for (int k = 0; k < table.ngram_size; ++k) {
        if (k == 0) {
            shifted[0] =
                combined_token(tokens, history, context, slot, slots, width, lane, current);
            continue;
        }
        const std::int32_t previous =
            combined_token(tokens, history, context, slot, slots, width, lane, current - k);
        // Token current-k is admitted only if none of current-k .. current-1 is EOS.
        closed     = closed || previous == table.eos_token_id;
        shifted[k] = closed ? table.eos_token_id : previous;
    }
    const std::int32_t heads = (table.ngram_size - 1) * table.heads_per_ngram;
    std::int32_t* out        = rows + item * heads;
    for (int n = 2; n <= table.ngram_size; ++n) {
        std::int64_t mixed = shifted[0] * table.multipliers[0];
        for (int k = 1; k < n; ++k) { mixed ^= shifted[k] * table.multipliers[k]; }
        const int first = (n - 2) * table.heads_per_ngram;
        for (int j = first; j < first + table.heads_per_ngram; ++j) {
            const std::int64_t modulus = table.moduli[j];
            std::int64_t row           = mixed % modulus;
            if (row < 0) { row += modulus; } // torch.remainder is floor-mod
            out[j] = static_cast<std::int32_t>(row + table.offsets[j]);
        }
    }
}

__global__ void ngram_history_advance_kernel(const std::int32_t* __restrict__ tokens,
                                             const std::int32_t* __restrict__ valid_columns,
                                             std::int32_t* __restrict__ history,
                                             const std::int32_t* __restrict__ source_slots,
                                             const std::int32_t* __restrict__ destination_slots,
                                             std::int32_t width, std::int32_t lanes,
                                             std::int32_t context) {
    const std::int32_t lane = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= lanes) { return; }
    std::int32_t count = width;
    if (valid_columns != nullptr) { count = max(0, min(width, valid_columns[lane])); }
    const std::int32_t source = source_slots[lane];
    std::int32_t values[kNgramMaximumSize];
    for (int r = 0; r < context; ++r) {
        const std::int32_t index = count + r; // over [history(context) ; tokens(count)]
        values[r]                = index < context
                                       ? history[static_cast<std::int64_t>(source) * context + index]
                                       : tokens[static_cast<std::int64_t>(lane) * width + (index - context)];
    }
    const std::int32_t destination = destination_slots[lane];
    for (int r = 0; r < context; ++r) {
        history[static_cast<std::int64_t>(destination) * context + r] = values[r];
    }
}

__global__ void gather_scaled_fp8_rows_kernel(const std::uint8_t* __restrict__ codes,
                                              const __nv_bfloat16* __restrict__ scales,
                                              const std::int32_t* __restrict__ rows,
                                              __nv_bfloat16* __restrict__ out, std::int32_t width,
                                              std::int32_t heads, std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(width) * heads * columns;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c   = static_cast<std::int32_t>(i % width);
        const std::int64_t jt  = i / width; // (column * heads + head)
        const std::int64_t row = rows[jt];
        __nv_fp8_e4m3 code;
        code.__x      = codes[row * width + c];
        const float v = static_cast<float>(code) * __bfloat162float(scales[row]);
        out[i]        = __float2bfloat16_rn(v);
    }
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
        total = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : 0.0F;
        total = warp_sum(total);
        if (lane == 0) { scratch[0] = total; }
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// One CTA per column.
__global__ void ple_gate_kernel(const __nv_bfloat16* __restrict__ key,
                                const __nv_bfloat16* __restrict__ query,
                                const __nv_bfloat16* __restrict__ value,
                                __nv_bfloat16* __restrict__ gated, std::int32_t width,
                                std::int32_t streams) {
    __shared__ float scratch[32];
    __shared__ float gate[16];
    const std::int64_t column = blockIdx.x;
    const std::int64_t wide   = static_cast<std::int64_t>(width) * streams;
    for (int s = 0; s < streams; ++s) {
        float dot               = 0.0F;
        const std::int64_t base = column * wide + static_cast<std::int64_t>(s) * width;
        for (int h = static_cast<int>(threadIdx.x); h < width; h += blockDim.x) {
            dot += __bfloat162float(key[base + h]) * __bfloat162float(query[base + h]);
        }
        const float g = block_sum(dot, scratch) * rsqrtf(static_cast<float>(width));
        if (threadIdx.x == 0) {
            const float root = sqrtf(fmaxf(fabsf(g), 1e-6F));
            const float sign = g > 0.0F ? 1.0F : (g < 0.0F ? -1.0F : 0.0F);
            gate[s]          = sigmoid(sign * root);
        }
    }
    __syncthreads();
    for (std::int64_t i = threadIdx.x; i < wide; i += blockDim.x) {
        const std::int32_t s = static_cast<std::int32_t>(i / width);
        const std::int32_t h = static_cast<std::int32_t>(i - static_cast<std::int64_t>(s) * width);
        gated[column * wide + i] =
            __float2bfloat16_rn(gate[s] * __bfloat162float(value[column * width + h]));
    }
}

__global__ void ple_dilated_conv_kernel(const __nv_bfloat16* __restrict__ normed,
                                        const __nv_bfloat16* __restrict__ gated,
                                        const __nv_bfloat16* __restrict__ weight,
                                        const __nv_bfloat16* __restrict__ states,
                                        const std::int32_t* __restrict__ source_slots,
                                        __nv_bfloat16* __restrict__ residual, std::int32_t channels,
                                        std::int32_t width, std::int32_t lanes, std::int32_t taps,
                                        std::int32_t dilation) {
    const std::int64_t total   = static_cast<std::int64_t>(channels) * width * lanes;
    const std::int32_t history = (taps - 1) * dilation;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c    = static_cast<std::int32_t>(i % channels);
        const std::int64_t wb   = i / channels;
        const std::int32_t w    = static_cast<std::int32_t>(wb % width);
        const std::int32_t lane = static_cast<std::int32_t>(wb / width);
        const std::int64_t slot = source_slots[lane];
        float sum               = 0.0F;
        for (int j = 0; j < taps; ++j) {
            const std::int32_t m = w - (taps - 1 - j) * dilation;
            float x;
            if (m >= 0) {
                x = __bfloat162float(
                    normed[(static_cast<std::int64_t>(lane) * width + m) * channels + c]);
            } else {
                const std::int32_t row = history + m;
                x                      = __bfloat162float(states[slot * history * channels +
                                            static_cast<std::int64_t>(row) * channels + c]);
            }
            sum += __bfloat162float(weight[static_cast<std::int64_t>(j) * channels + c]) * x;
        }
        residual[i] = __float2bfloat16_rn(__bfloat162float(residual[i]) +
                                          __bfloat162float(gated[i]) + silu(sum));
    }
}

__global__ void ple_conv_advance_kernel(const __nv_bfloat16* __restrict__ normed,
                                        const std::int32_t* __restrict__ valid_columns,
                                        __nv_bfloat16* __restrict__ states,
                                        const std::int32_t* __restrict__ source_slots,
                                        const std::int32_t* __restrict__ destination_slots,
                                        std::int32_t channels, std::int32_t width,
                                        std::int32_t lanes, std::int32_t history) {
    const std::int64_t total = static_cast<std::int64_t>(channels) * lanes;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c    = static_cast<std::int32_t>(i % channels);
        const std::int32_t lane = static_cast<std::int32_t>(i / channels);
        std::int32_t count      = width;
        if (valid_columns != nullptr) { count = max(0, min(width, valid_columns[lane])); }
        const std::int64_t source = source_slots[lane];
        __nv_bfloat16 values[kMaximumHistory];
        for (int r = 0; r < history; ++r) {
            const std::int32_t index = count + r;
            values[r]                = index < history
                                           ? states[(source * history + index) * channels + c]
                                           : normed[(static_cast<std::int64_t>(lane) * width + (index - history)) *
                                         channels +
                                     c];
        }
        const std::int64_t destination = destination_slots[lane];
        for (int r = 0; r < history; ++r) {
            states[(destination * history + r) * channels + c] = values[r];
        }
    }
}

} // namespace

void ngram_hash_rows_launch(const Tensor& tokens, const Tensor& history, const Tensor& source_slots,
                            const NgramHashTable& table, Tensor& rows, cudaStream_t stream) {
    const std::int32_t width = tokens.ne[0];
    const std::int32_t lanes = tokens.ne[1];
    ngram_hash_rows_kernel<<<grid_for(static_cast<std::int64_t>(width) * lanes), kBlock, 0,
                             stream>>>(static_cast<const std::int32_t*>(tokens.data),
                                       static_cast<const std::int32_t*>(history.data),
                                       static_cast<const std::int32_t*>(source_slots.data),
                                       kernel_table(table), static_cast<std::int32_t*>(rows.data),
                                       width, lanes, history.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ngram_history_advance_launch(const Tensor& tokens, const Tensor* valid_columns,
                                  Tensor& history, const Tensor& source_slots,
                                  const Tensor& destination_slots, cudaStream_t stream) {
    const std::int32_t lanes = tokens.ne[1];
    ngram_history_advance_kernel<<<(lanes + 31) / 32, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(tokens.data),
        valid_columns != nullptr ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr,
        static_cast<std::int32_t*>(history.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data), tokens.ne[0], lanes,
        history.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void gather_scaled_fp8_rows_launch(const Weight& table, const Tensor& rows, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t heads   = rows.ne[0];
    const std::int64_t columns = rows.ne[1];
    gather_scaled_fp8_rows_kernel<<<grid_for(out.numel()), kBlock, 0, stream>>>(
        static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales),
        static_cast<const std::int32_t*>(rows.data), static_cast<__nv_bfloat16*>(out.data), table.k,
        heads, columns);
    CUDA_CHECK(cudaGetLastError());
}

void ple_gate_launch(const Tensor& key, const Tensor& query, const Tensor& value,
                     std::int32_t streams, Tensor& gated, cudaStream_t stream) {
    const std::int64_t columns = value.numel() / value.ne[0];
    ple_gate_kernel<<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data), static_cast<const __nv_bfloat16*>(query.data),
        static_cast<const __nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(gated.data),
        value.ne[0], streams);
    CUDA_CHECK(cudaGetLastError());
}

void ple_dilated_conv_launch(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                             std::int32_t dilation, const Tensor& states,
                             const Tensor& source_slots, Tensor& residual, cudaStream_t stream) {
    ple_dilated_conv_kernel<<<grid_for(normed.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normed.data),
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<__nv_bfloat16*>(residual.data), normed.ne[0], normed.ne[1], normed.ne[2],
        weight.ne[1], dilation);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_advance_launch(const Tensor& normed, const Tensor* valid_columns,
                             std::int32_t history, Tensor& states, const Tensor& source_slots,
                             const Tensor& destination_slots, cudaStream_t stream) {
    const std::int32_t channels = normed.ne[0];
    const std::int32_t lanes    = normed.ne[2];
    ple_conv_advance_kernel<<<grid_for(static_cast<std::int64_t>(channels) * lanes), kBlock, 0,
                              stream>>>(
        static_cast<const __nv_bfloat16*>(normed.data),
        valid_columns != nullptr ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr,
        static_cast<__nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data), channels, normed.ne[1], lanes,
        history);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
