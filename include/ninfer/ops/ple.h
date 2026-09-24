#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kNgramMaximumSize  = 4;
inline constexpr int kNgramMaximumHeads = 32;

/**
 * Qwen4Exp n-gram hash constants of one PLE layer. Head j of n-gram n (n in [2, ngram_size]) is
 * `(n-2)*heads_per_ngram + j`; its table row is `(XOR_{k<n} ctx_k * multipliers[k]) mod
 * moduli[head] + offsets[head]` on signed 64-bit products.
 */
struct NgramHashTable {
    std::int32_t ngram_size      = 0;
    std::int32_t heads_per_ngram = 0;
    std::int32_t eos_token_id    = 0;
    std::array<std::int64_t, kNgramMaximumSize> multipliers{};
    std::array<std::int64_t, kNgramMaximumHeads> moduli{};
    std::array<std::int64_t, kNgramMaximumHeads> offsets{};

    [[nodiscard]] constexpr std::int32_t heads() const noexcept {
        return (ngram_size - 1) * heads_per_ngram;
    }
};

/**
 * Table rows of the n-gram hash for `tokens [W,B]` I32.
 *
 * `history` is contiguous I32 `[ngram_size-1, S]` state: column s holds the ngram_size-1 tokens
 * that precede the sequence occupying slot s, oldest first (EOS at sequence start). Lane b reads
 * slot `source_slots[b]`. For position p of the combined sequence `[history ; tokens]` the context
 * token at distance k is used when none of the k tokens ending just before p is EOS, and EOS
 * otherwise. Output `rows` is I32 `[heads, W, B]`. The result is exact; there is no floating point.
 */
void ngram_hash_rows(const Tensor& tokens, const Tensor& history, const Tensor& source_slots,
                     const NgramHashTable& table, Tensor& rows, cudaStream_t stream);

/**
 * Advance n-gram history: slot `destination_slots[b]` receives the last ngram_size-1 tokens of
 * `[history(source_slots[b]) ; tokens[0:count_b, b]]`, where count_b is `valid_columns[b]`
 * clamped to [0,W] when `valid_columns` is given and W otherwise. Source and destination may be the
 * same slot.
 */
void ngram_history_advance(const Tensor& tokens, const Tensor* valid_columns, Tensor& history,
                           const Tensor& source_slots, const Tensor& destination_slots,
                           cudaStream_t stream);

/**
 * Gather row-scaled FP8 table rows: `out[j*R + c, t] = fp8(table[rows[j,t], c]) * scale[rows[j,t]]`
 * for rows of width R. `table` is a complete FP8_E4M3FN_ROW_BF16 weight whose code and scale planes
 * are device-addressable (device or mapped pinned host memory). `rows` is I32 `[heads, T]`; `out`
 * is contiguous BF16 `[heads*R, T]`. The represented value is exact; BF16 rounding is the only
 * error.
 */
void gather_scaled_fp8_rows(const Weight& table, const Tensor& rows, Tensor& out,
                            cudaStream_t stream);

/**
 * PLE stream gate. `key` and `query` are grouped-normalized contiguous BF16 `[S*H, T]`, `value`
 * contiguous BF16 `[H, T]`, `gated` contiguous BF16 `[S*H, T]`:
 *
 *   g[s,t]         = sum_h key[s*H+h,t] * query[s*H+h,t] / sqrt(H)
 *   g'[s,t]        = sign(g) * sqrt(max(|g|, 1e-6))
 *   gated[s*H+h,t] = sigmoid(g'[s,t]) * value[h,t].
 */
void ple_gate(const Tensor& key, const Tensor& query, const Tensor& value, std::int32_t streams,
              Tensor& gated, cudaStream_t stream);

/**
 * Causal dilated depthwise convolution with SiLU, added to the residual with the gated values.
 *
 * `normed` and `gated` are contiguous BF16 `[C, W, B]`; `weight` is contiguous BF16 `[K, C]` (tap
 * major, tap K-1 multiplies the current column); `states` is contiguous BF16 `[(K-1)*dilation*C,
 * S]`, slot s holding the (K-1)*dilation previous `normed` columns oldest first. Lane b reads slot
 * `source_slots[b]`. For every valid column w:
 *
 *   conv[c,w,b]      = sum_j weight[j,c] * x[w - (K-1-j)*dilation]   over [state ; normed]
 *   residual[c,w,b] += gated[c,w,b] + silu(conv[c,w,b]).
 *
 * The state is not modified; ple_conv_advance commits it.
 */
void ple_dilated_conv(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                      std::int32_t dilation, const Tensor& states, const Tensor& source_slots,
                      Tensor& residual, cudaStream_t stream);

/**
 * Advance convolution history like ngram_history_advance: slot `destination_slots[b]` receives the
 * last (K-1)*dilation columns of `[states(source_slots[b]) ; normed[:, 0:count_b, b]]`.
 */
void ple_conv_advance(const Tensor& normed, const Tensor* valid_columns, std::int32_t history,
                      Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                      cudaStream_t stream);

} // namespace ninfer::ops
