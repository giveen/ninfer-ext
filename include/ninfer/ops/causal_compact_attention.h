#pragma once

#include "core/tensor.h"
#include "ninfer/ops/attention_geometry.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Causal attention over Gemma 4's compact global KV representation.
 *
 * A compact row stores the value vector of R = rotary_dim entries followed by the 2 * rope_angles
 * rotated key dims, low run then high run, exactly as `compact_kv_rows` writes them. With
 * `rot` = [0,rope_angles) union [rd/2, rd/2 + rope_angles) and `idx` the index of a rotated dim in
 * the stored run, and with q already scaled by w_kn on the non-rotated dims,
 *
 *   kvh          = h / (Hq / Hkv)
 *   score[k]     = scale * ( sum_{d not in rot} q[d,h,i] * row[d,kvh,k]
 *                          + sum_{d in rot}    q[d,h,i] * row[rd + idx(d),kvh,k] )
 *   ideal[:,h,i] = sum_k probability[k] * row[j,kvh,k], j < rd        (the stored value vector)
 *
 * `q` and `out` are contiguous BF16 [rd,Hq,T,B], `kv` is contiguous BF16 [rd + 2*rope_angles,Hkv,S,B],
 * and `position_q` and `position_k` are contiguous I32 [T,B] and [S,B]. Visibility is purely causal:
 * `0 <= position_q[i] - position_k[k]`. A query row with no visible key writes exact BF16 zero.
 *
 * The query prescale by w_kn is deliberately not part of this Op: it is one `scale_columns` call on
 * the non-rotated dims, and keeping it out means this Op's oracle is the attention formula alone.
 *
 * `indexed_keys` states that key k holds position k (the global cache stores each row at its own
 * position) and that unwritten keys are invisible, which lets the scan stop after the latest query's
 * position; the result is the same as without it whenever the statement holds.
 *
 * The route is tensor-core flash attention (ops/kernel/gemma_flash_attention.cuh) over a widened
 * query that makes q' . row the score above, for (rotary_dim, rope_angles) of (512,64), (128,16) and
 * (64,8). Few query rows split the keys across blocks when `workspace` holds
 * `causal_compact_attention_workspace_bytes`; an empty or smaller workspace selects the one-pass
 * route, which computes the same result. The oracle evaluates the
 * formula naively in FP64 from the represented BF16 values; the BF16 output is promoted for
 * comparison and storage rounding belongs to the Op criterion. All tensors are contiguous,
 * non-overlapping, and every output element is written.
 */
void causal_compact_attention(const Tensor& q, const Tensor& kv, const Tensor& position_q,
                              const Tensor& position_k, AttentionHeadGeometry geometry,
                              std::int32_t rotary_dim, std::int32_t rope_angles, float scale,
                              bool indexed_keys, Tensor& out, const Tensor& workspace,
                              cudaStream_t stream);

std::size_t causal_compact_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t rotary_dim,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch);

} // namespace ninfer::ops
