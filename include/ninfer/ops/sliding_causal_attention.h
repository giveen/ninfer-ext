#pragma once

#include "core/tensor.h"
#include "ninfer/ops/attention_geometry.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Causal sliding-window grouped-query attention over an explicit K/V pair.
 *
 *   visible(k,i) <=> 0 <= position_q[i] - position_k[k] < window   (self included)
 *   kvh          =  h / (Hq / Hkv)
 *   score[k]     = scale * dot(q[:,h,i], k[:,kvh,k])
 *   ideal[:,h,i] = sum_k probability[k] * v[:,kvh,k], over exactly the visible keys
 *
 * `q` and `out` are contiguous BF16 [D,Hq,T,B], `k` and `v` are contiguous BF16 [D,Hkv,S,B], and
 * `position_q` and `position_k` are contiguous I32 [T,B] and [S,B]. Only the window bound is
 * one-sided: a key strictly ahead of the query is excluded even inside the window, which makes this
 * causal rather than symmetric. A query row with no visible key writes exact BF16 zero, and the
 * window may exceed the key count. `window` is at most 4096, which is what keeps the visible keys in
 * shared memory.
 *
 * The route is tensor-core flash attention (ops/kernel/gemma_flash_attention.cuh): blocks of 16
 * query rows that share a KV head stream 32-key tiles with an online Softmax, and a tile no row can
 * see is skipped. `head_dim` is 32, 64, 128 or 256. When the rows alone give too few blocks (decode),
 * the keys are split across blocks and merged by a second kernel; that needs `workspace` of at least
 * `sliding_causal_attention_workspace_bytes` bytes, and an empty or smaller workspace selects the
 * one-pass route, which computes the same result.
 *
 * The oracle evaluates the formula naively in FP64 from the represented BF16 q/k/v, with a stable
 * Softmax over exactly the visible set. The BF16 output is promoted for comparison and storage
 * rounding belongs to the Op criterion. q/k/v/positions are unchanged and every output element is
 * written. All tensors are contiguous and must not overlap except that k may alias v.
 */
void sliding_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& position_q, const Tensor& position_k,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              Tensor& out, const Tensor& workspace, cudaStream_t stream);

// Workspace bytes the key split needs for these extents; zero when the route does not split.
std::size_t sliding_causal_attention_workspace_bytes(AttentionHeadGeometry geometry,
                                                     std::int32_t query_tokens,
                                                     std::int32_t key_tokens, std::int32_t batch);

} // namespace ninfer::ops
