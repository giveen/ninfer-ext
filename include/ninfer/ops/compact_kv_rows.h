#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * Assembles the compact KV rows of the global attention layers:
 *
 *   ideal[d, h, t]         = v[d, h, t]          for d < R
 *   ideal[R + i, h, t]     = k[i, h, t]          for i < P
 *   ideal[R + P + i, h, t] = k[R/2 + i, h, t]    for i < P
 *
 * `v` and `k` are contiguous BF16 with equal shapes whose leading extent is R, and `out` is
 * contiguous BF16 with the same trailing extents and a leading extent of R + 2P. The registered
 * profile is [R,Hkv,T,B]: one row per (KV head, token, batch), so the batch multiplies the row
 * count and nothing else changes. The first R entries of a compact row hold the value vector; the 2P that
 * follow hold the rotated key dimensions in low-then-high order, which is the pair structure a
 * rotation of P pairs over an R-wide head produces: dims [0,P) are followed by dims [R/2,R/2+P).
 * The dimensions an R-wide rotation leaves alone are not stored, because the value vector stands in
 * for them once the query is prescaled: the representation is the plan's 640 values per row for
 * R = 512 and P = 64. `1 <= P <= R/2`. Every output entry is a bit-exact copy of its source entry,
 * so the oracle is the exact source value and no tolerance applies. `out` must not overlap `v` or
 * `k`, and `v` and `k` may overlap each other. The Op uses no workspace or persistent state.
 */
void compact_kv_rows(const Tensor& v, const Tensor& k, Tensor& out, std::int32_t rotary_dim,
                     std::int32_t rotary_pairs, cudaStream_t stream);

} // namespace ninfer::ops
