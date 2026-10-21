#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Maps an index column back to global ids:
 *
 *   out[t] = table[index[t]]  for every t in [0,count).
 *
 * `index` and `out` are contiguous I32 with rank 1 and extent `count`; `table` is contiguous I32
 * with rank 1 and extent `table_rows > 0`. Every entry of `index` must be a row of `table`; the
 * caller establishes that range, as it establishes the table's own ids. `out` must not overlap
 * `index` or `table`. The Op has no workspace and no state beyond writing all of `out`, so it can
 * run inside a captured graph.
 */
void gather_ids(const Tensor& index, const Tensor& table, Tensor& out, std::int32_t count,
                std::int32_t table_rows, cudaStream_t stream);

} // namespace ninfer::ops
