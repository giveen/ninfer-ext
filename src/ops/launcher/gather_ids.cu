// Implements: include/ninfer/ops/gather_ids.h
// Match: contiguous I32 index/table/out with the count already validated.
// Algorithm assumptions: one thread per index entry; the mapping is a dependent load, so the
// launch is sized by the (always tiny) column count and never tiled.
#include "ops/launcher/gather_ids.h"

#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kGatherIdsBlock = 128;

__global__ void gather_ids_kernel(const std::int32_t* __restrict__ index,
                                  const std::int32_t* __restrict__ table,
                                  std::int32_t* __restrict__ out, std::int32_t count) {
    const std::int32_t t = blockIdx.x * kGatherIdsBlock + threadIdx.x;
    if (t < count) { out[t] = table[index[t]]; }
}

} // namespace

void gather_ids_launch(const Tensor& index, const Tensor& table, Tensor& out, std::int32_t count,
                       cudaStream_t stream) {
    if (count == 0) { return; }
    const int blocks = (count + kGatherIdsBlock - 1) / kGatherIdsBlock;
    gather_ids_kernel<<<blocks, kGatherIdsBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(index.data), static_cast<const std::int32_t*>(table.data),
        static_cast<std::int32_t*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
