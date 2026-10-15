#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::kimi_delta_attention {

inline constexpr int kStateDim         = 128;
inline constexpr int kChunkSize        = 16;
inline constexpr int kChunkedMinTokens = 12;
inline constexpr float kQkL2NormEps    = 1.0e-6F;

struct Arguments {
    const __nv_bfloat16 *q, *k, *v, *g, *beta;
    const float *a_log, *dt_bias, *state_in;
    float* state_out;
    __nv_bfloat16* out;
    int qk_heads, value_heads, tokens;
    float lower_bound, scale;
};

inline bool valid_heads(int qk, int value) { return qk > 0 && value >= qk && value % qk == 0; }

__host__ __device__ inline int chunk_count(int tokens) { return div_up(tokens, kChunkSize); }

// One producer/consumer packet. The vector and square tiles use the swizzles below in both
// global and shared memory, so recurrence can asynchronously copy whole packets without repacking.
struct alignas(256) Chunk {
    __nv_bfloat16 kd[kChunkSize * kStateDim];
    __nv_bfloat16 qd[kChunkSize * kStateDim];
    __nv_bfloat16 kr[kChunkSize * kStateDim];
    float gamma[kStateDim];
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
};

static_assert(sizeof(Chunk) == 14848);

__host__ __device__ constexpr int vector_index(int row, int col) {
    return row * kStateDim + (col ^ (row * 8));
}

// Kr time rows are interleaved {0,4,1,5,2,6,3,7} for transposed BF16 MMA loads.
__host__ __device__ constexpr int restored_index(int row, int col) {
    const int physical = (row & ~7) + (row & 3) * 2 + ((row >> 2) & 1);
    return vector_index(physical, col);
}

__host__ __device__ constexpr int square_index(int row, int col) {
    return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4));
}

// Bytes of the chunked workspace: `value_heads` head-major runs of `chunk_count(tokens)` chunks,
// matching the kernels' `workspace + head * chunk_count(tokens) + chunk` indexing. The region must
// start 256-byte aligned because both launch paths reinterpret it as `Chunk*`.

inline std::size_t chunked_workspace_bytes(int value_heads, int tokens) {
    if (value_heads <= 0 || tokens <= 0) {
        throw std::invalid_argument("kimi delta attention workspace needs positive geometry");
    }
    return Tensor(nullptr, DType::U8,
                  {static_cast<int>(sizeof(Chunk)), value_heads, chunk_count(tokens)})
        .bytes();
}

void launch_recurrent(const Arguments& args, cudaStream_t stream);
void launch_batch_update(const Arguments& args, const std::int32_t* slots, int batch,
                         cudaStream_t stream);
void launch_prepare(const Arguments& args, Chunk* workspace, cudaStream_t stream);
void launch_chunk_recurrence(const Arguments& args, const Chunk* workspace,
                             DeviceExecutionView execution);
int chunk_value_tile(int value_heads, int multiprocessor_count);

} // namespace ninfer::ops::detail::kimi_delta_attention
