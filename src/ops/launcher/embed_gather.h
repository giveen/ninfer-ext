#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {

// The embedding scale is a value in the weight dtype, so it is resolved to BF16 once at the Op
// boundary: every gather kernel multiplies by the same value, and a caller's unrounded float does
// not change the result.
inline float bf16_rounded_scale(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return value; } // NaN keeps its payload
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    float rounded;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

enum class Q8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, float embed_scale,
                               Tensor& out, cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, float embed_scale, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q8_launch(const Tensor& ids, const Weight& table, float embed_scale, Tensor& out,
                            cudaStream_t stream);
void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, float embed_scale,
                             Tensor& out, cudaStream_t stream);
void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, float embed_scale,
                                 Tensor& out, Q8EmbedRoute route, cudaStream_t stream);
const char* q8_embed_route_name(Q8EmbedRoute route);

} // namespace ninfer::ops::detail
