// ninfer::ops - the dense BF16 fallback route, for a problem no registered shape matches.
//
// This is a translation unit of its own so the route is one symbol a reviewer can follow, rather
// than a definition folded into another shape's file: it is what makes every Nemotron-H projection
// compute at all (see bf16_generic.cuh), and the hand-written per-shape kernels this registry is
// built around are still the intended route for a width a model actually runs.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/bf16/bf16_generic.cuh"
#include "ops/linear/bf16/bf16_shapes.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void bf16_generic_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = weight.n;
    const std::int32_t k      = weight.k;
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int64_t owners =
        static_cast<std::int64_t>(n) *
        ((tokens + kBf16GenericTokensPerThread - 1) / kBf16GenericTokensPerThread);
    const std::int64_t blocks = (owners + kThreads - 1) / kThreads;
    bf16_generic_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), n, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
