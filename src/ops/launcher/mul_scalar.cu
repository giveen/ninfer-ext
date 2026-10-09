// Implements: include/ninfer/ops/mul_scalar.h
// Finite dispatch: aligned BF16x8 production route, BF16x2 fallback, then scalar fallback for
// two-byte-aligned sliced storage.
#include "ops/launcher/mul_scalar.h"

#include "ops/common/math.h"
#include "ops/kernel/mul_scalar.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void mul_scalar_launch(Tensor& x, float factor, cudaStream_t stream) {
    constexpr int block   = 256;
    constexpr int maxGrid = 16384;
    const std::int64_t n  = x.numel();
    const auto address    = reinterpret_cast<std::uintptr_t>(x.data);
    if (n == 0) { return; }

    if ((address & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0 &&
        n <= kBf16x8CacheSizedMaxElements) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            maxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
        mul_scalar_bf16x8_kernel<<<grid, block, 0, stream>>>(static_cast<Bf16x8Pack*>(x.data), packs,
                                                            factor);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const bool paired = (address & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block * kMulScalarPairsPerThread)),
            std::numeric_limits<int>::max()));
        mul_scalar_bf16x2_kernel<<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat162*>(x.data), pairs,
            static_cast<__nv_bfloat16*>(x.data) + pairs * 2, (n & 1) != 0, factor);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    mul_scalar_kernel<<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n, factor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
