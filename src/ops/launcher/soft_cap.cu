// Implements: include/ninfer/ops/soft_cap.h
// Finite dispatch: aligned BF16x8 production route, BF16x2 fallback, then scalar fallback for
// two-byte-aligned sliced storage.
#include "ops/launcher/soft_cap.h"

#include "ops/common/math.h"
#include "ops/kernel/soft_cap.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void soft_cap_launch(Tensor& x, float cap, cudaStream_t stream) {
    constexpr int block   = 256;
    constexpr int maxGrid = 16384;
    const std::int64_t n  = x.numel();
    const auto address    = reinterpret_cast<std::uintptr_t>(x.data);

    if ((address & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0 &&
        n <= kBf16x8CacheSizedMaxElements) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            maxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
        soft_cap_bf16x8_kernel<<<grid, block, 0, stream>>>(static_cast<Bf16x8Pack*>(x.data), packs,
                                                          cap);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const bool paired = (address & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block * kSoftCapPairsPerThread)),
            std::numeric_limits<int>::max()));
        auto* x2                 = static_cast<__nv_bfloat162*>(x.data);
        auto* tail               = static_cast<__nv_bfloat16*>(x.data) + pairs * 2;
        soft_cap_bf16x2_kernel<<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0, cap);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(
        std::min<std::int64_t>(div_up(n, static_cast<std::int64_t>(block)),
                               std::numeric_limits<int>::max()));
    soft_cap_scalar_kernel<<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n,
                                                       cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
