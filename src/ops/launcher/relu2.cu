// ninfer::ops - relu2 launcher: storage-alignment routes for one elementwise activation.
#include "ops/launcher/relu2.h"

#include "ops/common/math.h"
#include "ops/kernel/relu2.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void relu2_launch(Tensor& x, cudaStream_t stream) {
    constexpr int block   = 256;
    constexpr int maxGrid = 16384;
    const std::int64_t n  = x.numel();
    const auto address    = reinterpret_cast<std::uintptr_t>(x.data);
    if ((address & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0 &&
        n <= kBf16x8CacheSizedMaxElements) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            maxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
        relu2_bf16x8_kernel<block>
            <<<grid, block, 0, stream>>>(static_cast<Bf16x8Pack*>(x.data), packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool paired = (address & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block * kRelu2PairsPerThread)),
            std::numeric_limits<int>::max()));
        relu2_bf16x2_kernel<block><<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat162*>(x.data), pairs,
            static_cast<__nv_bfloat16*>(x.data) + pairs * 2, (n & 1) != 0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    relu2_scalar_kernel<block><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
