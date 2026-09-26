#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_a16_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

constexpr int kMmaMaxTokens = 16;

// Through T = 16 the Tensor Core kernel streams the weight once at close to the single-token rate;
// one 8-token tile covers T <= 8 and two tiles cover the rest.
template <class Geometry, int TokenTiles>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = Nvfp4A16MmaSchedule<TokenTiles == 1 ? 1 : 2, TokenTiles, 8>;
    using Tile     = Nvfp4A16MmaRowTile<Geometry, Schedule, Nvfp4AddResidualEpilogue,
                                        Nvfp4ContiguousOutput>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    const Tile tile{{output, Geometry::kOutputRows}, {output, Geometry::kOutputRows}};
    nvfp4_a16_mma_kernel<Geometry, Schedule, Tile><<<Tile::kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor, tile,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int ActiveTokens>
void launch_simt(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = Nvfp4SimtSchedule<4, 1, 2, (ActiveTokens <= 20) ? 8 : 16, ActiveTokens, 1,
                                       Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const float inverse       = 1.0F / weight.weight_scale_divisor;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_simt_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_simt<Geometry, kMmaMaxTokens + 1 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
struct Launchers {
    static constexpr auto kSimt =
        make_simt_launchers<Geometry>(std::make_index_sequence<32 - kMmaMaxTokens>{});

    static void launch(const Tensor& x, const Weight& weight, Tensor& residual,
                       cudaStream_t stream) {
        const std::int32_t tokens = x.ne[1];
        if (tokens <= 8) return launch_mma<Geometry, 1>(x, weight, residual, stream);
        if (tokens <= kMmaMaxTokens) return launch_mma<Geometry, 2>(x, weight, residual, stream);
        kSimt[static_cast<std::size_t>(tokens - kMmaMaxTokens - 1)](x, weight, residual, stream);
    }
};

} // namespace

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        Launchers<Nvfp4N5120K6144>::launch(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        Launchers<Nvfp4N5120K17408>::launch(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
