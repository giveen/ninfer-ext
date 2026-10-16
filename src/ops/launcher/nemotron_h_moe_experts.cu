// Implements: include/ninfer/ops/nemotron_h_moe_experts.h
//
// Two routes, instantiated once per bank format: the expert bank is a reader, so a kernel body is
// the same for every format a reader exists for.
//
//  * `Route::PerToken` is one CTA per token walking the K selected experts serially. It matches
//    every geometry and every format, needs no workspace, and is the route a wide call takes: at T
//    blocks of `kThreads` threads the grid is the token count, which is the whole GPU for a
//    prefill.
//  * `Route::Narrow` is the same arithmetic split by output row, one block per (token, rank, row
//    tile), for a call narrow enough that a per-token CTA would occupy a single SM. It needs the
//    caller's workspace for each rank's activation and its raw down partial, which the query
//    reports; a wide call needs neither and reports zero.
//
// The narrow route is matched on the plane-addressed NVFP4 form -- the registered form this
// family's banks are stored in -- and on a token count and a pair of widths that keep its tiling
// and its staging exact (see `route_for`). Everything else keeps the per-token route, so no caller
// that predates the narrow route changes behaviour.
#include "ops/launcher/nemotron_h_moe_experts.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/nemotron_h_moe_experts.cuh"

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads                   = 128;
constexpr std::size_t kStaticSharedBytes = 48 * 1024;

// The kernel's shared layout: the activation row, the intermediate product, then the accumulator
// row.
std::size_t expert_shared_bytes(std::int32_t latent, std::int32_t intermediate) {
    const std::size_t floats =
        static_cast<std::size_t>(latent) + intermediate + static_cast<std::size_t>(latent);
    return floats * sizeof(float);
}

// The narrow route's rows per block. Total warps are fixed by the geometry (one thread per output
// row), so the block size only decides how they are grouped: at this family's widths a 64-row up
// tile is 40 blocks per (token, rank) and a 32-row latent tile -- the up bank is `intermediate`
// rows and the down bank only `latent`, so the down phase needs the smaller tile to reach a
// comparable block count -- is 32, which is ~1000 blocks over the two phases for a single token.
constexpr int kNarrowUpThreads   = 64;
constexpr int kNarrowDownThreads = 32;
constexpr int kNarrowSumThreads  = 128;

// The token count the narrow route covers, which is the engine's decode lane count: above it the
// grid of the per-token route is already wide enough to fill the GPU.
constexpr std::int32_t kNarrowTokens = 8;

enum class Route : std::uint8_t { PerToken, Narrow };

// The narrow route matches the registered format, a narrow call, and widths its chunked loops and
// its staging are exact at: the 64-column chunk divides both widths, so a row's reduction has the
// four chains and no tail -- the per-token kernel's tail is a chained continuation of the last
// chain and a chunked loop cannot reproduce it -- and a width that is a multiple of the chunk is a
// multiple of 8, so the activation staging moves whole 16-byte packs. The plane-addressed wrapper
// already refuses widths that are not multiples of 128, so these conditions are a property of that
// route rather than a restriction on it; stating them here keeps a future low-bit route that admits
// a coarser width on the per-token kernel.
Route route_for(QType qtype, std::int32_t latent, std::int32_t intermediate, std::int32_t tokens) {
    if (qtype != QType::NVFP4 || tokens > kNarrowTokens) { return Route::PerToken; }
    if (latent % kNarrowChunkColumns != 0 || intermediate % kNarrowChunkColumns != 0) {
        return Route::PerToken;
    }
    return Route::Narrow;
}

std::int32_t tile_count(std::int32_t extent, int rows_per_block) {
    return (extent + rows_per_block - 1) / rows_per_block;
}

// Where the narrow route's two buffers live inside the caller's workspace: every selected
// (token, rank)'s activation product at the workspace's own start, then its raw down partial on a
// 256-byte boundary so that neither depends on the geometry's parity.
struct NarrowCarve {
    std::size_t partial_offset_floats = 0;
    std::size_t bytes                 = 0;
};

NarrowCarve narrow_carve(std::int32_t latent, std::int32_t intermediate, std::int32_t top_k,
                         std::int32_t tokens) {
    const std::size_t slots   = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(top_k);
    const std::size_t product = slots * static_cast<std::size_t>(intermediate);
    NarrowCarve carve;
    carve.partial_offset_floats = (product + 63) & ~static_cast<std::size_t>(63);
    carve.bytes =
        (carve.partial_offset_floats + slots * static_cast<std::size_t>(latent)) * sizeof(float);
    return carve;
}

template <class UpReader, class DownReader>
void launch_route(const UpReader& up, const DownReader& down, const void* x, const void* expert_ids,
                  const void* expert_weights, std::int32_t latent, std::int32_t intermediate,
                  std::int32_t top_k, std::int32_t tokens, void* out, cudaStream_t stream) {
    const std::size_t shared = expert_shared_bytes(latent, intermediate);
    if (shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(nemotron_h_moe_experts_kernel<UpReader, DownReader>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    }
    nemotron_h_moe_experts_kernel<UpReader, DownReader><<<tokens, kThreads, shared, stream>>>(
        up, down, latent, intermediate, top_k, tokens, static_cast<const std::int32_t*>(expert_ids),
        static_cast<const float*>(expert_weights), static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

// The narrow route: three launches on the caller's stream, each of which the next depends on. The
// Op remains graph-capturable -- nothing here synchronizes, allocates or reads a value back to the
// host -- and the two buffers are the caller's, so the sequence is one captured graph like any
// other narrow call.
void launch_narrow(const Nvfp4BankReader& up, const Nvfp4BankReader& down, const void* x,
                   const void* expert_ids, const void* expert_weights, std::int32_t latent,
                   std::int32_t intermediate, std::int32_t top_k, std::int32_t tokens,
                   void* workspace, void* out, cudaStream_t stream) {
    if (workspace == nullptr) {
        throw std::invalid_argument(
            "nemotron_h_moe_experts: the narrow route needs the workspace the query reports");
    }
    const NarrowCarve carve   = narrow_carve(latent, intermediate, top_k, tokens);
    float* const product      = static_cast<float*>(workspace);
    float* const partial      = static_cast<float*>(workspace) + carve.partial_offset_floats;
    const auto* const ids     = static_cast<const std::int32_t*>(expert_ids);
    const auto* const weights = static_cast<const float*>(expert_weights);

    const std::size_t up_shared = static_cast<std::size_t>(latent) * sizeof(__nv_bfloat16);
    if (up_shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(nemotron_h_moe_experts_narrow_up_kernel<kNarrowUpThreads>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(up_shared)));
    }
    const dim3 up_grid(static_cast<unsigned>(tile_count(intermediate, kNarrowUpThreads)),
                       static_cast<unsigned>(tokens), static_cast<unsigned>(top_k));
    nemotron_h_moe_experts_narrow_up_kernel<kNarrowUpThreads>
        <<<up_grid, kNarrowUpThreads, up_shared, stream>>>(
            up, latent, intermediate, tokens, top_k, ids, static_cast<const __nv_bfloat16*>(x),
            product);
    CUDA_CHECK(cudaGetLastError());

    const std::size_t down_shared = static_cast<std::size_t>(intermediate) * sizeof(float);
    if (down_shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(
            nemotron_h_moe_experts_narrow_down_kernel<kNarrowDownThreads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(down_shared)));
    }
    const dim3 down_grid(static_cast<unsigned>(tile_count(latent, kNarrowDownThreads)),
                         static_cast<unsigned>(tokens), static_cast<unsigned>(top_k));
    nemotron_h_moe_experts_narrow_down_kernel<kNarrowDownThreads>
        <<<down_grid, kNarrowDownThreads, down_shared, stream>>>(down, latent, intermediate, tokens,
                                                                 top_k, ids, product, partial);
    CUDA_CHECK(cudaGetLastError());

    const dim3 sum_grid(static_cast<unsigned>(tile_count(latent, kNarrowSumThreads)),
                        static_cast<unsigned>(tokens));
    nemotron_h_moe_experts_narrow_rank_sum_kernel<kNarrowSumThreads>
        <<<sum_grid, kNarrowSumThreads, 0, stream>>>(partial, weights, latent, top_k, tokens,
                                                     static_cast<__nv_bfloat16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t nemotron_h_moe_experts_workspace(QType qtype, std::int32_t latent, std::int32_t experts,
                                             std::int32_t intermediate, std::int32_t top_k,
                                             std::int32_t tokens) {
    (void)experts;
    // Non-zero exactly when the narrow route exists for this format and these widths: the
    // per-token route asks for nothing -- it holds a token's activation and its product in shared
    // memory and reads a bank that is already addressed where it lies -- and the row-tiled route
    // needs the two buffers the carve below lays out.
    if (qtype != QType::NVFP4 || latent % kNarrowChunkColumns != 0 ||
        intermediate % kNarrowChunkColumns != 0) {
        return 0;
    }
    // A *wide* call of this format needs none of it either, but a caller sizes one arena for every
    // call it makes -- the engine plans its pass arena at the prefill width and then runs its
    // decode calls through that storage -- so above the narrow route's token count the reported
    // size is the widest narrow call's rather than zero. That keeps the query non-decreasing in T,
    // which is what lets a T=1 call fit an arena sized at the widest chunk.
    const std::int32_t covered = tokens < kNarrowTokens ? tokens : kNarrowTokens;
    return narrow_carve(latent, intermediate, top_k, covered).bytes;
}

void nemotron_h_moe_experts_launch(const void* x, const void* expert_ids,
                                   const void* expert_weights, const NemotronHMoeBankLaunch& up,
                                   const NemotronHMoeBankLaunch& down, std::int32_t latent,
                                   std::int32_t experts, std::int32_t intermediate,
                                   std::int32_t top_k, std::int32_t tokens, void* workspace,
                                   void* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (up.qtype != down.qtype) {
        throw std::invalid_argument("nemotron_h_moe_experts: the two banks must share one format");
    }
    // The whole-matrix form has the per-token route only: the row-tiled route is the
    // plane-addressed form's, and a bank this entry point would take it for is refused rather than
    // run at a per-token grid the caller did not ask for.
    if (route_for(up.qtype, latent, intermediate, tokens) == Route::Narrow) {
        throw std::invalid_argument(
            "nemotron_h_moe_experts: this entry point has no row-tiled route");
    }

    if (up.qtype == QType::BF16) {
        const Bf16BankReader up_reader{static_cast<const __nv_bfloat16*>(up.payload), up.columns};
        const Bf16BankReader down_reader{static_cast<const __nv_bfloat16*>(down.payload),
                                         down.columns};
        launch_route(up_reader, down_reader, x, expert_ids, expert_weights, latent, intermediate,
                     top_k, tokens, out, stream);
        return;
    }
    (void)workspace;
    (void)experts;
    throw std::invalid_argument("nemotron_h_moe_experts: unsupported bank format");
}

void nemotron_h_moe_experts_planar_launch(const void* x, const void* expert_ids,
                                          const void* expert_weights, const ExpertWeights& staged,
                                          ExpertBankSpan span, std::int32_t latent,
                                          std::int32_t intermediate, std::int32_t top_k,
                                          std::int32_t tokens, void* workspace, void* out,
                                          cudaStream_t stream) {
    if (tokens <= 0) { return; }
    // Planes: 0 gate/up codes, 1 gate/up scales, 2 down codes, 3 down scales. The up bank is
    // `experts * intermediate` rows of `latent` columns and the down bank `experts * latent` rows
    // of `intermediate` columns, so an expert's plane run is `stride[p]` bytes in both.
    const Nvfp4BankReader up{reinterpret_cast<const std::uint8_t*>(staged.base[0]),
                             reinterpret_cast<const std::uint8_t*>(staged.base[1]),
                             staged.gate_up_divisors,
                             staged.stride[0],
                             staged.stride[1],
                             intermediate,
                             span.experts,
                             staged.gate_up_divisor_rows,
                             latent};
    const Nvfp4BankReader down{reinterpret_cast<const std::uint8_t*>(staged.base[2]),
                               reinterpret_cast<const std::uint8_t*>(staged.base[3]),
                               staged.down_divisors,
                               staged.stride[2],
                               staged.stride[3],
                               latent,
                               span.experts,
                               staged.down_divisor_rows,
                               intermediate};
    if (route_for(QType::NVFP4, latent, intermediate, tokens) == Route::Narrow) {
        launch_narrow(up, down, x, expert_ids, expert_weights, latent, intermediate, top_k, tokens,
                      workspace, out, stream);
        return;
    }
    launch_route(up, down, x, expert_ids, expert_weights, latent, intermediate, top_k, tokens, out,
                 stream);
}

} // namespace ninfer::ops::detail
