// Implements: include/ninfer/ops/nemotron_h_moe_experts.h
//
// Two routes, instantiated once per bank format: the expert bank is a reader, so a kernel body is
// the same for every format a reader exists for.
//
//  * `Route::PerToken` is one CTA per token walking the K selected experts serially. It matches
//    every geometry and every format, needs no workspace, and is the route of every call outside
//    `route_for`'s domain.
//  * `Route::RowTiled` is the same arithmetic split by output row, one block per (token, rank, row
//    tile), with the call's token axis walked in tiles. It takes the calls this family's scheduler
//    issues -- the whole decode lane count up to its widest prefill chunk -- and needs the caller's
//    workspace for each rank's activation and its raw down partial, one tile's worth, which the
//    query reports.
//
// The row-tiled route is matched on the plane-addressed NVFP4 form -- the registered form this
// family's banks are stored in -- and on a pair of widths that keep its tiling and its staging
// exact (see `route_for`). Everything else keeps the per-token route, so no caller is served by a
// route it was not served by before except where the row-tiled route replaces a per-token grid with
// one that reads the same bytes in bulk.
#include "ops/launcher/nemotron_h_moe_experts.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/nemotron_h_moe_experts.cuh"

#include <algorithm>
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

// The row-tiled route's rows per block. Total warps are fixed by the geometry (one thread per
// output row), so the block size only decides how they are grouped: at this family's widest
// geometry an up row tile of 64 rows is 40 blocks per (token, rank) and a down tile of 32 rows --
// the up bank is `intermediate` rows and the down bank only `latent`, so the down phase needs the
// smaller tile to reach a comparable block count -- is 32, which over one tile of 64 tokens and
// nine ranks is ~41k blocks across the two phases: many waves of the whole GPU rather than the one
// block on one SM a per-token CTA would leave it with.
constexpr int kRowTiledUpThreads   = 64;
constexpr int kRowTiledDownThreads = 32;
constexpr int kRowTiledSumThreads  = 128;
// Decode: one warp per output row. Four rows per block keeps a 128-thread CTA and the shared
// activation stage unchanged while giving the row's 32 lanes the whole chain.
constexpr int kRowTiledWarpRowsPerCta = 4;

// The tile the row-tiled route walks a call's token axis in, and the widest call it takes.
//
// The tile bounds the route's workspace: one tile's `top_k` activation products over `intermediate`
// and its `top_k` raw down partials over `latent` is ~8 MiB at this family's widest geometry, where
// the same two buffers for the whole 4096-token prefill chunk are ~530 MB -- which would come out
// of the residency the offload path spends on device-resident banks. 64 tokens is also several
// waves of the route's blocks at those widths, so a tile's three kernels run the GPU out rather
// than run into a launch gap between them.
constexpr std::int32_t kRowTiledTileTokens = 64;

// The widest call the row-tiled route takes: this family's own prefill chunk, whose default is 4096
// and whose candidates are 4096/2048/1024 (`models/nemotron_h/program.h`), so every call this
// family's scheduler issues runs the row-tiled route. A wider call keeps the per-token route, which
// is the route with no width limit; it is also what leaves that route reachable through the public
// entry point at a width the identity test can compare this one against.
constexpr std::int32_t kRowTiledMaxTokens = 4096;

// The tokens the row-tiled route's buffers cover for a call of `tokens`: one tile, or the call
// itself when it is narrower. Both the workspace the query reports and the offsets the launch
// writes at come from this, so a call can never be handed less storage than its own kernels write.
constexpr std::int32_t row_tiled_covered_tokens(std::int32_t tokens) {
    return tokens < kRowTiledTileTokens ? tokens : kRowTiledTileTokens;
}

enum class Route : std::uint8_t { PerToken, RowTiled };

// The row-tiled route matches the registered format, a call no wider than this family's prefill
// chunk, and widths its chunked loops and its staging are exact at: the 64-column chunk divides
// both widths, so a row's reduction has the four chains and no tail -- the per-token kernel's tail
// is a chained continuation of the last chain and a chunked loop cannot reproduce it -- and a width
// that is a multiple of the chunk is a multiple of 8, so the activation staging moves whole 16-byte
// packs. The plane-addressed wrapper already refuses widths that are not multiples of 128, so these
// conditions are a property of that route rather than a restriction on it; stating them here keeps
// a future low-bit route that admits a coarser width on the per-token kernel.
Route route_for(QType qtype, std::int32_t latent, std::int32_t intermediate, std::int32_t tokens) {
    if (qtype != QType::NVFP4 || tokens > kRowTiledMaxTokens) { return Route::PerToken; }
    if (latent % kRowTiledChunkColumns != 0 || intermediate % kRowTiledChunkColumns != 0) {
        return Route::PerToken;
    }
    return Route::RowTiled;
}

std::int32_t tile_count(std::int32_t extent, int rows_per_block) {
    return (extent + rows_per_block - 1) / rows_per_block;
}

// Where the row-tiled route's two buffers live inside the caller's workspace: one token tile's
// selected (token, rank) activation products at the workspace's own start, then their raw down
// partials on a 256-byte boundary so that neither depends on the geometry's parity. The tile is the
// only thing the layout depends on, so a call of any width uses the same buffers.
struct RowTiledCarve {
    std::size_t partial_offset_floats = 0;
    std::size_t bytes                 = 0;
};

RowTiledCarve row_tiled_carve(std::int32_t latent, std::int32_t intermediate, std::int32_t top_k,
                              std::int32_t tokens) {
    const std::size_t slots   = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(top_k);
    const std::size_t product = slots * static_cast<std::size_t>(intermediate);
    RowTiledCarve carve;
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

// The row-tiled route: for each tile of the call's token axis, three launches on the caller's
// stream, each of which the next depends on. Every tile runs on the same two buffers -- which is
// what bounds the route's workspace independently of how wide the call is -- so the tiles are
// ordered by the stream rather than by a barrier, and a call no wider than one tile is one pass
// through the loop. The Op remains graph-capturable -- nothing here synchronizes, allocates or
// reads a value back to the host -- so the whole walk is one captured graph like any other call.
//
// A tile's launches are handed `x`, `expert_ids`, `expert_weights` and `out` offset to the tile's
// first token, while the routing arrays keep the call's own token count as their rank stride; the
// kernels index the workspace tile-locally (see the kernel header).
void launch_row_tiled(const Nvfp4BankReader& up, const Nvfp4BankReader& down, const void* x,
                      const void* expert_ids, const void* expert_weights, std::int32_t latent,
                      std::int32_t intermediate, std::int32_t top_k, std::int32_t tokens,
                      void* workspace, void* out, cudaStream_t stream) {
    if (workspace == nullptr) {
        throw std::invalid_argument(
            "nemotron_h_moe_experts: the row-tiled route needs the workspace the query reports");
    }
    // The buffers are the ones the query reports for this call -- see `row_tiled_covered_tokens` --
    // so a call narrower than one tile gets the tile capped at its own width, which is what keeps
    // the kernel's writes inside the storage the wrapper validated.
    const RowTiledCarve carve =
        row_tiled_carve(latent, intermediate, top_k, row_tiled_covered_tokens(tokens));
    float* const product     = static_cast<float*>(workspace);
    float* const partial     = static_cast<float*>(workspace) + carve.partial_offset_floats;
    const auto* const ids    = static_cast<const std::int32_t*>(expert_ids);
    const auto* const routed = static_cast<const float*>(expert_weights);
    const auto* const input  = static_cast<const __nv_bfloat16*>(x);
    auto* const output       = static_cast<__nv_bfloat16*>(out);

    const std::size_t up_shared = static_cast<std::size_t>(latent) * sizeof(__nv_bfloat16);
    if (up_shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(
            nemotron_h_moe_experts_row_tiled_up_kernel<kRowTiledUpThreads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(up_shared)));
    }
    const std::size_t down_shared = static_cast<std::size_t>(intermediate) * sizeof(float);
    if (down_shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(
            nemotron_h_moe_experts_row_tiled_down_kernel<kRowTiledDownThreads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(down_shared)));
    }
    const bool warp_route = tokens == 1;
    if (warp_route) {
        if (up_shared > kStaticSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(
                nemotron_h_moe_experts_warp_up_kernel<kRowTiledWarpRowsPerCta>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(up_shared)));
        }
        if (down_shared > kStaticSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(
                nemotron_h_moe_experts_warp_down_kernel<kRowTiledWarpRowsPerCta>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(down_shared)));
        }
    }

    for (std::int32_t base = 0; base < tokens; base += kRowTiledTileTokens) {
        const std::int32_t count      = std::min(kRowTiledTileTokens, tokens - base);
        const auto* const tile_ids    = ids + base;
        const auto* const tile_routed = routed + base;
        const auto* const tile_x      = input + static_cast<std::size_t>(base) * latent;
        auto* const tile_out          = output + static_cast<std::size_t>(base) * latent;

        if (warp_route) {
            constexpr int kThreads = kRowTiledWarpRowsPerCta * 32;
            const dim3 up_grid(
                static_cast<unsigned>(tile_count(intermediate, kRowTiledWarpRowsPerCta)),
                static_cast<unsigned>(count), static_cast<unsigned>(top_k));
            nemotron_h_moe_experts_warp_up_kernel<kRowTiledWarpRowsPerCta>
                <<<up_grid, kThreads, up_shared, stream>>>(up, latent, intermediate, tokens, top_k,
                                                           tile_ids, tile_x, product);
            CUDA_CHECK(cudaGetLastError());

            const dim3 down_grid(static_cast<unsigned>(tile_count(latent, kRowTiledWarpRowsPerCta)),
                                 static_cast<unsigned>(count), static_cast<unsigned>(top_k));
            nemotron_h_moe_experts_warp_down_kernel<kRowTiledWarpRowsPerCta>
                <<<down_grid, kThreads, down_shared, stream>>>(
                    down, latent, intermediate, tokens, top_k, tile_ids, product, partial);
            CUDA_CHECK(cudaGetLastError());
        } else {
            const dim3 up_grid(static_cast<unsigned>(tile_count(intermediate, kRowTiledUpThreads)),
                               static_cast<unsigned>(count), static_cast<unsigned>(top_k));
            nemotron_h_moe_experts_row_tiled_up_kernel<kRowTiledUpThreads>
                <<<up_grid, kRowTiledUpThreads, up_shared, stream>>>(
                    up, latent, intermediate, tokens, top_k, tile_ids, tile_x, product);
            CUDA_CHECK(cudaGetLastError());

            const dim3 down_grid(static_cast<unsigned>(tile_count(latent, kRowTiledDownThreads)),
                                 static_cast<unsigned>(count), static_cast<unsigned>(top_k));
            nemotron_h_moe_experts_row_tiled_down_kernel<kRowTiledDownThreads>
                <<<down_grid, kRowTiledDownThreads, down_shared, stream>>>(
                    down, latent, intermediate, tokens, top_k, tile_ids, product, partial);
            CUDA_CHECK(cudaGetLastError());
        }

        const dim3 sum_grid(static_cast<unsigned>(tile_count(latent, kRowTiledSumThreads)),
                            static_cast<unsigned>(count));
        nemotron_h_moe_experts_row_tiled_rank_sum_kernel<kRowTiledSumThreads>
            <<<sum_grid, kRowTiledSumThreads, 0, stream>>>(partial, tile_routed, latent, top_k,
                                                           tokens, tile_out);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

std::size_t nemotron_h_moe_experts_workspace(QType qtype, std::int32_t latent, std::int32_t experts,
                                             std::int32_t intermediate, std::int32_t top_k,
                                             std::int32_t tokens) {
    (void)experts;
    // Non-zero exactly when the row-tiled route exists for this format and these widths: the
    // per-token route asks for nothing -- it holds a token's activation and its product in shared
    // memory and reads a bank that is already addressed where it lies -- and the row-tiled route
    // needs the two buffers the carve below lays out.
    if (qtype != QType::NVFP4 || latent % kRowTiledChunkColumns != 0 ||
        intermediate % kRowTiledChunkColumns != 0) {
        return 0;
    }
    // The route needs one token tile's storage whatever the call's width -- a wider call simply
    // walks more tiles through the same buffers -- so the reported size is capped by the tile and
    // never falls as T grows. That is what lets a caller size one arena for every call it makes:
    // the engine plans its pass arena at the prefill chunk and then runs its decode calls through
    // that storage, and the tile's buffers fit any of them. It is also what the refusal in the
    // wrapper is checked against, so a tile's storage really is reserved rather than assumed.
    const std::int32_t covered = row_tiled_covered_tokens(tokens);
    return row_tiled_carve(latent, intermediate, top_k, covered).bytes;
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
    if (route_for(up.qtype, latent, intermediate, tokens) == Route::RowTiled) {
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
    if (route_for(QType::NVFP4, latent, intermediate, tokens) == Route::RowTiled) {
        launch_row_tiled(up, down, x, expert_ids, expert_weights, latent, intermediate, top_k,
                         tokens, workspace, out, stream);
        return;
    }
    launch_route(up, down, x, expert_ids, expert_weights, latent, intermediate, top_k, tokens, out,
                 stream);
}

} // namespace ninfer::ops::detail
