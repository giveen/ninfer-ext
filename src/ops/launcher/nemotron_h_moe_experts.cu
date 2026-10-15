// Implements: include/ninfer/ops/nemotron_h_moe_experts.h
//
// One route, one CTA per token, instantiated once per bank format: the expert bank is a reader, so
// the kernel body is the same for every format a reader exists for. This is the route the Op is
// admitted with -- it is exact, it needs no workspace and it covers every geometry -- and the
// banks this family actually serves (240 to 496 experts at 1536 to 2560 intermediate) are large
// enough that a streamed route, where one CTA per (row, token) walks the selected experts over
// every SM, is the next step. That route adds a Route and a case below rather than reshaping the
// Op: the workspace query already exists for it, and `Route::PerToken` stays the fallback.
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

enum class Route : std::uint8_t { PerToken };

// The per-token route matches every format and every extent today, so its match facts are the
// extent being positive. A streamed route is matched on the format and on the bank being large
// enough to pay for the staging, which is the same shape of decision the Gemma 4 op makes.
Route route_for(QType qtype, std::int32_t tokens) {
    (void)qtype;
    (void)tokens;
    return Route::PerToken;
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

} // namespace

std::size_t nemotron_h_moe_experts_workspace(QType qtype, std::int32_t latent, std::int32_t experts,
                                             std::int32_t intermediate, std::int32_t top_k,
                                             std::int32_t tokens) {
    switch (route_for(qtype, tokens)) {
    case Route::PerToken:
        break;
    }
    return 0;
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

    switch (route_for(up.qtype, tokens)) {
    case Route::PerToken:
        break;
    }

    if (up.qtype == QType::BF16) {
        const Bf16BankReader up_reader{static_cast<const __nv_bfloat16*>(up.payload), up.columns};
        const Bf16BankReader down_reader{static_cast<const __nv_bfloat16*>(down.payload),
                                         down.columns};
        launch_route(up_reader, down_reader, x, expert_ids, expert_weights, latent, intermediate,
                     top_k, tokens, out, stream);
        return;
    }
    // Only the per-token route's launch happens above, and it reads neither the expert count nor a
    // workspace; a registered-format route consumes both.
    (void)workspace;
    (void)experts;
    throw std::invalid_argument("nemotron_h_moe_experts: unsupported bank format");
}

} // namespace ninfer::ops::detail
