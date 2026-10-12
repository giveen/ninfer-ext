// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// One CTA per token, with the token's activation row, one selected expert's gate and up vectors,
// the product and the running output row in shared memory. The bank format is a reader, so the
// kernel body is instantiated once per format rather than duplicated; a wider geometry only needs
// the dynamic shared opt-in below.
#include "ops/launcher/gemma_moe_experts.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/gemma_moe_experts.cuh"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads                   = 128;
constexpr std::size_t kStaticSharedBytes = 48 * 1024;

// The kernel's shared layout: the activation row, then the gate, up and product vectors, then the
// accumulator row.
std::size_t expert_shared_bytes(std::int32_t hidden, std::int32_t intermediate) {
    const std::size_t floats = static_cast<std::size_t>(hidden) + 3U * intermediate + hidden;
    return floats * sizeof(float);
}

template <class GateUpReader, class DownReader>
void launch_route(const GateUpReader& gate_up, const DownReader& down, const void* x,
                  const void* expert_ids, const void* expert_weights, std::int32_t hidden,
                  std::int32_t intermediate, std::int32_t top_k, std::int32_t tokens, void* out,
                  cudaStream_t stream) {
    const std::size_t shared = expert_shared_bytes(hidden, intermediate);
    if (shared > kStaticSharedBytes) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma_moe_experts_kernel<GateUpReader, DownReader>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    }
    gemma_moe_experts_kernel<<<tokens, kThreads, shared, stream>>>(
        gate_up, down, hidden, intermediate, top_k, tokens,
        static_cast<const std::int32_t*>(expert_ids), static_cast<const float*>(expert_weights),
        static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                              std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k,
                              std::int32_t tokens, void* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }

    if (gate_up.qtype == QType::BF16 && down.qtype == QType::BF16) {
        const Bf16BankReader gate_up_reader{static_cast<const __nv_bfloat16*>(gate_up.payload),
                                            gate_up.columns};
        const Bf16BankReader down_reader{static_cast<const __nv_bfloat16*>(down.payload),
                                         down.columns};
        launch_route(gate_up_reader, down_reader, x, expert_ids, expert_weights, hidden,
                     intermediate, top_k, tokens, out, stream);
        return;
    }
    if (gate_up.qtype == QType::NVFP4 && down.qtype == QType::NVFP4) {
        const Nvfp4BankReader gate_up_reader{static_cast<const std::uint8_t*>(gate_up.payload),
                                             static_cast<const std::uint8_t*>(gate_up.scales),
                                             gate_up.divisors, gate_up.divisor_rows,
                                             gate_up.columns};
        const Nvfp4BankReader down_reader{static_cast<const std::uint8_t*>(down.payload),
                                          static_cast<const std::uint8_t*>(down.scales),
                                          down.divisors, down.divisor_rows, down.columns};
        launch_route(gate_up_reader, down_reader, x, expert_ids, expert_weights, hidden,
                     intermediate, top_k, tokens, out, stream);
        return;
    }
    throw std::invalid_argument("gemma_moe_experts: the two banks must share one format");
}

} // namespace ninfer::ops::detail
