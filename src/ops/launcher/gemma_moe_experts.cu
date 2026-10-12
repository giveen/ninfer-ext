// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// An NVFP4 pair at a registered geometry streams over every SM in two stages (see the kernel
// header). Any other pair takes one CTA per token, with the token's activation row, one selected
// expert's gate and up vectors, the product and the running output row in shared memory. The bank format is a reader, so the
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

SparseMoePlanes stream_planes(const GemmaMoeBankLaunch& bank) {
    return {static_cast<const std::uint8_t*>(bank.payload), nullptr,
            static_cast<const std::uint8_t*>(bank.scales), bank.divisors, bank.divisor_rows};
}

// Two hidden rows a stage-two CTA: NVFP4 interleaves rows r and r+1 inside one 32-byte sector of
// its scale tile, which is why the 35B's decode takes the same pair.
constexpr int kStreamDownRows = 2;

template <int Hidden, int Intermediate, int TopK>
void launch_stream_route(const void* x, const void* expert_ids, const void* expert_weights,
                         const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                         std::int32_t tokens, float* products, void* out, cudaStream_t stream) {
    const auto* ids = static_cast<const std::int32_t*>(expert_ids);
    gemma_moe_stream_gate_up_kernel<Hidden, Intermediate, TopK>
        <<<dim3(Intermediate, tokens), TopK * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), ids, stream_planes(gate_up), products, tokens);
    CUDA_CHECK(cudaGetLastError());
    gemma_moe_stream_down_kernel<Hidden, Intermediate, TopK, kStreamDownRows>
        <<<dim3(Hidden / kStreamDownRows, tokens), TopK * 32, 0, stream>>>(
            ids, static_cast<const float*>(expert_weights), products, stream_planes(down),
            static_cast<__nv_bfloat16*>(out), tokens);
    CUDA_CHECK(cudaGetLastError());
}

// The geometries the streamed route is instantiated for: the 26B-A4B's hidden 2816, stored
// intermediate 768 and top-8.
bool is_gemma_26b_geometry(std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k) {
    return hidden == 2816 && intermediate == 768 && top_k == 8;
}

} // namespace

bool gemma_moe_experts_streams(QType qtype, std::int32_t hidden, std::int32_t intermediate,
                               std::int32_t top_k) {
    return qtype == QType::NVFP4 && is_gemma_26b_geometry(hidden, intermediate, top_k);
}

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                              std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k,
                              std::int32_t tokens, float* products, void* out,
                              cudaStream_t stream) {
    if (tokens <= 0) { return; }

    if (gate_up.qtype == down.qtype &&
        gemma_moe_experts_streams(gate_up.qtype, hidden, intermediate, top_k)) {
        launch_stream_route<2816, 768, 8>(x, expert_ids, expert_weights, gate_up, down, tokens,
                                          products, out, stream);
        return;
    }

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
