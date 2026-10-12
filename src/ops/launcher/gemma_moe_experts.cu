// Implements: include/ninfer/ops/gemma_moe_experts.h
//
// Three routes. At the registered geometry an NVFP4 pair, and an int4 group-32 row pair below
// kGroupedMinTokens, stream over every SM in two stages; an int4 group-32 row pair at or above it
// takes the grouped tensor-core route (see the two kernel headers). Any other pair takes one CTA
// per token, with the token's activation row, one selected expert's gate and up vectors, the
// product and the running output row in shared memory; the bank format is a reader, so that kernel
// body is instantiated once per format, and a wider geometry only needs the dynamic shared opt-in.
#include "ops/launcher/gemma_moe_experts.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/gemma_moe_experts.cuh"
#include "ops/kernel/gemma_moe_experts_grouped.cuh"

#include <algorithm>
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

template <class Codec, int Hidden, int Intermediate, int TopK>
void launch_stream_route(const void* x, const void* expert_ids, const void* expert_weights,
                         const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                         std::int32_t tokens, float* products, void* out, cudaStream_t stream) {
    const auto* ids = static_cast<const std::int32_t*>(expert_ids);
    gemma_moe_stream_gate_up_kernel<Codec, Hidden, Intermediate, TopK>
        <<<dim3(Intermediate, tokens), TopK * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), ids, stream_planes(gate_up), products, tokens);
    CUDA_CHECK(cudaGetLastError());
    gemma_moe_stream_down_kernel<Codec, Hidden, Intermediate, TopK, kStreamDownRows>
        <<<dim3(Hidden / kStreamDownRows, tokens), TopK * 32, 0, stream>>>(
            ids, static_cast<const float*>(expert_weights), products, stream_planes(down),
            static_cast<__nv_bfloat16*>(out), tokens);
    CUDA_CHECK(cudaGetLastError());
}


// The geometry the streamed and grouped routes are instantiated for: the 26B-A4B's hidden 2816,
// stored intermediate 768 and top-8.
constexpr int kHidden       = 2816;
constexpr int kIntermediate = 768;
constexpr int kTopK         = 8;

// From this many tokens an int4 group-32 row pair takes the grouped route; below it each token
// streams its own experts' rows.
constexpr std::int32_t kGroupedMinTokens = 16;
// The grouped route works through a pass in slices of at most this many tokens, which bounds its
// workspace by the slice rather than by the pass.
constexpr std::int32_t kGroupedSliceTokens = 1024;

enum class Route : std::uint8_t { PerToken, Streamed, Grouped };

Route route_for(QType qtype, std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k,
                std::int32_t tokens) {
    if (hidden != kHidden || intermediate != kIntermediate || top_k != kTopK) {
        return Route::PerToken;
    }
    if (qtype == QType::NVFP4) { return Route::Streamed; }
    if (qtype == QType::Q4_G32_FP16_ROWS) {
        return tokens >= kGroupedMinTokens ? Route::Grouped : Route::Streamed;
    }
    return Route::PerToken;
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The grouped route's workspace over one slice, in carve order.
struct GroupedCarve {
    int* counts                = nullptr; // [E]
    int* offsets               = nullptr; // [E + 1]
    int* job_experts           = nullptr; // [max jobs]
    int* job_columns           = nullptr; // [max jobs]
    int* job_count             = nullptr; // [1]
    int* packed_token          = nullptr; // [K * T]
    int* inverse               = nullptr; // [K * T]
    __nv_bfloat16* products    = nullptr; // [K * T, I]
    __nv_bfloat16* expert_out  = nullptr; // [K * T, H]
    std::size_t bytes          = 0;
    int max_jobs               = 0;
};

GroupedCarve carve_grouped(void* base, std::int32_t experts, std::int32_t tokens) {
    const std::int32_t assignments = kTopK * tokens;
    GroupedCarve carve;
    // One job per full 64 assignments of the slice, plus at most one partial job per expert.
    carve.max_jobs = assignments / kGroupedBN + std::min(experts, assignments);
    auto* cursor   = static_cast<std::uint8_t*>(base);
    auto take      = [&](std::size_t bytes) {
        std::uint8_t* const at = cursor;
        if (cursor != nullptr) { cursor += align256(bytes); }
        carve.bytes += align256(bytes);
        return at;
    };
    carve.counts       = reinterpret_cast<int*>(take(sizeof(int) * experts));
    carve.offsets      = reinterpret_cast<int*>(take(sizeof(int) * (experts + 1)));
    carve.job_experts  = reinterpret_cast<int*>(take(sizeof(int) * carve.max_jobs));
    carve.job_columns  = reinterpret_cast<int*>(take(sizeof(int) * carve.max_jobs));
    carve.job_count    = reinterpret_cast<int*>(take(sizeof(int)));
    carve.packed_token = reinterpret_cast<int*>(take(sizeof(int) * assignments));
    carve.inverse      = reinterpret_cast<int*>(take(sizeof(int) * assignments));
    carve.products     = reinterpret_cast<__nv_bfloat16*>(
        take(sizeof(__nv_bfloat16) * static_cast<std::size_t>(assignments) * kIntermediate));
    carve.expert_out = reinterpret_cast<__nv_bfloat16*>(
        take(sizeof(__nv_bfloat16) * static_cast<std::size_t>(assignments) * kHidden));
    return carve;
}

void launch_grouped_route(const void* x, const void* expert_ids, const void* expert_weights,
                          const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                          std::int32_t experts, std::int32_t tokens, void* workspace, void* out,
                          cudaStream_t stream) {
    const auto* ids     = static_cast<const std::int32_t*>(expert_ids);
    const auto* weights = static_cast<const float*>(expert_weights);
    const auto* input   = static_cast<const __nv_bfloat16*>(x);
    auto* output        = static_cast<__nv_bfloat16*>(out);
    const auto* gate_up_codes  = static_cast<const std::uint8_t*>(gate_up.payload);
    const auto* gate_up_scales = static_cast<const std::uint8_t*>(gate_up.scales);
    const auto* down_codes     = static_cast<const std::uint8_t*>(down.payload);
    const auto* down_scales    = static_cast<const std::uint8_t*>(down.scales);
    for (std::int32_t first = 0; first < tokens; first += kGroupedSliceTokens) {
        const std::int32_t count = std::min(kGroupedSliceTokens, tokens - first);
        const GroupedSlice slice{tokens, first, count, kTopK};
        const GroupedCarve carve = carve_grouped(workspace, experts, count);

        gemma_moe_grouped_count_kernel<<<experts, kRouteThreads, 0, stream>>>(ids, slice,
                                                                              carve.counts);
        CUDA_CHECK(cudaGetLastError());
        gemma_moe_grouped_plan_kernel<<<1, 32, 0, stream>>>(carve.counts, experts, carve.offsets,
                                                            carve.job_experts, carve.job_columns,
                                                            carve.job_count);
        CUDA_CHECK(cudaGetLastError());
        gemma_moe_grouped_place_kernel<<<experts, kRouteThreads, 0, stream>>>(
            ids, slice, carve.offsets, carve.packed_token, carve.inverse);
        CUDA_CHECK(cudaGetLastError());

        const GroupedJobs jobs{carve.job_experts, carve.job_columns, carve.job_count,
                               carve.offsets};
        const auto* slice_input = input + static_cast<std::int64_t>(first) * kHidden;
        gemma_moe_grouped_mma_kernel<GroupedProjection::GateUp, kHidden, kIntermediate>
            <<<dim3(kIntermediate / 32, carve.max_jobs), kGroupedThreads, 0, stream>>>(
                slice_input, carve.packed_token, gate_up_codes, gate_up_scales, jobs,
                carve.products);
        CUDA_CHECK(cudaGetLastError());
        gemma_moe_grouped_mma_kernel<GroupedProjection::Down, kHidden, kIntermediate>
            <<<dim3(kHidden / kGroupedBM, carve.max_jobs), kGroupedThreads, 0, stream>>>(
                carve.products, nullptr, down_codes, down_scales, jobs, carve.expert_out);
        CUDA_CHECK(cudaGetLastError());
        gemma_moe_grouped_combine_kernel<kHidden, kTopK><<<count, kHidden / 8, 0, stream>>>(
            carve.expert_out, carve.inverse, weights, slice,
            output + static_cast<std::int64_t>(first) * kHidden);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

std::size_t gemma_moe_experts_workspace(QType qtype, std::int32_t hidden, std::int32_t experts,
                                        std::int32_t intermediate, std::int32_t top_k,
                                        std::int32_t tokens) {
    switch (route_for(qtype, hidden, intermediate, top_k, tokens)) {
    case Route::Streamed:
        return static_cast<std::size_t>(top_k) * intermediate * tokens * sizeof(float);
    case Route::Grouped:
        return carve_grouped(nullptr, experts, std::min(tokens, kGroupedSliceTokens)).bytes;
    case Route::PerToken:
        break;
    }
    return 0;
}

void gemma_moe_experts_launch(const void* x, const void* expert_ids, const void* expert_weights,
                              const GemmaMoeBankLaunch& gate_up, const GemmaMoeBankLaunch& down,
                              std::int32_t hidden, std::int32_t experts, std::int32_t intermediate,
                              std::int32_t top_k, std::int32_t tokens, void* workspace, void* out,
                              cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (gate_up.qtype != down.qtype) {
        throw std::invalid_argument("gemma_moe_experts: the two banks must share one format");
    }

    switch (route_for(gate_up.qtype, hidden, intermediate, top_k, tokens)) {
    case Route::Grouped:
        launch_grouped_route(x, expert_ids, expert_weights, gate_up, down, experts, tokens,
                             workspace, out, stream);
        return;
    case Route::Streamed:
        if (gate_up.qtype == QType::NVFP4) {
            launch_stream_route<Nvfp4Codec, kHidden, kIntermediate, kTopK>(
                x, expert_ids, expert_weights, gate_up, down, tokens,
                static_cast<float*>(workspace), out, stream);
        } else {
            launch_stream_route<Q4RowsCodec, kHidden, kIntermediate, kTopK>(
                x, expert_ids, expert_weights, gate_up, down, tokens,
                static_cast<float*>(workspace), out, stream);
        }
        return;
    case Route::PerToken:
        break;
    }

    if (gate_up.qtype == QType::Q4_G32_FP16_ROWS) {
        const Q4RowsBankReader gate_up_reader{static_cast<const std::uint8_t*>(gate_up.payload),
                                              static_cast<const std::uint8_t*>(gate_up.scales),
                                              gate_up.columns};
        const Q4RowsBankReader down_reader{static_cast<const std::uint8_t*>(down.payload),
                                           static_cast<const std::uint8_t*>(down.scales),
                                           down.columns};
        launch_route(gate_up_reader, down_reader, x, expert_ids, expert_weights, hidden,
                     intermediate, top_k, tokens, out, stream);
        return;
    }
    if (gate_up.qtype == QType::BF16) {
        const Bf16BankReader gate_up_reader{static_cast<const __nv_bfloat16*>(gate_up.payload),
                                            gate_up.columns};
        const Bf16BankReader down_reader{static_cast<const __nv_bfloat16*>(down.payload),
                                         down.columns};
        launch_route(gate_up_reader, down_reader, x, expert_ids, expert_weights, hidden,
                     intermediate, top_k, tokens, out, stream);
        return;
    }
    if (gate_up.qtype == QType::NVFP4) {
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
    throw std::invalid_argument("gemma_moe_experts: unsupported bank format");
}

} // namespace ninfer::ops::detail
