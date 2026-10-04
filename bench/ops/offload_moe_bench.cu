// Routed-expert decode benchmark for the public moe_experts contract (Qwen4Exp geometry).
//
// A bank of 512 random experts in either layout (NVFP4 or EXL3 at a chosen rate) is read through the
// staged route (slot = expert id) with fresh random top-10 routing per call and the L2 flushed
// between calls, so every call streams its experts from DRAM as a decode round does. Timing is the
// median GPU time of the whole moe_experts call (job build, gate/up, down, merge).
//
//   ninfer_offload_moe_bench [--format nvfp4|exl3] [--half-bits N] [--tokens 1,4,8] [--iters N]
#include "ninfer/ops/offload_moe.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t H = ops::kOffloadMoeHidden;
constexpr std::int32_t E = ops::kOffloadMoeExperts;
constexpr std::int32_t K = ops::kOffloadMoeTopK;

template <class T>
T* device_copy(const std::vector<T>& host) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return static_cast<T*>(p);
}

std::vector<std::int32_t> parse_list(const std::string& text) {
    std::vector<std::int32_t> out;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t comma = text.find(',', at);
        out.push_back(std::stoi(text.substr(at, comma == std::string::npos ? comma : comma - at)));
        if (comma == std::string::npos) { break; }
        at = comma + 1;
    }
    return out;
}

ops::ExpertWeights make_bank(bool exl3, int half_bits, std::mt19937_64& rng,
                             std::size_t* bank_bytes) {
    ops::ExpertWeights bank;
    bank.layout = exl3 ? ops::exl3_expert_layout(half_bits, half_bits) : ops::nvfp4_expert_layout();
    *bank_bytes = 0;
    for (int p = 0; p < bank.layout.planes; ++p) {
        const std::size_t stride = static_cast<std::size_t>(bank.layout.plane_bytes[p]);
        std::vector<std::uint8_t> bytes(stride * E);
        const bool scale_plane = exl3 ? (p == 1 || p == 2 || p == 4 || p == 5) : (p == 1 || p == 3);
        if (scale_plane && exl3) {
            auto* f = reinterpret_cast<float*>(bytes.data());
            for (std::size_t i = 0; i < bytes.size() / 4; ++i) { f[i] = (rng() & 1) ? 1.0F : -1.0F; }
        } else if (scale_plane) {
            std::fill(bytes.begin(), bytes.end(), 0x38); // E4M3 1.0
        } else {
            for (std::size_t i = 0; i + 8 <= bytes.size(); i += 8) {
                const std::uint64_t w = rng();
                std::memcpy(&bytes[i], &w, 8);
            }
        }
        void* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, bytes.size()));
        CUDA_CHECK(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        bank.base[p]   = static_cast<const std::byte*>(d);
        bank.stride[p] = static_cast<std::int64_t>(stride);
        *bank_bytes += bytes.size();
    }
    if (!exl3) {
        bank.gate_up_divisors     = device_copy(std::vector<float>(2 * E, 1.0F));
        bank.gate_up_divisor_rows = ops::kOffloadMoeIntermediate;
        bank.down_divisors        = device_copy(std::vector<float>(E, 1.0F));
        bank.down_divisor_rows    = H;
    }
    return bank;
}

} // namespace

int main(int argc, char** argv) {
    bool exl3 = false;
    int half_bits = 8, iters = 100;
    std::vector<std::int32_t> tokens = {1, 4, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const auto next          = [&]() -> std::string {
            if (++i >= argc) { std::fprintf(stderr, "%s needs a value\n", argv[i - 1]); std::exit(2); }
            return argv[i];
        };
        if (a == "--format") {
            const std::string v = next();
            if (v != "nvfp4" && v != "exl3") { std::fprintf(stderr, "format is nvfp4 or exl3\n"); return 2; }
            exl3 = v == "exl3";
        } else if (a == "--half-bits") {
            half_bits = std::stoi(next());
        } else if (a == "--tokens") {
            tokens = parse_list(next());
        } else if (a == "--iters") {
            iters = std::stoi(next());
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    std::mt19937_64 rng(0xBE7C4U);
    std::size_t bank_bytes = 0;
    const ops::ExpertWeights bank = make_bank(exl3, half_bits, rng, &bank_bytes);
    const double expert_bytes     = static_cast<double>(bank.layout.slot_bytes);

    void* flush = nullptr;
    constexpr std::size_t kFlushBytes = 256ULL << 20;
    CUDA_CHECK(cudaMalloc(&flush, kFlushBytes));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    std::printf("format %s%s, expert slot %.0f B\n", exl3 ? "exl3 half-bits " : "nvfp4",
                exl3 ? std::to_string(half_bits).c_str() : "", expert_bytes);
    std::printf("%6s %12s %14s %12s\n", "T", "median us", "distinct exp", "GB/s (touched)");
    for (const std::int32_t t : tokens) {
        WorkspaceArena workspace(ops::moe_experts_workspace_bytes(t, E, bank.layout));
        std::vector<float> x(static_cast<std::size_t>(H) * t);
        std::vector<std::uint16_t> xb(x.size()), shared(x.size());
        for (auto& v : xb) {
            const float f = static_cast<float>(static_cast<int>(rng() % 2000) - 1000) / 500.0F;
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            v = static_cast<std::uint16_t>(u >> 16);
        }
        std::fill(shared.begin(), shared.end(), 0);
        auto* dx = device_copy(xb);
        auto* dshared = device_copy(shared);
        auto* dsg = device_copy(std::vector<float>(t, 0.5F));
        auto* dw  = device_copy(std::vector<float>(static_cast<std::size_t>(K) * t, 0.1F));
        void* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dy, static_cast<std::size_t>(H) * t * 2));
        std::vector<float> times;
        double distinct_sum = 0;
        for (int it = 0; it < iters + 5; ++it) {
            std::vector<std::int32_t> ids(static_cast<std::size_t>(K) * t);
            std::vector<std::int32_t> pool(E);
            for (std::int32_t col = 0; col < t; ++col) {
                std::iota(pool.begin(), pool.end(), 0);
                std::shuffle(pool.begin(), pool.end(), rng);
                for (int k = 0; k < K; ++k) { ids[static_cast<std::size_t>(col) * K + k] = pool[k]; }
            }
            std::vector<std::int32_t> uniq(ids);
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            auto* did = device_copy(ids);
            Tensor tx(dx, DType::BF16, {H, t});
            Tensor tshared(dshared, DType::BF16, {H, t});
            Tensor tids(did, DType::I32, {K, t});
            Tensor tw(dw, DType::FP32, {K, t});
            Tensor tsg(dsg, DType::FP32, {t});
            Tensor ty(dy, DType::BF16, {H, t});
            CUDA_CHECK(cudaMemsetAsync(flush, it & 0xFF, kFlushBytes, stream));
            CUDA_CHECK(cudaEventRecord(begin, stream));
            ops::moe_experts(tx, tids, tids, tw, tsg, tshared, bank, E, workspace, ty, stream);
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            if (it >= 5) {
                times.push_back(ms * 1000.0F);
                distinct_sum += static_cast<double>(uniq.size());
            }
            CUDA_CHECK(cudaFree(did));
        }
        std::sort(times.begin(), times.end());
        const double median   = times[times.size() / 2];
        const double distinct = distinct_sum / static_cast<double>(times.size());
        std::printf("%6d %12.1f %14.1f %12.0f\n", t, median, distinct,
                    distinct * expert_bytes / (median * 1e-6) / 1e9);
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dshared));
        CUDA_CHECK(cudaFree(dsg));
        CUDA_CHECK(cudaFree(dw));
        CUDA_CHECK(cudaFree(dy));
    }
    return 0;
}
