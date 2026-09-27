#include "quantize/exl3/hessian.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

// Two accumulations (a token count that is not a multiple of the staging depth, then a second
// batch) must equal the FP64 Gram matrix of the BF16-decoded inputs.
void test_accumulation() {
    constexpr std::int64_t kK    = 256;
    const std::int64_t batches[] = {301, 64};
    std::mt19937_64 rng(0x6e55);
    std::normal_distribution<float> normal;

    std::vector<double> oracle(kK * kK, 0.0);
    float* d_h = nullptr;
    cudaMalloc(&d_h, kK * kK * sizeof(float));
    cudaMemset(d_h, 0, kK * kK * sizeof(float));
    double scale = 0.0;
    for (const std::int64_t tokens : batches) {
        std::vector<__nv_bfloat16> x(static_cast<std::size_t>(tokens * kK));
        for (auto& value : x) { value = __float2bfloat16(normal(rng) * 3.0f); }
        for (std::int64_t t = 0; t < tokens; ++t) {
            for (std::int64_t i = 0; i < kK; ++i) {
                const double xi = __bfloat162float(x[static_cast<std::size_t>(t * kK + i)]);
                for (std::int64_t j = 0; j < kK; ++j) {
                    oracle[static_cast<std::size_t>(i * kK + j)] +=
                        xi * __bfloat162float(x[static_cast<std::size_t>(t * kK + j)]);
                }
            }
        }
        __nv_bfloat16* d_x = nullptr;
        cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        ninfer::quantize::exl3::accumulate_hessian(d_x, tokens, kK, d_h, nullptr);
        cudaDeviceSynchronize();
        cudaFree(d_x);
    }
    std::vector<float> h(kK * kK);
    cudaMemcpy(h.data(), d_h, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    for (std::int64_t i = 0; i < kK; ++i) {
        scale = std::max(scale, oracle[static_cast<std::size_t>(i * kK + i)]);
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < h.size(); ++i) {
        worst = std::max(worst, std::abs(h[i] - oracle[i]) / scale);
    }
    // FP32 accumulation of 365 products per entry: error relative to the largest diagonal entry.
    expect(worst < 1e-5, "accumulated Hessian differs from the FP64 Gram matrix");
    for (std::int64_t i = 0; i < kK; ++i) {
        for (std::int64_t j = 0; j < i; ++j) {
            if (h[static_cast<std::size_t>(i * kK + j)] !=
                h[static_cast<std::size_t>(j * kK + i)]) {
                expect(false, "accumulated Hessian is not exactly symmetric");
                return;
            }
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    test_accumulation();
    if (failures == 0) { std::cout << "OK exl3 hessian\n"; }
    return failures == 0 ? 0 : 1;
}
