#include "artifact/exl3_trellis.h"
#include "quantize/exl3/ldlq.h"
#include "quantize/exl3/pipeline.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

namespace q = ninfer::quantize::exl3;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

constexpr std::int64_t kK = 256;
constexpr std::int64_t kN = 384;

double h128(std::int64_t i, std::int64_t j) {
    return (std::popcount(static_cast<unsigned>((i & j) & 127)) & 1 ? -1.0 : 1.0) /
           std::sqrt(128.0);
}

// Independent decode of the stored representation, in FP64:
// W = diag(su) · H128_k · Z · H128_n · diag(sv), Z[k][n] = mul1(state).
std::vector<double> decode(const std::vector<std::uint16_t>& states, const std::vector<float>& su,
                           const std::vector<float>& sv) {
    const std::int64_t tiles_k = kK / 16;
    std::vector<double> z(kK * kN);
    for (std::int64_t nt = 0; nt < kN / 16; ++nt) {
        for (std::int64_t kt = 0; kt < tiles_k; ++kt) {
            for (int t = 0; t < 256; ++t) {
                const std::uint16_t s =
                    states[static_cast<std::size_t>((nt * tiles_k + kt) * 256 + t)];
                z[static_cast<std::size_t>((kt * 16 + q::tile_k(t)) * kN + nt * 16 +
                                           q::tile_n(t))] = ninfer::artifact::exl3_mul1_value(s);
            }
        }
    }
    std::vector<double> left(kK * kN, 0.0), out(kK * kN, 0.0);
    for (std::int64_t r = 0; r < kK; ++r) {
        const std::int64_t block = r / 128 * 128;
        for (std::int64_t i = 0; i < 128; ++i) {
            const double hv = h128(r - block, i);
            for (std::int64_t c = 0; c < kN; ++c) {
                left[static_cast<std::size_t>(r * kN + c)] +=
                    hv * z[static_cast<std::size_t>((block + i) * kN + c)];
            }
        }
    }
    for (std::int64_t r = 0; r < kK; ++r) {
        for (std::int64_t c = 0; c < kN; ++c) {
            const std::int64_t block = c / 128 * 128;
            double sum               = 0.0;
            for (std::int64_t i = 0; i < 128; ++i) {
                sum += left[static_cast<std::size_t>(r * kN + block + i)] * h128(block + i, c);
            }
            out[static_cast<std::size_t>(r * kN + c)] =
                su[static_cast<std::size_t>(r)] * sum * sv[static_cast<std::size_t>(c)];
        }
    }
    return out;
}

void test_pipeline(int half_bits, double max_proxy) {
    std::mt19937_64 rng(0xe1e3 + static_cast<unsigned>(half_bits));
    std::normal_distribution<float> normal;
    // Calibration-like H (mean XᵀX, correlated, outlier channels) and a weight with a few large
    // output columns.
    std::vector<float> h(kK * kK, 0.0f);
    constexpr int kTokens = 1024;
    std::vector<double> hacc(kK * kK, 0.0);
    for (int t = 0; t < kTokens; ++t) {
        const double shared = normal(rng);
        std::vector<double> x(kK);
        for (std::int64_t i = 0; i < kK; ++i) {
            x[static_cast<std::size_t>(i)] = (i % 29 == 0 ? 10.0 : 1.0) * (normal(rng) + shared);
        }
        for (std::int64_t i = 0; i < kK; ++i) {
            for (std::int64_t j = 0; j < kK; ++j) {
                hacc[static_cast<std::size_t>(i * kK + j)] +=
                    x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)] / kTokens;
            }
        }
    }
    for (std::size_t i = 0; i < h.size(); ++i) { h[i] = static_cast<float>(hacc[i]); }
    std::vector<float> w(kK * kN);
    for (std::int64_t r = 0; r < kK; ++r) {
        for (std::int64_t c = 0; c < kN; ++c) {
            w[static_cast<std::size_t>(r * kN + c)] =
                0.02f * normal(rng) * (c % 41 == 0 ? 6.0f : 1.0f);
        }
    }

    float *d_w = nullptr, *d_h = nullptr, *d_su = nullptr, *d_sv = nullptr, *d_wq = nullptr;
    std::uint16_t* d_states = nullptr;
    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_h, h.size() * sizeof(float));
    cudaMalloc(&d_su, kK * sizeof(float));
    cudaMalloc(&d_sv, kN * sizeof(float));
    cudaMalloc(&d_wq, w.size() * sizeof(float));
    cudaMalloc(&d_states, w.size() * sizeof(std::uint16_t));
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    q::TensorOptions options;
    options.bitrate_half_bits = half_bits;
    options.seed              = 7;
    const q::TensorReport report =
        q::quantize_tensor(d_w, d_h, kK, kN, options, d_states, d_su, d_sv, d_wq, nullptr);
    std::vector<std::uint16_t> states(w.size());
    std::vector<float> su(kK), sv(kN), wq(w.size()), h_after(h.size());
    cudaMemcpy(states.data(), d_states, states.size() * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(su.data(), d_su, su.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sv.data(), d_sv, sv.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wq.data(), d_wq, wq.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_after.data(), d_h, h_after.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (void* p :
         {static_cast<void*>(d_w), static_cast<void*>(d_h), static_cast<void*>(d_su),
          static_cast<void*>(d_sv), static_cast<void*>(d_wq), static_cast<void*>(d_states)}) {
        cudaFree(p);
    }

    std::cout << "K=" << half_bits / 2.0 << " proxy rotated " << report.proxy_error_rotated
              << ", original " << report.proxy_error_before_refit << ", after refit "
              << report.proxy_error << ", global scale " << report.global_scale << ", out scales "
              << report.out_scales << '\n';
    expect(h_after == h, "quantize_tensor modified its input Hessian");
    const std::vector<double> decoded = decode(states, su, sv);
    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < wq.size(); ++i) {
        worst = std::max(worst, std::abs(decoded[i] - wq[i]));
        scale = std::max(scale, std::abs(decoded[i]));
    }
    expect(worst <= 1e-4 * scale, "stored states and scales do not decode to the reconstruction");
    expect(report.proxy_error <= report.proxy_error_before_refit * 1.001,
           "scale refit increased the proxy error");
    expect(report.proxy_error < max_proxy, "proxy error is implausibly high");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    test_pipeline(4, 0.08);
    test_pipeline(8, 0.006);
    if (failures == 0) { std::cout << "OK exl3 pipeline\n"; }
    return failures == 0 ? 0 : 1;
}
