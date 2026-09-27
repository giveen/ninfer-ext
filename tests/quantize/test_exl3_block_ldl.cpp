#include "quantize/exl3/block_ldl.h"

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

using Matrix = std::vector<double>;

// FP64 oracle: unblocked Cholesky, then each 16-wide block column multiplied by the inverse of
// its diagonal block (so H = L D Lᵀ with identity diagonal blocks in L).
Matrix reference_block_ldl(const Matrix& h, std::int64_t n) {
    Matrix l(static_cast<std::size_t>(n * n), 0.0);
    const auto at = [n](Matrix& m, std::int64_t r, std::int64_t c) -> double& {
        return m[static_cast<std::size_t>(r * n + c)];
    };
    for (std::int64_t j = 0; j < n; ++j) {
        double s = h[static_cast<std::size_t>(j * n + j)];
        for (std::int64_t m = 0; m < j; ++m) { s -= at(l, j, m) * at(l, j, m); }
        at(l, j, j) = std::sqrt(s);
        for (std::int64_t i = j + 1; i < n; ++i) {
            double t = h[static_cast<std::size_t>(i * n + j)];
            for (std::int64_t m = 0; m < j; ++m) { t -= at(l, i, m) * at(l, j, m); }
            at(l, i, j) = t / at(l, j, j);
        }
    }
    constexpr int kB = ninfer::quantize::exl3::kLdlBlock;
    Matrix out       = l;
    for (std::int64_t c0 = 0; c0 < n; c0 += kB) {
        // inverse of the lower-triangular diagonal block
        double inv[kB][kB] = {};
        for (int c = 0; c < kB; ++c) {
            for (int r = c; r < kB; ++r) {
                double s = r == c ? 1.0 : 0.0;
                for (int m = c; m < r; ++m) { s -= at(l, c0 + r, c0 + m) * inv[m][c]; }
                inv[r][c] = s / at(l, c0 + r, c0 + r);
            }
        }
        for (std::int64_t r = c0; r < n; ++r) {
            for (int j = 0; j < kB; ++j) {
                double s = 0.0;
                for (int m = 0; m < kB; ++m) { s += at(l, r, c0 + m) * inv[m][j]; }
                at(out, r, c0 + j) = s;
            }
        }
    }
    return out;
}

// A calibration-like Hessian: correlated activations with a few outlier channels, damped by
// 2.5% of the mean diagonal (the quantizer's default).
Matrix make_hessian(std::int64_t n, std::mt19937_64& rng) {
    std::normal_distribution<double> normal;
    const std::int64_t tokens = 2 * n;
    std::vector<double> x(static_cast<std::size_t>(tokens * n));
    for (std::int64_t t = 0; t < tokens; ++t) {
        const double shared = normal(rng);
        for (std::int64_t i = 0; i < n; ++i) {
            const double gain                      = i % 37 == 0 ? 8.0 : 1.0;
            x[static_cast<std::size_t>(t * n + i)] = gain * (normal(rng) + 0.5 * shared);
        }
    }
    Matrix h(static_cast<std::size_t>(n * n), 0.0);
    for (std::int64_t t = 0; t < tokens; ++t) {
        for (std::int64_t i = 0; i < n; ++i) {
            for (std::int64_t j = 0; j < n; ++j) {
                h[static_cast<std::size_t>(i * n + j)] +=
                    x[static_cast<std::size_t>(t * n + i)] * x[static_cast<std::size_t>(t * n + j)];
            }
        }
    }
    double mean_diag = 0.0;
    for (std::int64_t i = 0; i < n; ++i) { mean_diag += h[static_cast<std::size_t>(i * n + i)]; }
    mean_diag /= static_cast<double>(n);
    for (std::int64_t i = 0; i < n; ++i) {
        h[static_cast<std::size_t>(i * n + i)] += 0.025 * mean_diag;
    }
    return h;
}

bool run_gpu(const Matrix& h, std::int64_t n, std::vector<float>& l) {
    std::vector<float> host(h.begin(), h.end());
    float* device = nullptr;
    cudaMalloc(&device, host.size() * sizeof(float));
    cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    const bool ok = ninfer::quantize::exl3::block_ldl(device, n, nullptr);
    l.resize(host.size());
    cudaMemcpy(l.data(), device, l.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);
    return ok;
}

void test_matches_fp64_reference() {
    std::mt19937_64 rng(0x1d1);
    for (const std::int64_t n : {std::int64_t{128}, std::int64_t{320}}) {
        const Matrix h      = make_hessian(n, rng);
        const Matrix oracle = reference_block_ldl(h, n);
        std::vector<float> l;
        expect(run_gpu(h, n, l), "block LDL rejected a positive-definite Hessian");
        double worst = 0.0;
        for (std::int64_t r = 0; r < n; ++r) {
            for (std::int64_t c = 0; c < n; ++c) {
                const auto i = static_cast<std::size_t>(r * n + c);
                if (c > r && l[i] != 0.0f) {
                    expect(false, "block LDL left a nonzero above the diagonal");
                    return;
                }
                if (r / 16 == c / 16 && l[i] != (r == c ? 1.0f : 0.0f)) {
                    expect(false, "block LDL diagonal blocks are not exact identities");
                    return;
                }
                worst = std::max(worst, std::abs(l[i] - oracle[i]));
            }
        }
        std::cout << "n=" << n << " max |L - L_fp64| " << worst << '\n';
        expect(worst < 1e-3, "block LDL differs from the FP64 reference");
    }
}

void test_rejects_indefinite() {
    constexpr std::int64_t kN = 128;
    Matrix h(kN * kN, 0.0);
    for (std::int64_t i = 0; i < kN; ++i) { h[static_cast<std::size_t>(i * kN + i)] = 1.0; }
    h[static_cast<std::size_t>(100 * kN + 100)] = -1.0;
    std::vector<float> l;
    expect(!run_gpu(h, kN, l), "block LDL accepted an indefinite matrix");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    test_matches_fp64_reference();
    test_rejects_indefinite();
    if (failures == 0) { std::cout << "OK exl3 block LDL\n"; }
    return failures == 0 ? 0 : 1;
}
