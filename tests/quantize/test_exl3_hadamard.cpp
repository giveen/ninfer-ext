#include "quantize/exl3/hadamard.h"

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

// Oracle: the explicit Sylvester matrix, H[i][j] = (-1)^popcount(i & j) / sqrt(128), in FP64.
double h128(int i, int j) {
    return (std::popcount(static_cast<unsigned>(i & j)) & 1 ? -1.0 : 1.0) / std::sqrt(128.0);
}

std::vector<float> run(const std::vector<float>& input, std::int64_t rows, std::int64_t cols,
                       bool by_rows) {
    float* device = nullptr;
    cudaMalloc(&device, input.size() * sizeof(float));
    cudaMemcpy(device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (by_rows) {
        q::hadamard128_rows(device, rows, cols, nullptr);
    } else {
        q::hadamard128_cols(device, rows, cols, nullptr);
    }
    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), device, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);
    return output;
}

void test_transforms() {
    std::mt19937_64 rng(0x4ada);
    std::normal_distribution<float> normal;
    // Non-multiple-of-32 column count exercises the column kernel's partial panel.
    constexpr std::int64_t kRows = 256;
    constexpr std::int64_t kCols = 384;
    std::vector<float> x(static_cast<std::size_t>(kRows * kCols));
    for (float& value : x) { value = normal(rng); }

    const auto rows_out = run(x, kRows, kCols, true);
    double worst        = 0.0;
    for (std::int64_t r = 0; r < kRows; ++r) {
        for (std::int64_t c = 0; c < kCols; ++c) {
            const std::int64_t block = c / 128 * 128;
            double sum               = 0.0;
            for (int i = 0; i < 128; ++i) {
                sum += x[static_cast<std::size_t>(r * kCols + block + i)] *
                       h128(i, static_cast<int>(c - block));
            }
            worst =
                std::max(worst, std::abs(sum - rows_out[static_cast<std::size_t>(r * kCols + c)]));
        }
    }
    expect(worst < 1e-5, "row Hadamard differs from the FP64 Sylvester product");

    for (const std::int64_t cols : {kCols, std::int64_t{40}}) {
        std::vector<float> y(static_cast<std::size_t>(kRows * cols));
        for (float& value : y) { value = normal(rng); }
        const auto cols_out = run(y, kRows, cols, false);
        worst               = 0.0;
        for (std::int64_t r = 0; r < kRows; ++r) {
            const std::int64_t block = r / 128 * 128;
            for (std::int64_t c = 0; c < cols; ++c) {
                double sum = 0.0;
                for (int i = 0; i < 128; ++i) {
                    sum += h128(static_cast<int>(r - block), i) *
                           y[static_cast<std::size_t>((block + i) * cols + c)];
                }
                worst = std::max(worst,
                                 std::abs(sum - cols_out[static_cast<std::size_t>(r * cols + c)]));
            }
        }
        expect(worst < 1e-5, "column Hadamard differs from the FP64 Sylvester product");
    }

    // Involution: applying the row transform twice restores the input.
    const auto twice = run(rows_out, kRows, kCols, true);
    worst            = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::abs(twice[i] - x[i])));
    }
    expect(worst < 1e-5, "H128 applied twice does not restore the input");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    test_transforms();
    if (failures == 0) { std::cout << "OK exl3 hadamard\n"; }
    return failures == 0 ? 0 : 1;
}
