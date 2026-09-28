// EXL3 mul1 Linear qualification: the production Op against an FP64 oracle built from the same
// stored planes. The oracle independently decodes the trellis, applies the explicit 128-point
// Sylvester Hadamards, and accumulates every dot product in double precision.

#include "artifact/exl3_trellis.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace art = ninfer::artifact;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

int tile_k(int t) { return 2 * ((t >> 3) & 3) + (t & 1) + 8 * ((t >> 1) & 1); }
int tile_n(int t) { return (t >> 5) + 8 * ((t >> 2) & 1); }

std::vector<double> hadamard128() {
    std::vector<double> h(128 * 128);
    for (int a = 0; a < 128; ++a) {
        for (int b = 0; b < 128; ++b) {
            h[a * 128 + b] = (__builtin_popcount(a & b) & 1) ? -1.0 : 1.0;
        }
    }
    const double scale = 1.0 / std::sqrt(128.0);
    for (double& value : h) { value *= scale; }
    return h;
}

int run_case(std::int32_t n, std::int32_t k, int half_bits, std::int32_t columns,
             std::uint32_t seed) {
    std::mt19937 generator(seed);
    const auto uniform = [&] { return std::uniform_real_distribution<float>(-2.0F, 2.0F)(generator); };

    const std::size_t tiles_n = static_cast<std::size_t>(n) / 16;
    const std::size_t tiles_k = static_cast<std::size_t>(k) / 16;
    const std::size_t tile_bytes = static_cast<std::size_t>(16 * half_bits);
    const std::size_t trellis_bytes = tiles_n * tiles_k * tile_bytes;
    std::vector<std::byte> trellis(trellis_bytes);
    for (auto& byte : trellis) { byte = static_cast<std::byte>(generator() & 0xFF); }
    std::vector<float> su(k), sv(n);
    for (float& value : su) { value = 0.25F + std::abs(uniform()); }
    for (float& value : sv) { value = 0.25F + std::abs(uniform()); }

    // Packed payload: trellis, then su, then sv, at the layout's plane offsets.
    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(n),
                                            static_cast<std::uint64_t>(k)};
    const auto geometry = ninfer::weight_geometry(ninfer::QType::EXL3_MUL1,
                                                  ninfer::QuantLayout::TrellisT16, shape, 1,
                                                  static_cast<std::uint64_t>(half_bits));
    std::vector<std::byte> payload(geometry.bytes);
    std::memcpy(payload.data(), trellis.data(), trellis_bytes);
    std::memcpy(payload.data() + geometry.input_scale_offset, su.data(), su.size() * sizeof(float));
    std::memcpy(payload.data() + geometry.output_scale_offset, sv.data(),
                sv.size() * sizeof(float));

    // Input activation and device output.
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(k) * columns);
    for (auto& value : x) { value = __float2bfloat16(uniform()); }

    std::byte* d_payload = nullptr;
    __nv_bfloat16* d_x   = nullptr;
    __nv_bfloat16* d_out = nullptr;
    check(cudaMalloc(&d_payload, payload.size()), "cudaMalloc(payload)");
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    check(cudaMalloc(&d_out, static_cast<std::size_t>(n) * columns * sizeof(__nv_bfloat16)),
          "cudaMalloc(out)");
    check(cudaMemcpy(d_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice),
          "copy payload");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");

    ninfer::WeightParent parent{geometry, d_payload};
    ninfer::WeightView view{geometry.shape, {{&parent, 0, geometry.elements}}};
    const ninfer::Weight weight = ninfer::native_weight(view);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, columns});
    ninfer::Tensor output(d_out, ninfer::DType::BF16, {n, columns});

    const std::size_t scratch =
        ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::EXL3_MUL1, n, k,
                                                     ninfer::ops::LinearPolicy::A16Only, columns,
                                                     columns);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::ops::linear(input, weight, output, ninfer::ops::LinearPolicy::A16Only, workspace,
                        nullptr);
    check(cudaDeviceSynchronize(), "linear");

    std::vector<__nv_bfloat16> device_out(static_cast<std::size_t>(n) * columns);
    check(cudaMemcpy(device_out.data(), d_out, device_out.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy out");

    // FP64 oracle from the stored planes.
    std::vector<double> z(static_cast<std::size_t>(k) * n, 0.0);
    for (std::size_t nt = 0; nt < tiles_n; ++nt) {
        for (std::size_t kt = 0; kt < tiles_k; ++kt) {
            const std::byte* tile = trellis.data() + (nt * tiles_k + kt) * tile_bytes;
            std::array<std::uint16_t, art::kExl3WeightsPerTile> states{};
            art::exl3_unpack_trellis_tile(std::span<const std::byte>(tile, tile_bytes), half_bits,
                                          states);
            for (int t = 0; t < 256; ++t) {
                const std::size_t i = kt * 16 + static_cast<std::size_t>(tile_k(t));
                const std::size_t j = nt * 16 + static_cast<std::size_t>(tile_n(t));
                z[i * n + j] = static_cast<double>(art::exl3_mul1_value(states[t]));
            }
        }
    }
    const std::vector<double> h = hadamard128();
    std::vector<double> wz(static_cast<std::size_t>(k) * n, 0.0); // H_k Z H_n
    std::vector<double> temp(static_cast<std::size_t>(k) * n, 0.0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
            double sum = 0.0;
            for (std::size_t b = 0; b < 128; ++b) {
                sum += z[(i / 128 * 128 + b) * n + j] * h[(i % 128) * 128 + b];
            }
            temp[i * n + j] = sum;
        }
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
            double sum = 0.0;
            for (std::size_t b = 0; b < 128; ++b) {
                sum += temp[i * n + (j / 128 * 128 + b)] * h[(j % 128) * 128 + b];
            }
            wz[i * n + j] = sum;
        }
    }

    double max_abs = 0.0, max_error = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
        for (std::int32_t c = 0; c < columns; ++c) {
            double reference = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                const double w = su[i] * sv[j] * wz[i * n + j];
                reference += w * static_cast<double>(__bfloat162float(x[i + c * k]));
            }
            const double actual = static_cast<double>(__bfloat162float(device_out[j + c * n]));
            max_abs     = std::max(max_abs, std::abs(reference));
            max_error   = std::max(max_error, std::abs(actual - reference));
        }
    }
    const double relative = max_abs > 0.0 ? max_error / max_abs : max_error;
    if (relative > 2.0e-2) {
        std::cerr << "EXL3 linear N=" << n << " K=" << k << " T=" << columns
                  << " relative error " << relative << '\n';
        ++failures;
    }

    check(cudaFree(d_out), "cudaFree(out)");
    check(cudaFree(d_x), "cudaFree(x)");
    check(cudaFree(d_payload), "cudaFree(payload)");
    return 0;
}

// The model's real parent shapes are far wider than the small exhaustive cases. For a sampled set of
// output rows this oracle costs O(k*128 + columns*k*128) per row instead of the full O(k*n*128), so
// the same FP64 decode can qualify [248320, 5120] on one GPU.
void run_sampled_case(std::int32_t n, std::int32_t k, int half_bits, std::int32_t columns,
                      std::uint32_t seed) {
    std::mt19937 generator(seed);
    const auto uniform = [&] { return std::uniform_real_distribution<float>(-2.0F, 2.0F)(generator); };

    const std::size_t tiles_n = static_cast<std::size_t>(n) / 16;
    const std::size_t tiles_k = static_cast<std::size_t>(k) / 16;
    const std::size_t tile_bytes = static_cast<std::size_t>(16 * half_bits);
    std::vector<std::byte> trellis(tiles_n * tiles_k * tile_bytes);
    for (auto& byte : trellis) { byte = static_cast<std::byte>(generator() & 0xFF); }
    std::vector<float> su(k), sv(n);
    for (float& value : su) { value = 0.25F + std::abs(uniform()); }
    for (float& value : sv) { value = 0.25F + std::abs(uniform()); }

    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(n),
                                            static_cast<std::uint64_t>(k)};
    const auto geometry = ninfer::weight_geometry(ninfer::QType::EXL3_MUL1,
                                                  ninfer::QuantLayout::TrellisT16, shape, 1,
                                                  static_cast<std::uint64_t>(half_bits));
    std::vector<std::byte> payload(geometry.bytes);
    std::memcpy(payload.data(), trellis.data(), trellis.size());
    std::memcpy(payload.data() + geometry.input_scale_offset, su.data(), su.size() * sizeof(float));
    std::memcpy(payload.data() + geometry.output_scale_offset, sv.data(), sv.size() * sizeof(float));

    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(k) * columns);
    for (auto& value : x) { value = __float2bfloat16(uniform()); }

    std::byte* d_payload = nullptr;
    __nv_bfloat16* d_x   = nullptr;
    __nv_bfloat16* d_out = nullptr;
    check(cudaMalloc(&d_payload, payload.size()), "cudaMalloc(payload)");
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    check(cudaMalloc(&d_out, static_cast<std::size_t>(n) * columns * sizeof(__nv_bfloat16)),
          "cudaMalloc(out)");
    check(cudaMemcpy(d_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice),
          "copy payload");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");

    ninfer::WeightParent parent{geometry, d_payload};
    ninfer::WeightView view{geometry.shape, {{&parent, 0, geometry.elements}}};
    const ninfer::Weight weight = ninfer::native_weight(view);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, columns});
    ninfer::Tensor output(d_out, ninfer::DType::BF16, {n, columns});
    const std::size_t scratch = ninfer::ops::linear_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, columns, columns);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::ops::linear(input, weight, output, ninfer::ops::LinearPolicy::A16Only, workspace,
                        nullptr);
    check(cudaDeviceSynchronize(), "linear");

    std::vector<__nv_bfloat16> device_out(static_cast<std::size_t>(n) * columns);
    check(cudaMemcpy(device_out.data(), d_out, device_out.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy out");

    const std::vector<double> h = hadamard128();
    const std::vector<std::int32_t> rows{0, 1, 15, 16, 127, 128, 129, n / 2, n - 2, n - 1};
    double max_abs = 0.0, max_error = 0.0;
    for (const std::int32_t j : rows) {
        const std::int64_t nb = j / 128;
        std::vector<double> zblock(static_cast<std::size_t>(k) * 128, 0.0);
        for (std::size_t ntl = 0; ntl < 8; ++ntl) {
            for (std::size_t kt = 0; kt < tiles_k; ++kt) {
                const std::byte* tile =
                    trellis.data() + ((nb * 8 + ntl) * tiles_k + kt) * tile_bytes;
                std::array<std::uint16_t, art::kExl3WeightsPerTile> states{};
                art::exl3_unpack_trellis_tile(std::span<const std::byte>(tile, tile_bytes),
                                              half_bits, states);
                for (int t = 0; t < 256; ++t) {
                    zblock[(kt * 16 + static_cast<std::size_t>(tile_k(t))) * 128 + ntl * 16 +
                           static_cast<std::size_t>(tile_n(t))] =
                        static_cast<double>(art::exl3_mul1_value(states[t]));
                }
            }
        }
        std::vector<double> y(k, 0.0);
        for (std::int32_t i = 0; i < k; ++i) {
            double sum = 0.0;
            for (int jl = 0; jl < 128; ++jl) {
                sum += zblock[static_cast<std::size_t>(i) * 128 + jl] * h[jl * 128 + (j % 128)];
            }
            y[static_cast<std::size_t>(i)] = sum;
        }
        for (std::int32_t c = 0; c < columns; ++c) {
            std::vector<double> a(k, 0.0);
            for (std::int32_t i = 0; i < k; ++i) {
                double sum = 0.0;
                for (int ib = 0; ib < 128; ++ib) {
                    const std::int32_t src = i / 128 * 128 + ib;
                    sum += h[(i % 128) * 128 + ib] * su[static_cast<std::size_t>(src)] *
                           static_cast<double>(
                               __bfloat162float(x[static_cast<std::size_t>(src) + c * k]));
                }
                a[static_cast<std::size_t>(i)] = sum;
            }
            double reference = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                reference += a[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
            }
            reference *= sv[static_cast<std::size_t>(j)];
            const double actual = static_cast<double>(
                __bfloat162float(device_out[j + c * n]));
            max_abs   = std::max(max_abs, std::abs(reference));
            max_error = std::max(max_error, std::abs(actual - reference));
        }
    }
    const double relative = max_abs > 0.0 ? max_error / max_abs : max_error;
    if (relative > 3.0e-2) {
        std::cerr << "EXL3 sampled linear N=" << n << " K=" << k << " T=" << columns
                  << " relative error " << relative << '\n';
        ++failures;
    }
    check(cudaFree(d_out), "cudaFree(out)");
    check(cudaFree(d_x), "cudaFree(x)");
    check(cudaFree(d_payload), "cudaFree(payload)");
}

void run_sampled_cases() {
    run_sampled_case(256, 128, 8, 2, 201U);      // oracle self-check against the exhaustive case
    run_sampled_case(384, 256, 8, 2, 202U);
    run_sampled_case(14336, 5120, 10, 2, 211U);  // attention q/key/gate/value parent
    run_sampled_case(14336, 5120, 10, 32, 241U); // same parent through the tensor-core path
    run_sampled_case(16384, 5120, 8, 2, 223U);   // GDN q/key/value/z parent
    run_sampled_case(34816, 5120, 8, 2, 227U);   // MLP gate/up parent
    run_sampled_case(5120, 6144, 10, 2, 229U);   // attention output
    run_sampled_case(5120, 17408, 8, 2, 233U);   // MLP down
    run_sampled_case(248320, 5120, 12, 1, 239U); // vocabulary head
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        for (const int rate : {3, 4, 8}) {
            run_case(128, 128, rate, 1, 101U);
            run_case(128, 128, rate, 7, 103U);
            run_case(256, 128, rate, 64, 107U);
            run_case(384, 256, rate, 3, 109U);
        }
        run_sampled_cases();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " EXL3 mul1 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXL3 Linear: " << error.what() << '\n';
        return 1;
    }
}
