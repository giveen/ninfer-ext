// EXL3 fused-consumer qualification: `linear_add`, `linear_swiglu`, `attn_input_proj` and
// `gdn_input_proj` decode one EXL3 parent and apply their epilogue. The reference is the validated
// `ops::linear` on the same parent, so this isolates the consumers' row mapping and epilogue from
// the decode itself (which `ninfer_linear_exl3_a16_test` qualifies against an FP64 oracle).

#include "artifact/exl3_trellis.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
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

struct Exl3Parent {
    std::vector<std::byte> host;
    std::byte* device = nullptr;
    ninfer::WeightGeometry geometry{};
    std::optional<ninfer::WeightParent> parent;
    std::optional<ninfer::WeightView> view;
    ninfer::Weight weight{};

    Exl3Parent(std::int32_t n, std::int32_t k, int half_bits, std::uint32_t seed) {
        std::mt19937 generator(seed);
        const std::size_t tiles_n = static_cast<std::size_t>(n) / 16;
        const std::size_t tiles_k = static_cast<std::size_t>(k) / 16;
        const std::size_t tile_bytes = static_cast<std::size_t>(16 * half_bits);
        std::vector<std::byte> trellis(tiles_n * tiles_k * tile_bytes);
        for (auto& byte : trellis) { byte = static_cast<std::byte>(generator() & 0xFF); }
        std::vector<float> su(k), sv(n);
        const auto uniform = [&] { return std::uniform_real_distribution<float>(0.1F, 1.5F)(generator); };
        for (float& value : su) { value = uniform(); }
        for (float& value : sv) { value = uniform(); }

        const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(n),
                                                 static_cast<std::uint64_t>(k)};
        geometry = ninfer::weight_geometry(ninfer::QType::EXL3_MUL1, ninfer::QuantLayout::TrellisT16,
                                           shape, 1, static_cast<std::uint64_t>(half_bits));
        host.resize(geometry.bytes);
        std::memcpy(host.data(), trellis.data(), trellis.size());
        std::memcpy(host.data() + geometry.input_scale_offset, su.data(), su.size() * sizeof(float));
        std::memcpy(host.data() + geometry.output_scale_offset, sv.data(), sv.size() * sizeof(float));
        check(cudaMalloc(&device, host.size()), "cudaMalloc(parent)");
        check(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice), "copy parent");
        parent.emplace(ninfer::WeightParent{geometry, device});
        view.emplace(ninfer::WeightView{geometry.shape, {{&*parent, 0, geometry.elements}}});
        weight = ninfer::native_weight(*view);
    }

    ~Exl3Parent() {
        if (device != nullptr) { (void)cudaFree(device); }
    }
};

std::vector<__nv_bfloat16> host_bf16(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    const auto uniform = [&] { return std::uniform_real_distribution<float>(-1.0F, 1.0F)(generator); };
    std::vector<__nv_bfloat16> out(count);
    for (auto& value : out) { value = __float2bfloat16(uniform()); }
    return out;
}

// Runs the validated `ops::linear` on the parent and returns its BF16 [n, t] output on the host.
std::vector<__nv_bfloat16> linear_reference(const Exl3Parent& parent, const __nv_bfloat16* x,
                                            std::int32_t k, std::int32_t n, std::int32_t t) {
    const std::size_t scratch = ninfer::ops::linear_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, t, t);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    __nv_bfloat16* d_out = nullptr;
    check(cudaMalloc(&d_out, static_cast<std::size_t>(n) * t * sizeof(__nv_bfloat16)),
          "cudaMalloc(ref out)");
    ninfer::Tensor input(const_cast<__nv_bfloat16*>(x), ninfer::DType::BF16, {k, t});
    ninfer::Tensor output(d_out, ninfer::DType::BF16, {n, t});
    ninfer::ops::linear(input, parent.weight, output, ninfer::ops::LinearPolicy::A16Only, workspace,
                        nullptr);
    check(cudaDeviceSynchronize(), "linear reference");
    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(n) * t);
    check(cudaMemcpy(host.data(), d_out, host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy ref out");
    check(cudaFree(d_out), "cudaFree(ref out)");
    return host;
}

double max_relative(const std::vector<__nv_bfloat16>& actual,
                    const std::vector<__nv_bfloat16>& expected) {
    double scale = 0.0, error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(__bfloat162float(actual[i]));
        const double e = static_cast<double>(__bfloat162float(expected[i]));
        scale = std::max(scale, std::abs(e));
        error = std::max(error, std::abs(a - e));
    }
    return scale > 0.0 ? error / scale : error;
}

void test_linear_add(std::int32_t n, std::int32_t k, int half_bits, std::int32_t t,
                     std::uint32_t seed) {
    Exl3Parent parent(n, k, half_bits, seed);
    const auto x = host_bf16(static_cast<std::size_t>(k) * t, seed + 1U);
    const auto residual = host_bf16(static_cast<std::size_t>(n) * t, seed + 2U);
    const auto reference = linear_reference(parent, x.data(), k, n, t);

    __nv_bfloat16 *d_x = nullptr, *d_res = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    check(cudaMalloc(&d_res, residual.size() * sizeof(__nv_bfloat16)), "cudaMalloc(res)");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");
    check(cudaMemcpy(d_res, residual.data(), residual.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice),
          "copy residual");
    const std::size_t scratch = ninfer::ops::linear_add_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, t, t);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, t});
    ninfer::Tensor out(d_res, ninfer::DType::BF16, {n, t});
    ninfer::ops::linear_add(input, parent.weight, out, ninfer::ops::LinearPolicy::A16Only, workspace,
                            nullptr);
    check(cudaDeviceSynchronize(), "linear_add");
    std::vector<__nv_bfloat16> host(residual.size());
    check(cudaMemcpy(host.data(), d_res, host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy linear_add");
    std::vector<__nv_bfloat16> expected(residual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = __float2bfloat16(__bfloat162float(reference[i]) +
                                       __bfloat162float(residual[i]));
    }
    const double relative = max_relative(host, expected);
    if (relative > 2.0e-2) {
        std::cerr << "EXL3 linear_add N=" << n << " K=" << k << " T=" << t
                  << " relative error " << relative << '\n';
        ++failures;
    }
    check(cudaFree(d_res), "cudaFree(res)");
    check(cudaFree(d_x), "cudaFree(x)");
}

void test_linear_swiglu(std::int32_t n, std::int32_t k, int half_bits, std::int32_t t,
                        std::uint32_t seed) {
    Exl3Parent parent(n, k, half_bits, seed);
    const auto x = host_bf16(static_cast<std::size_t>(k) * t, seed + 1U);
    const auto reference = linear_reference(parent, x.data(), k, n, t);

    __nv_bfloat16 *d_x = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    check(cudaMalloc(&d_out, static_cast<std::size_t>(n / 2) * t * sizeof(__nv_bfloat16)),
          "cudaMalloc(out)");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");
    const std::size_t scratch = ninfer::ops::linear_swiglu_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, t, t);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, t});
    ninfer::Tensor out(d_out, ninfer::DType::BF16, {n / 2, t});
    ninfer::ops::linear_swiglu(input, parent.weight, out, ninfer::ops::LinearPolicy::A16Only,
                               workspace, nullptr);
    check(cudaDeviceSynchronize(), "linear_swiglu");
    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(n / 2) * t);
    check(cudaMemcpy(host.data(), d_out, host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy linear_swiglu");
    std::vector<__nv_bfloat16> expected(host.size());
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t j = 0; j < n / 2; ++j) {
            const double gate = __bfloat162float(reference[j + static_cast<std::size_t>(c) * n]);
            const double up =
                __bfloat162float(reference[(n / 2 + j) + static_cast<std::size_t>(c) * n]);
            const double silu = gate / (1.0 + std::exp(-gate));
            expected[j + static_cast<std::size_t>(c) * (n / 2)] =
                __float2bfloat16(static_cast<float>(silu * up));
        }
    }
    const double relative = max_relative(host, expected);
    if (relative > 2.0e-2) {
        double host_max = 0.0, expected_max = 0.0;
        for (std::size_t i = 0; i < host.size(); ++i) {
            host_max = std::max(host_max, std::abs(static_cast<double>(__bfloat162float(host[i]))));
            expected_max =
                std::max(expected_max, std::abs(static_cast<double>(__bfloat162float(expected[i]))));
        }
        std::cerr << "EXL3 linear_swiglu N=" << n << " K=" << k << " T=" << t
                  << " relative error " << relative << " host_max " << host_max << " expected_max "
                  << expected_max << '\n';
        std::cerr << "  host[0..3] " << __bfloat162float(host[0]) << ' '
                  << __bfloat162float(host[1]) << ' ' << __bfloat162float(host[2]) << ' '
                  << __bfloat162float(host[3]) << "  expected[0..3] "
                  << __bfloat162float(expected[0]) << ' ' << __bfloat162float(expected[1]) << ' '
                  << __bfloat162float(expected[2]) << ' ' << __bfloat162float(expected[3]) << '\n';
        const double r0 = __bfloat162float(reference[0]);
        const double r1 = __bfloat162float(reference[static_cast<std::size_t>(n / 2)]);
        std::cerr << "  reference gate[0]=" << r0 << " up[0]=" << r1
                  << " silu(gate)*up=" << (r0 / (1.0 + std::exp(-r0))) * r1
                  << " silu(up)*gate=" << (r1 / (1.0 + std::exp(-r1))) * r0 << '\n';
        ++failures;
    }
    check(cudaFree(d_out), "cudaFree(out)");
    check(cudaFree(d_x), "cudaFree(x)");
}

void test_attn_input_proj(std::int32_t t, int half_bits, std::uint32_t seed) {
    constexpr std::int32_t k = 5120, q_rows = 6144, kv_rows = 1024, n = 14336;
    Exl3Parent parent(n, k, half_bits, seed);
    const auto x = host_bf16(static_cast<std::size_t>(k) * t, seed + 1U);
    const auto reference = linear_reference(parent, x.data(), k, n, t);

    __nv_bfloat16 *d_x = nullptr, *d_q = nullptr, *d_g = nullptr, *d_k = nullptr, *d_v = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    for (auto** p : {&d_q, &d_g}) {
        check(cudaMalloc(p, static_cast<std::size_t>(q_rows) * t * sizeof(__nv_bfloat16)),
              "cudaMalloc(q/gate)");
    }
    for (auto** p : {&d_k, &d_v}) {
        check(cudaMalloc(p, static_cast<std::size_t>(kv_rows) * t * sizeof(__nv_bfloat16)),
              "cudaMalloc(k/v)");
    }
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");
    const std::size_t scratch = ninfer::ops::attn_input_proj_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, t, t);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, t});
    ninfer::Tensor q(d_q, ninfer::DType::BF16, {q_rows, t});
    ninfer::Tensor gate(d_g, ninfer::DType::BF16, {q_rows, t});
    ninfer::Tensor key(d_k, ninfer::DType::BF16, {kv_rows, t});
    ninfer::Tensor value(d_v, ninfer::DType::BF16, {kv_rows, t});
    ninfer::ops::attn_input_proj(input, parent.weight, q, gate, key, value,
                                 ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    check(cudaDeviceSynchronize(), "attn_input_proj");
    const auto copy = [&](__nv_bfloat16* device, std::size_t count) {
        std::vector<__nv_bfloat16> host(count);
        check(cudaMemcpy(host.data(), device, count * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost),
              "copy slice");
        return host;
    };
    const auto q_host = copy(d_q, static_cast<std::size_t>(q_rows) * t);
    const auto k_host = copy(d_k, static_cast<std::size_t>(kv_rows) * t);
    const auto g_host = copy(d_g, static_cast<std::size_t>(q_rows) * t);
    const auto v_host = copy(d_v, static_cast<std::size_t>(kv_rows) * t);
    const auto slice = [&](std::int32_t row, std::int32_t rows) {
        std::vector<__nv_bfloat16> out(static_cast<std::size_t>(rows) * t);
        for (std::int32_t c = 0; c < t; ++c) {
            for (std::int32_t r = 0; r < rows; ++r) {
                out[r + static_cast<std::size_t>(c) * rows] =
                    reference[(row + r) + static_cast<std::size_t>(c) * n];
            }
        }
        return out;
    };
    const double q_err = max_relative(q_host, slice(0, q_rows));
    const double k_err = max_relative(k_host, slice(q_rows, kv_rows));
    const double g_err = max_relative(g_host, slice(q_rows + kv_rows, q_rows));
    const double v_err = max_relative(v_host, slice(q_rows + kv_rows + q_rows, kv_rows));
    if (std::max({q_err, k_err, g_err, v_err}) > 1.0e-3) {
        std::cerr << "EXL3 attn_input_proj T=" << t << " errors q=" << q_err << " k=" << k_err
                  << " gate=" << g_err << " value=" << v_err << '\n';
        ++failures;
    }
    check(cudaFree(d_v), "cudaFree(v)");
    check(cudaFree(d_k), "cudaFree(k)");
    check(cudaFree(d_g), "cudaFree(gate)");
    check(cudaFree(d_q), "cudaFree(q)");
    check(cudaFree(d_x), "cudaFree(x)");
}

void test_gdn_input_proj(std::int32_t t, int half_bits, std::uint32_t seed) {
    constexpr std::int32_t k = 5120, qkv_rows = 10240, z_rows = 6144, n = 16384;
    Exl3Parent parent(n, k, half_bits, seed);
    const auto x = host_bf16(static_cast<std::size_t>(k) * t, seed + 1U);
    const auto reference = linear_reference(parent, x.data(), k, n, t);

    __nv_bfloat16 *d_x = nullptr, *d_qkv = nullptr, *d_z = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)), "cudaMalloc(x)");
    check(cudaMalloc(&d_qkv, static_cast<std::size_t>(qkv_rows) * t * sizeof(__nv_bfloat16)),
          "cudaMalloc(qkv)");
    check(cudaMalloc(&d_z, static_cast<std::size_t>(z_rows) * t * sizeof(__nv_bfloat16)),
          "cudaMalloc(z)");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");
    const std::size_t scratch = ninfer::ops::gdn_input_proj_workspace_capacity_bytes(
        ninfer::QType::EXL3_MUL1, n, k, ninfer::ops::LinearPolicy::A16Only, t, t);
    ninfer::DeviceArena workspace(scratch == 0 ? 1 : scratch);
    ninfer::Tensor input(d_x, ninfer::DType::BF16, {k, t});
    ninfer::Tensor qkv(d_qkv, ninfer::DType::BF16, {qkv_rows, t});
    ninfer::Tensor z(d_z, ninfer::DType::BF16, {z_rows, t});
    ninfer::ops::gdn_input_proj(input, parent.weight, qkv, z, ninfer::ops::LinearPolicy::A16Only,
                                workspace, nullptr);
    check(cudaDeviceSynchronize(), "gdn_input_proj");
    std::vector<__nv_bfloat16> qkv_host(static_cast<std::size_t>(qkv_rows) * t);
    std::vector<__nv_bfloat16> z_host(static_cast<std::size_t>(z_rows) * t);
    check(cudaMemcpy(qkv_host.data(), d_qkv, qkv_host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy qkv");
    check(cudaMemcpy(z_host.data(), d_z, z_host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy z");
    std::vector<__nv_bfloat16> qkv_ref(static_cast<std::size_t>(qkv_rows) * t);
    std::vector<__nv_bfloat16> z_ref(static_cast<std::size_t>(z_rows) * t);
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t r = 0; r < qkv_rows; ++r) {
            qkv_ref[r + static_cast<std::size_t>(c) * qkv_rows] =
                reference[r + static_cast<std::size_t>(c) * n];
        }
        for (std::int32_t r = 0; r < z_rows; ++r) {
            z_ref[r + static_cast<std::size_t>(c) * z_rows] =
                reference[(qkv_rows + r) + static_cast<std::size_t>(c) * n];
        }
    }
    const double qkv_err = max_relative(qkv_host, qkv_ref);
    const double z_err = max_relative(z_host, z_ref);
    if (std::max(qkv_err, z_err) > 1.0e-3) {
        std::cerr << "EXL3 gdn_input_proj T=" << t << " errors qkv=" << qkv_err
                  << " z=" << z_err << '\n';
        ++failures;
    }
    check(cudaFree(d_z), "cudaFree(z)");
    check(cudaFree(d_qkv), "cudaFree(qkv)");
    check(cudaFree(d_x), "cudaFree(x)");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        test_linear_add(5120, 17408, 8, 2, 301U);
        test_linear_swiglu(34816, 5120, 8, 2, 303U);
        test_attn_input_proj(2, 10, 305U);
        test_gdn_input_proj(2, 8, 307U);
        // The 3.5 bpw artifact is attention K=9 with GDN K=7 -- the odd half rates the consumers take
        // in production, and the combination that had never been run.
        test_linear_add(5120, 17408, 7, 2, 311U);
        test_linear_swiglu(34816, 5120, 7, 2, 313U);
        test_attn_input_proj(2, 9, 317U);
        test_gdn_input_proj(2, 7, 319U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " EXL3 fused consumers\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXL3 consumers: " << error.what() << '\n';
        return 1;
    }
}
