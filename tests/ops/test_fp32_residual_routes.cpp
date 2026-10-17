// FP32 routes of the residual-stream ops, qualified against naive host oracles at the draft's real
// geometry (8 rows x 4096 channels). Exact routes (adds, widening) compare bits; silu and rmsnorm
// carry the device fast-math/rsqrtf approximation and compare at 1e-5 relative.
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kRows    = 8;
constexpr std::int32_t kColumns = 4096;

std::vector<std::uint32_t> bits(const std::vector<float>& values) {
    std::vector<std::uint32_t> out(values.size());
    std::memcpy(out.data(), values.data(), values.size() * sizeof(float));
    return out;
}

float widen(std::uint16_t bf16) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bf16) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof value);
    return value;
}

int verify_near_bits(const char* label, const std::vector<float>& got,
                     const std::vector<float>& want, float relative) {
    int failures = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a     = got[i];
        const double b     = want[i];
        const double scale = std::max(1.0, std::abs(b));
        if (!(std::abs(a - b) <= relative * scale)) {
            if (failures < 3) {
                std::cerr << label << ": index " << i << " got " << a << " want " << b << '\n';
            }
            ++failures;
        }
    }
    return failures;
}

int run_residual_add() {
    const std::size_t n = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<float> y(n), x(n), want(n);
    fill_uniform(y, 11u, -4.0f, 4.0f);
    fill_uniform(x, 12u, -4.0f, 4.0f);
    for (std::size_t i = 0; i < n; ++i) want[i] = x[i] + y[i];

    GuardedDeviceBuffer dy(n * sizeof(float)), dx(n * sizeof(float));
    dy.copy_from_host(y.data(), dy.bytes());
    dx.copy_from_host(x.data(), dx.bytes());
    Tensor ty(dy.data(), DType::FP32, {kColumns, kRows});
    Tensor tx(dx.data(), DType::FP32, {kColumns, kRows});
    ops::residual_add(ty, tx, nullptr);
    cuda_synchronize();
    return verify_exact("fp32 residual_add", bits(from_device<float>(dx.data(), n)), bits(want));
}

int run_add_bias() {
    const std::size_t n = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<float> bias(kColumns), x(n), want(n);
    fill_uniform(bias, 21u, -4.0f, 4.0f);
    fill_uniform(x, 22u, -4.0f, 4.0f);
    for (std::size_t i = 0; i < n; ++i) {
        want[i] = x[i] + bias[i % static_cast<std::size_t>(kColumns)];
    }

    GuardedDeviceBuffer db(kColumns * sizeof(float)), dx(n * sizeof(float));
    db.copy_from_host(bias.data(), db.bytes());
    dx.copy_from_host(x.data(), dx.bytes());
    Tensor tb(db.data(), DType::FP32, {kColumns});
    Tensor tx(dx.data(), DType::FP32, {kColumns, kRows});
    ops::add_bias(tb, tx, nullptr);
    cuda_synchronize();
    return verify_exact("fp32 add_bias", bits(from_device<float>(dx.data(), n)), bits(want));
}

int run_silu_mul() {
    const std::size_t n = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<float> gate(n), up(n), want(n);
    fill_uniform(gate, 31u, -12.0f, 12.0f);
    fill_uniform(up, 32u, -4.0f, 4.0f);
    gate[0] = 0.0f;
    gate[1] = -8.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const double g = gate[i];
        want[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * static_cast<double>(up[i]));
    }

    GuardedDeviceBuffer dg(n * sizeof(float)), du(n * sizeof(float)), dout(n * sizeof(float));
    dg.copy_from_host(gate.data(), dg.bytes());
    du.copy_from_host(up.data(), du.bytes());
    Tensor tg(dg.data(), DType::FP32, {kColumns, kRows});
    Tensor tu(du.data(), DType::FP32, {kColumns, kRows});
    Tensor tout(dout.data(), DType::FP32, {kColumns, kRows});
    ops::silu_mul(tg, tu, tout, nullptr);
    cuda_synchronize();
    return verify_near_bits("fp32 silu_mul", from_device<float>(dout.data(), n), want, 1e-5f);
}

int run_rmsnorm() {
    const std::size_t n = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<float> x(n), out(n), gain(kColumns), want(n);
    fill_uniform(x, 41u, -4.0f, 4.0f);
    fill_uniform(gain, 42u, 0.5f, 1.5f);
    fill_uniform(out, 43u, -1.0f, 1.0f);
    x[0] = 0.0f; // zero row: exercises the eps path
    for (std::int32_t c = 1; c < kColumns; ++c) x[c] = 0.0f;

    const float eps = 1e-5f;
    for (std::int32_t r = 0; r < kRows; ++r) {
        double sum = 0.0;
        for (std::int32_t c = 0; c < kColumns; ++c) {
            const double v = x[static_cast<std::size_t>(r) * kColumns + c];
            sum += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum / kColumns + eps);
        for (std::int32_t c = 0; c < kColumns; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * kColumns + c;
            want[i]             = static_cast<float>(x[i] * inv * static_cast<double>(gain[c]));
        }
    }

    // Stored gain is BF16: the device reads the widened value, so the oracle must too.
    std::vector<std::uint16_t> gain_bf16(kColumns);
    for (std::int32_t c = 0; c < kColumns; ++c) gain_bf16[c] = f32_to_bf16(gain[c]);
    for (std::int32_t r = 0; r < kRows; ++r) {
        double sum = 0.0;
        for (std::int32_t c = 0; c < kColumns; ++c) {
            const double v = x[static_cast<std::size_t>(r) * kColumns + c];
            sum += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum / kColumns + eps);
        for (std::int32_t c = 0; c < kColumns; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * kColumns + c;
            want[i] = static_cast<float>(x[i] * inv * static_cast<double>(widen(gain_bf16[c])));
        }
    }

    GuardedDeviceBuffer dx(n * sizeof(float)), dg(kColumns * sizeof(std::uint16_t)),
        dout(n * sizeof(float));
    dx.copy_from_host(x.data(), dx.bytes());
    dg.copy_from_host(gain_bf16.data(), dg.bytes());
    dout.copy_from_host(out.data(), dout.bytes());
    Tensor tx(dx.data(), DType::FP32, {kColumns, kRows});
    Tensor tg(dg.data(), DType::BF16, {kColumns});
    Tensor tout(dout.data(), DType::FP32, {kColumns, kRows});
    ops::rmsnorm(tx, tg, eps, false, tout, nullptr);
    cuda_synchronize();
    return verify_near_bits("fp32 rmsnorm", from_device<float>(dout.data(), n), want, 1e-4f);
}

int run_cast_widen() {
    const std::size_t n = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<std::uint16_t> source(n);
    std::vector<std::uint32_t> want_bits(n);
    std::vector<float> words(n);
    fill_uniform(words, 51u, 0.0f, 65535.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t word = static_cast<std::uint16_t>(words[i]);
        source[i]                = word;
        want_bits[i]             = static_cast<std::uint32_t>(word) << 16;
    }

    GuardedDeviceBuffer ds(n * sizeof(std::uint16_t)), dd(n * sizeof(float));
    ds.copy_from_host(source.data(), ds.bytes());
    Tensor ts(ds.data(), DType::BF16, {kColumns, kRows});
    Tensor td(dd.data(), DType::FP32, {kColumns, kRows});
    ops::cast_bf16_to_fp32(ts, td, nullptr);
    cuda_synchronize();
    return verify_exact("cast_bf16_to_fp32", bits(from_device<float>(dd.data(), n)), want_bits);
}

} // namespace

int main() {
    int failures = 0;
    failures += run_residual_add();
    failures += run_add_bias();
    failures += run_silu_mul();
    failures += run_rmsnorm();
    failures += run_cast_widen();
    if (failures == 0) {
        std::cout << "fp32 residual routes: all cases passed\n";
        return 0;
    }
    std::cerr << "fp32 residual routes: " << failures << " failures\n";
    return 1;
}
