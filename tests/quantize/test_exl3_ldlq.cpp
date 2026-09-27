#include "exl3_viterbi_reference.h"

#include "artifact/exl3_trellis.h"
#include "quantize/exl3/block_ldl.h"
#include "quantize/exl3/ldlq.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

namespace q   = ninfer::quantize::exl3;
namespace ref = ninfer::test::exl3;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

constexpr std::int64_t kK = 128;
constexpr std::int64_t kN = 128;
constexpr int kHalfBits   = 4;
constexpr float kScale    = 1.0f / 147.8f;

// Correlated, outlier-heavy Hessian (damped by 2.5% of the mean diagonal).
std::vector<double> make_hessian(std::mt19937_64& rng) {
    std::normal_distribution<double> normal;
    std::vector<double> h(kK * kK, 0.0);
    for (int t = 0; t < 4 * kK; ++t) {
        const double shared = normal(rng);
        std::vector<double> x(kK);
        for (std::int64_t i = 0; i < kK; ++i) {
            x[static_cast<std::size_t>(i)] = (i % 23 == 0 ? 6.0 : 1.0) * (normal(rng) + shared);
        }
        for (std::int64_t i = 0; i < kK; ++i) {
            for (std::int64_t j = 0; j < kK; ++j) {
                h[static_cast<std::size_t>(i * kK + j)] +=
                    x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)];
            }
        }
    }
    double mean = 0.0;
    for (std::int64_t i = 0; i < kK; ++i) { mean += h[static_cast<std::size_t>(i * kK + i)]; }
    for (std::int64_t i = 0; i < kK; ++i) {
        h[static_cast<std::size_t>(i * kK + i)] += 0.025 * mean / kK;
    }
    return h;
}

// tr((W - Wq)ᵀ H (W - Wq)) / tr(Wᵀ H W)
double proxy_error(const std::vector<double>& h, const std::vector<float>& w,
                   const std::vector<double>& wq) {
    double num = 0.0, den = 0.0;
    for (std::int64_t c = 0; c < kN; ++c) {
        for (std::int64_t i = 0; i < kK; ++i) {
            const double ei =
                w[static_cast<std::size_t>(i * kN + c)] - wq[static_cast<std::size_t>(i * kN + c)];
            const double wi = w[static_cast<std::size_t>(i * kN + c)];
            for (std::int64_t j = 0; j < kK; ++j) {
                const double hij = h[static_cast<std::size_t>(i * kK + j)];
                num += ei * hij *
                       (w[static_cast<std::size_t>(j * kN + c)] -
                        wq[static_cast<std::size_t>(j * kN + c)]);
                den += wi * hij * w[static_cast<std::size_t>(j * kN + c)];
            }
        }
    }
    return num / den;
}

// FP64 host LDLQ with the reference two-pass encoder and the same tile order.
std::vector<double> host_ldlq(const std::vector<float>& w, const std::vector<float>& l,
                              std::vector<std::uint16_t>& states) {
    const ref::TrellisSpec spec = ref::trellis_spec(16, 256, kHalfBits);
    const ref::Codebook mul1    = [](std::uint32_t s) {
        return ninfer::artifact::exl3_mul1_value(static_cast<std::uint16_t>(s)) *
               static_cast<double>(kScale);
    };
    std::vector<double> wq(kK * kN, 0.0);
    states.assign(static_cast<std::size_t>(kK * kN), 0);
    const std::int64_t tiles_k = kK / 16;
    for (std::int64_t strip = tiles_k - 1; strip >= 0; --strip) {
        for (std::int64_t tile = 0; tile < kN / 16; ++tile) {
            std::vector<double> target(256);
            for (int t = 0; t < 256; ++t) {
                const std::int64_t row = strip * 16 + q::tile_k(t);
                const std::int64_t col = tile * 16 + q::tile_n(t);
                double comp            = 0.0;
                for (std::int64_t r = (strip + 1) * 16; r < kK; ++r) {
                    comp += static_cast<double>(l[static_cast<std::size_t>(r * kK + row)]) *
                            (w[static_cast<std::size_t>(r * kN + col)] -
                             wq[static_cast<std::size_t>(r * kN + col)]);
                }
                target[static_cast<std::size_t>(t)] =
                    w[static_cast<std::size_t>(row * kN + col)] + comp;
            }
            const ref::TrellisPath path = ref::encode_two_pass(spec, target, mul1);
            for (int t = 0; t < 256; ++t) {
                const std::int64_t row = strip * 16 + q::tile_k(t);
                const std::int64_t col = tile * 16 + q::tile_n(t);
                wq[static_cast<std::size_t>(row * kN + col)] =
                    mul1(path.states[static_cast<std::size_t>(t)]);
                states[static_cast<std::size_t>((tile * tiles_k + strip) * 256 + t)] =
                    static_cast<std::uint16_t>(path.states[static_cast<std::size_t>(t)]);
            }
        }
    }
    return wq;
}

struct GpuResult {
    std::vector<std::uint16_t> states;
    std::vector<float> wq;
};

GpuResult gpu_ldlq(const std::vector<float>& w, const std::vector<float>& l) {
    float *d_w = nullptr, *d_l = nullptr, *d_wq = nullptr;
    std::uint16_t* d_states   = nullptr;
    void* d_scratch           = nullptr;
    const std::size_t scratch = q::ldlq_scratch_bytes(kK, kN, kHalfBits);
    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_l, l.size() * sizeof(float));
    cudaMalloc(&d_wq, w.size() * sizeof(float));
    cudaMalloc(&d_states, w.size() * sizeof(std::uint16_t));
    cudaMalloc(&d_scratch, scratch);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice);
    q::ldlq_quantize(d_w, d_l, kK, kN, kHalfBits, kScale, d_states, d_wq, d_scratch, scratch,
                     nullptr);
    GpuResult out{std::vector<std::uint16_t>(w.size()), std::vector<float>(w.size())};
    cudaMemcpy(out.states.data(), d_states, out.states.size() * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out.wq.data(), d_wq, out.wq.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_l), static_cast<void*>(d_wq),
                    static_cast<void*>(d_states), d_scratch}) {
        cudaFree(p);
    }
    return out;
}

void test_ldlq() {
    std::mt19937_64 rng(0x1d1a);
    std::normal_distribution<float> normal;
    const std::vector<double> h = make_hessian(rng);
    std::vector<float> w(kK * kN);
    for (float& value : w) { value = normal(rng); }

    // L from the (separately qualified) GPU block LDL.
    std::vector<float> l(h.begin(), h.end());
    float* d_h = nullptr;
    cudaMalloc(&d_h, l.size() * sizeof(float));
    cudaMemcpy(d_h, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice);
    expect(q::block_ldl(d_h, kK, nullptr), "block LDL rejected the test Hessian");
    cudaMemcpy(l.data(), d_h, l.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_h);

    const GpuResult gpu = gpu_ldlq(w, l);
    std::vector<std::uint16_t> host_states;
    const std::vector<double> host_wq = host_ldlq(w, l, host_states);

    // Decoded values and states must agree through the tile order.
    const std::int64_t tiles_k = kK / 16;
    for (std::int64_t tile = 0; tile < kN / 16; ++tile) {
        for (std::int64_t strip = 0; strip < tiles_k; ++strip) {
            for (int t = 0; t < 256; ++t) {
                const std::uint16_t s =
                    gpu.states[static_cast<std::size_t>((tile * tiles_k + strip) * 256 + t)];
                const float value =
                    static_cast<float>(ninfer::artifact::exl3_mul1_value(s) * kScale);
                const std::int64_t row = strip * 16 + q::tile_k(t);
                const std::int64_t col = tile * 16 + q::tile_n(t);
                if (gpu.wq[static_cast<std::size_t>(row * kN + col)] != value) {
                    expect(false, "LDLQ decoded matrix disagrees with its states");
                    return;
                }
            }
        }
    }

    std::vector<double> gpu_wq(gpu.wq.begin(), gpu.wq.end());
    const double gpu_err  = proxy_error(h, w, gpu_wq);
    const double host_err = proxy_error(h, w, host_wq);
    std::size_t same      = 0;
    for (std::size_t i = 0; i < host_states.size(); ++i) {
        same += gpu.states[i] == host_states[i];
    }

    // Plain trellis quantization (no error feedback) of the same W: L = I.
    std::vector<float> identity(kK * kK, 0.0f);
    for (std::int64_t i = 0; i < kK; ++i) { identity[static_cast<std::size_t>(i * kK + i)] = 1.0f; }
    const GpuResult plain = gpu_ldlq(w, identity);
    std::vector<double> plain_wq(plain.wq.begin(), plain.wq.end());
    const double plain_err = proxy_error(h, w, plain_wq);

    std::cout << "proxy error: GPU LDLQ " << gpu_err << ", FP64 LDLQ " << host_err
              << ", no feedback " << plain_err << "; identical states "
              << static_cast<double>(same) / static_cast<double>(host_states.size()) << '\n';
    expect(std::abs(gpu_err - host_err) <= 0.02 * host_err,
           "GPU LDLQ proxy error differs from the FP64 LDLQ by more than 2%");
    expect(gpu_err < 0.8 * plain_err, "LDLQ error feedback does not beat plain quantization");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    test_ldlq();
    if (failures == 0) { std::cout << "OK exl3 LDLQ\n"; }
    return failures == 0 ? 0 : 1;
}
