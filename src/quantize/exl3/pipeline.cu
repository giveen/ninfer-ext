#include "quantize/exl3/pipeline.h"

#include "core/device.h"
#include "quantize/exl3/block_ldl.h"
#include "quantize/exl3/hadamard.h"
#include "quantize/exl3/ldlq.h"
#include "quantize/exl3/linalg.h"
#include "quantize/exl3/trellis_encoder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ninfer::quantize::exl3 {
namespace {

// Unit-variance mul1 codebook: bytesum(s * 0x83DCD12D) - 510 has standard deviation ~147.8.
constexpr float kCodebookUnit = 1.0f / 147.8f;

// LDLQ error feedback enlarges the effective target slightly at low rates; the scale search
// quantizes sampled tiles pre-enlarged by this factor (values measured by exllamav3).
float ldlq_drift(int half_bits) {
    switch (half_bits) {
    case 2:
        return 1.08f;
    case 3:
        return 1.035f;
    case 4:
        return 1.018f;
    case 5:
        return 1.009f;
    case 6:
        return 1.004f;
    default:
        return 1.0f;
    }
}

template <class T>
class DeviceArray {
public:
    DeviceArray(std::size_t count, cudaStream_t stream) : count_(count), stream_(stream) {
        CUDA_CHECK(cudaMallocAsync(&data_, std::max<std::size_t>(count, 1) * sizeof(T), stream));
    }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    ~DeviceArray() { cudaFreeAsync(data_, stream_); }

    T* get() const { return data_; }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpyAsync(data_, host.data(), host.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
    }

    void copy_from(const T* device, std::size_t count) {
        CUDA_CHECK(
            cudaMemcpyAsync(data_, device, count * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
    }

    std::vector<T> download(std::size_t count) const {
        std::vector<T> host(count);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return host;
    }

private:
    T* data_ = nullptr;
    std::size_t count_;
    cudaStream_t stream_;
};

std::vector<float> diagonal(const float* m, std::int64_t n, cudaStream_t stream) {
    std::vector<float> out(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy2DAsync(out.data(), sizeof(float), m,
                                 static_cast<std::size_t>(n + 1) * sizeof(float), sizeof(float),
                                 static_cast<std::size_t>(n), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

__global__ void tile_mean_square_kernel(const float* __restrict__ w, std::int64_t n,
                                        std::int64_t tiles_n, float* __restrict__ out) {
    __shared__ float partial[256];
    const std::int64_t tile = blockIdx.x;
    const std::int64_t kt = tile / tiles_n, nt = tile % tiles_n;
    const int t   = static_cast<int>(threadIdx.x);
    const float v = w[(kt * 16 + t / 16) * n + nt * 16 + t % 16];
    partial[t]    = v * v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) { partial[t] += partial[t + stride]; }
        __syncthreads();
    }
    if (t == 0) { out[tile] = partial[0] / 256.0f; }
}

__global__ void gather_tiles_kernel(const float* __restrict__ w, std::int64_t n,
                                    const int2* __restrict__ tiles, float multiplier,
                                    float* __restrict__ out) {
    const int2 at = tiles[blockIdx.x];
    const int t   = static_cast<int>(threadIdx.x);
    out[static_cast<std::int64_t>(blockIdx.x) * 256 + t] =
        w[(static_cast<std::int64_t>(at.x) * 16 + tile_k(t)) * n + at.y * 16 + tile_n(t)] *
        multiplier;
}

// Global scale: a coarse grid on a third of the sampled tiles, then a fine grid on all of them,
// refined by parabolic interpolation. Sampled tiles are a wrapped diagonal of width 3 (every tile
// row and column appears) plus the highest- and lowest-RMS tiles.
float search_global_scale(const float* wr, std::int64_t k, std::int64_t n, int half_bits,
                          cudaStream_t stream) {
    const std::int64_t tiles_k = k / 16, tiles_n = n / 16;
    std::vector<int2> sample;
    const std::int64_t diagonal_length = std::max(tiles_k, tiles_n);
    for (std::int64_t i = 0; i < diagonal_length; ++i) {
        for (int width = 0; width < 3; ++width) {
            sample.push_back(
                make_int2(static_cast<int>(i % tiles_k), static_cast<int>((i + width) % tiles_n)));
        }
    }
    {
        DeviceArray<float> ms(static_cast<std::size_t>(tiles_k * tiles_n), stream);
        tile_mean_square_kernel<<<static_cast<unsigned>(tiles_k * tiles_n), 256, 0, stream>>>(
            wr, n, tiles_n, ms.get());
        CUDA_CHECK(cudaGetLastError());
        const std::vector<float> host = ms.download(static_cast<std::size_t>(tiles_k * tiles_n));
        std::vector<std::int64_t> order(host.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
            return host[static_cast<std::size_t>(a)] > host[static_cast<std::size_t>(b)];
        });
        const auto extremes = std::min<std::size_t>(
            std::max<std::size_t>(8, static_cast<std::size_t>(diagonal_length * 3 / 16)),
            (order.size() + 1) / 2);
        for (std::size_t i = 0; i < extremes; ++i) {
            for (const std::int64_t tile : {order[i], order[order.size() - 1 - i]}) {
                sample.push_back(
                    make_int2(static_cast<int>(tile / tiles_n), static_cast<int>(tile % tiles_n)));
            }
        }
    }
    const float drift = ldlq_drift(half_bits);

    const auto evaluate = [&](const std::vector<int2>& tiles, const std::vector<float>& scales) {
        const auto count = static_cast<std::int64_t>(tiles.size());
        const auto batch = count * static_cast<std::int64_t>(scales.size());
        DeviceArray<int2> list(tiles.size(), stream);
        list.upload(tiles);
        DeviceArray<float> target(static_cast<std::size_t>(count * 256), stream);
        gather_tiles_kernel<<<static_cast<unsigned>(count), 256, 0, stream>>>(wr, n, list.get(),
                                                                              drift, target.get());
        DeviceArray<float> input(static_cast<std::size_t>(batch * 256), stream);
        for (std::size_t s = 0; s < scales.size(); ++s) {
            gather_tiles_kernel<<<static_cast<unsigned>(count), 256, 0, stream>>>(
                wr, n, list.get(), drift * scales[s],
                input.get() + static_cast<std::int64_t>(s) * count * 256);
        }
        CUDA_CHECK(cudaGetLastError());
        const int kBlocks               = trellis_encoder_blocks();
        const std::size_t scratch_bytes = trellis_encoder_scratch_bytes(kBlocks, half_bits);
        DeviceArray<std::uint8_t> scratch(scratch_bytes, stream);
        DeviceArray<std::uint16_t> states(static_cast<std::size_t>(batch * 256), stream);
        DeviceArray<float> decoded(static_cast<std::size_t>(batch * 256), stream);
        encode_trellis_tiles(input.get(), batch, half_bits, kCodebookUnit, states.get(),
                             decoded.get(), scratch.get(), scratch_bytes, kBlocks, stream);
        const auto host_target  = target.download(static_cast<std::size_t>(count * 256));
        const auto host_decoded = decoded.download(static_cast<std::size_t>(batch * 256));
        std::vector<double> mse(scales.size(), 0.0);
        for (std::size_t s = 0; s < scales.size(); ++s) {
            double sum = 0.0;
            for (std::int64_t i = 0; i < count * 256; ++i) {
                const double d = host_decoded[static_cast<std::size_t>(
                                     static_cast<std::int64_t>(s) * count * 256 + i)] /
                                     scales[s] -
                                 host_target[static_cast<std::size_t>(i)];
                sum += d * d;
            }
            mse[s] = sum / static_cast<double>(count * 256);
        }
        return mse;
    };

    std::vector<float> coarse;
    for (int i = 0; i < 10; ++i) { coarse.push_back(0.1f + 0.2f * static_cast<float>(i)); }
    std::vector<int2> subset;
    for (std::size_t i = 0; i < sample.size(); i += 3) { subset.push_back(sample[i]); }
    const auto coarse_mse = evaluate(subset, coarse);
    const float center    = coarse[static_cast<std::size_t>(
        std::min_element(coarse_mse.begin(), coarse_mse.end()) - coarse_mse.begin())];
    constexpr float kStep = 0.075f;
    std::vector<float> fine;
    for (int i = 0; i < 5; ++i) { fine.push_back(center + kStep * static_cast<float>(i - 2)); }
    const auto fine_mse = evaluate(sample, fine);
    const auto best = static_cast<std::size_t>(std::min_element(fine_mse.begin(), fine_mse.end()) -
                                               fine_mse.begin());
    double offset   = 0.0;
    if (best > 0 && best < 4) {
        const double y0 = fine_mse[best - 1], y1 = fine_mse[best], y2 = fine_mse[best + 1];
        const double denom = y0 - 2.0 * y1 + y2;
        offset             = denom > 0.0 ? std::clamp(0.5 * (y0 - y2) / denom, -0.5, 0.5) : 0.0;
    }
    return std::max(fine[best] + static_cast<float>(offset) * kStep, 0.01f);
}

// Conjugate gradients for the SPD system a x = b, starting from x = 1 (the refit factors are
// near one). Returns false if it does not converge.
bool solve_spd(const float* a, const float* b, float* x, std::int64_t n, cudaStream_t stream) {
    DeviceArray<float> r(static_cast<std::size_t>(n), stream),
        p(static_cast<std::size_t>(n), stream), ap(static_cast<std::size_t>(n), stream);
    const std::vector<float> ones(static_cast<std::size_t>(n), 1.0f);
    CUDA_CHECK(cudaMemcpyAsync(x, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    gemm(a, x, ap.get(), n, 1, n, false, stream);
    r.copy_from(b, static_cast<std::size_t>(n));
    axpy(r.get(), ap.get(), -1.0f, n, stream);
    p.copy_from(r.get(), static_cast<std::size_t>(n));
    double rs          = dot(r.get(), r.get(), n, stream);
    const double limit = 1e-12 * dot(b, b, n, stream);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        if (rs <= limit) { return true; }
        gemm(a, p.get(), ap.get(), n, 1, n, false, stream);
        const double pap = dot(p.get(), ap.get(), n, stream);
        if (!(pap > 0.0)) { return false; }
        const auto alpha = static_cast<float>(rs / pap);
        axpy(x, p.get(), alpha, n, stream);
        axpy(r.get(), ap.get(), -alpha, n, stream);
        const double rs_next = dot(r.get(), r.get(), n, stream);
        xpby(p.get(), r.get(), static_cast<float>(rs_next / rs), n, stream);
        rs = rs_next;
    }
    return rs <= limit * 1e4;
}

std::vector<float> reciprocal(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = 1.0f / v[i]; }
    return out;
}

} // namespace

TensorReport quantize_tensor(const float* w, const float* h, std::int64_t k, std::int64_t n,
                             const TensorOptions& options, std::uint16_t* states, float* su_out,
                             float* sv_out, float* wq, cudaStream_t stream) {
    if (k <= 0 || n <= 0 || k % 128 || n % 128) {
        throw std::invalid_argument("EXL3 tensor quantization needs k % 128 == 0 and n % 128 == 0");
    }
    const int half_bits = options.bitrate_half_bits;
    const auto kk = static_cast<std::size_t>(k * k), kn = static_cast<std::size_t>(k * n);
    TensorReport report;

    // Damping and input skew.
    std::vector<float> diag = diagonal(h, k, stream);
    const double mean_diag =
        std::accumulate(diag.begin(), diag.end(), 0.0) / static_cast<double>(k);
    DeviceArray<float> hd(kk, stream);
    hd.copy_from(h, kk);
    add_diagonal(hd.get(), k, static_cast<float>(options.damping * mean_diag), stream);
    std::vector<double> root(diag.size());
    for (std::size_t i = 0; i < diag.size(); ++i) {
        root[i] = std::sqrt(std::max(0.0, diag[i] + options.damping * mean_diag));
    }
    std::sort(root.begin(), root.end(), std::greater<>());
    const double top =
        std::accumulate(root.begin(), root.begin() + static_cast<std::ptrdiff_t>(k / 50), 0.0);
    const double all  = std::accumulate(root.begin(), root.end(), 0.0);
    report.out_scales = all > 0.0 && top / all < options.out_scale_skew_limit;

    // Random signs.
    std::mt19937_64 rng(options.seed);
    const auto sign = [&] { return (rng() & 1U) ? 1.0f : -1.0f; };
    std::vector<float> su(static_cast<std::size_t>(k)), sv(static_cast<std::size_t>(n));
    for (float& s : su) { s = sign(); }
    for (float& s : sv) { s = sign(); }

    // Rotated Hessian and its block LDL (with progressive damping on failure).
    DeviceArray<float> vec_k(static_cast<std::size_t>(k), stream),
        vec_n(static_cast<std::size_t>(n), stream);
    vec_k.upload(su);
    DeviceArray<float> hrot(kk, stream);
    hrot.copy_from(hd.get(), kk);
    scale_rows_cols(hrot.get(), k, k, vec_k.get(), vec_k.get(), stream);
    hadamard128_rows(hrot.get(), k, k, stream);
    hadamard128_cols(hrot.get(), k, k, stream);
    DeviceArray<float> l(kk, stream);
    for (;; ++report.damping_retries) {
        if (report.damping_retries > 10) {
            throw std::runtime_error("EXL3: Hessian is not positive definite after damping");
        }
        l.copy_from(hrot.get(), kk);
        add_diagonal(l.get(), k,
                     static_cast<float>(2.0 * options.damping * mean_diag * report.damping_retries),
                     stream);
        if (block_ldl(l.get(), k, stream)) { break; }
    }

    // Regularize W: output scales and signs, output Hadamard, input scales and signs, input
    // Hadamard, global scale.
    DeviceArray<float> wr(kn, stream);
    wr.copy_from(w, kn);
    col_rms(wr.get(), k, n, vec_n.get(), stream);
    std::vector<float> out_rms = vec_n.download(static_cast<std::size_t>(n));
    const double mean_out =
        std::accumulate(out_rms.begin(), out_rms.end(), 0.0) / static_cast<double>(n);
    for (std::size_t c = 0; c < sv.size(); ++c) {
        float scale = mean_out > 1e-30 ? static_cast<float>(out_rms[c] / mean_out) : 1.0f;
        if (std::abs(scale) < 1e-30f) { scale = 0.1f; }
        if (report.out_scales) { sv[c] *= scale; }
    }
    vec_n.upload(reciprocal(sv));
    scale_rows_cols(wr.get(), k, n, nullptr, vec_n.get(), stream);
    hadamard128_rows(wr.get(), k, n, stream);
    row_rms(wr.get(), k, n, vec_k.get(), stream);
    std::vector<float> in_rms = vec_k.download(static_cast<std::size_t>(k));
    for (std::size_t r = 0; r < su.size(); ++r) {
        su[r] *= std::abs(in_rms[r]) < 1e-30f ? 0.1f : in_rms[r];
    }
    vec_k.upload(reciprocal(su));
    scale_rows_cols(wr.get(), k, n, vec_k.get(), nullptr, stream);
    hadamard128_cols(wr.get(), k, n, stream);
    report.global_scale = search_global_scale(wr.get(), k, n, half_bits, stream);
    axpy(wr.get(), wr.get(), report.global_scale - 1.0f, static_cast<std::int64_t>(kn), stream);
    for (float& s : su) { s /= report.global_scale; }

    // LDLQ in the rotated domain.
    DeviceArray<float> qr(kn, stream);
    {
        const std::size_t scratch_bytes = ldlq_scratch_bytes(k, n, half_bits);
        DeviceArray<std::uint8_t> scratch(scratch_bytes, stream);
        ldlq_quantize(wr.get(), l.get(), k, n, half_bits, kCodebookUnit, states, qr.get(),
                      scratch.get(), scratch_bytes, stream);
    }
    DeviceArray<float> e(kn, stream), he(kn, stream);
    const auto proxy = [&](const float* hess, const float* ref, const float* approx) {
        e.copy_from(ref, kn);
        axpy(e.get(), approx, -1.0f, static_cast<std::int64_t>(kn), stream);
        gemm(hess, e.get(), he.get(), k, n, k, false, stream);
        const double num = dot(e.get(), he.get(), static_cast<std::int64_t>(kn), stream);
        gemm(hess, ref, he.get(), k, n, k, false, stream);
        return num / std::max(dot(ref, he.get(), static_cast<std::int64_t>(kn), stream), 1e-30);
    };
    report.proxy_error_rotated = proxy(hrot.get(), wr.get(), qr.get());

    // Back to the original domain: wq = diag(su) H_k Q H_n diag(sv).
    CUDA_CHECK(cudaMemcpyAsync(wq, qr.get(), kn * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    hadamard128_cols(wq, k, n, stream);
    vec_k.upload(su);
    scale_rows_cols(wq, k, n, vec_k.get(), nullptr, stream);
    hadamard128_rows(wq, k, n, stream);
    vec_n.upload(sv);
    scale_rows_cols(wq, k, n, nullptr, vec_n.get(), stream);
    report.proxy_error_before_refit = proxy(hd.get(), w, wq);

    // Refit su and sv in the Hessian metric with the trellis fixed.
    {
        DeviceArray<float> hw(kn, stream), hq(kn, stream), a(kk, stream);
        gemm(hd.get(), w, hw.get(), k, n, k, false, stream);
        for (int round = 0; round < options.refit_rounds; ++round) {
            gemm(hd.get(), wq, hq.get(), k, n, k, false, stream);
            col_dot(wq, hw.get(), k, n, vec_n.get(), stream);
            const std::vector<float> num = vec_n.download(static_cast<std::size_t>(n));
            col_dot(wq, hq.get(), k, n, vec_n.get(), stream);
            const std::vector<float> den = vec_n.download(static_cast<std::size_t>(n));
            std::vector<float> c(static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < c.size(); ++i) {
                c[i] = den[i] > 1e-30f ? num[i] / den[i] : 1.0f;
                sv[i] *= c[i];
            }
            vec_n.upload(c);
            scale_rows_cols(wq, k, n, nullptr, vec_n.get(), stream);

            gemm(wq, wq, a.get(), k, k, n, true, stream);
            multiply(a.get(), hd.get(), static_cast<std::int64_t>(kk), stream);
            const std::vector<float> a_diag = diagonal(a.get(), k, stream);
            add_diagonal(
                a.get(), k,
                static_cast<float>(1e-6 * std::accumulate(a_diag.begin(), a_diag.end(), 0.0) /
                                   static_cast<double>(k)),
                stream);
            DeviceArray<float> b(static_cast<std::size_t>(k), stream);
            row_dot(wq, hw.get(), k, n, b.get(), stream);
            if (!solve_spd(a.get(), b.get(), vec_k.get(), k, stream)) { continue; }
            std::vector<float> r = vec_k.download(static_cast<std::size_t>(k));
            for (std::size_t i = 0; i < r.size(); ++i) {
                if (!std::isfinite(r[i]) || r[i] <= 0.0f) { r[i] = 1.0f; }
                su[i] *= r[i];
            }
            vec_k.upload(r);
            scale_rows_cols(wq, k, n, vec_k.get(), nullptr, stream);
        }
    }
    report.proxy_error = proxy(hd.get(), w, wq);

    for (float& s : sv) { s *= kCodebookUnit; }
    CUDA_CHECK(cudaMemcpyAsync(su_out, su.data(), su.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(sv_out, sv.data(), sv.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return report;
}

} // namespace ninfer::quantize::exl3
