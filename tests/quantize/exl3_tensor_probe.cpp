// Runs the EXL3 single-tensor pipeline on raw FP32 inputs and prints its report as JSON.
// Used by tools/exl3/compare_tensor.py to compare against exllamav3 on the same W and H.
//
// usage: ninfer_exl3_tensor_probe K N HALF_BITS SEED W.f32 H.f32
//   W.f32: [K][N] row-major (rows are input channels); H.f32: [K][K] mean XᵀX (undamped).

#include "quantize/exl3/pipeline.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::vector<float> out(count);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(out.data()),
                 static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("cannot read " + path);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0] << " K N HALF_BITS SEED W.f32 H.f32\n";
        return 2;
    }
    const std::int64_t k = std::stoll(argv[1]), n = std::stoll(argv[2]);
    ninfer::quantize::exl3::TensorOptions options;
    options.bitrate_half_bits = std::stoi(argv[3]);
    options.seed              = std::stoull(argv[4]);
    const auto w              = read_f32(argv[5], static_cast<std::size_t>(k * n));
    const auto h              = read_f32(argv[6], static_cast<std::size_t>(k * k));

    float *d_w = nullptr, *d_h = nullptr, *d_su = nullptr, *d_sv = nullptr, *d_wq = nullptr;
    std::uint16_t* d_states = nullptr;
    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_h, h.size() * sizeof(float));
    cudaMalloc(&d_su, static_cast<std::size_t>(k) * sizeof(float));
    cudaMalloc(&d_sv, static_cast<std::size_t>(n) * sizeof(float));
    cudaMalloc(&d_wq, w.size() * sizeof(float));
    cudaMalloc(&d_states, w.size() * sizeof(std::uint16_t));
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    const auto start  = std::chrono::steady_clock::now();
    const auto report = ninfer::quantize::exl3::quantize_tensor(d_w, d_h, k, n, options, d_states,
                                                                d_su, d_sv, d_wq, nullptr);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"proxy_error_rotated\": " << report.proxy_error_rotated
              << ", \"proxy_error_before_refit\": " << report.proxy_error_before_refit
              << ", \"proxy_error\": " << report.proxy_error
              << ", \"global_scale\": " << report.global_scale
              << ", \"out_scales\": " << (report.out_scales ? "true" : "false")
              << ", \"damping_retries\": " << report.damping_retries << ", \"seconds\": " << seconds
              << "}\n";
    for (void* p :
         {static_cast<void*>(d_w), static_cast<void*>(d_h), static_cast<void*>(d_su),
          static_cast<void*>(d_sv), static_cast<void*>(d_wq), static_cast<void*>(d_states)}) {
        cudaFree(p);
    }
    return 0;
}
