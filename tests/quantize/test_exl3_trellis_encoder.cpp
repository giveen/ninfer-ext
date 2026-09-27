#include "exl3_viterbi_reference.h"

#include "artifact/exl3_trellis.h"
#include "quantize/exl3/trellis_encoder.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

namespace ref = ninfer::test::exl3;
namespace q   = ninfer::quantize::exl3;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

bool cuda_ok(cudaError_t error) {
    if (error == cudaSuccess) { return true; }
    std::cerr << "CUDA: " << cudaGetErrorString(error) << '\n';
    ++failures;
    return false;
}

void test_rate(int half_bits, bool with_reference) {
    constexpr int kTiles   = 24;
    constexpr int kBlocks  = 7; // fewer blocks than tiles exercises the tile stride
    constexpr float kScale = 1.0f / 147.8f;
    std::mt19937_64 rng(0xC0DE + static_cast<unsigned>(half_bits));
    std::normal_distribution<float> normal;
    std::vector<float> tiles(kTiles * 256);
    for (float& value : tiles) { value = normal(rng); }

    const std::size_t scratch_bytes = q::trellis_encoder_scratch_bytes(kBlocks, half_bits);
    float* d_tiles                  = nullptr;
    float* d_decoded                = nullptr;
    std::uint16_t* d_states         = nullptr;
    void* d_scratch                 = nullptr;
    if (!cuda_ok(cudaMalloc(&d_tiles, tiles.size() * sizeof(float))) ||
        !cuda_ok(cudaMalloc(&d_decoded, tiles.size() * sizeof(float))) ||
        !cuda_ok(cudaMalloc(&d_states, tiles.size() * sizeof(std::uint16_t))) ||
        !cuda_ok(cudaMalloc(&d_scratch, scratch_bytes))) {
        return;
    }
    cuda_ok(
        cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(float), cudaMemcpyHostToDevice));
    q::encode_trellis_tiles(d_tiles, kTiles, half_bits, kScale, d_states, d_decoded, d_scratch,
                            scratch_bytes, kBlocks, nullptr);
    std::vector<std::uint16_t> states(tiles.size());
    std::vector<float> decoded(tiles.size());
    cuda_ok(cudaMemcpy(states.data(), d_states, states.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost));
    cuda_ok(cudaMemcpy(decoded.data(), d_decoded, decoded.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    cudaFree(d_tiles);
    cudaFree(d_decoded);
    cudaFree(d_states);
    cudaFree(d_scratch);

    const ref::Codebook mul1 = [&](std::uint32_t s) {
        return static_cast<double>(
                   ninfer::artifact::exl3_mul1_value(static_cast<std::uint16_t>(s))) *
               static_cast<double>(kScale);
    };
    const ref::TrellisSpec spec = ref::trellis_spec(16, 256, half_bits);
    int identical               = 0;
    double gpu_mse              = 0.0;
    double ref_mse              = 0.0;
    double worst                = 0.0;
    for (int tile = 0; tile < kTiles; ++tile) {
        const std::span<const std::uint16_t> ring(states.data() + tile * 256, 256);
        std::vector<std::uint32_t> ring32(ring.begin(), ring.end());
        std::vector<double> target(tiles.begin() + tile * 256, tiles.begin() + (tile + 1) * 256);
        expect(ref::is_circular_path(spec, ring32), "GPU encoder returned a broken ring");
        std::array<std::uint16_t, 256> ring_array{};
        std::copy(ring.begin(), ring.end(), ring_array.begin());
        std::array<std::uint16_t, 256> unpacked{};
        ninfer::artifact::exl3_unpack_trellis_tile(
            ninfer::artifact::exl3_pack_trellis_tile(ring_array,
                                                     static_cast<std::uint64_t>(half_bits)),
            static_cast<std::uint64_t>(half_bits), unpacked);
        expect(unpacked == ring_array, "GPU ring does not survive the tile codec");
        for (int i = 0; i < 256; ++i) {
            const float exact = static_cast<float>(
                ninfer::artifact::exl3_mul1_value(ring[static_cast<std::size_t>(i)]) * kScale);
            if (decoded[static_cast<std::size_t>(tile * 256 + i)] != exact) {
                expect(false, "GPU decoded values differ from mul1(state) * scale");
                break;
            }
        }
        const double gpu_cost = ref::path_cost(ring32, target, mul1);
        gpu_mse += gpu_cost / 256.0 / kTiles;
        if (with_reference) {
            const ref::TrellisPath oracle = ref::encode_two_pass(spec, target, mul1);
            identical += oracle.states == ring32 ? 1 : 0;
            ref_mse += oracle.cost / 256.0 / kTiles;
            worst = std::max(worst, (gpu_cost - oracle.cost) / oracle.cost);
        }
    }
    std::cout << "K=" << half_bits / 2.0 << " GPU MSE " << gpu_mse;
    if (with_reference) {
        std::cout << " FP64 reference MSE " << ref_mse << " identical rings " << identical << '/'
                  << kTiles << " worst relative excess " << worst;
        // FP32 path costs may resolve near-ties differently from FP64, but they must reach the
        // same quality: the mean within 0.2% and no tile more than 2% worse.
        expect(gpu_mse <= ref_mse * 1.002, "GPU encoder MSE exceeds the FP64 reference");
        expect(worst <= 0.02, "a GPU tile is much worse than the FP64 reference");
    }
    std::cout << '\n';
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    for (const int half_bits : {2, 3, 4, 5, 6, 8}) { test_rate(half_bits, true); }
    test_rate(16, false);
    if (failures == 0) { std::cout << "OK exl3 trellis encoder\n"; }
    return failures == 0 ? 0 : 1;
}
