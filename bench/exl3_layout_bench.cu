#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr std::uint32_t kMul1 = 0x83DCD12DU;

__device__ __forceinline__ std::uint32_t mul1(std::uint32_t state) {
    const std::uint32_t product = state * kMul1;
    return (product & 0xFFU) + ((product >> 8) & 0xFFU) + ((product >> 16) & 0xFFU) +
           ((product >> 24) & 0xFFU);
}

template <bool OutputMajor>
__global__ void scan_gemv_tiles(const std::uint32_t* __restrict__ tiles,
                                std::uint32_t* __restrict__ output,
                                std::uint32_t tiles_n, std::uint32_t tiles_k,
                                std::uint32_t words_per_tile) {
    const auto n_tile = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
    const auto warp = static_cast<std::uint32_t>(threadIdx.x) >> 5U;
    std::uint32_t sum = 0;
    for (std::uint32_t k_tile = warp; k_tile < tiles_k; k_tile += 4) {
        const auto tile_index = OutputMajor ? n_tile * tiles_k + k_tile
                                            : k_tile * tiles_n + n_tile;
        const auto* words = tiles + static_cast<std::uint64_t>(tile_index) * words_per_tile;
        for (std::uint32_t word = lane; word < words_per_tile; word += 32) {
            const auto packed = words[word];
            sum += mul1(packed & 0xFFFFU) + mul1(packed >> 16);
        }
    }
    output[static_cast<std::uint64_t>(n_tile) * 128 + threadIdx.x] = sum;
}

template <int RateHalf, bool MsbFirst>
__device__ __forceinline__ std::uint32_t read_window(const std::uint32_t* tile,
                                                      std::uint32_t weight) {
    constexpr std::uint32_t words = 4 * RateHalf;
    constexpr std::uint32_t total_bits = 128 * RateHalf;
    constexpr std::uint32_t low_width = RateHalf / 2;
    const auto count = weight + 1;
    const auto end = count * low_width + ((RateHalf & 1) ? count / 2 : 0);
    const auto begin = end + total_bits - 16;
    const auto wrapped_begin = begin >= total_bits ? begin - total_bits : begin;
    const auto word_index = wrapped_begin / 32;
    const auto shift = wrapped_begin % 32;
    auto lo = tile[word_index];
    auto hi = tile[word_index + 1 == words ? 0 : word_index + 1];
    if constexpr (MsbFirst) {
        lo = __byte_perm(lo, 0, 0x0123);
        hi = __byte_perm(hi, 0, 0x0123);
        return __funnelshift_l(lo, hi, shift) >> 16;
    } else {
        return __funnelshift_r(lo, hi, shift) & 0xFFFFU;
    }
}

template <int RateHalf, bool MsbFirst>
__global__ void decode_bit_order(const std::uint32_t* __restrict__ tiles,
                                 std::uint32_t* __restrict__ output) {
    constexpr std::uint32_t words = 4 * RateHalf;
    const auto weight = static_cast<std::uint32_t>(threadIdx.x);
    const auto* tile = tiles + static_cast<std::uint64_t>(blockIdx.x) * words;
    output[static_cast<std::uint64_t>(blockIdx.x) * 256 + weight] =
        mul1(read_window<RateHalf, MsbFirst>(tile, weight));
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

struct Result {
    const char* order;
    double milliseconds;
    double gib_per_second;
};

struct BitOrderResult {
    const char* order;
    double milliseconds;
    double giga_states_per_second;
};

template <int RateHalf, bool MsbFirst>
BitOrderResult measure_bit_order(const std::uint32_t* tiles, std::uint32_t* output,
                                 std::uint32_t tile_count) {
    constexpr int warmups = 5;
    constexpr int iterations = 50;
    for (int i = 0; i < warmups; ++i) {
        decode_bit_order<RateHalf, MsbFirst><<<tile_count, 256>>>(tiles, output);
    }
    check(cudaGetLastError(), "bit-order warmup launch");
    check(cudaDeviceSynchronize(), "bit-order warmup synchronize");
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "cudaEventCreate(bit start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(bit stop)");
    check(cudaEventRecord(start), "cudaEventRecord(bit start)");
    for (int i = 0; i < iterations; ++i) {
        decode_bit_order<RateHalf, MsbFirst><<<tile_count, 256>>>(tiles, output);
    }
    check(cudaGetLastError(), "bit-order timed launch");
    check(cudaEventRecord(stop), "cudaEventRecord(bit stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(bit stop)");
    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(bit order)");
    check(cudaEventDestroy(start), "cudaEventDestroy(bit start)");
    check(cudaEventDestroy(stop), "cudaEventDestroy(bit stop)");
    constexpr std::uint32_t tile_count_const = 4096;
    const auto milliseconds = static_cast<double>(elapsed_ms) / iterations;
    const auto giga_states = static_cast<double>(tile_count_const) * 256.0 / 1.0e9;
    return {MsbFirst ? "MSB-first" : "LSB-first", milliseconds,
            giga_states / (milliseconds / 1000.0)};
}

template <int RateHalf>
void benchmark_bit_order() {
    constexpr std::uint32_t tile_count = 4096;
    constexpr std::uint32_t words_per_tile = 4 * RateHalf;
    std::vector<std::uint32_t> host(static_cast<std::size_t>(tile_count) * words_per_tile);
    std::uint32_t state = 0xA511E9B3U;
    for (auto& word : host) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        word = state;
    }
    std::uint32_t* tiles = nullptr;
    std::uint32_t* output = nullptr;
    check(cudaMalloc(&tiles, host.size() * sizeof(std::uint32_t)), "cudaMalloc(bit tiles)");
    check(cudaMalloc(&output, static_cast<std::size_t>(tile_count) * 256 * sizeof(std::uint32_t)),
          "cudaMalloc(bit output)");
    check(cudaMemcpy(tiles, host.data(), host.size() * sizeof(std::uint32_t),
                     cudaMemcpyHostToDevice),
          "copy bit tiles");
    const auto lsb = measure_bit_order<RateHalf, false>(tiles, output, tile_count);
    const auto msb = measure_bit_order<RateHalf, true>(tiles, output, tile_count);
    std::printf("bitrate_half_bits=%d: %s %.3f ms %.2f Gstates/s; %s %.3f ms %.2f Gstates/s; "
                "MSB/LSB time %.3fx\n",
                RateHalf, lsb.order, lsb.milliseconds, lsb.giga_states_per_second, msb.order,
                msb.milliseconds, msb.giga_states_per_second,
                msb.milliseconds / lsb.milliseconds);
    check(cudaFree(output), "cudaFree(bit output)");
    check(cudaFree(tiles), "cudaFree(bit tiles)");
}

template <bool OutputMajor>
Result measure(const std::uint32_t* device_tiles, std::uint32_t* device_output,
               std::uint32_t tiles_n, std::uint32_t tiles_k,
               std::uint32_t words_per_tile, std::size_t payload_bytes) {
    constexpr int warmups = 5;
    constexpr int iterations = 25;
    for (int i = 0; i < warmups; ++i) {
        scan_gemv_tiles<OutputMajor><<<tiles_n, 128>>>(device_tiles, device_output, tiles_n,
                                                     tiles_k, words_per_tile);
    }
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup synchronize");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        scan_gemv_tiles<OutputMajor><<<tiles_n, 128>>>(device_tiles, device_output, tiles_n,
                                                     tiles_k, words_per_tile);
    }
    check(cudaGetLastError(), "timed launch");
    check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::vector<std::uint32_t> output(tiles_n * 128);
    check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost),
          "copy output checksum");
    const auto checksum = std::accumulate(output.begin(), output.end(), std::uint64_t{0});
    if (checksum == 0) { std::fprintf(stderr, "unexpected zero checksum\n"); }

    const auto milliseconds = static_cast<double>(elapsed_ms) / iterations;
    const auto gib = static_cast<double>(payload_bytes) / (1024.0 * 1024.0 * 1024.0);
    return {OutputMajor ? "output-major" : "k-major", milliseconds,
            gib / (milliseconds / 1000.0)};
}

void benchmark(std::uint32_t n, std::uint32_t k, std::uint32_t bitrate_half_bits) {
    const auto tiles_n = n / 16;
    const auto tiles_k = k / 16;
    const auto tile_bytes = 16 * bitrate_half_bits;
    const auto words_per_tile = tile_bytes / sizeof(std::uint32_t);
    const auto payload_bytes = static_cast<std::size_t>(tiles_n) * tiles_k * tile_bytes;
    std::vector<std::uint32_t> host(payload_bytes / sizeof(std::uint32_t));
    std::uint32_t state = 0x9E3779B9U;
    for (auto& word : host) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        word = state;
    }

    std::uint32_t* device_tiles = nullptr;
    std::uint32_t* device_output = nullptr;
    check(cudaMalloc(&device_tiles, payload_bytes), "cudaMalloc(tiles)");
    check(cudaMalloc(&device_output, static_cast<std::size_t>(tiles_n) * 128 * sizeof(std::uint32_t)),
          "cudaMalloc(output)");
    check(cudaMemcpy(device_tiles, host.data(), payload_bytes, cudaMemcpyHostToDevice),
          "copy tiles");

    const auto output_major = measure<true>(device_tiles, device_output, tiles_n, tiles_k,
                                            words_per_tile, payload_bytes);
    const auto k_major = measure<false>(device_tiles, device_output, tiles_n, tiles_k,
                                        words_per_tile, payload_bytes);
    std::printf("N=%u K=%u bitrate_half_bits=%u bytes=%zu: %s %.3f ms %.1f GiB/s; "
                "%s %.3f ms %.1f GiB/s; output/k %.3fx\n",
                n, k, bitrate_half_bits, payload_bytes, output_major.order,
                output_major.milliseconds, output_major.gib_per_second, k_major.order,
                k_major.milliseconds, k_major.gib_per_second,
                k_major.milliseconds / output_major.milliseconds);
    check(cudaFree(device_output), "cudaFree(output)");
    check(cudaFree(device_tiles), "cudaFree(tiles)");
}

} // namespace

int main() {
    check(cudaSetDevice(0), "cudaSetDevice");
    benchmark(17408, 5120, 8); // MLP gate/up, 4.0 bits/weight.
    benchmark(17408, 5120, 7); // MLP gate/up, 3.5 bits/weight.
    benchmark(12288, 5120, 8); // Attention query+gate parent.
    benchmark(5120, 6144, 8);  // Attention output projection.
    benchmark_bit_order<7>();
    benchmark_bit_order<8>();
    return 0;
}
