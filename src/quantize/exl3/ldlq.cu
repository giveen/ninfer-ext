#include "quantize/exl3/ldlq.h"

#include "core/device.h"
#include "quantize/exl3/trellis_encoder.h"

#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kStrip    = 16;
constexpr int kCompCols = 64;

struct Scratch {
    float* comp;         // [16][n]
    float* target;       // [tiles_n][256]
    float* decoded;      // [tiles_n][256]
    std::uint16_t* ring; // [tiles_n][256]
    void* encoder;
    std::size_t encoder_bytes;
    int encoder_blocks;
};

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int encoder_blocks(std::int64_t n) {
    const int device_blocks = trellis_encoder_blocks();
    return static_cast<int>(n / 16 < device_blocks ? n / 16 : device_blocks);
}

Scratch carve(void* base, std::int64_t n, int half_bits) {
    auto* p       = static_cast<std::uint8_t*>(base);
    const auto tn = static_cast<std::size_t>(n / 16);
    Scratch s{};
    s.comp = reinterpret_cast<float*>(p);
    p += align256(kStrip * static_cast<std::size_t>(n) * sizeof(float));
    s.target = reinterpret_cast<float*>(p);
    p += align256(tn * 256 * sizeof(float));
    s.decoded = reinterpret_cast<float*>(p);
    p += align256(tn * 256 * sizeof(float));
    s.ring = reinterpret_cast<std::uint16_t*>(p);
    p += align256(tn * 256 * sizeof(std::uint16_t));
    s.encoder        = p;
    s.encoder_blocks = encoder_blocks(n);
    s.encoder_bytes  = trellis_encoder_scratch_bytes(s.encoder_blocks, half_bits);
    return s;
}

// comp[i][c] = sum over rows r >= r0 of L[r][col0 + i] * (W - Wq)[r][c]; one CTA per 64 columns.
__global__ void __launch_bounds__(256)
    compensation_kernel(const float* __restrict__ w, const float* __restrict__ wq,
                        const float* __restrict__ l, std::int64_t k, std::int64_t n,
                        std::int64_t col0, std::int64_t r0, float* __restrict__ comp) {
    __shared__ float ls[kStrip][kStrip + 1];
    __shared__ float es[kStrip][kCompCols];
    const std::int64_t c0 = static_cast<std::int64_t>(blockIdx.x) * kCompCols;
    const int tc          = static_cast<int>(threadIdx.x % kCompCols);
    const int ti          = static_cast<int>(threadIdx.x / kCompCols); // 0..3, rows ti, ti+4, ...
    float acc[4]          = {};
    for (std::int64_t rb = r0; rb < k; rb += kStrip) {
        for (int i = static_cast<int>(threadIdx.x); i < kStrip * kStrip; i += blockDim.x) {
            const std::int64_t r       = rb + i / kStrip;
            ls[i / kStrip][i % kStrip] = r < k ? l[r * k + col0 + i % kStrip] : 0.0f;
        }
        for (int i = static_cast<int>(threadIdx.x); i < kStrip * kCompCols; i += blockDim.x) {
            const std::int64_t r             = rb + i / kCompCols;
            const std::int64_t c             = c0 + i % kCompCols;
            es[i / kCompCols][i % kCompCols] = r < k ? w[r * n + c] - wq[r * n + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < kStrip; ++m) {
            const float e = es[m][tc];
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[j] = fmaf(ls[m][ti + 4 * j], e, acc[j]); }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) { comp[(ti + 4 * j) * n + c0 + tc] = acc[j]; }
}

// target tiles of strip b: W[b rows] + comp, gathered into fragment order.
__global__ void gather_kernel(const float* __restrict__ w, const float* __restrict__ comp,
                              std::int64_t n, std::int64_t row0, float* __restrict__ target) {
    const std::int64_t tile = blockIdx.x;
    const int t             = static_cast<int>(threadIdx.x);
    const int kk            = tile_k(t);
    const std::int64_t c    = tile * 16 + tile_n(t);
    target[tile * 256 + t]  = w[(row0 + kk) * n + c] + comp[kk * n + c];
}

// Decoded values back into Wq rows of strip b; states into the output-major tile layout.
__global__ void scatter_kernel(const float* __restrict__ decoded,
                               const std::uint16_t* __restrict__ ring, std::int64_t n,
                               std::int64_t tiles_k, std::int64_t strip, float* __restrict__ wq,
                               std::uint16_t* __restrict__ states) {
    const std::int64_t tile                                      = blockIdx.x;
    const int t                                                  = static_cast<int>(threadIdx.x);
    wq[(strip * kStrip + tile_k(t)) * n + tile * 16 + tile_n(t)] = decoded[tile * 256 + t];
    states[(tile * tiles_k + strip) * 256 + t]                   = ring[tile * 256 + t];
}

} // namespace

std::size_t ldlq_scratch_bytes(std::int64_t k, std::int64_t n, int bitrate_half_bits) {
    if (k <= 0 || n <= 0 || k % kStrip || n % 128) {
        throw std::invalid_argument("LDLQ needs k % 16 == 0 and n % 128 == 0");
    }
    const auto tn = static_cast<std::size_t>(n / 16);
    return align256(kStrip * static_cast<std::size_t>(n) * sizeof(float)) +
           2 * align256(tn * 256 * sizeof(float)) + align256(tn * 256 * sizeof(std::uint16_t)) +
           trellis_encoder_scratch_bytes(encoder_blocks(n), bitrate_half_bits);
}

void ldlq_quantize(const float* w, const float* l, std::int64_t k, std::int64_t n,
                   int bitrate_half_bits, float scale, std::uint16_t* states, float* wq,
                   void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    if (scratch_bytes < ldlq_scratch_bytes(k, n, bitrate_half_bits)) {
        throw std::invalid_argument("LDLQ scratch is too small");
    }
    const Scratch s            = carve(scratch, n, bitrate_half_bits);
    const std::int64_t tiles_k = k / kStrip;
    const std::int64_t tiles_n = n / 16;
    const auto comp_blocks     = static_cast<unsigned>(n / kCompCols);
    for (std::int64_t strip = tiles_k - 1; strip >= 0; --strip) {
        const std::int64_t row0 = strip * kStrip;
        compensation_kernel<<<comp_blocks, 256, 0, stream>>>(w, wq, l, k, n, row0, row0 + kStrip,
                                                             s.comp);
        gather_kernel<<<static_cast<unsigned>(tiles_n), 256, 0, stream>>>(w, s.comp, n, row0,
                                                                          s.target);
        encode_trellis_tiles(s.target, tiles_n, bitrate_half_bits, scale, s.ring, s.decoded,
                             s.encoder, s.encoder_bytes, s.encoder_blocks, stream);
        scatter_kernel<<<static_cast<unsigned>(tiles_n), 256, 0, stream>>>(
            s.decoded, s.ring, n, tiles_k, strip, wq, states);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
