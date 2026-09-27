#include "quantize/exl3/trellis_encoder.h"

#include "core/device.h"

#include <cfloat>
#include <stdexcept>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kStateBits      = 16;
constexpr int kThreads        = 512;
constexpr int kLength         = kTrellisTileWeights;
constexpr std::uint32_t kMul1 = 0x83DCD12DU;

// Per-block scratch: two FP32 cost buffers indexed by the carry node (the part of a state that
// continues into the next one) and one back-pointer byte per node and step.
struct ScratchLayout {
    std::size_t nodes;       // 2^(16 - smallest step width)
    std::size_t cost_bytes;  // both cost buffers
    std::size_t back_bytes;  // kLength * nodes
    std::size_t block_bytes; // aligned total
};

ScratchLayout scratch_layout(int bitrate_half_bits) {
    if (bitrate_half_bits < 2 || bitrate_half_bits > 16) {
        throw std::invalid_argument("EXL3 trellis encoder bitrate_half_bits must be in [2, 16]");
    }
    const int min_width = bitrate_half_bits / 2;
    ScratchLayout layout{};
    layout.nodes       = std::size_t{1} << (kStateBits - min_width);
    layout.cost_bytes  = 2 * layout.nodes * sizeof(float);
    layout.back_bytes  = static_cast<std::size_t>(kLength) * layout.nodes;
    layout.block_bytes = (layout.cost_bytes + layout.back_bytes + 255) / 256 * 256;
    return layout;
}

__device__ __forceinline__ int step_width(int half_bits, int position) {
    return (half_bits >> 1) + ((half_bits & 1) & position);
}

__device__ __forceinline__ float mul1_value(std::uint32_t state, float scale) {
    const std::uint32_t product = state * kMul1;
    const int sum               = static_cast<int>(__dp4a(product, 0x01010101U, 0U));
    return static_cast<float>(sum - 510) * scale;
}

// One Viterbi pass over the ring in order roll, roll + 1, ... (mod 256). With overlap >= 0 the
// first state must continue from it and the ring closes onto it. Leaves the states in `out`.
__device__ void viterbi_pass(const float* target, int half_bits, float scale, int roll, int overlap,
                             float* cost_a, float* cost_b, std::uint8_t* back,
                             std::size_t nodes_capacity, std::uint16_t* out) {
    const int first_width = step_width(half_bits, roll);
    const int first_nodes = 1 << (kStateBits - first_width);
    for (int n = threadIdx.x; n < first_nodes; n += blockDim.x) {
        cost_a[n] = overlap < 0 || n == overlap ? 0.0f : FLT_MAX;
    }
    __syncthreads();

    float* previous = cost_a;
    float* next     = cost_b;
    for (int step = 0; step < kLength; ++step) {
        const int position             = (step + roll) & (kLength - 1);
        const int width                = step_width(half_bits, position);
        const int next_width           = step_width(half_bits, (position + 1) & (kLength - 1));
        const std::uint32_t carry_mask = (1U << (kStateBits - width)) - 1U;
        const int nodes                = 1 << (kStateBits - next_width);
        const float x                  = target[position];
        std::uint8_t* back_step        = back + static_cast<std::size_t>(step) * nodes_capacity;
        for (int n = threadIdx.x; n < nodes; n += blockDim.x) {
            float best               = FLT_MAX;
            int best_d               = 0;
            const std::uint32_t base = static_cast<std::uint32_t>(n) << next_width;
            for (int d = 0; d < (1 << next_width); ++d) {
                const std::uint32_t s = base | static_cast<std::uint32_t>(d);
                const float err       = mul1_value(s, scale) - x;
                const float c         = previous[s & carry_mask] + err * err;
                if (c < best) {
                    best   = c;
                    best_d = d;
                }
            }
            next[n]      = best;
            back_step[n] = static_cast<std::uint8_t>(best_d);
        }
        __syncthreads();
        float* t = previous;
        previous = next;
        next     = t;
    }

    // Pick the final carry node, then trace back on one thread.
    __shared__ float s_best[kThreads];
    __shared__ int s_node[kThreads];
    int final_node = overlap;
    if (overlap < 0) {
        const int nodes = 1 << (kStateBits - first_width);
        float best      = FLT_MAX;
        int node        = 0;
        for (int n = threadIdx.x; n < nodes; n += blockDim.x) {
            if (previous[n] < best) {
                best = previous[n];
                node = n;
            }
        }
        s_best[threadIdx.x] = best;
        s_node[threadIdx.x] = node;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other = s_best[threadIdx.x + stride];
                const int other_n = s_node[threadIdx.x + stride];
                if (other < s_best[threadIdx.x] ||
                    (other == s_best[threadIdx.x] && other_n < s_node[threadIdx.x])) {
                    s_best[threadIdx.x] = other;
                    s_node[threadIdx.x] = other_n;
                }
            }
            __syncthreads();
        }
        final_node = s_node[0];
    }
    if (threadIdx.x == 0) {
        std::uint32_t node = static_cast<std::uint32_t>(final_node);
        for (int step = kLength - 1; step >= 0; --step) {
            const int position    = (step + roll) & (kLength - 1);
            const int width       = step_width(half_bits, position);
            const int next_width  = step_width(half_bits, (position + 1) & (kLength - 1));
            const std::uint8_t d  = back[static_cast<std::size_t>(step) * nodes_capacity + node];
            const std::uint32_t s = (node << next_width) | d;
            out[position]         = static_cast<std::uint16_t>(s);
            node                  = s & ((1U << (kStateBits - width)) - 1U);
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kThreads)
    encode_tiles_kernel(const float* __restrict__ tiles, std::int64_t count, int half_bits,
                        float scale, std::uint16_t* __restrict__ states,
                        float* __restrict__ decoded, std::uint8_t* __restrict__ scratch,
                        std::size_t block_bytes, std::size_t nodes, std::size_t cost_bytes) {
    __shared__ float target[kLength];
    __shared__ std::uint16_t ring[kLength];
    std::uint8_t* block_scratch = scratch + static_cast<std::size_t>(blockIdx.x) * block_bytes;
    auto* cost_a                = reinterpret_cast<float*>(block_scratch);
    auto* cost_b                = cost_a + nodes;
    auto* back                  = block_scratch + cost_bytes;

    for (std::int64_t tile = blockIdx.x; tile < count; tile += gridDim.x) {
        const float* input = tiles + tile * kLength;
        for (int i = threadIdx.x; i < kLength; i += blockDim.x) { target[i] = input[i]; }
        __syncthreads();

        viterbi_pass(target, half_bits, scale, kLength / 2, -1, cost_a, cost_b, back, nodes, ring);
        const int first_width = step_width(half_bits, 0);
        const int overlap     = ring[kLength - 1] >> first_width;
        viterbi_pass(target, half_bits, scale, 0, overlap, cost_a, cost_b, back, nodes, ring);

        for (int i = threadIdx.x; i < kLength; i += blockDim.x) {
            states[tile * kLength + i]  = ring[i];
            decoded[tile * kLength + i] = mul1_value(ring[i], scale);
        }
        __syncthreads();
    }
}

} // namespace

std::size_t trellis_encoder_scratch_bytes(int blocks, int bitrate_half_bits) {
    if (blocks <= 0) { throw std::invalid_argument("EXL3 trellis encoder needs blocks >= 1"); }
    return scratch_layout(bitrate_half_bits).block_bytes * static_cast<std::size_t>(blocks);
}

void encode_trellis_tiles(const float* tiles, std::int64_t count, int bitrate_half_bits,
                          float scale, std::uint16_t* states, float* decoded, void* scratch,
                          std::size_t scratch_bytes, int blocks, cudaStream_t stream) {
    const ScratchLayout layout = scratch_layout(bitrate_half_bits);
    if (count < 0 || blocks <= 0 ||
        scratch_bytes < trellis_encoder_scratch_bytes(blocks, bitrate_half_bits)) {
        throw std::invalid_argument("EXL3 trellis encoder: invalid count, blocks or scratch");
    }
    if (count == 0) { return; }
    const auto grid = static_cast<unsigned>(blocks < count ? blocks : count);
    encode_tiles_kernel<<<grid, kThreads, 0, stream>>>(
        tiles, count, bitrate_half_bits, scale, states, decoded,
        static_cast<std::uint8_t*>(scratch), layout.block_bytes, layout.nodes, layout.cost_bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
