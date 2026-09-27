#include "quantize/exl3/trellis_encoder.h"

#include "core/device.h"

#include <cfloat>
#include <stdexcept>
#include <utility>

namespace ninfer::quantize::exl3 {
namespace {

constexpr int kStateBits      = 16;
constexpr int kThreads        = 1024;
constexpr int kLength         = kTrellisTileWeights;
constexpr std::uint32_t kMul1 = 0x83DCD12DU;
// Both FP32 cost buffers fit in shared memory from a smallest step of 3 bits (2 x 8192 floats);
// 2-bit smallest steps use one 16384-float buffer updated in place.
constexpr int kInPlaceWidth = 2;

// Per-block scratch: one back-pointer byte per carry node and step, plus the FP32 cost buffers
// indexed by carry node when they do not fit in shared memory (1-bit smallest steps).
struct ScratchLayout {
    std::size_t nodes;       // 2^(16 - smallest step width)
    bool shared_costs;       // cost buffers in shared memory
    std::size_t cost_bytes;  // both cost buffers (global scratch or shared memory)
    std::size_t back_bytes;  // kLength * nodes
    std::size_t block_bytes; // aligned global scratch per block
};

ScratchLayout scratch_layout(int bitrate_half_bits) {
    if (bitrate_half_bits < 2 || bitrate_half_bits > 16) {
        throw std::invalid_argument("EXL3 trellis encoder bitrate_half_bits must be in [2, 16]");
    }
    const int min_width = bitrate_half_bits / 2;
    ScratchLayout layout{};
    layout.nodes             = std::size_t{1} << (kStateBits - min_width);
    layout.shared_costs      = min_width >= kInPlaceWidth;
    layout.cost_bytes        = (min_width == kInPlaceWidth ? 1 : 2) * layout.nodes * sizeof(float);
    layout.back_bytes        = static_cast<std::size_t>(kLength) * layout.nodes;
    const std::size_t global = layout.back_bytes + (layout.shared_costs ? 0 : layout.cost_bytes);
    layout.block_bytes       = (global + 255) / 256 * 256;
    return layout;
}

__host__ __device__ constexpr int step_width(int half_bits, int position) {
    return (half_bits >> 1) + ((half_bits & 1) & position);
}

__device__ __forceinline__ int mul1_integer(std::uint32_t product) {
    return static_cast<int>(__dp4a(product, 0x01010101U, 0U)) - 510;
}

__device__ __forceinline__ float mul1_value(std::uint32_t state, float scale) {
    return static_cast<float>(mul1_integer(state * kMul1)) * scale;
}

// One trellis step at a weight of width W followed by one of width V. The new state is
// s = (n << V) | d, where n = s >> V is the carry node into the next step and d the V bits it
// drops; s continues from the previous carry node c = s & (2^(16-W) - 1). Writing
// n = (hi << L) | lo with L = 16 - W - V and hi the W bits of n outside c gives c = (lo << V) | d.
// Work item i maps to hi = i mod 2^W, lo = i >> W, so a warp's cost reads are broadcasts or
// conflict-free; back pointers are stored by item so their writes coalesce.
template <int W, int V>
__device__ __forceinline__ void
trellis_step(const float* __restrict__ previous, float* __restrict__ next,
             std::uint8_t* __restrict__ back_step, float x, float scale) {
    constexpr int kItems   = 1 << (kStateBits - V);
    constexpr int kLoBits  = kStateBits - W - V;
    constexpr int kChoices = 1 << V;
#pragma unroll
    for (int base = 0; base < kItems; base += kThreads) {
        const int i = base + static_cast<int>(threadIdx.x);
        if (kItems < kThreads && i >= kItems) { break; }
        const std::uint32_t hi = static_cast<std::uint32_t>(i) & ((1U << W) - 1U);
        const std::uint32_t lo = static_cast<std::uint32_t>(i) >> W;
        const std::uint32_t n  = (hi << kLoBits) | lo;
        const float* carry     = previous + (lo << V);
        std::uint32_t product  = (n << V) * kMul1;
        float best             = FLT_MAX;
        int best_d             = 0;
        // value - x = bytesum * scale - (510 * scale + x)
        const float bias     = fmaf(510.0f, scale, x);
        const auto candidate = [&](int d, float previous_cost) {
            const float err =
                fmaf(static_cast<float>(__dp4a(product, 0x01010101U, 0U)), scale, -bias);
            const float c = fmaf(err, err, previous_cost);
            if (c < best) {
                best   = c;
                best_d = d;
            }
            product += kMul1;
        };
        if constexpr (kChoices >= 4) {
#pragma unroll 4
            for (int d = 0; d < kChoices; d += 4) {
                const float4 costs = *reinterpret_cast<const float4*>(carry + d);
                candidate(d, costs.x);
                candidate(d + 1, costs.y);
                candidate(d + 2, costs.z);
                candidate(d + 3, costs.w);
            }
        } else {
#pragma unroll
            for (int d = 0; d < kChoices; ++d) { candidate(d, carry[d]); }
        }
        next[n]      = best;
        back_step[i] = static_cast<std::uint8_t>(best_d);
    }
}

// The same step updating one cost buffer in place, for 2-bit smallest steps whose two buffers
// would not fit in shared memory. Each carry group lo feeds only the 2^W outputs (hi << L) | lo,
// so a thread loads its groups' 2^V carries into registers, the block synchronizes, and the
// outputs overwrite the buffer. Needs 2^L to be a multiple of the block size.
template <int W, int V>
__device__ __forceinline__ void trellis_step_in_place(float* __restrict__ costs,
                                                      std::uint8_t* __restrict__ back_step, float x,
                                                      float scale) {
    constexpr int kLoBits  = kStateBits - W - V;
    constexpr int kGroups  = (1 << kLoBits) / kThreads;
    constexpr int kChoices = 1 << V;
    static_assert(kGroups >= 1 && (1 << kLoBits) % kThreads == 0);
    float carry[kGroups][kChoices];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        const std::uint32_t lo = static_cast<std::uint32_t>(threadIdx.x) + g * kThreads;
#pragma unroll
        for (int d = 0; d < kChoices; d += 4) {
            const float4 c4 = *reinterpret_cast<const float4*>(costs + (lo << V) + d);
            carry[g][d]     = c4.x;
            carry[g][d + 1] = c4.y;
            carry[g][d + 2] = c4.z;
            carry[g][d + 3] = c4.w;
        }
    }
    __syncthreads();
    const float bias = fmaf(510.0f, scale, x); // value - x = bytesum * scale - bias
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        const std::uint32_t lo = static_cast<std::uint32_t>(threadIdx.x) + g * kThreads;
#pragma unroll
        for (std::uint32_t hi = 0; hi < (1U << W); ++hi) {
            const std::uint32_t n = (hi << kLoBits) | lo;
            std::uint32_t product = (n << V) * kMul1;
            float best            = FLT_MAX;
            int best_d            = 0;
#pragma unroll
            for (int d = 0; d < kChoices; ++d) {
                const float err =
                    fmaf(static_cast<float>(__dp4a(product, 0x01010101U, 0U)), scale, -bias);
                const float c = fmaf(err, err, carry[g][d]);
                if (c < best) {
                    best   = c;
                    best_d = d;
                }
                product += kMul1;
            }
            costs[n]                  = best;
            back_step[(lo << W) | hi] = static_cast<std::uint8_t>(best_d);
        }
    }
}

// One Viterbi pass over the ring in order roll, roll + 1, ... (mod 256). With overlap >= 0 the
// first state must continue from it and the ring closes onto it. Leaves the states in `out`.
template <int HalfBits>
__device__ void viterbi_pass(const float* target, float scale, int roll, int overlap, float* cost_a,
                             float* cost_b, std::uint8_t* back, std::size_t nodes_capacity,
                             std::uint16_t* out) {
    constexpr int kLow    = HalfBits / 2;
    constexpr int kHigh   = kLow + (HalfBits & 1);
    const int first_width = step_width(HalfBits, roll);
    const int first_nodes = 1 << (kStateBits - first_width);
    for (int n = threadIdx.x; n < first_nodes; n += blockDim.x) {
        cost_a[n] = overlap < 0 || n == overlap ? 0.0f : FLT_MAX;
    }
    __syncthreads();

    float* previous = cost_a;
    float* next     = cost_b;
    for (int step = 0; step < kLength; ++step) {
        const int position      = (step + roll) & (kLength - 1);
        std::uint8_t* back_step = back + static_cast<std::size_t>(step) * nodes_capacity;
        // Positions alternate parity around the ring (256 is even), so odd rates alternate
        // (high, low) and (low, high) steps.
        if constexpr (kLow == kInPlaceWidth) {
            if constexpr (kLow == kHigh) {
                trellis_step_in_place<kLow, kLow>(previous, back_step, target[position], scale);
            } else if (position & 1) {
                trellis_step_in_place<kHigh, kLow>(previous, back_step, target[position], scale);
            } else {
                trellis_step_in_place<kLow, kHigh>(previous, back_step, target[position], scale);
            }
            __syncthreads();
            continue;
        } else if constexpr (kLow == kHigh) {
            trellis_step<kLow, kLow>(previous, next, back_step, target[position], scale);
        } else if (position & 1) {
            trellis_step<kHigh, kLow>(previous, next, back_step, target[position], scale);
        } else {
            trellis_step<kLow, kHigh>(previous, next, back_step, target[position], scale);
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
        float best = FLT_MAX;
        int node   = 0;
        for (int n = threadIdx.x; n < first_nodes; n += blockDim.x) {
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
            const int position   = (step + roll) & (kLength - 1);
            const int width      = step_width(HalfBits, position);
            const int next_width = step_width(HalfBits, (position + 1) & (kLength - 1));
            const int lo_bits    = kStateBits - width - next_width;
            const std::uint32_t item =
                ((node & ((1U << lo_bits) - 1U)) << width) | (node >> lo_bits);
            const std::uint8_t d  = back[static_cast<std::size_t>(step) * nodes_capacity + item];
            const std::uint32_t s = (node << next_width) | d;
            out[position]         = static_cast<std::uint16_t>(s);
            node                  = s & ((1U << (kStateBits - width)) - 1U);
        }
    }
    __syncthreads();
}

template <int HalfBits>
__global__ void __launch_bounds__(kThreads)
    encode_tiles_kernel(const float* __restrict__ tiles, std::int64_t count, float scale,
                        std::uint16_t* __restrict__ states, float* __restrict__ decoded,
                        std::uint8_t* __restrict__ scratch, std::size_t block_bytes,
                        std::size_t nodes) {
    constexpr bool kSharedCosts = HalfBits / 2 >= kInPlaceWidth;
    constexpr bool kInPlace     = HalfBits / 2 == kInPlaceWidth;
    extern __shared__ float shared_costs[];
    __shared__ float target[kLength];
    __shared__ std::uint16_t ring[kLength];
    std::uint8_t* block_scratch = scratch + static_cast<std::size_t>(blockIdx.x) * block_bytes;
    float* cost_a               = nullptr;
    std::uint8_t* back          = block_scratch;
    if constexpr (kSharedCosts) {
        cost_a = shared_costs;
    } else {
        cost_a = reinterpret_cast<float*>(block_scratch);
        back   = block_scratch + 2 * nodes * sizeof(float);
    }
    float* cost_b = kInPlace ? cost_a : cost_a + nodes;

    for (std::int64_t tile = blockIdx.x; tile < count; tile += gridDim.x) {
        const float* input = tiles + tile * kLength;
        for (int i = threadIdx.x; i < kLength; i += blockDim.x) { target[i] = input[i]; }
        __syncthreads();

        viterbi_pass<HalfBits>(target, scale, kLength / 2, -1, cost_a, cost_b, back, nodes, ring);
        const int overlap = ring[kLength - 1] >> step_width(HalfBits, 0);
        viterbi_pass<HalfBits>(target, scale, 0, overlap, cost_a, cost_b, back, nodes, ring);

        for (int i = threadIdx.x; i < kLength; i += blockDim.x) {
            states[tile * kLength + i]  = ring[i];
            decoded[tile * kLength + i] = mul1_value(ring[i], scale);
        }
        __syncthreads();
    }
}

template <int HalfBits>
void launch(const float* tiles, std::int64_t count, float scale, std::uint16_t* states,
            float* decoded, std::uint8_t* scratch, const ScratchLayout& layout, unsigned grid,
            cudaStream_t stream) {
    const int shared = layout.shared_costs ? static_cast<int>(layout.cost_bytes) : 0;
    if (shared > 0) {
        CUDA_CHECK(cudaFuncSetAttribute(encode_tiles_kernel<HalfBits>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    }
    encode_tiles_kernel<HalfBits><<<grid, kThreads, shared, stream>>>(
        tiles, count, scale, states, decoded, scratch, layout.block_bytes, layout.nodes);
}

template <int... Rates>
void dispatch(int half_bits, std::integer_sequence<int, Rates...>, const float* tiles,
              std::int64_t count, float scale, std::uint16_t* states, float* decoded,
              std::uint8_t* scratch, const ScratchLayout& layout, unsigned grid,
              cudaStream_t stream) {
    const bool launched =
        ((half_bits == Rates + 2 ? (launch<Rates + 2>(tiles, count, scale, states, decoded, scratch,
                                                      layout, grid, stream),
                                    true)
                                 : false) ||
         ...);
    if (!launched) { throw std::invalid_argument("EXL3 trellis encoder: unsupported rate"); }
}

} // namespace

int trellis_encoder_blocks() {
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms; // one 1024-thread block per SM
}

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
    dispatch(bitrate_half_bits, std::make_integer_sequence<int, 15>{}, tiles, count, scale, states,
             decoded, static_cast<std::uint8_t*>(scratch), layout, grid, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::quantize::exl3
