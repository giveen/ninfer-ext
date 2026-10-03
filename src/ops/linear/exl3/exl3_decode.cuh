#pragma once

// EXL3 mul1 trellis decode shared by the dense linear and the routed-expert kernels. A tile is
// 16*half_bits bytes; every helper returns the integer mul1 codebook value (before the Hadamard
// and scale planes).

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

constexpr std::uint32_t kMul1Multiplier = 0x83DCD12DU;

__device__ __forceinline__ float mul1_value(std::uint16_t state) {
    // The byte sum of the multiplied state is the mul1 codebook; one dp4a replaces six shift/mask/add
    // ops, which matters because every weight of the model passes through here.
    const std::uint32_t product = static_cast<std::uint32_t>(state) * kMul1Multiplier;
    const std::uint32_t sum     = __dp4a(product, 0x01010101U, 0U);
    return static_cast<float>(static_cast<std::int32_t>(sum) - 510);
}

// 16-bit circular window of the tile's tail-biting bitstream that decodes weight `t`.
__device__ __forceinline__ std::uint16_t tile_state(const std::uint8_t* tile, int half_bits,
                                                    int t) {
    const int low   = half_bits >> 1;
    const int count = t + 1;
    const int end   = count * low + ((half_bits & 1) ? count / 2 : 0);
    const int total = 128 * half_bits;
    const int begin = (end + total - 16) % total;
    std::uint32_t state = 0;
    for (int b = 0; b < 16; ++b) {
        const int position = (begin + b) % total;
        state |= (static_cast<std::uint32_t>(tile[position >> 3] >> (position & 7)) & 1U) << b;
    }
    return static_cast<std::uint16_t>(state);
}

// A window's position in the tile, precomputed once per thread and packed into 16 bits (word in the
// low byte, shift in the high byte) so a 16-window plan costs eight registers instead of forty-eight.
// The state index, bitrate and tile width are loop-invariant, so the two integer modulos that located
// the window move out of the K loop entirely; only the two loads and the funnel shift remain.
__device__ __forceinline__ std::uint16_t exl3_window(int state, int bits, int words) {
    const int total = words * 32;
    int start       = (state + 1) * bits - 16;
    if (start < 0) { start += total; }
    return static_cast<std::uint16_t>((start >> 5) | ((start & 31) << 8));
}

__device__ __forceinline__ float exl3_window_value(const std::uint32_t* tile,
                                                   std::uint16_t window, int words) {
    const int word  = window & 0xFF;
    const int shift = (window >> 8) & 0x1F;
    const int next  = (word + 1 == words) ? 0 : word + 1;
    return mul1_value(static_cast<std::uint16_t>(
        __funnelshift_r(tile[word], tile[next], shift) & 0xFFFFU));
}

// The eight B-fragment windows of one 16x16 tile at a 4-bit rate, resolved from the lane's word and
// its circular predecessor with one funnel shift and five bit-field extracts (exllamav3's
// dq8_regs_4bits). d[r] is the window for state 8*lane + r, which is the state the contraction
// indexes. The tile is 32 words, so `lane - 1` wraps inside it.
__device__ __forceinline__ void exl3_windows_4bit(const std::uint32_t* tile, int lane, float d[8]) {
    const std::uint32_t b = tile[lane];
    const std::uint32_t a = tile[(lane + 31) & 31];
    const std::uint32_t s = __funnelshift_r(a, b, 20);
    d[0] = mul1_value(static_cast<std::uint16_t>(s & 0xFFFFU));
    d[1] = mul1_value(static_cast<std::uint16_t>((s >> 4) & 0xFFFFU));
    d[2] = mul1_value(static_cast<std::uint16_t>((s >> 8) & 0xFFFFU));
    d[3] = mul1_value(static_cast<std::uint16_t>(b & 0xFFFFU));
    d[4] = mul1_value(static_cast<std::uint16_t>((b >> 4) & 0xFFFFU));
    d[5] = mul1_value(static_cast<std::uint16_t>((b >> 8) & 0xFFFFU));
    d[6] = mul1_value(static_cast<std::uint16_t>((b >> 12) & 0xFFFFU));
    d[7] = mul1_value(static_cast<std::uint16_t>((b >> 16) & 0xFFFFU));
}

// Odd half bits (KA + 0.5 bpw, mul1): positions alternate KA and KA+1 bits, so four consecutive
// windows share one 18 + 3*KA bit field and a lane's eight windows are two such fields. One funnel
// shift per field yields four windows by shifting the 32-bit result, the structure exllamav3's
// dq8_half uses. The field must fit the funnel result, i.e. 18 + 3*KA <= 32 (up to 4.5 bpw, where
// exllamav3's half-rate kernels stop); wider half rates keep the bitwise decode.
__device__ __forceinline__ bool exl3_half_fast(int half_bits) {
    return (half_bits & 1) != 0 && 18 + 3 * (half_bits >> 1) <= 32;
}

__device__ __forceinline__ void exl3_windows_half(const std::uint32_t* tile, int lane,
                                                  int half_bits, float d[8]) {
    const int ka     = half_bits >> 1;
    const int span   = 18 + 3 * ka;
    const int words  = 4 * half_bits;
    const int total  = words * 32;
    const int off[4] = {0, ka + 1, 2 * ka + 1, 3 * ka + 2};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int t0    = 8 * lane + 4 * g;
        const int end   = (t0 + 4) * ka + (t0 + 4) / 2; // end bit of the group's last window
        int begin       = end - span;
        if (begin < 0) { begin += total; }
        const int w0    = begin >> 5;
        const int shift = begin & 31;
        const int w1    = (w0 + 1 == words) ? 0 : w0 + 1;
        const std::uint32_t field = __funnelshift_r(tile[w0], tile[w1], shift);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            d[4 * g + j] = mul1_value(static_cast<std::uint16_t>((field >> off[j]) & 0xFFFFU));
        }
    }
}


// One lane's view of a 16x16 trellis tile at any rate: the eight codebook values d[r] of states
// 8*lane + r, which are the m16n8k16 B-fragment elements (d[0..3] feed the first n8 half, d[4..7]
// the second). The per-lane window offsets of the even rates are resolved once at construction.
// Raw tile words held in registers between the load and the decode.
struct Exl3TileWords {
    std::uint32_t w[4];
};

struct Exl3TileDecoder {
    int half_bits;
    int bits;
    int words;
    bool fast; // even rate: eight windows from two 64-bit funnel reads
    int wlo0 = 0, wn0 = 0, sft0 = 0, wlo1 = 0, wn1 = 0, sft1 = 0;

    __device__ __forceinline__ Exl3TileDecoder(int half_bits_in, int lane)
        : half_bits(half_bits_in), bits(half_bits_in >> 1), words(4 * half_bits_in),
          fast((half_bits_in & 1) == 0) {
        if (fast && bits != 4) {
            const int total = words * 32;
            const int s0    = ((8 * lane + 1) * bits - 16 + total) % total;
            const int s1    = ((8 * lane + 5) * bits - 16 + total) % total;
            wlo0            = s0 >> 5;
            sft0            = s0 & 31;
            wn0             = (wlo0 + 1 == words) ? 0 : wlo0 + 1;
            wlo1            = s1 >> 5;
            sft1            = s1 & 31;
            wn1             = (wlo1 + 1 == words) ? 0 : wlo1 + 1;
        }
    }

    // `tile` is the tile's 16*half_bits bytes, 4-byte aligned.
    __device__ __forceinline__ void decode(const std::uint8_t* tile, int lane, float d[8]) const {
        const auto* tile32 = reinterpret_cast<const std::uint32_t*>(tile);
        if (half_bits == 8) {
            exl3_windows_4bit(tile32, lane, d);
        } else if (fast) {
            const unsigned long long w0 =
                (static_cast<unsigned long long>(tile32[wn0]) << 32) | tile32[wlo0];
            const unsigned long long w1 =
                (static_cast<unsigned long long>(tile32[wn1]) << 32) | tile32[wlo1];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[i] = mul1_value(static_cast<std::uint16_t>((w0 >> (sft0 + bits * i)) & 0xFFFFU));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[4 + i] =
                    mul1_value(static_cast<std::uint16_t>((w1 >> (sft1 + bits * i)) & 0xFFFFU));
            }
        } else if (exl3_half_fast(half_bits)) {
            exl3_windows_half(tile32, lane, half_bits, d);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) { d[r] = mul1_value(tile_state(tile, half_bits, 8 * lane + r)); }
        }
    }

    // The words one lane reads of a tile at a prefetchable rate: the 4-bit rate (kHB == 8) reads its
    // own word and its circular predecessor; other even rates read the four words of their two
    // 64-bit funnel windows. Loading them ahead of the decode lets many tiles' loads overlap.
    template <int kHB>
    __device__ __forceinline__ Exl3TileWords load_words(const std::uint8_t* tile, int lane) const {
        const auto* tile32 = reinterpret_cast<const std::uint32_t*>(tile);
        Exl3TileWords r{};
        if constexpr (kHB == 8) {
            r.w[0] = tile32[lane];
            r.w[1] = tile32[(lane + 31) & 31];
        } else {
            r.w[0] = tile32[wlo0];
            r.w[1] = tile32[wn0];
            r.w[2] = tile32[wlo1];
            r.w[3] = tile32[wn1];
        }
        return r;
    }

    template <int kHB>
    __device__ __forceinline__ void decode_words(const Exl3TileWords& r, float d[8]) const {
        if constexpr (kHB == 8) {
            const std::uint32_t b = r.w[0];
            const std::uint32_t s = __funnelshift_r(r.w[1], b, 20);
            d[0] = mul1_value(static_cast<std::uint16_t>(s & 0xFFFFU));
            d[1] = mul1_value(static_cast<std::uint16_t>((s >> 4) & 0xFFFFU));
            d[2] = mul1_value(static_cast<std::uint16_t>((s >> 8) & 0xFFFFU));
            d[3] = mul1_value(static_cast<std::uint16_t>(b & 0xFFFFU));
            d[4] = mul1_value(static_cast<std::uint16_t>((b >> 4) & 0xFFFFU));
            d[5] = mul1_value(static_cast<std::uint16_t>((b >> 8) & 0xFFFFU));
            d[6] = mul1_value(static_cast<std::uint16_t>((b >> 12) & 0xFFFFU));
            d[7] = mul1_value(static_cast<std::uint16_t>((b >> 16) & 0xFFFFU));
        } else {
            const unsigned long long w0 = (static_cast<unsigned long long>(r.w[1]) << 32) | r.w[0];
            const unsigned long long w1 = (static_cast<unsigned long long>(r.w[3]) << 32) | r.w[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[i] = mul1_value(static_cast<std::uint16_t>((w0 >> (sft0 + bits * i)) & 0xFFFFU));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[4 + i] =
                    mul1_value(static_cast<std::uint16_t>((w1 >> (sft1 + bits * i)) & 0xFFFFU));
            }
        }
    }
};

} // namespace ninfer::ops::detail
