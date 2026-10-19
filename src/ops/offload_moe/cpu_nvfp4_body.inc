// Shared body of the host NVFP4 expert kernels; included by one translation unit per instruction
// set with NINFER_CPU_NVFP4_NS naming its namespace and NINFER_CPU_NVFP4_VNNI 0 or 1. See
// cpu_nvfp4.h for the arithmetic contract. Structure after Basalt's nvfp4_avx2.cpp (MIT): a 64-value
// block is decoded once per row and applied to every token.

#include "ops/offload_moe/cpu_nvfp4.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::cpu::NINFER_CPU_NVFP4_NS {
namespace {

// E4M3 (bias 7) as FP32, one entry per byte; 0x7F and 0xFF (NaN) never occur in a stored bank.
struct E4m3Table {
    float value[256];
    constexpr E4m3Table() : value{} {
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            const std::uint32_t exponent = (byte >> 3) & 0xF;
            const std::uint32_t mantissa = byte & 7;
            float v = 0.0F;
            if (exponent == 0) {
                v = static_cast<float>(mantissa) * (1.0F / 512.0F); // m/8 * 2^-6
            } else {
                v = std::bit_cast<float>(((exponent + 120) << 23) | (mantissa << 20));
            }
            value[byte] = (byte & 0x80) ? -v : v;
        }
    }
};
constexpr E4m3Table kE4m3{};

// e2m1 codes times two, so every value is an int8: {0, .5, 1, 1.5, 2, 3, 4, 6} -> {0..12}.
alignas(16) constexpr std::int8_t kE2m1Twice[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                                     0, -1, -2, -3, -4, -6, -8, -12};

constexpr int kMaxTokens = 4; // tokens per register pass; wider calls loop

// Byte offset of row `row`, 64-value block `block` in a K16M128x4 scale plane whose rows hold
// `blocks` such blocks: its four group scales are bytes 0..3 of that word.
inline std::int64_t scale_word(int row, int block, int blocks) {
    return (static_cast<std::int64_t>(row >> 7) * blocks + block) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4;
}

struct Block {
    __m256i magnitude_a, magnitude_b; // |w| of groups (0,2) and (1,3), lane-ordered like the codes
    __m256i sign_a, sign_b;
    __m256 scale_a, scale_b; // the matching group scales, four lanes each
};

inline Block decode(const std::uint8_t* codes, const std::uint8_t* scale) {
    const __m128i table128 = _mm_load_si128(reinterpret_cast<const __m128i*>(kE2m1Twice));
    const __m256i table    = _mm256_broadcastsi128_si256(table128);
    const __m256i nibble   = _mm256_set1_epi8(0x0F);
    const __m256i packed   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i lo       = _mm256_and_si256(packed, nibble);
    const __m256i hi       = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble);
    Block b;
    // Interleaving the nibbles restores element order per 128-bit lane: A holds elements 0..15 and
    // 32..47 (groups 0 and 2), B holds 16..31 and 48..63 (groups 1 and 3).
    b.sign_a      = _mm256_shuffle_epi8(table, _mm256_unpacklo_epi8(lo, hi));
    b.sign_b      = _mm256_shuffle_epi8(table, _mm256_unpackhi_epi8(lo, hi));
    b.magnitude_a = _mm256_sign_epi8(b.sign_a, b.sign_a);
    b.magnitude_b = _mm256_sign_epi8(b.sign_b, b.sign_b);
    const float s0 = kE4m3.value[scale[0]], s1 = kE4m3.value[scale[1]];
    const float s2 = kE4m3.value[scale[2]], s3 = kE4m3.value[scale[3]];
    b.scale_a = _mm256_setr_ps(s0, s0, s0, s0, s2, s2, s2, s2);
    b.scale_b = _mm256_setr_ps(s1, s1, s1, s1, s3, s3, s3, s3);
    return b;
}

// Four-product int32 sums of |w| * (sign(w) * y) per lane.
inline __m256i dot4(__m256i magnitude, __m256i signed_y) {
#if NINFER_CPU_NVFP4_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), magnitude, signed_y);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude, signed_y), _mm256_set1_epi16(1));
#endif
}

inline float horizontal_sum(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    r        = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r        = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Rows `rows[0..R)` (code row pointers, scale rows, all `blocks` 64-value blocks wide) against
// tokens `y[0..NT)`: sum[k][t] is twice the exact weight-times-activation dot product.
template <int R, int NT>
inline void rows_dot(const std::uint8_t* const* codes, const int* scale_rows,
                     const std::uint8_t* scales, int blocks, const Q8Row* y,
                     float (&sum)[R][kMaxTokens]) {
    __m256 acc_a[R][NT], acc_b[R][NT];
    for (int k = 0; k < R; ++k) {
        for (int t = 0; t < NT; ++t) { acc_a[k][t] = acc_b[k][t] = _mm256_setzero_ps(); }
    }
    for (int block = 0; block < blocks; ++block) {
        Block w[R];
        for (int k = 0; k < R; ++k) {
            w[k] = decode(codes[k] + block * 32, scales + scale_word(scale_rows[k], block, blocks));
        }
        for (int t = 0; t < NT; ++t) {
            const std::int8_t* q = y[t].codes + block * 64;
            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
            const __m256i ya = _mm256_permute2x128_si256(y0, y1, 0x20);
            const __m256i yb = _mm256_permute2x128_si256(y0, y1, 0x31);
            const float d0 = y[t].scales[2 * block], d1 = y[t].scales[2 * block + 1];
            const __m256 d = _mm256_setr_ps(d0, d0, d0, d0, d1, d1, d1, d1);
            for (int k = 0; k < R; ++k) {
                const __m256i pa = dot4(w[k].magnitude_a, _mm256_sign_epi8(ya, w[k].sign_a));
                const __m256i pb = dot4(w[k].magnitude_b, _mm256_sign_epi8(yb, w[k].sign_b));
                acc_a[k][t] =
                    _mm256_fmadd_ps(_mm256_mul_ps(d, w[k].scale_a), _mm256_cvtepi32_ps(pa), acc_a[k][t]);
                acc_b[k][t] =
                    _mm256_fmadd_ps(_mm256_mul_ps(d, w[k].scale_b), _mm256_cvtepi32_ps(pb), acc_b[k][t]);
            }
        }
    }
    for (int k = 0; k < R; ++k) {
        for (int t = 0; t < NT; ++t) {
            sum[k][t] = horizontal_sum(_mm256_add_ps(acc_a[k][t], acc_b[k][t]));
        }
    }
}

template <int NT>
void gate_up_pass(const Nvfp4Expert& e, int r0, int r1, const Q8Row* x, float* const* h) {
    constexpr int kBlocks = kNvfp4Hidden / 64;
    constexpr int kRowBytes = kNvfp4Hidden / 2;
    const float gate_scale = 0.5F * e.inv_gate, up_scale = 0.5F * e.inv_up;
    float sum[2][kMaxTokens];
    for (int r = r0; r < r1; ++r) {
        const int rows[2] = {r, r + kNvfp4Intermediate};
        const std::uint8_t* codes[2] = {e.gate_up_codes + static_cast<std::int64_t>(rows[0]) * kRowBytes,
                                        e.gate_up_codes + static_cast<std::int64_t>(rows[1]) * kRowBytes};
        rows_dot<2, NT>(codes, rows, e.gate_up_scales, kBlocks, x, sum);
        for (int t = 0; t < NT; ++t) {
            const float g = sum[0][t] * gate_scale, u = sum[1][t] * up_scale;
            h[t][r]       = g / (1.0F + std::exp(-g)) * u;
        }
    }
}

template <int NT>
void down_pass(const Nvfp4Expert& e, int r0, int r1, const Q8Row* hq, float* const* out) {
    constexpr int kBlocks = kNvfp4Intermediate / 64;
    constexpr int kRowBytes = kNvfp4Intermediate / 2;
    const float scale = 0.5F * e.inv_down;
    float sum[2][kMaxTokens];
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const int rows[2] = {r, r + 1};
        const std::uint8_t* codes[2] = {e.down_codes + static_cast<std::int64_t>(r) * kRowBytes,
                                        e.down_codes + static_cast<std::int64_t>(r + 1) * kRowBytes};
        rows_dot<2, NT>(codes, rows, e.down_scales, kBlocks, hq, sum);
        for (int t = 0; t < NT; ++t) {
            out[t][r]     = sum[0][t] * scale;
            out[t][r + 1] = sum[1][t] * scale;
        }
    }
    if (r < r1) {
        const int rows[1] = {r};
        const std::uint8_t* codes[1] = {e.down_codes + static_cast<std::int64_t>(r) * kRowBytes};
        float one[1][kMaxTokens];
        rows_dot<1, NT>(codes, rows, e.down_scales, kBlocks, hq, one);
        for (int t = 0; t < NT; ++t) { out[t][r] = one[0][t] * scale; }
    }
}

} // namespace

void gate_up_rows(const Nvfp4Expert& e, int r0, int r1, const Q8Row* x, int tokens,
                  float* const* h) {
    for (int t0 = 0; t0 < tokens; t0 += kMaxTokens) {
        const int n = tokens - t0 < kMaxTokens ? tokens - t0 : kMaxTokens;
        switch (n) {
        case 1: gate_up_pass<1>(e, r0, r1, x + t0, h + t0); break;
        case 2: gate_up_pass<2>(e, r0, r1, x + t0, h + t0); break;
        case 3: gate_up_pass<3>(e, r0, r1, x + t0, h + t0); break;
        default: gate_up_pass<4>(e, r0, r1, x + t0, h + t0); break;
        }
    }
}

void down_rows(const Nvfp4Expert& e, int r0, int r1, const Q8Row* hq, int tokens,
               float* const* out) {
    for (int t0 = 0; t0 < tokens; t0 += kMaxTokens) {
        const int n = tokens - t0 < kMaxTokens ? tokens - t0 : kMaxTokens;
        switch (n) {
        case 1: down_pass<1>(e, r0, r1, hq + t0, out + t0); break;
        case 2: down_pass<2>(e, r0, r1, hq + t0, out + t0); break;
        case 3: down_pass<3>(e, r0, r1, hq + t0, out + t0); break;
        default: down_pass<4>(e, r0, r1, hq + t0, out + t0); break;
        }
    }
}

} // namespace ninfer::ops::cpu::NINFER_CPU_NVFP4_NS
