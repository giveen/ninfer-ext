#pragma once

// The Mamba-2 selective scan evaluated naively in FP64, with the discrimination variants the test
// needs to prove that a case *pins* a decision rather than merely agreeing with it.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::test::mamba2_ref {

// `torch.nn.functional.softplus` with its default beta=1 and threshold=20 returns its input above
// the threshold, and this tree's device helper carries the same plateau, so the oracle does too: a
// recurrence that agreed only in the limit would disagree with the reference on a saturated step.
inline double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

// Round to nearest even on the BF16 grid, which is what a BF16 state image would store.
inline float round_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return value; }
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    float rounded;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

/**
 * Alternatives the contract rejects, evaluated from the same buffers. Each one is silent in the
 * sense that it produces a finite, plausibly-scaled result: a case whose correct and wrong readings
 * differ by a wide margin is a case that discriminates between them.
 *
 *   * `swap_state_extents` reads and writes the state with the value extent as its fastest axis
 *     (`value + value_dim * key`) instead of the key extent (`key + 128 * value`). The plane is
 *     `128 * value_dim` elements either way, so nothing else about the call changes.
 *   * `group_modulo` maps head `h` to group `h % groups` instead of `h / (heads / groups)`, which
 *     is the mixer's `repeat_interleave` of `n_groups` groups over the head axis.
 *   * `scaled_readout` multiplies the state contraction by `1/sqrt(128)`, the scale the tree's
 *     gated-linear-attention Ops lock their readout to. Mamba-2's is unscaled.
 *   * `bf16_state` rounds every state plane to the BF16 grid on load and after every token's
 *     update, which is what a lower-precision state image would do.
 */
struct Variant {
    bool swap_state_extents = false;
    bool group_modulo       = false;
    bool scaled_readout     = false;
    bool bf16_state         = false;
};

struct Inputs {
    std::int32_t value_dim = 0;
    std::int32_t heads     = 0;
    std::int32_t groups    = 0;
    std::int32_t tokens    = 0;
    float time_step_min    = 0.0F;

    // The exact FP32 values represented by the public BF16 tensors.
    std::vector<float> x;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> dt;
    // The exact public FP32 parameters and state.
    std::vector<float> dt_bias;
    std::vector<float> a_log;
    std::vector<float> d_skip;
    std::vector<float> state;
};

struct Result {
    std::vector<double> out;
    std::vector<double> final_state;
};

inline Result evaluate(const Inputs& in, const Variant& variant = {}) {
    constexpr std::int64_t kStateSize = 128;
    const std::int64_t V              = in.value_dim;
    const std::int64_t H              = in.heads;
    const std::int64_t G              = in.groups;
    const std::int64_t T              = in.tokens;
    if (V <= 0 || H <= 0 || G <= 0 || T <= 0 || H % G != 0) {
        throw std::invalid_argument("mamba2_ref: invalid geometry");
    }

    const std::size_t vector_size = static_cast<std::size_t>(V * H * T);
    const std::size_t group_size  = static_cast<std::size_t>(kStateSize * G * T);
    const std::size_t state_size  = static_cast<std::size_t>(kStateSize * V * H);
    if (in.x.size() != vector_size || in.b.size() != group_size || in.c.size() != group_size ||
        in.dt.size() != static_cast<std::size_t>(H * T) ||
        in.dt_bias.size() != static_cast<std::size_t>(H) ||
        in.a_log.size() != static_cast<std::size_t>(H) ||
        in.d_skip.size() != static_cast<std::size_t>(H) || in.state.size() != state_size) {
        throw std::invalid_argument("mamba2_ref: input size does not match geometry");
    }

    Result result;
    result.out.assign(vector_size, 0.0);
    result.final_state.assign(state_size, 0.0);

    // One head's state plane, `[value_dim, 128]` with the key extent fastest. The recurrence is
    // evaluated in the reference's own order: a full decay-and-rank-one pass over the plane, then
    // the readout contraction.
    std::vector<double> plane(static_cast<std::size_t>(kStateSize * V));
    for (std::int64_t head = 0; head < H; ++head) {
        const std::int64_t plane_base = head * V * kStateSize;
        for (std::int64_t value = 0; value < V; ++value) {
            for (std::int64_t key = 0; key < kStateSize; ++key) {
                const std::int64_t offset =
                    variant.swap_state_extents ? value + V * key : key + kStateSize * value;
                double loaded =
                    static_cast<double>(in.state[static_cast<std::size_t>(plane_base + offset)]);
                if (variant.bf16_state) { loaded = static_cast<double>(round_bf16(float(loaded))); }
                plane[static_cast<std::size_t>(offset)] = loaded;
            }
        }

        const double a = -std::exp(static_cast<double>(in.a_log[static_cast<std::size_t>(head)]));
        const double skip        = static_cast<double>(in.d_skip[static_cast<std::size_t>(head)]);
        const std::int64_t group = variant.group_modulo ? head % G : head / (H / G);

        for (std::int64_t token = 0; token < T; ++token) {
            const double raw_step =
                static_cast<double>(in.dt[static_cast<std::size_t>(token * H + head)]);
            const double step =
                std::max(softplus(raw_step +
                                  static_cast<double>(in.dt_bias[static_cast<std::size_t>(head)])),
                         static_cast<double>(in.time_step_min));
            const double decay        = std::exp(step * a);
            const std::int64_t g_base = (token * G + group) * kStateSize;
            const std::int64_t v_base = (token * H + head) * V;

            for (std::int64_t value = 0; value < V; ++value) {
                const double dx =
                    step * static_cast<double>(in.x[static_cast<std::size_t>(v_base + value)]);
                for (std::int64_t key = 0; key < kStateSize; ++key) {
                    const std::int64_t offset =
                        variant.swap_state_extents ? value + V * key : key + kStateSize * value;
                    double& element = plane[static_cast<std::size_t>(offset)];
                    element =
                        decay * element +
                        dx * static_cast<double>(in.b[static_cast<std::size_t>(g_base + key)]);
                    if (variant.bf16_state) {
                        element = static_cast<double>(round_bf16(float(element)));
                    }
                }
            }

            for (std::int64_t value = 0; value < V; ++value) {
                double readout = 0.0;
                for (std::int64_t key = 0; key < kStateSize; ++key) {
                    const std::int64_t offset =
                        variant.swap_state_extents ? value + V * key : key + kStateSize * value;
                    readout += plane[static_cast<std::size_t>(offset)] *
                               static_cast<double>(in.c[static_cast<std::size_t>(g_base + key)]);
                }
                const double raw =
                    static_cast<double>(in.x[static_cast<std::size_t>(v_base + value)]);
                const double contraction =
                    variant.scaled_readout ? readout / std::sqrt(double(kStateSize)) : readout;
                result.out[static_cast<std::size_t>(v_base + value)] = contraction + skip * raw;
            }
        }

        for (std::int64_t value = 0; value < V; ++value) {
            for (std::int64_t key = 0; key < kStateSize; ++key) {
                const std::int64_t offset =
                    variant.swap_state_extents ? value + V * key : key + kStateSize * value;
                result.final_state[static_cast<std::size_t>(plane_base + offset)] =
                    plane[static_cast<std::size_t>(offset)];
            }
        }
    }
    return result;
}

} // namespace ninfer::test::mamba2_ref
