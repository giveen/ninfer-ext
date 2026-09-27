#pragma once

// FP64 host reference for the EXL3 tail-biting bitshift-trellis encoder. It is the oracle for the
// production GPU encoder, not a production path: its cost is 2^S * 2^w per step.
//
// State convention (matches artifact/exl3_trellis.h): the state of weight t is the S-bit window
// ending after its w_t new bits, and the new bits enter at the top,
//     s_t = (s_{t-1} >> w_t) | (b_t << (S - w_t)),
// with the ring closed by s_{-1} = s_{L-1}. A weight's value is codebook(s_t).

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::test::exl3 {

struct TrellisSpec {
    int state_bits = 16;
    std::vector<int> step_widths; // one per weight; the ring length is its size
};

// Integer K: every step is K bits. Half-integer K = ka + 0.5 (bitrate_half_bits odd): even weights
// take ka bits and odd weights ka + 1, as in trellis_t16_v1.
[[nodiscard]] TrellisSpec trellis_spec(int state_bits, int ring_length, int bitrate_half_bits);

using Codebook = std::function<double(std::uint32_t state)>;

struct TrellisPath {
    std::vector<std::uint32_t> states;
    double cost = 0.0; // sum of squared errors against the target
};

[[nodiscard]] bool is_circular_path(const TrellisSpec& spec, std::span<const std::uint32_t> states);
[[nodiscard]] double path_cost(std::span<const std::uint32_t> states,
                               std::span<const double> target, const Codebook& codebook);

// Exact optimum over all circular paths: one constrained Viterbi per ring overlap value.
[[nodiscard]] TrellisPath encode_exact(const TrellisSpec& spec, std::span<const double> target,
                                       const Codebook& codebook);

// The production method: an unconstrained pass over the ring rotated by L/2 fixes the overlap
// entering weight 0, and a second pass constrained to that overlap closes the ring.
[[nodiscard]] TrellisPath encode_two_pass(const TrellisSpec& spec, std::span<const double> target,
                                          const Codebook& codebook);

} // namespace ninfer::test::exl3
