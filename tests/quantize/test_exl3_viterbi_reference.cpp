#include "exl3_viterbi_reference.h"

#include "artifact/exl3_trellis.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace ninfer::test::exl3;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

// Independent oracle: enumerate every bitstream of the ring and derive its circular states
// directly from the stream (the window convention of trellis_t16_v1), then keep the cheapest.
double brute_force_cost(const TrellisSpec& spec, std::span<const double> target,
                        const Codebook& codebook) {
    const std::size_t length = spec.step_widths.size();
    std::vector<std::size_t> end(length);
    std::size_t total = 0;
    for (std::size_t t = 0; t < length; ++t) {
        total += static_cast<std::size_t>(spec.step_widths[t]);
        end[t] = total;
    }
    double best = INFINITY;
    for (std::uint64_t stream = 0; stream < (std::uint64_t{1} << total); ++stream) {
        double cost = 0.0;
        for (std::size_t t = 0; t < length && cost < best; ++t) {
            std::uint32_t state = 0;
            for (int bit = 0; bit < spec.state_bits; ++bit) {
                const std::size_t at = (end[t] + total - static_cast<std::size_t>(spec.state_bits) +
                                        static_cast<std::size_t>(bit)) %
                                       total;
                state |= static_cast<std::uint32_t>((stream >> at) & 1U) << bit;
            }
            const double d = codebook(state) - target[t];
            cost += d * d;
        }
        best = std::min(best, cost);
    }
    return best;
}

void test_small_trellises_against_brute_force() {
    std::mt19937_64 rng(0x5eed);
    std::normal_distribution<double> normal;

    struct Case {
        int state_bits;
        int length;
        int half_bits;
    };

    for (const Case c : {Case{6, 8, 4}, Case{7, 10, 3}, Case{5, 12, 2}, Case{6, 6, 5}}) {
        std::vector<double> table(std::size_t{1} << c.state_bits);
        for (double& value : table) { value = normal(rng); }
        const Codebook codebook = [&](std::uint32_t s) { return table[s]; };
        const TrellisSpec spec  = trellis_spec(c.state_bits, c.length, c.half_bits);
        for (int trial = 0; trial < 4; ++trial) {
            std::vector<double> target(static_cast<std::size_t>(c.length));
            for (double& value : target) { value = normal(rng); }
            const double oracle     = brute_force_cost(spec, target, codebook);
            const TrellisPath exact = encode_exact(spec, target, codebook);
            expect(is_circular_path(spec, exact.states), "exact encoder returned a broken ring");
            expect(std::abs(exact.cost - oracle) <= 1e-12 * (1.0 + oracle),
                   "exact encoder cost differs from the brute-force optimum");
            expect(std::abs(path_cost(exact.states, target, codebook) - exact.cost) <= 1e-12,
                   "exact encoder reports a cost its states do not have");
            const TrellisPath approx = encode_two_pass(spec, target, codebook);
            expect(is_circular_path(spec, approx.states),
                   "two-pass encoder returned a broken ring");
            expect(approx.cost >= exact.cost - 1e-12, "two-pass encoder beat the exact optimum");
            expect(std::abs(path_cost(approx.states, target, codebook) - approx.cost) <= 1e-12,
                   "two-pass encoder reports a cost its states do not have");
        }
    }
}

// Full-size tile: S = 16, L = 256, mul1 codebook normalized to unit variance. The encoded states
// must pack into a trellis_t16_v1 tile and decode back unchanged.
void test_real_tiles_pack_through_the_codec() {
    std::mt19937_64 rng(0x7e11);
    std::normal_distribution<double> normal;
    const Codebook mul1 = [](std::uint32_t s) {
        return ninfer::artifact::exl3_mul1_value(static_cast<std::uint16_t>(s)) / 147.8;
    };
    for (const int half_bits : {4, 5, 6}) {
        const TrellisSpec spec = trellis_spec(16, 256, half_bits);
        constexpr int kTiles   = 8;
        double total_mse       = 0.0;
        TrellisPath path;
        for (int tile_index = 0; tile_index < kTiles; ++tile_index) {
            std::vector<double> target(256);
            for (double& value : target) { value = normal(rng); }
            path = encode_two_pass(spec, target, mul1);
            total_mse += path.cost / 256.0 / kTiles;
        }
        std::array<std::uint16_t, 256> states{};
        for (std::size_t t = 0; t < 256; ++t) {
            states[t] = static_cast<std::uint16_t>(path.states[t]);
        }
        const auto tile =
            ninfer::artifact::exl3_pack_trellis_tile(states, static_cast<std::uint64_t>(half_bits));
        std::array<std::uint16_t, 256> decoded{};
        ninfer::artifact::exl3_unpack_trellis_tile(tile, static_cast<std::uint64_t>(half_bits),
                                                   decoded);
        expect(decoded == states, "reference path does not survive the tile codec");
        // Loose quality guard against the Gaussian rate-distortion bound 2^-2K: the L=16 trellis
        // lands within about 20% of it (QTIP reports ~0.07 at K=2).
        const double bound = std::exp2(-static_cast<double>(half_bits));
        expect(total_mse >= bound * 0.9 && total_mse < bound * 1.35,
               "reference trellis MSE is outside the plausible range");
        std::cout << "K=" << half_bits / 2.0 << " unit-Gaussian MSE " << total_mse << " (bound "
                  << bound << ")\n";
    }
}

} // namespace

int main() {
    test_small_trellises_against_brute_force();
    test_real_tiles_pack_through_the_codec();
    if (failures == 0) { std::cout << "OK exl3 viterbi reference\n"; }
    return failures == 0 ? 0 : 1;
}
