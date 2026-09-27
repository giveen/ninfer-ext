#include "exl3_viterbi_reference.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::test::exl3 {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void validate(const TrellisSpec& spec, std::size_t target_size) {
    if (spec.state_bits < 2 || spec.state_bits > 16) {
        throw std::invalid_argument("reference trellis state must have 2..16 bits");
    }
    if (spec.step_widths.size() < 2 || spec.step_widths.size() != target_size) {
        throw std::invalid_argument("reference trellis ring and target lengths differ");
    }
    int total = 0;
    for (const int width : spec.step_widths) {
        if (width < 1 || width >= spec.state_bits) {
            throw std::invalid_argument("reference trellis step width must be in [1, S)");
        }
        total += width;
    }
    if (total < spec.state_bits) {
        throw std::invalid_argument("reference trellis ring holds fewer bits than one state");
    }
}

// Viterbi over the ring in the order rotation, rotation + 1, ... (mod L). When overlap is set, the
// state of the first visited weight must continue from it, and the last visited state must close
// the ring onto it; this only forms a ring when rotation is 0. Returns states by weight index.
std::optional<TrellisPath> viterbi(const TrellisSpec& spec, std::span<const double> target,
                                   const Codebook& codebook, std::size_t rotation,
                                   std::optional<std::uint32_t> overlap) {
    const int s_bits           = spec.state_bits;
    const std::uint32_t states = 1U << s_bits;
    const std::size_t length   = target.size();
    const auto position        = [&](std::size_t step) { return (step + rotation) % length; };
    const auto low_mask        = [&](int width) { return (1U << (s_bits - width)) - 1U; };

    std::vector<double> values(states);
    for (std::uint32_t s = 0; s < states; ++s) { values[s] = codebook(s); }
    const auto error = [&](std::size_t weight, std::uint32_t s) {
        const double d = values[s] - target[weight];
        return d * d;
    };

    // back[step][s] holds the low w bits of the predecessor state.
    std::vector<std::vector<std::uint8_t>> back(length, std::vector<std::uint8_t>(states));
    std::vector<double> cost(states), next(states);

    const std::size_t first = position(0);
    const int first_width   = spec.step_widths[first];
    for (std::uint32_t s = 0; s < states; ++s) {
        const bool allowed = !overlap || (s & low_mask(first_width)) == *overlap;
        cost[s]            = allowed ? error(first, s) : kInf;
    }
    for (std::size_t step = 1; step < length; ++step) {
        const std::size_t weight = position(step);
        const int width          = spec.step_widths[weight];
        const std::uint32_t mask = low_mask(width);
        for (std::uint32_t s = 0; s < states; ++s) {
            const std::uint32_t base = (s & mask) << width;
            double best              = kInf;
            std::uint32_t best_low   = 0;
            for (std::uint32_t low = 0; low < (1U << width); ++low) {
                if (cost[base | low] < best) {
                    best     = cost[base | low];
                    best_low = low;
                }
            }
            next[s]       = best + error(weight, s);
            back[step][s] = static_cast<std::uint8_t>(best_low);
        }
        cost.swap(next);
    }

    double best        = kInf;
    std::uint32_t last = 0;
    for (std::uint32_t s = 0; s < states; ++s) {
        const bool closes = !overlap || (s >> first_width) == *overlap;
        if (closes && cost[s] < best) {
            best = cost[s];
            last = s;
        }
    }
    if (best == kInf) { return std::nullopt; }

    TrellisPath path{std::vector<std::uint32_t>(length), best};
    std::uint32_t s = last;
    for (std::size_t step = length; step-- > 0;) {
        path.states[position(step)] = s;
        if (step == 0) { break; }
        const int width = spec.step_widths[position(step)];
        s               = ((s & low_mask(width)) << width) | back[step][s];
    }
    return path;
}

} // namespace

TrellisSpec trellis_spec(int state_bits, int ring_length, int bitrate_half_bits) {
    if (ring_length < 2 || bitrate_half_bits < 2) {
        throw std::invalid_argument("reference trellis spec needs L >= 2 and K >= 1");
    }
    TrellisSpec spec{state_bits, std::vector<int>(static_cast<std::size_t>(ring_length))};
    for (int weight = 0; weight < ring_length; ++weight) {
        spec.step_widths[static_cast<std::size_t>(weight)] =
            bitrate_half_bits / 2 + ((bitrate_half_bits & 1) && (weight & 1) ? 1 : 0);
    }
    return spec;
}

bool is_circular_path(const TrellisSpec& spec, std::span<const std::uint32_t> states) {
    if (states.size() != spec.step_widths.size()) { return false; }
    const int s_bits = spec.state_bits;
    for (std::size_t t = 0; t < states.size(); ++t) {
        const std::uint32_t previous = states[(t + states.size() - 1) % states.size()];
        const int width              = spec.step_widths[t];
        if (states[t] >= (1U << s_bits) ||
            (states[t] & ((1U << (s_bits - width)) - 1U)) != (previous >> width)) {
            return false;
        }
    }
    return true;
}

double path_cost(std::span<const std::uint32_t> states, std::span<const double> target,
                 const Codebook& codebook) {
    double cost = 0.0;
    for (std::size_t t = 0; t < states.size(); ++t) {
        const double d = codebook(states[t]) - target[t];
        cost += d * d;
    }
    return cost;
}

TrellisPath encode_exact(const TrellisSpec& spec, std::span<const double> target,
                         const Codebook& codebook) {
    validate(spec, target.size());
    const std::uint32_t overlaps = 1U << (spec.state_bits - spec.step_widths.front());
    std::optional<TrellisPath> best;
    for (std::uint32_t overlap = 0; overlap < overlaps; ++overlap) {
        auto path = viterbi(spec, target, codebook, 0, overlap);
        if (path && (!best || path->cost < best->cost)) { best = std::move(path); }
    }
    if (!best) { throw std::logic_error("reference trellis has no circular path"); }
    return *best;
}

TrellisPath encode_two_pass(const TrellisSpec& spec, std::span<const double> target,
                            const Codebook& codebook) {
    validate(spec, target.size());
    const std::size_t length    = target.size();
    const auto open             = viterbi(spec, target, codebook, length / 2, std::nullopt);
    const std::uint32_t overlap = open->states[length - 1] >> spec.step_widths.front();
    auto closed                 = viterbi(spec, target, codebook, 0, overlap);
    if (!closed) { throw std::logic_error("reference trellis second pass found no closed ring"); }
    return *closed;
}

} // namespace ninfer::test::exl3
