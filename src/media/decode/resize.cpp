#include "media/decode/resize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::media::decode {
namespace {

int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }

double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<float>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

} // namespace

void resize_bicubic(Image& input, int width, int height, const std::function<void()>& checkpoint) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("resize_bicubic: empty size");
    if (input.width == width && input.height == height) { return; }

    const struct {
        int w, h;
    } size{width, height};

    const Coefficients horizontal = coefficients(input.width, size.w);
    const Coefficients vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        if (y % 16 == 0 && checkpoint) { checkpoint(); }
        for (int x = 0; x < size.w; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<float, 3> value{};
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const float weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                temp.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }

    Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        if (y % 16 == 0 && checkpoint) { checkpoint(); }
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < size.w; ++x) {
            std::array<float, 3> value{};
            const std::uint8_t* source =
                temp.data() + (static_cast<std::size_t>(source_begin) * size.w + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(size.w) * 3) {
                const float weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                out.rgb.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    input = std::move(out);
}

} // namespace ninfer::media::decode
