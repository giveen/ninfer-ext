#include "ninfer/ops/scale_columns.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// BF16 output rounding alone can incur almost 2^-8 relative error, so the gross bound contains a
// single rounding of the exact FP64 product.
constexpr PointwiseCriterion scale_columns_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// This is the sole oracle. It consumes the exact logical BF16 values represented by the public
// inputs and evaluates the product naively in FP64 over the requested dim range; dimensions outside
// the range keep the represented input value, so a write there is a mismatch rather than a tolerance
// question.
std::vector<double> scale_columns_oracle(const std::vector<float>& x,
                                         const std::vector<float>& scale, std::int32_t d,
                                         std::int64_t rows, std::int32_t begin,
                                         std::int32_t end) {
    std::vector<double> expected(x.begin(), x.end());
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int32_t dim = begin; dim < end; ++dim) {
            const std::size_t index = static_cast<std::size_t>(row) * static_cast<std::size_t>(d) +
                                      static_cast<std::size_t>(dim);
            expected[index] = static_cast<double>(x[index]) *
                              static_cast<double>(scale[static_cast<std::size_t>(dim)]);
        }
    }
    return expected;
}

// Dimensions outside the requested range must be bit-exact, which is what lets a caller scale only
// the dims a partial rotation leaves alone.
int verify_untouched(const char* label, const std::vector<std::uint16_t>& storage,
                     const std::vector<std::uint16_t>& before, std::int32_t d, std::int64_t rows,
                     std::int32_t begin, std::int32_t end) {
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int32_t dim = 0; dim < d; ++dim) {
            if (dim >= begin && dim < end) { continue; }
            const std::size_t index = static_cast<std::size_t>(row) * static_cast<std::size_t>(d) +
                                      static_cast<std::size_t>(dim);
            if (storage[index] != before[index]) {
                std::cerr << label << ": dimension " << dim << " outside [" << begin << ", " << end
                          << ") changed at row " << row << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int run_case(const char* label, std::int32_t d, std::int32_t heads, std::int32_t tokens,
             std::int32_t begin, std::int32_t end, std::uint32_t seed, bool unaligned = false) {
    const std::int64_t rows  = static_cast<std::int64_t>(heads) * tokens;
    const std::size_t count  = static_cast<std::size_t>(rows) * static_cast<std::size_t>(d);
    const std::size_t offset = unaligned ? 1 : 0; // one BF16 element of leading storage

    std::vector<float> x(count), scale(d);
    fill_uniform(x, seed, -8.0f, 8.0f);
    fill_uniform(scale, seed + 1U, -1.5f, 1.5f);
    round_to_bf16(x);
    round_to_bf16(scale);

    const auto expected = scale_columns_oracle(x, scale, d, rows, begin, end);
    const auto x_bits   = encode_bf16(x);
    const auto s_bits   = encode_bf16(scale);

    GuardedDeviceBuffer device_x((offset + count) * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale((offset + static_cast<std::size_t>(d)) * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), count * sizeof(std::uint16_t), offset * sizeof(std::uint16_t));
    device_scale.copy_from_host(s_bits.data(), static_cast<std::size_t>(d) * sizeof(std::uint16_t),
                                offset * sizeof(std::uint16_t));

    auto* x_data = static_cast<std::uint8_t*>(device_x.data()) + offset * sizeof(std::uint16_t);
    auto* s_data =
        static_cast<std::uint8_t*>(device_scale.data()) + offset * sizeof(std::uint16_t);
    Tensor x_tensor(x_data, DType::BF16, {d, heads, tokens});
    Tensor scale_tensor(s_data, DType::BF16, {d});

    ops::scale_columns(scale_tensor, x_tensor, begin, end, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(x_data, count), expected,
                                    scale_columns_bf16_criterion());
    failures += verify_untouched(label, from_device<std::uint16_t>(x_data, count), x_bits, d, rows,
                                 begin, end);
    failures += verify_exact("scale_columns scale unchanged",
                             from_device<std::uint16_t>(s_data, static_cast<std::size_t>(d)),
                             s_bits);
    failures += device_x.verify_guards("scale_columns x");
    failures += device_scale.verify_guards("scale_columns scale");
    return failures;
}

// A range or a scale shape outside the contract is refused rather than scaled.
int verify_rejection(const char* label, std::int32_t d, std::int32_t heads, std::int32_t tokens,
                     std::int32_t begin, std::int32_t end, bool bad_scale_shape = false) {
    GuardedDeviceBuffer device_x(static_cast<std::size_t>(d) * heads * tokens *
                                 sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(static_cast<std::size_t>(d + 1) * sizeof(std::uint16_t));
    Tensor x_tensor(device_x.data(), DType::BF16, {d, heads, tokens});
    Tensor scale_tensor(device_scale.data(), DType::BF16,
                        bad_scale_shape ? std::initializer_list<std::int32_t>{d + 1}
                                        : std::initializer_list<std::int32_t>{d});
    try {
        ops::scale_columns(scale_tensor, x_tensor, begin, end, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": scale_columns accepted an invalid request\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma 4 global attention: the per-head scale of the weightless normalization by w_kn covers
    // the whole 512-wide head, and the query prescale covers the 384 dims the proportional rotation
    // leaves alone, in two contiguous runs.
    for (std::int32_t tokens : {1, 7, 128}) {
        failures += run_case("scale_columns global head [512,4]", 512, 4, tokens, 0, 512, 2101U);
        failures += run_case("scale_columns global non-rotated low [512,4]", 512, 4, tokens, 64, 256,
                             2102U);
        failures += run_case("scale_columns global non-rotated high [512,4]", 512, 4, tokens, 320,
                             512, 2103U);
        failures += run_case("scale_columns sliding head [256,16]", 256, 16, tokens, 0, 256, 2104U);
    }
    failures += run_case("scale_columns unaligned [512,4]", 512, 4, 7, 0, 512, 2201U, true);
    failures += run_case("scale_columns unaligned range [512,4]", 512, 4, 7, 64, 256, 2202U, true);
    failures += run_case("scale_columns odd range [256,16]", 256, 16, 3, 1, 8, 2203U);
    failures += run_case("scale_columns odd length [512,4]", 512, 4, 3, 63, 258, 2204U);
    failures += run_case("scale_columns single dim [512,4]", 512, 4, 5, 511, 512, 2205U);
    // A range that is even but not eight-aligned takes the BF16x2 route.
    failures += run_case("scale_columns pair route [512,4]", 512, 4, 7, 66, 256, 2208U);
    failures += run_case("scale_columns pair route unaligned [256,16]", 256, 16, 3, 66, 200, 2209U,
                         true);
    failures += run_case("scale_columns empty range [512,4]", 512, 4, 5, 128, 128, 2206U);
    failures += run_case("scale_columns near-zero [512,4]", 512, 4, 5, 0, 512, 2207U);

    failures += verify_rejection("scale_columns negative begin", 512, 4, 4, -1, 512);
    failures += verify_rejection("scale_columns end beyond d", 512, 4, 4, 0, 513);
    failures += verify_rejection("scale_columns begin after end", 512, 4, 4, 200, 100);
    failures += verify_rejection("scale_columns scale shape", 512, 4, 4, 0, 512, true);

    std::cout << (failures ? "FAIL" : "OK") << " scale_columns\n";
    return failures ? 1 : 0;
}
