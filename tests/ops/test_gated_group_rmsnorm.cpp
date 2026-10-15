#include "ninfer/ops/gated_group_rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// The checkpoint's own value: NemotronH builds its mamba output norm with the config's
// layer_norm_epsilon, and the tiny reference dump uses the same 1e-5.
constexpr float kGroupEps = 1.0e-5F;

// Measured on the calibration sweep: the FP64 oracle and the kernel differ only by the single BF16
// output rounding and the FP32 gate/rsqrt path, which peaks at 1.77e-3 relative L2 (the misaligned
// scalar case, 2.0e-3 here). The gross tail is set by the output grid rather than by the reduction:
// a BF16 rounding error is at most half an ulp of its own value, which is 1.95e-3 of the largest
// element but up to twice that of the largest reference when the error peak sits one binade below
// it, so 4.0e-3 bounds it for every case below.
constexpr ReductionCriterion gated_group_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 2.0e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 4.0e-3};
}

// The gate is applied before the reduction and the divisor is the mean square of the *gated*
// value, within each group; there is no mean subtraction.
std::vector<double> gated_group_rmsnorm_oracle(const std::vector<float>& input,
                                               const std::vector<float>& weight,
                                               const std::vector<float>& gate, const Shape& shape,
                                               std::int32_t group_size, float eps) {
    const auto activate = [](double value) { return value / (1.0 + std::exp(-value)); };
    std::vector<double> output(input.size());
    const std::int64_t rows   = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    const std::int32_t groups = shape.d / group_size;
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * shape.d;
        for (std::int32_t group = 0; group < groups; ++group) {
            const std::size_t base = row_base + static_cast<std::size_t>(group) * group_size;
            double mean_square     = 0.0;
            for (std::int32_t i = 0; i < group_size; ++i) {
                const double h = static_cast<double>(input[base + i]) * activate(gate[base + i]);
                mean_square += h * h;
            }
            const double inverse = 1.0 / std::sqrt(mean_square / group_size + eps);
            for (std::int32_t i = 0; i < group_size; ++i) {
                const double h = static_cast<double>(input[base + i]) * activate(gate[base + i]);
                output[base + i] =
                    static_cast<double>(weight[group * group_size + i]) * h * inverse;
            }
        }
    }
    return output;
}

// `group_scale_step` multiplies column `c` by `1 + step * (c / group_size)`, which gives every
// group a different mean square. It is how the case below discriminates the grouped reduction from
// a whole-row one: on homogeneous data the per-group mean squares nearly agree, so a whole-row
// implementation would still land close to the grouped result.
int run_case(const char* label, const Shape& shape, std::int32_t group_size, std::uint32_t seed,
             float input_scale = 4.0F, float eps = kGroupEps, bool bf16x2_unaligned = false,
             float group_scale_step = 0.0F) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    gate.front() = 0.0F;
    gate.back()  = -0.0F;
    if (group_scale_step != 0.0F) {
        for (std::size_t index = 0; index < count; ++index) {
            const auto column = static_cast<std::int32_t>(index % shape.d);
            input[index] *= 1.0F + group_scale_step * static_cast<float>(column / group_size);
        }
    }
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    const std::vector<double> reference =
        gated_group_rmsnorm_oracle(input, weight, gate, shape, group_size, eps);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceInput device_gate   = make_input(gate, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::gated_group_rmsnorm(input_tensor, weight_tensor, gate_tensor, group_size, eps,
                             output_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    gated_group_rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += verify_preserved(std::string(label) + " preserves gate", device_gate);
    return failures;
}

// The group size must divide the row width; a width it does not tile is a contract violation, not a
// silently truncated tail.
int refuse_case(const char* label, std::int32_t d, std::int32_t rows, std::int32_t group_size,
                float eps = kGroupEps) {
    const std::size_t n = static_cast<std::size_t>(d) * rows * sizeof(std::uint16_t);
    DeviceBuffer input_storage(n), gate_storage(n), weight_storage(d * sizeof(std::uint16_t));
    DeviceBuffer output_storage(n);
    Tensor x(input_storage.p, DType::BF16, {d, rows});
    Tensor weight(weight_storage.p, DType::BF16, {d});
    Tensor gate(gate_storage.p, DType::BF16, {d, rows});
    Tensor out(output_storage.p, DType::BF16, {d, rows});
    try {
        ops::gated_group_rmsnorm(x, weight, gate, group_size, eps, out, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "gated_group_rmsnorm accepted " << label << '\n';
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // The tiny reference dump's own geometry: 64 wide, group_size 32, 130 rows.
    failures +=
        run_case("gated_group_rmsnorm tiny reference [64,130] groups 32", {64, 130}, 32, 3401U);
    // The checkpoint's geometry: 8192 = 8 groups of 1024.
    failures +=
        run_case("gated_group_rmsnorm production [8192,4] groups 1024", {8192, 4}, 1024, 3402U);
    // A single group is the whole row with the gate applied first.
    failures +=
        run_case("gated_group_rmsnorm single group [8192,2] groups 8192", {8192, 2}, 8192, 3403U);
    failures += run_case("gated_group_rmsnorm single group [64,32] groups 64", {64, 32}, 64, 3404U);
    // A group narrower than the warp, and a group wider than one warp iteration.
    failures += run_case("gated_group_rmsnorm narrow group [40,32] groups 8", {40, 32}, 8, 3405U);
    failures +=
        run_case("gated_group_rmsnorm wide group [3072,8] groups 1536", {3072, 8}, 1536, 3406U);
    // Widths that are not a whole number of the fast-path row shapes (512, 1024, 2048): 160 is
    // 5x32, and 45 = 5x9 is odd, so its rows start on an odd element.
    failures += run_case("gated_group_rmsnorm odd width [160,16] groups 32", {160, 16}, 32, 3407U);
    failures += run_case("gated_group_rmsnorm odd width [45,16] groups 9", {45, 16}, 9, 3408U);
    // Odd group sizes take the scalar route; a group of one is its own degenerate case.
    failures += run_case("gated_group_rmsnorm odd group [96,8] groups 3", {96, 8}, 3, 3409U);
    failures += run_case("gated_group_rmsnorm group of one [64,32] groups 1", {64, 32}, 1, 3410U);
    // eps behaviour: a tiny input scale makes eps dominate the mean square, and a huge eps makes it
    // dominate every group.
    failures +=
        run_case("gated_group_rmsnorm near-zero [64,32] groups 32", {64, 32}, 32, 3411U, 1.0e-5F);
    failures += run_case("gated_group_rmsnorm eps-dominated [64,32] groups 32", {64, 32}, 32, 3412U,
                         4.0F, 1.0e6F);
    // Unaligned buffers cannot use the BF16x2 route, so this covers the scalar traversal at a
    // group size that would otherwise pair.
    failures += run_case("gated_group_rmsnorm unaligned [64,32] groups 32", {64, 32}, 32, 3413U,
                         4.0F, kGroupEps, true);
    // Per-group scales make the groups' mean squares differ by up to 64x, which is what separates
    // this reduction from a whole-row one; the homogeneous cases above would barely notice.
    failures += run_case("gated_group_rmsnorm uneven groups [8192,4] groups 1024", {8192, 4}, 1024,
                         3414U, 4.0F, kGroupEps, false, 1.0F);
    failures += run_case("gated_group_rmsnorm uneven groups [64,130] groups 32", {64, 130}, 32,
                         3415U, 4.0F, kGroupEps, false, 1.0F);

    failures += refuse_case("a group size that does not divide the width (100/30)", 100, 8, 30);
    failures +=
        refuse_case("a group size that does not divide the width (1025/1024)", 1025, 4, 1024);
    failures +=
        refuse_case("a group size that does not divide the width (4096/1536)", 4096, 4, 1536);
    failures += refuse_case("a zero group size", 64, 8, 0);
    failures += refuse_case("a negative group size", 64, 8, -32);
    failures += refuse_case("a group size wider than the row (64/128)", 64, 8, 128);
    failures += refuse_case("a zero eps", 64, 8, 32, 0.0F);
    failures += refuse_case("a negative eps", 64, 8, 32, -1.0e-5F);

    std::cout << (failures ? "FAIL" : "OK") << " gated_group_rmsnorm\n";
    return failures ? 1 : 0;
}
