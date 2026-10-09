#include "ninfer/ops/compact_kv_rows.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The layout is stated structurally and independently of the kernel: a compact row is the value row
// followed by the low rotated run and then the high rotated run. Every entry is a copy, so the
// comparison is exact.
std::vector<std::uint16_t> compact_kv_oracle(const std::vector<std::uint16_t>& v,
                                             const std::vector<std::uint16_t>& k,
                                             std::int32_t rotary_dim, std::int32_t rotary_pairs,
                                             std::int64_t rows) {
    const std::int32_t compact = rotary_dim + 2 * rotary_pairs;
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(compact) *
                                        static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t source_row   = static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(rotary_dim);
        const std::size_t compact_row  = static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(compact);
        // The value row verbatim.
        for (std::int32_t d = 0; d < rotary_dim; ++d) {
            expected[compact_row + static_cast<std::size_t>(d)] = v[source_row + d];
        }
        // The low rotated run, then the high rotated run.
        for (std::int32_t i = 0; i < rotary_pairs; ++i) {
            expected[compact_row + static_cast<std::size_t>(rotary_dim) + i] =
                k[source_row + i];
            expected[compact_row + static_cast<std::size_t>(rotary_dim) + rotary_pairs + i] =
                k[source_row + rotary_dim / 2 + i];
        }
    }
    return expected;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int run_case(const char* label, std::int32_t rotary_dim, std::int32_t rotary_pairs,
             std::int32_t heads, std::int32_t tokens, std::uint32_t seed) {
    const std::int64_t rows = static_cast<std::int64_t>(heads) * tokens;
    const std::int32_t compact = rotary_dim + 2 * rotary_pairs;
    const std::size_t source_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(rotary_dim);
    const std::size_t out_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(compact);

    // Distinguishable values in v and k, so a swapped source or a shifted index changes the bits.
    std::vector<float> v(source_count), k(source_count);
    fill_uniform(v, seed, -8.0f, 8.0f);
    fill_uniform(k, seed + 7919U, -8.0f, 8.0f);
    round_to_bf16(v);
    round_to_bf16(k);

    const auto v_bits = encode_bf16(v);
    const auto k_bits = encode_bf16(k);
    const auto expected = compact_kv_oracle(v_bits, k_bits, rotary_dim, rotary_pairs, rows);

    GuardedDeviceBuffer device_v(source_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_k(source_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(out_count * sizeof(std::uint16_t));
    device_v.copy_from_host(v_bits.data(), device_v.bytes());
    device_k.copy_from_host(k_bits.data(), device_k.bytes());
    device_out.fill(0x7d);

    Tensor v_tensor(device_v.data(), DType::BF16, {rotary_dim, heads, tokens});
    Tensor k_tensor(device_k.data(), DType::BF16, {rotary_dim, heads, tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {compact, heads, tokens});
    ops::compact_kv_rows(v_tensor, k_tensor, out_tensor, rotary_dim, rotary_pairs, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact(label, from_device<std::uint16_t>(device_out.data(), out_count), expected);
    failures += verify_exact("compact_kv_rows v unchanged",
                             from_device<std::uint16_t>(device_v.data(), source_count), v_bits);
    failures += verify_exact("compact_kv_rows k unchanged",
                             from_device<std::uint16_t>(device_k.data(), source_count), k_bits);
    failures += device_v.verify_guards("compact_kv_rows v");
    failures += device_k.verify_guards("compact_kv_rows k");
    failures += device_out.verify_guards("compact_kv_rows out");
    return failures;
}

// A tensor set outside the contract is refused rather than gathered.
int verify_rejection(const char* label, std::int32_t rotary_dim, std::int32_t rotary_pairs,
                     std::int32_t out_width, bool mismatched_source = false) {
    constexpr std::int32_t kHeads = 2, kTokens = 3;
    const std::size_t source_count =
        static_cast<std::size_t>(rotary_dim) * kHeads * kTokens;
    const std::size_t k_count =
        static_cast<std::size_t>(mismatched_source ? rotary_dim - 1 : rotary_dim) * kHeads * kTokens;
    const std::size_t out_count = static_cast<std::size_t>(out_width) * kHeads * kTokens;

    GuardedDeviceBuffer device_v(source_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_k(k_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(out_count * sizeof(std::uint16_t));
    Tensor v_tensor(device_v.data(), DType::BF16, {rotary_dim, kHeads, kTokens});
    Tensor k_tensor(device_k.data(), DType::BF16, {rotary_dim - (mismatched_source ? 1 : 0), kHeads,
                                                   kTokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {out_width, kHeads, kTokens});
    try {
        ops::compact_kv_rows(v_tensor, k_tensor, out_tensor, rotary_dim, rotary_pairs, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": compact_kv_rows accepted an invalid request\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma 4 global attention: 640 values per compact row for a 512-wide head rotating 64 pairs.
    for (std::int32_t tokens : {1, 7, 128}) {
        failures += run_case("compact_kv_rows global [640,4]", 512, 64, 4, tokens, 3101U);
    }
    // A different pair count, so the layout is not tied to the Gemma profile.
    failures += run_case("compact_kv_rows full span [512,16]", 256, 128, 16, 5, 3102U);
    failures += run_case("compact_kv_rows single pair [514,2]", 512, 1, 2, 3, 3103U);
    failures += run_case("compact_kv_rows odd head [128,8]", 64, 32, 8, 9, 3104U);
    // 65536 rows: one more than the largest grid dimension, so the row stride is exercised.
    failures += run_case("compact_kv_rows grid stride [128,8]", 64, 32, 8, 8192, 3105U);

    failures += verify_rejection("compact_kv_rows rotary_pairs=0", 512, 0, 512);
    failures += verify_rejection("compact_kv_rows rotary_pairs beyond half", 512, 257, 1026);
    failures += verify_rejection("compact_kv_rows out width", 512, 64, 639);
    failures += verify_rejection("compact_kv_rows v/k shapes", 512, 64, 640, true);

    std::cout << (failures ? "FAIL" : "OK") << " compact_kv_rows\n";
    return failures ? 1 : 0;
}
