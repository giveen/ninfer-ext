#include "evaluation.h"
#include "reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninfer::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

// BF16 bit patterns of a float, matching what the writer stores and the reader decodes.
std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

void put_u32(std::vector<unsigned char>& bytes, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * byte)) & 0xFFU));
    }
}

// Independent FP64 KL(P_ref || P_model) from the raw BF16 patterns.
double oracle_kl(const std::vector<std::uint16_t>& reference,
                 const std::vector<std::uint16_t>& model) {
    const auto value = [](std::uint16_t bits) {
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float result             = 0.0F;
        std::memcpy(&result, &wide, sizeof(result));
        return static_cast<double>(result);
    };
    const auto log_partition = [&](const std::vector<std::uint16_t>& row) {
        double max_value = -std::numeric_limits<double>::infinity();
        for (const std::uint16_t bits : row) { max_value = std::max(max_value, value(bits)); }
        double sum = 0.0;
        for (const std::uint16_t bits : row) { sum += std::exp(value(bits) - max_value); }
        return std::log(sum) + max_value;
    };
    const double reference_lse = log_partition(reference);
    const double model_lse     = log_partition(model);
    double divergence          = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double reference_logprob = value(reference[index]) - reference_lse;
        divergence += std::exp(reference_logprob) * (reference_logprob - (value(model[index]) - model_lse));
    }
    return divergence;
}

// The reference file is a contract between the Python writer and the C++ reader, so pin its bytes
// and check the reader and the KL against an FP64 oracle.
int check_reference_kl() {
    constexpr std::uint32_t vocab = 8;
    const std::vector<std::uint32_t> positions{3, 7};
    const std::vector<std::vector<float>> rows{{0.0F, 1.0F, 2.0F, 3.0F, 2.0F, 3.0F, 2.0F, 1.0F},
                                               {3.0F, 0.0F, 1.0F, 0.0F, 4.0F, 2.0F, 0.0F, 1.0F}};

    std::vector<unsigned char> bytes;
    const char magic[8] = {'N', 'I', 'N', 'F', 'K', 'L', '1', '\0'};
    bytes.insert(bytes.end(), magic, magic + 8);
    put_u32(bytes, 1);  // version
    put_u32(bytes, vocab);
    put_u32(bytes, static_cast<std::uint32_t>(rows.size()));
    put_u32(bytes, 4);  // context
    put_u32(bytes, 2);  // stride
    put_u32(bytes, 0);  // reserved
    for (int index = 0; index < 64; ++index) {
        bytes.push_back(static_cast<unsigned char>('a' + index % 26));
    }
    for (const std::uint32_t position : positions) { put_u32(bytes, position); }
    for (const auto& row : rows) {
        for (const float value : row) {
            const std::uint16_t bits = bf16_bits(value);
            bytes.push_back(static_cast<unsigned char>(bits & 0xFFU));
            bytes.push_back(static_cast<unsigned char>(bits >> 8));
        }
    }

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer-kl-reference-test.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) { throw std::runtime_error("cannot write the reference fixture"); }
    }

    int failures = 0;
    ninfer::perplexity::Reference reference;
    reference.load(path);
    failures += require(reference.vocab_size() == vocab, "reference vocab");
    failures += require(reference.context() == 4 && reference.stride() == 2, "reference protocol");
    failures += require(reference.rows() == 2, "reference row count");
    failures += require(reference.row(3) != nullptr && reference.row(7) != nullptr,
                        "reference row lookup");
    failures += require(reference.row(4) == nullptr, "reference row lookup misses absent targets");

    std::vector<std::uint16_t> first;
    std::vector<std::uint16_t> second;
    const std::uint16_t* first_row = reference.row(positions[0]);
    const std::uint16_t* other_row = reference.row(positions[1]);
    for (std::uint32_t value = 0; value < vocab; ++value) {
        first.push_back(first_row[value]);
        second.push_back(other_row[value]);
    }
    const double self = ninfer::perplexity::kl_divergence_row(first_row, first_row, vocab);
    failures += require(std::abs(self) < 1e-9, "KL(P||P) is zero");

    const double got    = ninfer::perplexity::kl_divergence_row(first_row, other_row, vocab);
    const double wanted = oracle_kl(first, second);
    failures += require(std::abs(got - wanted) < 1e-9, "KL matches the FP64 oracle");

    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    // A truncated or mis-magic'd file must be refused, not read as garbage.
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write("NINFKL1\0", 8);
        output.write("short", 5);
    }
    bool refused = false;
    try {
        ninfer::perplexity::Reference bad;
        bad.load(path);
    } catch (const std::exception&) { refused = true; }
    failures += require(refused, "truncated reference is refused");
    std::filesystem::remove(path, ignored);

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninfer::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninfer::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninfer::perplexity::ScoreAggregate a;
    ninfer::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    failures += check_reference_kl();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
