// Qualification of the contract's portable 128-bit arithmetic against the native type.
//
// On every toolchain that provides unsigned __int128 -- Linux, which is what this tree is developed
// and calibrated on -- ninfer::runtime::UInt128 IS that native type, so the two-word implementation
// the seam compiles for MSVC is never built or executed on the maintained platform and would first
// run on a user's machine, where a wrong carry would silently corrupt a checked contract price.
//
// This unit defines NINFER_UINT128_PORTABLE, which makes the seam compile the portable
// implementation for this translation unit, and holds it against unsigned __int128 used as the
// oracle. The oracle does not exist on MSVC, where the macro is inert, so the registered contract
// values are checked unconditionally and qualify the portable arithmetic on both platforms.
//
// The cases are the contract's own: the token-count attention work make_prefill_work builds, the two
// 64x64 products context_cost.cpp reduces, and the 32-bit attention count the context-cost fixture
// sums. They are driven by the magnitudes those callers actually see (token counts, payload bytes,
// Q32 nanosecond coefficients) and by the thresholds at which they saturate.

#define NINFER_UINT128_PORTABLE 1

#include "runtime/contract/resources.h"
#include "runtime/contract/uint128.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

// The portable two-word implementation, because NINFER_UINT128_PORTABLE is defined above.
using Wide = ninfer::runtime::UInt128;

// The contract's portable reductions, spelled exactly as context_cost.cpp spells them so that the
// comparisons below cover the checked arithmetic rather than a paraphrase of it.
std::uint64_t wide_saturating_product(std::uint64_t left, std::uint64_t right) {
    const Wide product = static_cast<Wide>(left) * right;
    return product > std::numeric_limits<std::uint64_t>::max()
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(product);
}

std::uint64_t wide_q32_product(std::uint64_t coefficient, std::uint64_t units) {
    if (coefficient == 0 || units == 0) { return 0; }
    const Wide product        = static_cast<Wide>(coefficient) * units;
    const Wide maximum_scaled = static_cast<Wide>(std::numeric_limits<std::uint64_t>::max()) << 32U;
    if (product >= maximum_scaled) { return std::numeric_limits<std::uint64_t>::max(); }
    return static_cast<std::uint64_t>((product + (1ULL << 32U) - 1U) >> 32U);
}

// The context-cost fixture counts attention pairs from 32-bit token counts. Both products of that
// count fit in 64 bits, so the count is the same whether the sum is accumulated in 64 or in 128
// bits: only the sum can overflow, and this form checks exactly the condition the 128-bit form
// reported.
std::uint64_t fixture_attention_pairs(std::uint32_t prefix, std::uint32_t suffix) {
    const std::uint64_t linear     = static_cast<std::uint64_t>(prefix) * suffix;
    const std::uint64_t triangular = static_cast<std::uint64_t>(suffix) * (suffix + 1ULL) / 2U;
    if (triangular > std::numeric_limits<std::uint64_t>::max() - linear) {
        throw std::overflow_error("prefill attention-pair count exceeds uint64");
    }
    return linear + triangular;
}

// Registered contract values. These hold on every toolchain, including MSVC, where the oracle below
// cannot exist, and the first two are the expectations tests/test_context_cost.cpp already pins.
void test_registered_contract_values() {
    constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t kQ32One      = 1ULL << 32U;

    expect(ninfer::runtime::make_prefill_work(100, 10, 0, 0, 1).attention_pairs == 1055,
           "the registered 100/10 prefill attention work changed");
    expect(ninfer::runtime::make_prefill_work(kMaximum, kMaximum, 0, 0, 1).attention_pairs ==
               kMaximum,
           "prefill attention work no longer saturates at the uint64 maximum");

    expect(wide_saturating_product(kMaximum, 2ULL) == kMaximum,
           "the saturating product no longer saturates");
    expect(wide_saturating_product(34816ULL, 5120ULL) == 34816ULL * 5120ULL,
           "the saturating product lost a product that fits in uint64");
    expect(wide_q32_product(kQ32One, 1000ULL) == 1000ULL,
           "a Q32 unit coefficient must return its operand");
    expect(wide_q32_product(kMaximum, kQ32One) == kMaximum,
           "the Q32 product must saturate at the representable maximum");

    expect(fixture_attention_pairs(1024U, 512U) == 1024ULL * 512ULL + 512ULL * 513ULL / 2U,
           "the context-cost attention-pair count changed");
}

#if defined(__SIZEOF_INT128__)

// The oracle: the native type this seam selects everywhere else.
using Oracle = unsigned __int128;

std::uint64_t oracle_high(Oracle value) { return static_cast<std::uint64_t>(value >> 64U); }
std::uint64_t oracle_low(Oracle value) { return static_cast<std::uint64_t>(value); }
Oracle oracle_words(std::uint64_t low, std::uint64_t high) {
    return (static_cast<Oracle>(high) << 64U) | static_cast<Oracle>(low);
}

using Words = std::pair<std::uint64_t, std::uint64_t>; // (low, high), as UInt128 stores them

Wide wide_of(const Words& words) { return Wide{words.first, words.second}; }
Oracle oracle_of(const Words& words) { return oracle_words(words.first, words.second); }

void expect_words(const Wide& value, Oracle oracle, const char* message) {
    expect(value.low() == oracle_low(oracle) && value.high() == oracle_high(oracle), message);
}

// Words that put a carry or a borrow on every boundary of the representation: carries out of the low
// word, chains through the 0xFFFFFFFF partial products, and the widest magnitudes.
std::vector<Words> boundary_words() {
    const std::vector<std::uint64_t> words{0ULL,
                                          1ULL,
                                          2ULL,
                                          0xFFFFFFFFULL,
                                          0x100000000ULL,
                                          0xFFFFFFFF00000000ULL,
                                          0x7FFFFFFFFFFFFFFFULL,
                                          0x8000000000000000ULL,
                                          0xFFFFFFFFFFFFFFFEULL,
                                          0xFFFFFFFFFFFFFFFFULL};
    std::vector<Words> out;
    out.reserve(words.size() * words.size());
    for (const std::uint64_t low : words) {
        for (const std::uint64_t high : words) { out.emplace_back(low, high); }
    }
    return out;
}

// Every operation the contract performs, over every pair of boundary values: addition carries,
// subtraction borrows, the truncated product (whose high word exercises the carry chain of the
// 32x32 partial products), shifts across the word boundary, and the ordering that decides whether a
// result saturates.
void compare_arithmetic(const Words& left, const Words& right) {
    const Wide wide_left      = wide_of(left);
    const Wide wide_right     = wide_of(right);
    const Oracle native_left  = oracle_of(left);
    const Oracle native_right = oracle_of(right);

    expect_words(wide_left + wide_right, native_left + native_right,
                 "addition differs from the native type");
    expect_words(wide_left - wide_right, native_left - native_right,
                 "subtraction differs from the native type");
    expect_words(wide_left * wide_right, native_left * native_right,
                 "truncated product differs from the native type");
    expect_words(~wide_left, ~native_left, "complement differs from the native type");

    for (const unsigned shift : {0U, 1U, 31U, 32U, 33U, 63U, 64U, 65U, 96U, 127U}) {
        expect_words(wide_left << shift, native_left << shift,
                     "left shift differs from the native type");
        expect_words(wide_left >> shift, native_left >> shift,
                     "right shift differs from the native type");
    }

    expect((wide_left == wide_right) == (native_left == native_right),
           "equality differs from the native type");
    expect((wide_left < wide_right) == (native_left < native_right),
           "less-than differs from the native type");
    expect((wide_left > wide_right) == (native_left > native_right),
           "greater-than differs from the native type");
    expect((wide_left <= wide_right) == (native_left <= native_right),
           "less-equal differs from the native type");
    expect((wide_left >= wide_right) == (native_left >= native_right),
           "greater-equal differs from the native type");
}

void test_word_boundary_arithmetic() {
    const std::vector<Words> values = boundary_words();
    for (const Words& left : values) {
        for (const Words& right : values) { compare_arithmetic(left, right); }
    }
}

// The contract expression itself, evaluated with the native type: the same tokens make_prefill_work
// uses, so a divergence is a divergence in UInt128 and nothing else.
std::uint64_t oracle_attention_pairs(std::uint64_t prefix_tokens, std::uint64_t suffix_tokens) {
    const Oracle suffix     = suffix_tokens;
    const Oracle linear     = static_cast<Oracle>(prefix_tokens) * suffix;
    const Oracle triangular = (suffix * (suffix + 1U)) >> 1U;
    const Oracle maximum    = ~static_cast<Oracle>(0);
    const Oracle attention  = triangular > maximum - linear ? maximum : linear + triangular;
    return attention > std::numeric_limits<std::uint64_t>::max()
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(attention);
}

// Token counts from startup sizing through the saturated extremes, including the registered 100/10
// request and both operands at the uint64 maximum.
void test_contract_attention_work() {
    const std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> cases{{100ULL, 10ULL},
                                                                    {0ULL, 0ULL},
                                                                    {0ULL, 1ULL},
                                                                    {1ULL, 0ULL},
                                                                    {1ULL, 1ULL},
                                                                    {0ULL, 131072ULL},
                                                                    {131071ULL, 131072ULL},
                                                                    {1ULL << 20U, 1ULL << 20U},
                                                                    {1ULL << 32U, 1ULL << 32U},
                                                                    {1ULL << 63U, 1ULL << 63U},
                                                                    {kMaximum - 1U, kMaximum - 1U},
                                                                    {kMaximum, 1ULL},
                                                                    {kMaximum, kMaximum}};
    for (const auto& [prefix, suffix] : cases) {
        const ninfer::runtime::PrefillWork work =
            ninfer::runtime::make_prefill_work(prefix, suffix, 0, 0, 1);
        expect(work.attention_pairs == oracle_attention_pairs(prefix, suffix),
               "prefill attention work differs from the native type");
    }
}

std::uint64_t oracle_saturating_product(std::uint64_t left, std::uint64_t right) {
    const Oracle product = static_cast<Oracle>(left) * right;
    return product > std::numeric_limits<std::uint64_t>::max()
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(product);
}

std::uint64_t oracle_q32_product(std::uint64_t coefficient, std::uint64_t units) {
    if (coefficient == 0 || units == 0) { return 0; }
    const Oracle product        = static_cast<Oracle>(coefficient) * units;
    const Oracle maximum_scaled = static_cast<Oracle>(std::numeric_limits<std::uint64_t>::max())
                                  << 32U;
    if (product >= maximum_scaled) { return std::numeric_limits<std::uint64_t>::max(); }
    return static_cast<std::uint64_t>((product + (1ULL << 32U) - 1U) >> 32U);
}

// The magnitudes a hardware calibration produces, plus the thresholds: the uint64 maximum for the
// saturating product and (uint64 maximum) << 32 for the Q32 product, where the result stops being
// representable.
void test_contract_cost_products() {
    const std::uint64_t kQ32One = 1ULL << 32U;
    const std::vector<std::uint64_t> operands{
        0ULL,
        1ULL,
        kQ32One,
        5120ULL,                                  // one Gemma 4 KV page payload
        1ULL << 20U,                              // one MiB
        1ULL << 32U,                              // 4 GiB
        1ULL << 40U,                              // 1 TiB
        103130ULL * kQ32One,                      // ~100 us per operation, in Q32
        34816ULL * kQ32One,                       // a prefill token count, in Q32
        std::numeric_limits<std::uint64_t>::max() >> 32U,
        (std::numeric_limits<std::uint64_t>::max() >> 32U) + 1ULL,
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint64_t>::max() - 1ULL,
        std::numeric_limits<std::uint64_t>::max()};
    for (const std::uint64_t left : operands) {
        for (const std::uint64_t right : operands) {
            expect(wide_saturating_product(left, right) == oracle_saturating_product(left, right),
                   "saturating product differs from the native type");
            expect(wide_q32_product(left, right) == oracle_q32_product(left, right),
                   "Q32 product differs from the native type");
        }
    }
}

std::uint64_t oracle_fixture_attention_pairs(std::uint32_t prefix, std::uint32_t suffix) {
    const Oracle pairs = static_cast<Oracle>(prefix) * suffix +
                         static_cast<Oracle>(suffix) * (suffix + 1ULL) / 2U;
    if (pairs > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("prefill attention-pair count exceeds uint64");
    }
    return static_cast<std::uint64_t>(pairs);
}

// The two largest 32-bit counts overflow uint64, so this also pins that the 64-bit form keeps
// reporting the overflow the 128-bit form reported.
void test_fixture_attention_pairs() {
    const std::uint32_t kMaximum = std::numeric_limits<std::uint32_t>::max();
    const std::vector<std::uint32_t> counts{0U,        1U,          2U,          8U, 1024U,
                                            131072U,   kMaximum / 2U, kMaximum - 1U, kMaximum};
    for (const std::uint32_t prefix : counts) {
        for (const std::uint32_t suffix : counts) {
            bool oracle_threw     = false;
            bool checked_threw    = false;
            std::uint64_t oracle  = 0;
            std::uint64_t checked = 0;
            try {
                oracle = oracle_fixture_attention_pairs(prefix, suffix);
            } catch (const std::overflow_error&) { oracle_threw = true; }
            try {
                checked = fixture_attention_pairs(prefix, suffix);
            } catch (const std::overflow_error&) { checked_threw = true; }
            expect(oracle_threw == checked_threw,
                   "the 64-bit attention count disagrees with the 128-bit count on overflow");
            if (!oracle_threw && !checked_threw) {
                expect(oracle == checked, "the 64-bit attention count differs from the 128-bit count");
            }
        }
    }
}

#endif // defined(__SIZEOF_INT128__)

} // namespace

int main() {
    test_registered_contract_values();
#if defined(__SIZEOF_INT128__)
    test_word_boundary_arithmetic();
    test_contract_attention_work();
    test_contract_cost_products();
    test_fixture_attention_pairs();
#else
    // MSVC has no native 128-bit type to compare against, so only the registered contract values
    // above qualify the portable implementation there.
    std::cout << "no native unsigned __int128 oracle on this toolchain; registered values only\n";
#endif
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "contract 128-bit arithmetic matches the native type\n";
    return 0;
}
