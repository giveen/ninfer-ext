#include "core/checked_math.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

// Runs `operation`, which must throw `Exception` and name `label` in its message.
template <typename Exception, typename Operation>
void expect_throws(Operation operation, const char* message, const char* label) {
    try {
        operation();
    } catch (const Exception& error) {
        expect(std::string(error.what()).find(label) != std::string::npos,
               "thrown message must carry the label");
        return;
    }
    expect(false, message);
}

void test_checked_add() {
    constexpr auto kU32Maximum = std::numeric_limits<std::uint32_t>::max();
    expect(ninfer::checked_add<std::uint32_t>(0, 0, "add") == 0, "zero add");
    expect(ninfer::checked_add<std::uint32_t>(kU32Maximum - 1U, 1U, "add") == kU32Maximum,
           "add up to the maximum must succeed");
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::checked_add<std::uint32_t>(std::numeric_limits<std::uint32_t>::max(), 1U,
                                                     "wide add overflows");
        },
        "add past the maximum must fail", "wide add overflows");

    // A narrow type promotes to int inside the helper; its own maximum still bounds the result.
    constexpr std::uint8_t kHigh  = 200;
    constexpr std::uint8_t kSteps = 55;
    expect(ninfer::checked_add<std::uint16_t>(kHigh, kSteps, "add") == 255,
           "narrow add must reach the promoted sum");
    expect(ninfer::checked_add<std::uint8_t>(kHigh, kSteps, "add") == 255,
           "narrow add must reach the type maximum");
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::checked_add<std::uint8_t>(std::numeric_limits<std::uint8_t>::max(),
                                                    static_cast<std::uint8_t>(1), "narrow add");
        },
        "narrow add past the maximum must fail", "narrow add");
}

void test_checked_mul() {
    constexpr auto kU32Maximum = std::numeric_limits<std::uint32_t>::max();
    expect(ninfer::checked_mul<std::uint32_t>(0, kU32Maximum, "mul") == 0,
           "a zero operand must short-circuit before the division");
    expect(ninfer::checked_mul<std::uint32_t>(kU32Maximum, 0, "mul") == 0,
           "a zero second operand must not overflow");
    expect(ninfer::checked_mul<std::uint32_t>(kU32Maximum, 1U, "mul") == kU32Maximum,
           "multiplying by one must succeed");
    expect(ninfer::checked_mul<std::uint32_t>(kU32Maximum / 2U, 2U, "mul") == kU32Maximum - 1U,
           "the largest even product must succeed");
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::checked_mul<std::uint32_t>(std::numeric_limits<std::uint32_t>::max(), 2U,
                                                     "wide mul overflows");
        },
        "product past the maximum must fail", "wide mul overflows");

    // 16 * 16 fits an int but not a uint8_t, so the checked type, not the promoted type, bounds it.
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::checked_mul<std::uint8_t>(static_cast<std::uint8_t>(16),
                                                    static_cast<std::uint8_t>(16), "narrow mul");
        },
        "narrow product past the maximum must fail", "narrow mul");
}

void test_align_up() {
    constexpr auto kU32Maximum = std::numeric_limits<std::uint32_t>::max();
    expect(ninfer::align_up<std::uint32_t>(0, 8, "align") == 0, "zero value must stay zero");
    expect(ninfer::align_up<std::uint32_t>(8, 8, "align") == 8, "aligned value must not move");
    expect(ninfer::align_up<std::uint32_t>(9, 8, "align") == 16, "value must round up");
    expect(ninfer::align_up<std::uint32_t>(7, 1, "align") == 7, "unit alignment must be a no-op");
    expect(ninfer::align_up<std::uint32_t>(kU32Maximum - 7U, 8, "align") == kU32Maximum - 7U,
           "already aligned maximum-adjacent value must not move");

    expect_throws<std::invalid_argument>(
        [] { (void)ninfer::align_up<std::uint32_t>(1, 0, "zero align"); },
        "zero alignment must fail", "zero align");
    expect_throws<std::invalid_argument>(
        [] { (void)ninfer::align_up<std::uint32_t>(1, 3, "odd align"); },
        "non-power-of-two alignment must fail", "odd align");

    // Rounding up adds alignment - 1 before masking, which must not wrap the value's own type.
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::align_up<std::uint32_t>(std::numeric_limits<std::uint32_t>::max(), 2,
                                                  "align overflow");
        },
        "alignment padding past the maximum must fail", "align overflow");
    expect_throws<std::overflow_error>(
        [] {
            (void)ninfer::align_up<std::uint8_t>(static_cast<std::uint8_t>(200),
                                                 static_cast<std::uint8_t>(64), "narrow align");
        },
        "narrow alignment padding past the maximum must fail", "narrow align");
}

} // namespace

int main() {
    test_checked_add();
    test_checked_mul();
    test_align_up();
    return failures == 0 ? 0 : 1;
}
