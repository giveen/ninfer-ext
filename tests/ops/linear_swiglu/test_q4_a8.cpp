#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // The W4A8 route is selected for large T; the small-T cases still exercise the A16 routes
        // under the (looser) A8 activation-compute criterion.
        constexpr std::array<std::int32_t, 10> kTokenCases{
            128, 255, 256, 257, 384, 512, 513, 640, 641, 1024,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q4_A8",
            {QType::Q4_G64_FP16, 34816, 5120, 17408, 1401U, ActivationCompute::A8}, kTokenCases,
            std::array<std::int32_t, 3>{256, 512, 1024});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
