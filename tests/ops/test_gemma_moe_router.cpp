// ninfer::ops::gemma_moe_router against its FP64 oracle.
//
// Two of the three cases are built by hand rather than sampled, because they pin the two points of
// the contract that a spot check cannot see: an exact tie at the top-k boundary resolves to the
// lower expert id, and the per-expert scale is applied after the renormalization, so the applied
// weights sum to the selected experts' mean scale rather than to one.
#include "ninfer/ops/gemma_moe_router.h"

#include "ops/direct_bf16_weight.h"
#include "ops/gemma_moe_router_ref.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1e-6F;
// The Op accumulates in FP32 from the same represented BF16 values the oracle uses in FP64, so a
// few ulp of relative disagreement is all the softmax and the renormalization can contribute.
constexpr PointwiseCriterion kWeightCriterion = {/*absolute*/ 1e-5, /*relative*/ 1e-3};

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

std::vector<std::uint16_t> scale_bits(std::size_t count, std::uint32_t seed, float low,
                                      float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return encode(values);
}

std::vector<std::uint16_t> constant_bits(std::size_t count, float value) {
    std::vector<float> values(count, value);
    return encode(values);
}

int run_case(const char* label, direct_bf16_weight::HostWeight router,
             const std::vector<std::uint16_t>& router_scale,
             const std::vector<std::uint16_t>& per_expert_scale,
             const std::vector<std::uint16_t>& x_bits, std::int32_t hidden, std::int32_t top_k,
             std::int32_t tokens, const std::vector<std::int32_t>* expected_ids = nullptr,
             double expected_weight_sum = 0.0) {
    const std::int32_t experts = router.n;
    const std::size_t ranks    = static_cast<std::size_t>(top_k) * tokens;

    const auto reference = gemma_moe_router_ref::run(x_bits, hidden, tokens, router, router_scale,
                                                     per_expert_scale, kEps, top_k);

    direct_bf16_weight::DeviceWeight router_device(router);
    GuardedDeviceBuffer x_device(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer router_scale_device(router_scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer expert_scale_device(per_expert_scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer ids_device(ranks * sizeof(std::int32_t));
    GuardedDeviceBuffer weights_device(ranks * sizeof(float));
    x_device.copy_from_host(x_bits.data(), x_device.bytes());
    router_scale_device.copy_from_host(router_scale.data(), router_scale_device.bytes());
    expert_scale_device.copy_from_host(per_expert_scale.data(), expert_scale_device.bytes());

    Tensor x_tensor(x_device.data(), DType::BF16, {hidden, tokens});
    Tensor ids_tensor(ids_device.data(), DType::I32, {top_k, tokens});
    Tensor weights_tensor(weights_device.data(), DType::FP32, {top_k, tokens});

    ops::GemmaMoeRouterWeights weights;
    weights.router           = router_device.view();
    weights.router_scale     = Tensor(router_scale_device.data(), DType::BF16, {hidden});
    weights.per_expert_scale = Tensor(expert_scale_device.data(), DType::BF16, {experts});
    const ops::GemmaMoeRouterGeometry geometry{hidden, experts, top_k, kEps};

    GuardedDeviceBuffer workspace_device(ops::gemma_moe_router_workspace_bytes(geometry, tokens));
    Tensor workspace(workspace_device.data(), DType::U8,
                     {static_cast<std::int32_t>(workspace_device.bytes())});
    ops::gemma_moe_router(x_tensor, weights, geometry, ids_tensor, weights_tensor, workspace,
                          nullptr);
    cuda_synchronize();

    int failures   = 0;
    const auto ids = from_device<std::int32_t>(ids_device.data(), static_cast<std::int64_t>(ranks));
    const std::vector<std::int32_t>& want_ids =
        expected_ids != nullptr ? *expected_ids : reference.ids;
    failures += verify_exact((std::string(label) + " expert ids").c_str(), ids, want_ids);
    const auto got_weights =
        from_device<float>(weights_device.data(), static_cast<std::int64_t>(ranks));
    const std::vector<double> got_as_double(got_weights.begin(), got_weights.end());
    failures += verify_pointwise(std::string(label) + " expert weights", got_as_double,
                                 reference.weights, kWeightCriterion);
    if (expected_weight_sum != 0.0) {
        double sum = 0.0;
        for (std::size_t rank = 0; rank < ranks; ++rank) { sum += got_weights[rank]; }
        if (std::abs(sum - expected_weight_sum) > 1e-3) {
            std::cerr << label << ": weight sum is " << sum << ", expected " << expected_weight_sum
                      << '\n';
            ++failures;
        }
    }
    failures += router_device.verify_preserved(label);
    return failures;
}

// The geometry the artifact stores: 128 experts over 2816 hidden, top-8. One token, a token count
// that leaves a partial tile of the projection's token tiling, and a prefill-sized pass.
int run_real_geometry_case(std::int32_t tokens) {
    constexpr std::int32_t kHidden  = 2816;
    constexpr std::int32_t kExperts = 128;
    constexpr std::int32_t kTopK    = 8;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(x, 11, -3.0F, 3.0F);
    round_to_bf16(x);
    return run_case(("real geometry T=" + std::to_string(tokens)).c_str(),
                    direct_bf16_weight::make_patterned(kExperts, kHidden, 7),
                    scale_bits(kHidden, 12, 0.8F, 1.2F), scale_bits(kExperts, 13, 0.5F, 1.5F),
                    encode(x), kHidden, kTopK, tokens);
}

// Two experts tie for the largest probability and one rank is available, so the lower id must win.
int run_tie_case() {
    constexpr std::int32_t kHidden = 8;
    direct_bf16_weight::HostWeight router;
    router.n = 4;
    router.k = kHidden;
    router.bits.resize(static_cast<std::size_t>(4) * kHidden);
    for (std::int32_t h = 0; h < kHidden; ++h) {
        // Rows 1 and 2 are identical and positive; rows 0 and 3 are their exact negation, so the
        // two projections are bitwise equal and strictly above the others.
        router.bits[static_cast<std::size_t>(0) * kHidden + h] = f32_to_bf16(-1.0F);
        router.bits[static_cast<std::size_t>(1) * kHidden + h] = f32_to_bf16(1.0F);
        router.bits[static_cast<std::size_t>(2) * kHidden + h] = f32_to_bf16(1.0F);
        router.bits[static_cast<std::size_t>(3) * kHidden + h] = f32_to_bf16(-1.0F);
    }
    const std::vector<std::int32_t> expected{1};
    return run_case("top-k tie", std::move(router), constant_bits(kHidden, 1.0F),
                    constant_bits(4, 1.0F), constant_bits(kHidden, 1.0F), kHidden, 1, 1, &expected);
}

// The two winners are experts 2 and 3, which are exactly tied, with per-expert scales 3 and 4. The
// weights are renormalized before the scale, so their sum is the mean of those scales and not one.
int run_scale_order_case() {
    constexpr std::int32_t kHidden = 8;
    direct_bf16_weight::HostWeight router;
    router.n = 4;
    router.k = kHidden;
    router.bits.resize(static_cast<std::size_t>(4) * kHidden);
    for (std::int32_t h = 0; h < kHidden; ++h) {
        for (std::int32_t e = 0; e < 4; ++e) {
            router.bits[static_cast<std::size_t>(e) * kHidden + h] =
                f32_to_bf16(e < 2 ? -1.0F : 1.0F);
        }
    }
    const std::vector<std::int32_t> expected{2, 3};
    return run_case("renormalize before the per-expert scale", std::move(router),
                    constant_bits(kHidden, 1.0F), encode({1.0F, 1.0F, 3.0F, 4.0F}),
                    constant_bits(kHidden, 1.0F), kHidden, 2, 1, &expected, 3.5);
}

} // namespace

int main() {
    int failures = 0;
    try {
        for (const std::int32_t tokens : {1, 5, 8, 300}) { failures += run_real_geometry_case(tokens); }
        failures += run_tie_case();
        failures += run_scale_order_case();
    } catch (const std::exception& error) {
        std::cerr << "gemma_moe_router: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) {
        std::cout << "OK: gemma_moe_router matches its oracle, the tie-break and the scale order\n";
    }
    return failures == 0 ? 0 : 1;
}
