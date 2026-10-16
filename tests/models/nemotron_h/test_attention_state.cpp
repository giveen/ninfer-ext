// The Nemotron-H attention state: the geometry it owns, the route over it, and the three things
// that are silent when they are wrong.
//
//   * Which extent is the head. This family's plane is `[head_dim, kv_heads, capacity, 1]`: the
//   head
//     width is the innermost axis, the KV head the next one, and the two differ (128 against 2), so
//     a swapped pair is visible here in a way it would not be in a family where both are 128. The
//     strides are asserted directly, and then the route is compared against a reference that reads
//     the same numbers under the documented index order.
//   * The order of the append against the query. A token's own key and value participate in its own
//     output, which the reference's own attention does by appending first. With exactly one visible
//     key the softmax is 1 and the output *is* that token's value, so the first query's output is
//     compared element by element with the value appended for it: an append-after-query route would
//     write exact zeros there instead.
//   * The visibility of slots no token has written. The state hands the Op the whole lane and lets
//     the positions decide, so an unwritten slot's bytes are never read. Every plane is filled with
//     a huge finite value before the appends here: if a single unwritten slot were visible, the
//     score of that key would dominate every softmax and the comparison would fail by orders of
//     magnitude rather than by a rounding.
//
// The state needs no artifact: it is driven by a config and by operands this test generates, and
// the geometry is the checkpoint's own attention shape (head 128, 2 KV heads) because the Op's
// launcher admits 32, 64, 128 and 256 only.
#include "core/device.h"
#include "models/nemotron_h/attention_state.h"
#include "ninfer/ops/sliding_causal_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace nemotron = ninfer::models::nemotron_h;

constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kQueryHeads = 4;
constexpr std::int32_t kKvHeads    = 2;
constexpr std::int32_t kLayers     = 2;
constexpr std::int32_t kLanes      = 2;
constexpr std::int32_t kCapacity   = 8;
constexpr std::int32_t kTokens     = 3;

// A BF16 value large enough that a softmax which saw it would be dominated by it, and finite so a
// visible garbage key is an error rather than a NaN that could look like a bug in the comparison.
constexpr std::uint16_t kDirty = 0x7f00;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void check_bytes(std::size_t actual, std::size_t expected, const std::string& what) {
    if (actual != expected) {
        std::cerr << "FAIL: " << what << ": got " << actual << " B, expected " << expected
                  << " B\n";
        ++failures;
    }
}

template <class Fn>
void expect_refusal(const std::string& label, Fn&& fn, const std::string& expected_text) {
    try {
        fn();
        std::cerr << "FAIL: " << label << ": the call was accepted\n";
        ++failures;
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(expected_text) == std::string::npos) {
            std::cerr << "FAIL: " << label << ": refused with another reason: " << error.what()
                      << '\n';
            ++failures;
        }
    }
}

nemotron::AttentionConfig attention_config(std::int32_t head_dim = kHeadDim) {
    nemotron::AttentionConfig attention;
    attention.num_attention_heads = kQueryHeads;
    attention.num_key_value_heads = kKvHeads;
    attention.head_dim            = head_dim;
    attention.rope_theta          = 0.0F;
    return attention;
}

// Deterministic operands, in the flattened element order of the tensors they fill: `(d, h, t)` for
// a query, `(d, g, s)` for a key or a value. Rounded to BF16 because that is what the Op reads and
// what the reference must be computed from.
std::vector<float> pattern(std::size_t count, float phase) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::sin(0.37F * static_cast<float>(i) + phase) * 2.0F;
    }
    test::round_to_bf16(values);
    return values;
}

// The reference: exactly the causal set, this model's own scale, no window and no rotation.
//   score(k) = scale * sum_d q[d,h,t] * k[d,g(k),s]
//   out[d,h,t] = sum_{s <= t} softmax_k(score) * v[d,g(k),s]
std::vector<double> reference(const std::vector<double>& q, const std::vector<double>& k,
                              const std::vector<double>& v, std::int32_t tokens) {
    const double scale            = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    const std::int32_t group_size = kQueryHeads / kKvHeads;
    std::vector<double> out(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens, 0.0);
    std::vector<double> scores(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kQueryHeads; ++h) {
            const std::int32_t g = h / group_size;
            double best          = -std::numeric_limits<double>::infinity();
            for (std::int32_t s = 0; s <= t; ++s) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += q[static_cast<std::size_t>(t * kQueryHeads + h) * kHeadDim + d] *
                           k[static_cast<std::size_t>(s * kKvHeads + g) * kHeadDim + d];
                }
                scores[static_cast<std::size_t>(s)] = scale * dot;
                best = std::max(best, scores[static_cast<std::size_t>(s)]);
            }
            double total = 0.0;
            std::vector<double> weights(static_cast<std::size_t>(tokens), 0.0);
            for (std::int32_t s = 0; s <= t; ++s) {
                weights[static_cast<std::size_t>(s)] =
                    std::exp(scores[static_cast<std::size_t>(s)] - best);
                total += weights[static_cast<std::size_t>(s)];
            }
            for (std::int32_t s = 0; s <= t; ++s) {
                const double weight = weights[static_cast<std::size_t>(s)] / total;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    out[static_cast<std::size_t>(t * kQueryHeads + h) * kHeadDim + d] +=
                        weight * v[static_cast<std::size_t>(s * kKvHeads + g) * kHeadDim + d];
                }
            }
        }
    }
    return out;
}

double relative_l2(const std::vector<double>& actual, const std::vector<double>& expected) {
    double error = 0.0;
    double norm  = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double difference = actual[i] - expected[i];
        error += difference * difference;
        norm += expected[i] * expected[i];
    }
    return norm == 0.0 ? std::sqrt(error) : std::sqrt(error / norm);
}

std::vector<double> to_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

// The geometry the state owns, and the multiplication by lanes, checked against the arithmetic the
// header states.
void layout_arithmetic() {
    const std::size_t plane         = static_cast<std::size_t>(kHeadDim) * kKvHeads * kCapacity * 2;
    const std::size_t expected_keys = static_cast<std::size_t>(kLayers) * kLanes * plane;
    const std::size_t expected_positions = static_cast<std::size_t>(kLanes) * kCapacity * 4;
    std::cout << "one plane " << plane << " B, keys " << expected_keys << " B, values "
              << expected_keys << " B, positions " << expected_positions << " B\n";

    nemotron::AttentionState two(attention_config(), kLayers, kCapacity, 2);
    nemotron::AttentionState one(attention_config(), kLayers, kCapacity, 1);
    check_bytes(two.keys_bytes(), expected_keys, "the K planes of two lanes");
    check_bytes(two.values_bytes(), expected_keys, "the V planes of two lanes");
    check_bytes(two.positions_bytes(), expected_positions, "the position rows of two lanes");
    check_bytes(one.keys_bytes(), expected_keys / 2, "one lane's K planes");
    check_bytes(one.positions_bytes(), expected_positions / 2, "one lane's position row");
    check_bytes(nemotron::AttentionState::device_bytes(attention_config(), kLayers, kCapacity, 2),
                two.allocated_bytes(), "the pure byte report and the constructed state");
    check(two.allocated_bytes() >= two.keys_bytes() + two.values_bytes() + two.positions_bytes(),
          "the allocation holds the three classes");
    check_bytes(two.layers(), kLayers, "the layer count");
    check_bytes(two.lanes(), kLanes, "the lane count");
    check_bytes(two.capacity(), kCapacity, "the capacity");

    // The two extents that must not swap: the head width is the innermost axis of a plane and the
    // KV head the next one. A swapped pair keeps every byte count identical, so the strides are
    // what says which is which.
    const nemotron::AttentionStateView view = two.layer(0, 0);
    check(view.keys.dtype == DType::BF16 && view.keys.is_contiguous(),
          "a K plane is contiguous BF16");
    check_bytes(static_cast<std::size_t>(view.keys.ne[0]), kHeadDim, "the plane's head extent");
    check_bytes(static_cast<std::size_t>(view.keys.ne[1]), kKvHeads, "the plane's KV head extent");
    check_bytes(static_cast<std::size_t>(view.keys.ne[2]), kCapacity, "the plane's token extent");
    check_bytes(static_cast<std::size_t>(view.keys.nb[0]), sizeof(std::uint16_t),
                "the head width is the innermost axis of a K plane");
    check_bytes(static_cast<std::size_t>(view.keys.nb[1]), kHeadDim * sizeof(std::uint16_t),
                "the KV head is the next axis of a K plane");
    check_bytes(static_cast<std::size_t>(view.keys.nb[2]),
                static_cast<std::size_t>(kHeadDim) * kKvHeads * sizeof(std::uint16_t),
                "a token is one run of head-dimension rows");
    check_bytes(view.positions.bytes(), expected_positions / kLanes, "a lane's position row");
    check(view.positions.dtype == DType::I32 && view.positions.is_contiguous(),
          "a position row is contiguous I32");
    check(view.positions.ne[0] == kCapacity && view.positions.ne[1] == 1,
          "a position row is [capacity, 1]");

    // The Op's own key-split workspace at this family's full context, which is the number a caller
    // admitting a decode step has to hold: the state hands the Op the whole lane, so the split is
    // sized by the capacity rather than by how much of it is filled.
    const std::size_t decode_workspace = ops::sliding_causal_attention_workspace_bytes(
        ops::AttentionHeadGeometry{kHeadDim, 32, kKvHeads}, 1, 262144, 1);
    std::cout << "one decode token over 262144 keys: the Op's split workspace is "
              << decode_workspace << " B\n";
    check(decode_workspace > 0, "a decode step over a full context needs the split workspace");
    check(decode_workspace < 16ULL * 1024 * 1024,
          "and it is small enough to hold beside the state");
}

// The route: the append, the query after it, the lane's own positions, and the reference.
void route_matches_the_reference(DeviceExecutionView execution) {
    nemotron::AttentionState state(attention_config(), kLayers, kCapacity, kLanes);
    state.reset_all(execution.stream);
    test::cuda_synchronize(execution.stream);

    // Fill every plane with a value a softmax could not absorb, so an unwritten slot that leaked
    // into the attention would dominate every score.
    std::vector<std::uint16_t> dirty(static_cast<std::size_t>(state.keys_bytes()) / 2, kDirty);
    for (const Tensor& plane : {state.layer(0, 0).keys, state.layer(0, 0).values}) {
        test::cuda_check(cudaMemcpyAsync(plane.data, dirty.data(), plane.bytes(),
                                         cudaMemcpyHostToDevice, execution.stream),
                         "dirtying a plane");
    }
    test::cuda_check(cudaMemsetAsync(state.layer(1, 0).keys.data, 0x7f,
                                     state.layer(1, 0).keys.bytes(), execution.stream),
                     "dirtying the second lane's plane");
    test::cuda_synchronize(execution.stream);

    // The lane's positions, before any token: every slot unwritten.
    const nemotron::AttentionStateView fresh = state.layer(0, 0);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(kCapacity), 0);
    test::cuda_check(cudaMemcpy(positions.data(), fresh.positions.data,
                                static_cast<std::size_t>(kCapacity) * sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost),
                     "reading the lane's positions");
    check(std::all_of(positions.begin(), positions.end(),
                      [](std::int32_t value) { return value == nemotron::kUnwrittenPosition; }),
          "a reset lane's every slot is unwritten");

    const std::vector<float> q =
        pattern(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens, 0.0F);
    const std::vector<float> k =
        pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens, 1.1F);
    const std::vector<float> v =
        pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens, 2.2F);
    DeviceBuffer dq = test::to_device_bf16(q);
    DeviceBuffer dk = test::to_device_bf16(k);
    DeviceBuffer dv = test::to_device_bf16(v);
    DeviceBuffer dout(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens *
                      sizeof(std::uint16_t));
    const Tensor query(dq.p, DType::BF16, {kHeadDim, kQueryHeads, kTokens, 1});
    const Tensor keys(dk.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    const Tensor values(dv.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    Tensor out(dout.p, DType::BF16, {kHeadDim, kQueryHeads, kTokens, 1});

    state.append(0, 0, keys, values, kTokens, execution.stream);
    check(state.pending(0) == kTokens, "the lane has a pending step");
    check(state.filled(0) == 0, "the lane's sequence is still empty until it advances");
    state.attend(0, 0, query, out, Tensor{}, execution);
    test::cuda_synchronize(execution.stream);

    // The query's own positions are the appended ones.
    std::vector<std::int32_t> appended(static_cast<std::size_t>(kCapacity), 0);
    test::cuda_check(cudaMemcpy(appended.data(), state.layer(0, 0).positions.data,
                                static_cast<std::size_t>(kCapacity) * sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost),
                     "reading the lane's positions after the append");
    bool positions_ok = true;
    for (std::int32_t slot = 0; slot < kCapacity; ++slot) {
        const std::int32_t expected = slot < kTokens ? slot : nemotron::kUnwrittenPosition;
        positions_ok = positions_ok && appended[static_cast<std::size_t>(slot)] == expected;
    }
    check(positions_ok, "the appended tokens hold their own positions and the rest stay unwritten");

    const std::vector<double> actual =
        test::from_device_bf16(dout.p, static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens);
    const std::vector<double> expected =
        reference(to_double(q), to_double(k), to_double(v), kTokens);
    const double rel_l2 = relative_l2(actual, expected);
    std::cout << "the route against the unbounded causal reference: relative L2 " << rel_l2 << '\n';
    check(rel_l2 < 2.0e-2, "the route's output equals the causal reference");

    // Append-before-query, exactly: with one visible key the softmax is 1 and the output *is* the
    // value appended for that token. An append-after-query route writes exact BF16 zero here.
    double max_difference = 0.0;
    for (std::int32_t h = 0; h < kQueryHeads; ++h) {
        const std::int32_t g = h / (kQueryHeads / kKvHeads);
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const double got  = actual[static_cast<std::size_t>(h) * kHeadDim + d];
            const double want = static_cast<double>(v[static_cast<std::size_t>(g) * kHeadDim + d]);
            max_difference    = std::max(max_difference, std::fabs(got - want));
        }
    }
    std::cout << "the first query's own value: max absolute difference " << max_difference << '\n';
    check(max_difference < 1.0e-3, "the first query's output is its own value, appended first");
}

// Nothing aliases between lanes: an append to one lane's plane leaves another lane's bytes alone.
void lanes_do_not_alias(DeviceExecutionView execution) {
    nemotron::AttentionState state(attention_config(), kLayers, kCapacity, kLanes);
    state.reset_all(execution.stream);
    test::cuda_synchronize(execution.stream);

    const std::size_t plane_bytes = static_cast<std::size_t>(kHeadDim) * kKvHeads * kCapacity * 2;
    const auto lane0              = state.layer(0, 0);
    const auto lane1              = state.layer(1, 0);
    const auto lane0_layer1       = state.layer(0, 1);
    const auto offset = [](void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
    check_bytes(offset(lane1.keys.data) - offset(lane0.keys.data), 2 * plane_bytes,
                "lane 1's K plane follows lane 0's by exactly two planes");
    check_bytes(offset(lane0_layer1.keys.data) - offset(lane0.keys.data), plane_bytes,
                "layer 1 follows layer 0 within a lane");

    std::vector<std::uint16_t> dirty(plane_bytes / 2, kDirty);
    test::cuda_check(cudaMemcpyAsync(lane1.keys.data, dirty.data(), lane1.keys.bytes(),
                                     cudaMemcpyHostToDevice, execution.stream),
                     "dirtying lane 1's K plane");
    test::cuda_synchronize(execution.stream);

    const std::vector<float> k =
        pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens, 3.3F);
    const std::vector<float> v =
        pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens, 4.4F);
    DeviceBuffer dk = test::to_device_bf16(k);
    DeviceBuffer dv = test::to_device_bf16(v);
    const Tensor keys(dk.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    const Tensor values(dv.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    state.append(0, 0, keys, values, kTokens, execution.stream);
    test::cuda_synchronize(execution.stream);

    std::vector<std::uint16_t> readback(plane_bytes / 2, 0);
    test::cuda_check(
        cudaMemcpy(readback.data(), lane1.keys.data, lane1.keys.bytes(), cudaMemcpyDeviceToHost),
        "reading lane 1's K plane back");
    bool intact = true;
    for (const std::uint16_t value : readback) { intact = intact && value == kDirty; }
    check(intact, "an append to lane 0 leaves lane 1's plane untouched");
}

// Every misuse refuses by name, and a head width the Op's launcher does not admit is refused by the
// route rather than padded here.
void refusals(DeviceExecutionView execution) {
    nemotron::AttentionState state(attention_config(), kLayers, kCapacity, kLanes);
    state.reset_all(execution.stream);
    test::cuda_synchronize(execution.stream);

    const std::vector<float> k =
        pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens, 5.5F);
    DeviceBuffer dk = test::to_device_bf16(k);
    DeviceBuffer dv = test::to_device_bf16(k);
    const Tensor keys(dk.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    const Tensor values(dv.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
    const Tensor wrong_shape(dk.p, DType::BF16, {kHeadDim, kKvHeads, kTokens - 1, 1});
    DeviceBuffer dout(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens *
                      sizeof(std::uint16_t));
    Tensor out(dout.p, DType::BF16, {kHeadDim, kQueryHeads, kTokens, 1});

    expect_refusal(
        "attending before any append", [&] { state.attend(0, 0, out, out, Tensor{}, execution); },
        "has nothing appended");
    expect_refusal(
        "a layer that has not appended this step",
        [&] {
            state.append(0, 0, keys, values, kTokens, execution.stream);
            state.attend(0, 1, out, out, Tensor{}, execution);
        },
        "before its query runs");
    expect_refusal(
        "appending the same layer twice",
        [&] { state.append(0, 0, keys, values, kTokens, execution.stream); },
        "already appended this step");
    expect_refusal(
        "a layer appending a different token count",
        [&] { state.append(0, 1, wrong_shape, wrong_shape, kTokens - 1, execution.stream); },
        "writes the same tokens");
    expect_refusal(
        "advancing a step a layer never appended", [&] { state.advance(0); },
        "did not append this step");
    expect_refusal(
        "an operand of the wrong shape",
        [&] { state.append(1, 0, wrong_shape, wrong_shape, kTokens, execution.stream); },
        "must be contiguous BF16");
    expect_refusal(
        "a layer beyond the state", [&] { state.append(0, 9, keys, values, 1, nullptr); },
        "is not one of this state's 2 attention layers");
    expect_refusal(
        "a lane beyond the state", [&] { (void)state.filled(2); },
        "is not one of this state's 2 lanes");

    // A head width the launcher does not admit: the state owns it, the route refuses it, and the
    // message names the carry the composition uses for its passes.
    nemotron::AttentionState narrow(attention_config(16), 1, kCapacity, 1);
    check_bytes(narrow.keys_bytes(), static_cast<std::size_t>(16) * kKvHeads * kCapacity * 2,
                "a non-admitted head width is still ownable");
    expect_refusal(
        "a head width the launcher does not admit", [&] { (void)narrow.workspace_bytes(1); },
        "zero-carry");
}

// A full lane's capacity is refused rather than wrapped, and a step's advance is what makes the
// tokens part of the sequence.
void capacity_and_advance(DeviceExecutionView execution) {
    nemotron::AttentionState state(attention_config(), 1, 2, 1);
    state.reset_all(execution.stream);
    test::cuda_synchronize(execution.stream);

    const std::vector<float> k = pattern(static_cast<std::size_t>(kHeadDim) * kKvHeads * 2, 6.6F);
    DeviceBuffer dk            = test::to_device_bf16(k);
    DeviceBuffer dv            = test::to_device_bf16(k);
    const Tensor keys(dk.p, DType::BF16, {kHeadDim, kKvHeads, 2, 1});
    const Tensor values(dv.p, DType::BF16, {kHeadDim, kKvHeads, 2, 1});
    const Tensor more(dk.p, DType::BF16, {kHeadDim, kKvHeads, 1, 1});

    state.append(0, 0, keys, values, 2, execution.stream);
    state.advance(0);
    check(state.filled(0) == 2, "advancing moves the lane's sequence past the step");
    check(state.pending(0) == 0, "an advanced step has no pending tokens");
    expect_refusal(
        "appending past the capacity", [&] { state.append(0, 0, more, more, 1, execution.stream); },
        "exceeds that");
    expect_refusal(
        "advancing with no step", [&] { state.advance(0); }, "no appended step to advance");
    // A reset lane starts empty again.
    state.reset(0, execution.stream);
    test::cuda_synchronize(execution.stream);
    check(state.filled(0) == 0 && state.pending(0) == 0, "a reset lane is empty");
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "nemotron_h attention state: no CUDA device, skipping\n";
        return 77;
    }
    try {
        DeviceContext context(0);
        const DeviceExecutionView execution = context.execution_view();
        layout_arithmetic();
        route_matches_the_reference(execution);
        lanes_do_not_alias(execution);
        refusals(execution);
        capacity_and_advance(execution);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " nemotron_h attention state check(s) failed\n";
        return 1;
    }
    std::cout << "nemotron_h attention state: ok\n";
    return 0;
}
