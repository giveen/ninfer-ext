// The Nemotron-H Program: the state it owns, the numbers it reports, and the refusals it makes.
//
// Three things are checked here and they are different in kind.
//
// The first is arithmetic. The state is three classes with three different geometries -- a Mamba
// recurrent plane, a Mamba convolution history and the attention K/V -- and every byte the Program
// reports is derived from the config and the lane count rather than from the checkpoint's own
// numbers, so the test states each formula in the config's own terms and compares it with what the
// Program planned. The classes are also checked to be linear in the lane count, which is the
// property admission depends on: a lane costs the same whether it is the first or the eighth. The
// config used for it is deliberately unlike the checkpoint's in every extent, so a number that came
// from a constant rather than from the config cannot pass.
//
// The second is the state's own identity: the recurrent plane's key extent is the one
// `ops::mamba2_scan` pins (128), not the config's narrower `state_size`, because a narrower state
// is carried by zero-extending the key axis and the plane the Op reads has to be that extent; and a
// lane's frontier is the *zero* state after `create`, which is what the landed prefill forward
// starts from. Both are asserted against the state on the device, not against a comment.
//
// The third is that every entry point that would compute refuses with the declared message. The
// landed forward is a zero-state prefill: it keeps no state between calls, so driving a lane
// through it would compute a wrong model rather than lose a feature. That has to be a refusal
// naming the missing piece, never a successful return -- and an argument the caller can fix has to
// be refused as the argument error it is, before the missing piece.
//
// The Program is driven by a hand-bound `Model` -- the config's own shape with no weights, since
// nothing in the state's construction reads a parameter -- so this needs no artifact. The one case
// that needs the real checkpoint's config reads it from the artifact's cold directory without
// materializing its 45 GB, and prints a skip when NINFER_NEMOTRON_ARTIFACT is unset.
#include "artifact/reader.h"
#include "core/device.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/forward.h"
#include "models/nemotron_h/program.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace nemotron = ninfer::models::nemotron_h;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void check_bytes(std::size_t actual, std::size_t expected, const std::string& what) {
    if (actual != expected) {
        std::cerr << "FAIL: " << what << ": planned " << actual << " B, expected " << expected
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

// The scaled-down hybrid this test plans state for: 2 Mamba, 2 attention and 1 MoE block.
nemotron::MoeBankConfig tiny_bank() {
    nemotron::MoeBankConfig bank;
    bank.num_experts              = 8;
    bank.experts_per_token        = 2;
    bank.intermediate_size        = 32;
    bank.shared_intermediate_size = 32;
    bank.latent_size              = 16;
    return bank;
}

nemotron::TextConfig synthetic_config() {
    nemotron::TextConfig config;
    config.hidden_size             = 64;
    config.vocab_size              = 256;
    config.max_position_embeddings = 4096;
    config.tie_word_embeddings     = false;
    config.rms_norm_eps            = 1.0e-5F;
    config.block_kinds             = {nemotron::BlockKind::Mamba, nemotron::BlockKind::Attention,
                                      nemotron::BlockKind::Moe, nemotron::BlockKind::Mamba,
                                      nemotron::BlockKind::Attention};
    nemotron::MambaConfig mamba;
    mamba.num_heads = 4;
    mamba.head_dim  = 16;
    // Narrower than the scan's own 128, so the recurrent plane's extent can only come from the
    // Op's contract; the assertion on it is in state_identity().
    mamba.state_size    = 8;
    mamba.n_groups      = 2;
    mamba.conv_kernel   = 4;
    mamba.chunk_size    = 128;
    mamba.time_step_min = 0.001F;
    mamba.conv_bias     = true;
    config.mamba        = mamba;
    nemotron::AttentionConfig attention;
    attention.num_attention_heads = 4;
    attention.num_key_value_heads = 2;
    attention.head_dim            = 16;
    attention.rope_theta          = 0.0F;
    config.attention              = attention;
    nemotron::RouterConfig router;
    router.scaling_factor  = 1.0F;
    router.normalize_top_k = true;
    router.correction_bias = true;
    config.router          = router;
    config.moe_banks       = {std::nullopt, std::nullopt, tiny_bank(), std::nullopt, std::nullopt};
    return config;
}

struct Shape {
    std::uint32_t lanes    = 1;
    std::uint32_t capacity = 128;
    std::optional<std::uint32_t> chunk;
};

EngineOptions options_for(const Shape& shape) {
    EngineOptions options;
    options.device          = 0;
    options.max_context     = shape.capacity;
    options.max_concurrency = shape.lanes;
    options.prefill_chunk   = shape.chunk;
    options.kv_cache        = KvCacheStorage::BFloat16;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(shape.capacity);
    return options;
}

// The three state classes in the config's own terms, so a number that came from anywhere else
// cannot match. A class whose mixer the config does not have is zero.
struct ExpectedState {
    std::size_t conv      = 0;
    std::size_t recurrent = 0;
    // The attention state's three storage classes: the K and V planes and the lane's positions.
    std::size_t keys      = 0;
    std::size_t values    = 0;
    std::size_t positions = 0;

    [[nodiscard]] std::size_t kv() const noexcept { return keys + values + positions; }
};

ExpectedState expected_state(const nemotron::TextConfig& config, const Shape& shape) {
    std::uint64_t mamba_layers     = 0;
    std::uint64_t attention_layers = 0;
    for (const nemotron::BlockKind kind : config.block_kinds) {
        if (kind == nemotron::BlockKind::Mamba) { ++mamba_layers; }
        if (kind == nemotron::BlockKind::Attention) { ++attention_layers; }
    }
    const std::uint64_t slots = shape.lanes;
    ExpectedState expected;
    if (config.mamba) {
        const nemotron::MambaConfig& mamba = *config.mamba;
        // conv: [channels, conv_kernel - 1, lanes] as BF16, one plane per Mamba layer.
        const std::uint64_t conv_per_layer =
            mamba.conv_channels() * (mamba.conv_kernel - 1) * slots * sizeof(std::uint16_t);
        // recurrent: [kMambaScanStateExtent, head_dim, heads, lanes] as FP32.
        const std::uint64_t recurrent_per_layer = std::uint64_t(nemotron::kMambaScanStateExtent) *
                                                  mamba.head_dim * mamba.num_heads * slots *
                                                  sizeof(float);
        expected.conv      = static_cast<std::size_t>(mamba_layers * conv_per_layer);
        expected.recurrent = static_cast<std::size_t>(mamba_layers * recurrent_per_layer);
    }
    if (config.attention) {
        const nemotron::AttentionConfig& attention = *config.attention;
        // The attention state is the config's own geometry with no padding and no page rounding:
        // contiguous BF16 [head_dim, kv_heads, capacity, 1] keys and the same for values, per
        // attention layer per lane, plus one I32 position row of `capacity` per lane.
        const std::uint64_t plane_bytes = std::uint64_t(attention.head_dim) *
                                          attention.num_key_value_heads * shape.capacity *
                                          sizeof(std::uint16_t);
        expected.keys   = static_cast<std::size_t>(attention_layers * slots * plane_bytes);
        expected.values = static_cast<std::size_t>(attention_layers * slots * plane_bytes);
        expected.positions =
            static_cast<std::size_t>(slots * std::uint64_t(shape.capacity) * sizeof(std::int32_t));
    }
    return expected;
}

void report(const std::string& label, const nemotron::StateMemorySummary& summary) {
    std::cout << label << ": mamba recurrent " << summary.mamba_recurrent_bytes << " B, mamba conv "
              << summary.mamba_conv_bytes << " B, attention KV " << summary.attention_kv_bytes
              << " B, state allocation " << summary.state_allocation_bytes << " B, workspace "
              << summary.workspace_bytes << " B, device total " << summary.device_bytes() << " B\n";
}

// The Program's own numbers against the config's arithmetic, and the classes against a lane count
// that multiplies them.
void state_arithmetic(const nemotron::TextConfig& config, DeviceContext& context) {
    const Shape one;
    const ExpectedState expected = expected_state(config, one);
    {
        nemotron::Model model(config, {});
        const EngineOptions options = options_for(one);
        nemotron::Program program(model, options, context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        report("one lane, 128 tokens", summary);
        check_bytes(summary.mamba_conv_bytes, expected.conv, "one lane: Mamba convolution history");
        check_bytes(summary.mamba_recurrent_bytes, expected.recurrent,
                    "one lane: Mamba recurrent plane");
        check_bytes(summary.attention_kv_bytes, expected.kv(), "one lane: attention K/V");
        const nemotron::AttentionState* attention = program.attention_state();
        check(attention != nullptr, "one lane: the attention state exists for this config");
        if (attention != nullptr) {
            check_bytes(attention->keys_bytes(), expected.keys, "one lane: the attention K planes");
            check_bytes(attention->values_bytes(), expected.values,
                        "one lane: the attention V planes");
            check_bytes(attention->positions_bytes(), expected.positions,
                        "one lane: the attention position rows");
            check(attention->filled(0) == 0, "a fresh Program holds no tokens for any lane");
        }

        // The two planning paths -- the pure report a caller sizes with and the layouts the
        // constructor allocated -- must agree, or admission would be sized from one geometry and
        // hold another.
        check_bytes(nemotron::Program::device_bytes(config, options), summary.device_bytes(),
                    "the pure device_bytes report and the constructed state");
        check(summary.state_allocation_bytes >= summary.state_bytes(),
              "the allocation holds the three classes");
        const std::size_t regions = 2 * config.num_hidden_layers() + 2;
        check(summary.state_allocation_bytes - summary.state_bytes() <= 256 * regions,
              "the allocation's alignment padding is bounded by one alignment per region");

        check(program.lanes() == 1 && program.admission_capacity() == 1,
              "one lane is one admission");
        check_bytes(static_cast<std::size_t>(program.capacity()), one.capacity,
                    "the lane's window is the configured capacity");
    }
    // Three lanes: every class is the one-lane cost times three, because a lane owns its own slot
    // and its own page run.
    {
        Shape three;
        three.lanes = 3;
        three.chunk = 128;
        Shape single;
        single.chunk = 128;
        nemotron::Model one_model(config, {});
        nemotron::Program one_lane(one_model, options_for(single), context);
        const nemotron::StateMemorySummary baseline = one_lane.memory_summary();
        nemotron::Model model(config, {});
        const EngineOptions options = options_for(three);
        nemotron::Program program(model, options, context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        report("three lanes, 128 tokens", summary);
        check_bytes(summary.mamba_conv_bytes, 3 * baseline.mamba_conv_bytes,
                    "three lanes: the convolution history multiplies");
        check_bytes(summary.mamba_recurrent_bytes, 3 * baseline.mamba_recurrent_bytes,
                    "three lanes: the recurrent plane multiplies");
        check_bytes(summary.attention_kv_bytes, 3 * baseline.attention_kv_bytes,
                    "three lanes: the attention K/V multiplies");
        const ExpectedState three_expected = expected_state(config, three);
        check_bytes(summary.mamba_conv_bytes, three_expected.conv,
                    "three lanes: the convolution history");
        check_bytes(summary.attention_kv_bytes, three_expected.kv(),
                    "three lanes: the attention K/V");
    }
    // A wider prefill chunk widens only the reported workspace.
    {
        Shape wide;
        wide.chunk = 512;
        nemotron::Model model(config, {});
        const EngineOptions options = options_for(wide);
        nemotron::Program program(model, options, context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        check_bytes(nemotron::Program::device_bytes(config, options), summary.device_bytes(),
                    "a wider chunk's device total");
        check(summary.workspace_bytes > 0, "a configured chunk reports a workspace");
        check_bytes(summary.state_bytes(),
                    summary.mamba_recurrent_bytes + summary.mamba_conv_bytes +
                        summary.attention_kv_bytes,
                    "the three classes sum to the state");
    }
}

// The recurrent plane is the Op's operand, and a lane's frontier after create is its zero state.
void state_identity(const nemotron::TextConfig& config, DeviceExecutionView execution,
                    DeviceContext& context) {
    nemotron::Model model(config, {});
    nemotron::Program program(model, options_for(Shape{}), context);
    const LinearAttentionStatePool* pool = program.mamba_state();
    check(pool != nullptr, "the mamba state pool exists for this config");
    if (pool == nullptr) { return; }
    const LinearAttentionStatePoolSpec& spec = pool->spec();
    check_bytes(static_cast<std::size_t>(spec.key_head_dim),
                static_cast<std::size_t>(nemotron::kMambaScanStateExtent),
                "the recurrent plane's key extent is the scan's, not the config's state size");
    check_bytes(static_cast<std::size_t>(spec.value_head_dim), config.mamba->head_dim,
                "the plane's value extent");
    check_bytes(spec.value_heads, config.mamba->num_heads, "the plane's head count");
    check_bytes(static_cast<std::size_t>(spec.conv_channels), config.mamba->conv_channels(),
                "the convolution's channels");
    check_bytes(static_cast<std::size_t>(spec.conv_width), config.mamba->conv_kernel - 1,
                "the convolution's history width");
    check_bytes(static_cast<std::size_t>(spec.slot_count), 1, "one slot per lane");
    check(spec.conv_dtype == DType::BF16,
          "the convolution history is the BF16 the convolution Op declares its state with");
    check_bytes(pool->recurrent_slot(0, 0).bytes(),
                static_cast<std::size_t>(nemotron::kMambaScanStateExtent) * config.mamba->head_dim *
                    config.mamba->num_heads * sizeof(float),
                "one lane's recurrent plane");

    // The recurrent plane's own extent order, pinned with a check rather than a comment: the scan
    // indexes its state as `(h * value_dim + v) * kStateDim + k`, so the key extent is the
    // innermost axis, the value extent the row, and the head the outermost one. Elsewhere in this
    // tree both extents are 128, which is exactly why a swap would be invisible; here the key
    // extent is the scan's own 128 against this config's 16-wide head, so the strides say which is
    // which.
    const Tensor recurrent_plane = pool->recurrent_slot(0, 0);
    check(recurrent_plane.dtype == DType::FP32 && recurrent_plane.is_contiguous(),
          "the recurrent slot is the contiguous FP32 state the scan declares");
    check_bytes(static_cast<std::size_t>(recurrent_plane.ne[0]),
                static_cast<std::size_t>(nemotron::kMambaScanStateExtent),
                "the key extent is the plane's first axis");
    check_bytes(static_cast<std::size_t>(recurrent_plane.ne[1]), config.mamba->head_dim,
                "the value extent is the plane's second axis");
    check_bytes(static_cast<std::size_t>(recurrent_plane.nb[0]), sizeof(float),
                "the key extent is the innermost axis of the recurrent plane");
    check_bytes(static_cast<std::size_t>(recurrent_plane.nb[1]),
                static_cast<std::size_t>(nemotron::kMambaScanStateExtent) * sizeof(float),
                "the value extent is one run of keys, so a swap would change this stride");
    check_bytes(static_cast<std::size_t>(recurrent_plane.nb[2]),
                static_cast<std::size_t>(nemotron::kMambaScanStateExtent) * config.mamba->head_dim *
                    sizeof(float),
                "a head is one run of value rows");

    // The convolution state's own convention, pinned with a check rather than a comment.
    // `causal_conv1d_silu` declares a contiguous BF16 [C, 3] state, oldest to newest, in this
    // tree's axis order (the innermost axis first), so a channel is the fastest element of a tap:
    // element (c, t) sits at the flat index `t * C + c`. The reference's cached convolution buffer
    // is a *different* layout -- four raw inputs per channel, `c * 4 + k` -- so a state imported
    // from a reference dump is the transpose of this, and the two are not interchangeable. The
    // pool's slot is the Op's layout, and its strides are what says so: a slot addressed the other
    // way round would still be a BF16 [C, 3] tensor of the right size.
    const Tensor conv = pool->conv_slot(0, 0);
    check(conv.dtype == DType::BF16 && conv.is_contiguous(),
          "the convolution slot is the contiguous BF16 state the Op declares");
    check_bytes(static_cast<std::size_t>(conv.ne[0]), config.mamba->conv_channels(),
                "the convolution state's channel extent");
    check_bytes(static_cast<std::size_t>(conv.ne[1]), config.mamba->conv_kernel - 1,
                "the convolution state's history width");
    check_bytes(static_cast<std::size_t>(conv.nb[0]), sizeof(std::uint16_t),
                "a channel is the innermost element of a tap");
    check_bytes(static_cast<std::size_t>(conv.nb[1]),
                static_cast<std::size_t>(config.mamba->conv_channels()) * sizeof(std::uint16_t),
                "a tap is one run of channels, so the taps are the outer axis");

    // Dirty both frontiers, then create the lane and read them back: create has to hand a new
    // sequence the zero state, or a reissued lane would start from another request's state. The
    // history is the *raw* convolution input, never the activated output, so zeroing it is what the
    // reference's own `has_previous_state == false` means: a zero tap contributes nothing to the
    // convolution sum.
    const Tensor recurrent = pool->recurrent_slot(0, 0);
    const std::vector<std::byte> pattern(recurrent.bytes(), std::byte{0x7f});
    test::cuda_check(cudaMemcpyAsync(recurrent.data, pattern.data(), pattern.size(),
                                     cudaMemcpyHostToDevice, execution.stream),
                     "dirtying the lane's recurrent frontier");
    const std::vector<std::byte> conv_pattern(conv.bytes(), std::byte{0x7f});
    test::cuda_check(cudaMemcpyAsync(conv.data, conv_pattern.data(), conv_pattern.size(),
                                     cudaMemcpyHostToDevice, execution.stream),
                     "dirtying the lane's convolution history");
    program.create(0, execution);
    test::cuda_synchronize(execution.stream);
    const auto read_frontier = [](const Tensor& frontier, const char* what) {
        std::vector<std::byte> readback(frontier.bytes(), std::byte{0xff});
        test::cuda_check(
            cudaMemcpy(readback.data(), frontier.data, readback.size(), cudaMemcpyDeviceToHost),
            what);
        return readback;
    };
    check(read_frontier(recurrent, "reading the lane's recurrent frontier back") ==
              std::vector<std::byte>(recurrent.bytes(), std::byte{0}),
          "create hands the lane the zero recurrent plane the scan starts from");
    check(read_frontier(conv, "reading the lane's convolution history back") ==
              std::vector<std::byte>(conv.bytes(), std::byte{0}),
          "create hands the lane the zero convolution history the convolution starts from");
}

// A state class the config's mixers do not need is absent and reports zero, so the numbers are the
// config's rather than a fixed set.
void absent_state_classes(const nemotron::TextConfig& config, DeviceContext& context) {
    {
        nemotron::TextConfig no_attention = config;
        no_attention.block_kinds          = {nemotron::BlockKind::Mamba, nemotron::BlockKind::Moe};
        no_attention.moe_banks            = {std::nullopt, tiny_bank()};
        nemotron::Model model(no_attention, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        check(program.attention_state() == nullptr, "no attention block: no attention state");
        check_bytes(summary.attention_kv_bytes, 0, "no attention block: no KV bytes");
        check_bytes(summary.mamba_recurrent_bytes, expected_state(no_attention, Shape{}).recurrent,
                    "no attention block: the Mamba classes are unchanged");
    }
    {
        nemotron::TextConfig no_mamba = config;
        no_mamba.block_kinds          = {nemotron::BlockKind::Attention, nemotron::BlockKind::Moe};
        no_mamba.mamba                = std::nullopt;
        no_mamba.moe_banks            = {std::nullopt, tiny_bank()};
        nemotron::Model model(no_mamba, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        check(program.mamba_state() == nullptr, "no Mamba block: no state pool");
        check_bytes(summary.mamba_recurrent_bytes, 0, "no Mamba block: no recurrent bytes");
        check_bytes(summary.mamba_conv_bytes, 0, "no Mamba block: no convolution bytes");
        check_bytes(summary.attention_kv_bytes, expected_state(no_mamba, Shape{}).kv(),
                    "no Mamba block: the KV class is unchanged");
    }
    {
        // Only MoE blocks: no state at all, and no allocation.
        nemotron::TextConfig only_moe = config;
        only_moe.block_kinds          = {nemotron::BlockKind::Moe, nemotron::BlockKind::Moe};
        only_moe.mamba                = std::nullopt;
        only_moe.attention            = std::nullopt;
        only_moe.moe_banks            = {tiny_bank(), tiny_bank()};
        nemotron::Model model(only_moe, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        const nemotron::StateMemorySummary summary = program.memory_summary();
        check(program.mamba_state() == nullptr && program.attention_state() == nullptr,
              "a config with no state-holding mixer owns no state");
        check_bytes(summary.state_bytes(), 0,
                    "a config with no state-holding mixer costs no state");
        check_bytes(summary.state_allocation_bytes, 0, "and allocates nothing");
    }
}

// Every geometry the state cannot hold exactly, and every entry point that would compute, refuses
// by name.
void refusals(const nemotron::TextConfig& config, DeviceExecutionView execution,
              DeviceContext& context) {
    {
        nemotron::Model model(config, {});
        EngineOptions no_lanes   = options_for(Shape{});
        no_lanes.max_concurrency = 0;
        expect_refusal(
            "zero lanes", [&] { nemotron::Program(model, no_lanes, context); }, "max_concurrency");
        EngineOptions no_context = options_for(Shape{});
        no_context.max_context   = 0;
        expect_refusal(
            "zero capacity", [&] { nemotron::Program(model, no_context, context); }, "max_context");
        EngineOptions quantized = options_for(Shape{});
        quantized.kv_cache      = KvCacheStorage::Int8Group64;
        expect_refusal(
            "a quantized KV geometry", [&] { nemotron::Program(model, quantized, context); },
            "BF16 because that is the dtype");
    }
    {
        nemotron::TextConfig wide = config;
        wide.mamba->state_size    = 256;
        nemotron::Model model(wide, {});
        expect_refusal(
            "a Mamba state wider than the scan's key extent",
            [&] { nemotron::Program(model, options_for(Shape{}), context); },
            "wider than the 128 key extent");
    }
    {
        nemotron::TextConfig no_shape = config;
        no_shape.mamba                = std::nullopt;
        nemotron::Model model(no_shape, {});
        expect_refusal(
            "a Mamba block without the mamba shape",
            [&] { nemotron::Program(model, options_for(Shape{}), context); },
            "requires the mamba shape");
    }
    {
        nemotron::TextConfig no_shape = config;
        no_shape.attention            = std::nullopt;
        nemotron::Model model(no_shape, {});
        expect_refusal(
            "an attention block without the attention shape",
            [&] { nemotron::Program(model, options_for(Shape{}), context); },
            "requires the attention shape");
    }
    // The computing entry points. Each names itself and the missing piece in full, so a caller --
    // or a log -- can tell which one was driven and what is absent.
    {
        nemotron::Model model(config, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        program.create(0, execution);
        const std::vector<std::int32_t> ids{1, 2, 3};
        const std::vector<std::int32_t> lanes{0};
        const std::vector<std::int32_t> one{7};
        const std::string gap(nemotron::kMissingStatePersistence);
        std::cout << "a computing entry point refuses with: nemotron_h Program: <entry> cannot "
                     "run: "
                  << gap << '\n';
        expect_refusal(
            "prefill", [&] { program.prefill(0, ids, execution); },
            "nemotron_h Program: prefill cannot run: " + gap);
        expect_refusal(
            "decode by lane set", [&] { program.decode(lanes, one, execution); },
            "nemotron_h Program: decode cannot run: " + gap);
        expect_refusal(
            "decode by lane", [&] { program.decode(0, 7, execution); },
            "nemotron_h Program: decode cannot run: " + gap);
        expect_refusal(
            "commit", [&] { program.commit(0, 1, execution); },
            "nemotron_h Program: commit cannot run: " + gap);
        expect_refusal(
            "discard", [&] { program.discard(0); },
            "nemotron_h Program: discard cannot run: " + gap);
        // An argument the caller can fix is refused as that argument error, before the missing
        // piece: a driver that cannot be called is not evidence that the state is right.
        expect_refusal(
            "an empty prefill",
            [&] { program.prefill(0, std::span<const std::int32_t>(), execution); },
            "at least one token");
        expect_refusal(
            "a decode set of mismatched length", [&] { program.decode(ids, one, execution); },
            "one id per lane");
        expect_refusal(
            "a lane beyond the Program", [&] { program.decode(7, 1, execution); },
            "is not one of this Program's 1 lanes");
    }
    // A lane that holds no request cannot be driven: the state it would read is not a sequence's.
    {
        nemotron::Model model(config, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        expect_refusal(
            "a prefill on a lane with no request",
            [&] { program.prefill(0, std::vector<std::int32_t>{1}, execution); },
            "holds no request");
    }
    // The lane lifecycle: one request per lane, and a lane that holds none cannot be finished.
    {
        nemotron::Model model(config, {});
        nemotron::Program program(model, options_for(Shape{}), context);
        expect_refusal(
            "finishing a lane that holds no request", [&] { program.finish(0); },
            "holds no request");
        expect_refusal(
            "aborting a lane that holds no request", [&] { program.abort(0); }, "holds no request");
        program.create(0, execution);
        expect_refusal(
            "creating a second request on one lane", [&] { program.create(0, execution); },
            "already holds a request");
        program.finish(0);
        expect_refusal(
            "finishing the same request twice", [&] { program.finish(0); }, "holds no request");
        program.create(0, execution);
        program.abort(0);
        expect_refusal(
            "a negative lane", [&] { program.create(-1, execution); },
            "is not one of this Program's 1 lanes");
    }
}

// The checkpoint's own geometry, hand-built: the 88-block hybrid the plan documents (40 Mamba, 40
// MoE, 8 attention in eight eleven-block cycles) with the real mixer shapes. Only the mixer counts
// and the two shapes decide the state, but every MoE block carries its bank because the workspace
// planner walks them.
nemotron::TextConfig checkpoint_config() {
    nemotron::TextConfig config;
    config.hidden_size             = 4096;
    config.vocab_size              = 131072;
    config.max_position_embeddings = 262144;
    config.tie_word_embeddings     = false;
    config.rms_norm_eps            = 1.0e-5F;
    nemotron::MambaConfig mamba;
    mamba.num_heads     = 128;
    mamba.head_dim      = 64;
    mamba.state_size    = 128;
    mamba.n_groups      = 8;
    mamba.conv_kernel   = 4;
    mamba.chunk_size    = 128;
    mamba.time_step_min = 0.001F;
    mamba.conv_bias     = true;
    config.mamba        = mamba;
    nemotron::AttentionConfig attention;
    attention.num_attention_heads = 32;
    attention.num_key_value_heads = 2;
    attention.head_dim            = 128;
    attention.rope_theta          = 0.0F;
    config.attention              = attention;
    nemotron::RouterConfig router;
    router.scaling_factor  = 5.0F;
    router.normalize_top_k = true;
    router.correction_bias = true;
    config.router          = router;
    // The plan's bank widths, cycled over the MoE blocks.
    const std::uint32_t experts[5]      = {240, 304, 368, 432, 496};
    const std::uint32_t intermediate[5] = {2048, 2240, 2304, 2560, 2560};
    config.block_kinds.reserve(88);
    config.moe_banks.reserve(88);
    for (std::uint32_t cycle = 0; cycle < 8; ++cycle) {
        for (std::uint32_t index = 0; index < 5; ++index) {
            config.block_kinds.push_back(nemotron::BlockKind::Mamba);
            config.moe_banks.push_back(std::nullopt);
        }
        for (std::uint32_t index = 0; index < 5; ++index) {
            nemotron::MoeBankConfig bank;
            bank.num_experts              = experts[index];
            bank.experts_per_token        = 9;
            bank.intermediate_size        = intermediate[index];
            bank.shared_intermediate_size = 5376;
            bank.latent_size              = 1024;
            config.block_kinds.push_back(nemotron::BlockKind::Moe);
            config.moe_banks.push_back(bank);
        }
        config.block_kinds.push_back(nemotron::BlockKind::Attention);
        config.moe_banks.push_back(std::nullopt);
    }
    return config;
}

// The state the plan records for this geometry, at the window a request is admitted with. The same
// assertions run for the hand-built geometry and for the artifact's own config, so the artifact
// case adds only that the checkpoint really carries these numbers.
void checkpoint_numbers(DeviceContext& context, const nemotron::TextConfig& config,
                        const std::string& source) {
    std::uint32_t mamba_layers     = 0;
    std::uint32_t attention_layers = 0;
    for (const nemotron::BlockKind kind : config.block_kinds) {
        if (kind == nemotron::BlockKind::Mamba) { ++mamba_layers; }
        if (kind == nemotron::BlockKind::Attention) { ++attention_layers; }
    }
    std::cout << source << ": " << config.num_hidden_layers() << " blocks, " << mamba_layers
              << " Mamba, " << attention_layers << " attention\n";
    check(config.num_hidden_layers() == 88, source + ": the block count");
    check(mamba_layers == 40 && attention_layers == 8, source + ": the mixer counts");

    // The window the plan's own per-sequence figures are quoted at: a 2048-token request, which is
    // the engine's default maximum context.
    Shape admitted;
    admitted.capacity = 2048;
    admitted.chunk    = 1024;
    nemotron::Model model(config, {});
    const EngineOptions options = options_for(admitted);
    nemotron::Program program(model, options, context);
    const nemotron::StateMemorySummary summary = program.memory_summary();
    report(source + ", one lane of 2048 tokens", summary);
    const ExpectedState expected = expected_state(config, admitted);
    check_bytes(summary.mamba_recurrent_bytes, expected.recurrent,
                source + ": the recurrent plane per lane");
    check_bytes(summary.mamba_conv_bytes, expected.conv, source + ": the convolution history");
    check_bytes(summary.attention_kv_bytes, expected.kv(), source + ": the attention K/V");
    check_bytes(summary.mamba_recurrent_bytes, 160ULL * 1024 * 1024,
                source + ": the plan's 160 MiB of recurrent state per sequence");
    check_bytes(summary.mamba_conv_bytes, 2457600,
                source + ": the plan's 2.34 MiB of history per sequence");
    // 8 attention layers x 2048 tokens x (2 planes x 2 KV heads x 128 head dim x 2 bytes) = 8 KB
    // per token; the lane's positions add one I32 per token.
    check_bytes(expected.keys + expected.values, 8ULL * 1024 * 2048,
                source + ": the plan's 8 KB per token across the 8 attention layers");
    check_bytes(expected.positions, 2048ULL * sizeof(std::int32_t),
                source + ": one position per token");
    check(summary.mamba_recurrent_bytes >= 40 * 4ULL * 1024 * 1024,
          source + ": at least 4 MiB of recurrent plane per Mamba layer per lane");

    // The full context costs 2 GiB of K/V planes for one lane, which is the plan's figure and is
    // checked through the pure planner rather than by allocating it: the state is the one-lane cost
    // at a 262144-token window, and the workspace is the same for any window of the same chunk.
    Shape full;
    full.capacity                     = 262144;
    full.chunk                        = admitted.chunk;
    const ExpectedState full_expected = expected_state(config, full);
    check_bytes(nemotron::Program::device_bytes(config, options_for(full)),
                full_expected.conv + full_expected.recurrent + full_expected.kv() +
                    summary.workspace_bytes,
                source + ": the full-context footprint");
    check_bytes(full_expected.keys + full_expected.values, (1ULL << 31),
                source + ": 2 GiB of attention K/V at the full 262144-token context");
}

// The artifact's own config, read from its cold directory without materializing its 45 GB. Prints a
// skip when the artifact is unset, like the loader's own test.
void artifact_numbers(DeviceContext& context) {
    const char* path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout
            << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set, so the artifact's config is not read\n";
        return;
    }
    artifact::Reader reader(path);
    checkpoint_numbers(context, nemotron::parse_config(reader.directory()).text,
                       "the artifact's config");
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "nemotron_h program: no CUDA device, skipping\n";
        return 77;
    }
    try {
        const nemotron::TextConfig config = synthetic_config();
        DeviceContext context(0);
        const DeviceExecutionView execution = context.execution_view();
        state_arithmetic(config, context);
        state_identity(config, execution, context);
        absent_state_classes(config, context);
        refusals(config, execution, context);
        checkpoint_numbers(context, checkpoint_config(), "the plan's geometry, hand-built");
        artifact_numbers(context);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " nemotron_h program check(s) failed\n";
        return 1;
    }
    std::cout << "nemotron_h program: ok\n";
    return 0;
}
