#include "artifact/exl3_trellis.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"
#include "artifact/schema.h"

#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

template <typename Exception, typename Operation>
void expect_throws(Operation operation, const char* message) {
    try {
        operation();
    } catch (const Exception&) { return; }
    expect(false, message);
}

std::vector<std::byte> make_tile(std::uint64_t rate_half_bits) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(16 * rate_half_bits));
    std::uint32_t state = 0x9E3779B9U;
    for (auto& byte : bytes) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        byte = static_cast<std::byte>(state & 0xFFU);
    }
    return bytes;
}

std::uint16_t reference_state(std::span<const std::byte> tile, std::uint64_t rate_half_bits,
                              std::size_t weight) {
    const auto low = static_cast<std::size_t>(rate_half_bits / 2);
    const auto total_bits = tile.size() * 8;
    std::size_t end = 0;
    for (std::size_t i = 0; i <= weight; ++i) {
        end += low;
        if ((rate_half_bits & 1U) && (i & 1U)) { ++end; }
    }
    const auto begin = (end + total_bits - 16) % total_bits;
    std::uint16_t result = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto position = (begin + i) % total_bits;
        const auto byte = std::to_integer<std::uint8_t>(tile[position / 8]);
        result |= static_cast<std::uint16_t>(((byte >> (position % 8)) & 1U) << i);
    }
    return result;
}

void test_trellis_round_trips() {
    for (std::uint64_t rate = 2; rate <= 16; ++rate) {
        const auto tile = make_tile(rate);
        std::array<std::uint16_t, ninfer::artifact::kExl3WeightsPerTile> states{};
        ninfer::artifact::exl3_unpack_trellis_tile(tile, rate, states);
        for (std::size_t i = 0; i < states.size(); ++i) {
            expect(states[i] == reference_state(tile, rate, i),
                   "decoded trellis state must match independent circular-bit oracle");
        }
        expect(ninfer::artifact::exl3_pack_trellis_tile(states, rate) == tile,
               "packed circular states must reproduce every tile bit exactly");
    }
    expect_throws<std::invalid_argument>(
        [] { (void)ninfer::artifact::exl3_pack_trellis_tile({}, 8); },
        "packer must reject an incomplete tile");
    expect_throws<std::invalid_argument>(
        [] { (void)ninfer::artifact::exl3_trellis_state({}, 8, 0); },
        "state decoder must reject a mismatched tile length");
    expect_throws<std::invalid_argument>(
        [] { (void)ninfer::artifact::exl3_trellis_state(std::vector<std::byte>(32), 1, 0); },
        "state decoder must reject bitrates below one bit per weight");
    expect_throws<std::out_of_range>(
        [] { (void)ninfer::artifact::exl3_trellis_state(std::vector<std::byte>(128), 8, 256); },
        "state decoder must reject tile indices outside the tile");
}

void test_mul1_codebook() {
    expect(ninfer::artifact::exl3_mul1_value(0) == -510,
           "mul1 codebook zero state must map to -510");
    for (std::uint32_t state = 0; state <= 0xFFFFU; ++state) {
        const auto product = state * 0x83DCD12DU;
        const auto byte_sum = (product & 0xFFU) + ((product >> 8) & 0xFFU) +
                              ((product >> 16) & 0xFFU) + ((product >> 24) & 0xFFU);
        expect(ninfer::artifact::exl3_mul1_value(static_cast<std::uint16_t>(state)) ==
                   static_cast<std::int32_t>(byte_sum) - 510,
               "mul1 codebook must equal the exact unsigned-word definition");
    }
}

void test_trellis_geometry() {
    using ninfer::QType;
    using ninfer::QuantLayout;
    using ninfer::weight_geometry;
    const std::array<std::uint64_t, 2> small_shape{128, 128};
    const auto half_rate = weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                           small_shape, 1, 3);
    expect(half_rate.tile_bytes == 48 && half_rate.trellis_bytes == 3072,
           "half-bit bitrate must size each 16x16 tile exactly");
    expect(half_rate.input_scale_offset == 3072 && half_rate.input_scale_bytes == 512 &&
               half_rate.output_scale_offset == 3584 && half_rate.output_scale_bytes == 512 &&
               half_rate.bytes == 4096,
           "EXL3 planes must use aligned trellis, input-scale and output-scale regions");

    const std::array<std::uint64_t, 2> qwen_mlp{17408, 5120};
    const auto real_shape = weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                            qwen_mlp, 1, 8);
    expect(real_shape.tiles_n == 1088 && real_shape.tiles_k == 320 &&
               real_shape.trellis_bytes == 44564480 && real_shape.bytes == 44654592,
           "Qwen3.8 MLP geometry must match its real matrix shape");
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                    small_shape, 1, 1); },
        "EXL3 layout must reject rates below one bit per weight");
    const std::array<std::uint64_t, 2> invalid_shape{128, 129};
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                    invalid_shape, 1, 8); },
        "EXL3 layout must reject dimensions not Hadamard aligned");
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::Q4_G64_FP16, QuantLayout::TrellisT16,
                                    small_shape, 1, 8); },
        "EXL3 layout must reject unrelated numeric formats");
    std::vector<std::byte> payload(half_rate.bytes);
    ninfer::WeightParent parent{half_rate, payload.data()};
    ninfer::WeightView view{{128, 128}, {{&parent, 0, half_rate.elements}}};
    const ninfer::Weight native = ninfer::native_weight(view);
    expect(native.qtype == QType::EXL3_MUL1 && native.qdata == payload.data() &&
               native.input_scales == payload.data() + half_rate.input_scale_offset &&
               native.scales == payload.data() + half_rate.output_scale_offset &&
               native.bitrate_half_bits == 3 && native.n == 128 && native.k == 128,
           "the native Weight bridge must expose the EXL3 trellis and both scale planes");
    expect_throws<std::invalid_argument>(
        [&] { (void)ninfer::weight_row_planes(view.parts.front()); },
        "the generic row-plane view must not misinterpret EXL3 scales");
    // Stacked matrices: the trellis plane and svh[N] are the whole object's, suh repeats per set.
    const std::array<std::uint64_t, 2> stacked_shape{1024, 256};
    const auto single = weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                        stacked_shape, 1, 8);
    const auto stacked = weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                         stacked_shape, 4, 8);
    expect(stacked.divisor_count == 4 && stacked.trellis_bytes == single.trellis_bytes &&
               stacked.input_scale_offset == single.input_scale_offset &&
               stacked.input_scale_bytes == 4 * single.input_scale_bytes &&
               stacked.output_scale_bytes == single.output_scale_bytes &&
               stacked.output_scale_offset % 256 == 0 &&
               stacked.bytes == stacked.output_scale_offset + stacked.output_scale_bytes,
           "stacked EXL3 matrices must repeat only the input scales");
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                    stacked_shape, 3, 8); },
        "a scale set count that does not divide the rows must be rejected");
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                    stacked_shape, 16, 8); },
        "a scale set smaller than a 128-row Hadamard block must be rejected");
    expect_throws<std::invalid_argument>(
        [&] { (void)weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit,
                                    stacked_shape, 2); },
        "only NVFP4 and EXL3 may carry several divisors or scale sets");
    {
        std::vector<std::byte> bank(stacked.bytes);
        ninfer::WeightParent bank_parent{stacked, bank.data()};
        ninfer::WeightView bank_view{{1024, 256}, {{&bank_parent, 0, stacked.elements}}};
        expect_throws<std::invalid_argument>(
            [&] { (void)ninfer::native_weight(bank_view); },
            "a stacked EXL3 bank must not be handed to a dense Linear as one matrix");
    }
    const std::array<std::uint64_t, 2> overflow_shape{
        std::numeric_limits<std::uint64_t>::max(), 128};
    expect_throws<std::overflow_error>(
        [&] { (void)weight_geometry(QType::EXL3_MUL1, QuantLayout::TrellisT16,
                                    overflow_shape, 1, 8); },
        "EXL3 geometry must reject unrepresentable dimensions");
}

void test_registered_format_and_layout() {
    using ninfer::QType;
    using ninfer::QuantLayout;
    using ninfer::artifact::format_name;
    using ninfer::artifact::layout_name;
    using ninfer::artifact::parse_format;
    using ninfer::artifact::parse_layout;
    expect(parse_format("exl3_mul1") == QType::EXL3_MUL1 &&
               format_name(QType::EXL3_MUL1) == "exl3_mul1",
           "C++ format registry must round-trip exl3_mul1");
    expect(parse_layout("trellis_t16_v1") == QuantLayout::TrellisT16 &&
               layout_name(QuantLayout::TrellisT16) == "trellis_t16_v1",
           "C++ layout registry must round-trip trellis_t16_v1");
}

void test_tensor_object_bitrate_contract() {
    ninfer::artifact::TensorObject object{
        "weight/exl3", {128, 128}, "exl3_mul1", "trellis_t16_v1", 0, 4096, 1, 3};
    const auto geometry = ninfer::artifact::describe_tensor(object);
    expect(geometry.bitrate_half_bits == 3 && geometry.bytes == 4096,
           "artifact tensor metadata must select the bitrate-specific geometry");
    object.bitrate_half_bits = 17;
    expect_throws<ninfer::artifact::ArtifactError>(
        [&] { (void)ninfer::artifact::describe_tensor(object); },
        "artifact tensor metadata must reject out-of-range bitrates");
    object.bitrate_half_bits = 3;
    object.bytes = 4095;
    expect_throws<ninfer::artifact::ArtifactError>(
        [&] { (void)ninfer::artifact::describe_tensor(object); },
        "artifact tensor metadata must reject inconsistent encoded sizes");
}

void test_directory_round_trips_bitrate_metadata() {
    using ninfer::artifact::Json;
    using ninfer::artifact::parse_directory;
    Json root = Json::object();
    root["components"] = {{"text", {{"config", Json::object()}}}};
    root["objects"] = Json::array({Json{{"id", "weight/exl3"},
                                        {"kind", "tensor"},
                                        {"shape", {128, 128}},
                                        {"format", "exl3_mul1"},
                                        {"layout", "trellis_t16_v1"},
                                        {"offset", 0},
                                        {"bytes", 4096},
                                        {"bitrate_half_bits", 3}}});
    root["bindings"] = {{"text/weight", {{"object", "weight/exl3"}}}};
    root["uses"] = Json::array();
    root["files"] = Json::array({Json{{"path", nullptr}, {"payload_bytes", 4096}}});
    auto directory = parse_directory(root, "test.ninfer");
    const auto handle = directory.object_index.at("weight/exl3");
    const auto geometry = ninfer::artifact::describe_tensor(directory.tensor(handle));
    expect(geometry.bitrate_half_bits == 3 && geometry.bytes == 4096,
           "C++ artifact parser must preserve per-tensor EXL3 bitrate metadata");

    root["objects"][0].erase("bitrate_half_bits");
    directory = parse_directory(root, "test.ninfer");
    const auto missing_rate_handle = directory.object_index.at("weight/exl3");
    expect_throws<ninfer::artifact::ArtifactError>(
        [&] { (void)ninfer::artifact::describe_tensor(directory.tensor(missing_rate_handle)); },
        "EXL3 tensor objects must not omit bitrate metadata");
}

} // namespace

int main() {
    test_trellis_round_trips();
    test_mul1_codebook();
    test_trellis_geometry();
    test_registered_format_and_layout();
    test_tensor_object_bitrate_contract();
    test_directory_round_trips_bitrate_metadata();
    return failures == 0 ? 0 : 1;
}
