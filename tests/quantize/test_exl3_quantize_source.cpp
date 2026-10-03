// CPU-only tests for the EXL3 quantizer app's host surface: option parsing, source naming and
// rates, Hessian files, the safetensors writer, tile packing, and the weight transpose. No CUDA
// call is made, so this runs without a device.

#include "hessian_io.h"
#include "options.h"
#include "parameter_reader.h"
#include "source_writer.h"

#include "artifact/exl3_trellis.h"

#include <nlohmann/json.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace app = ninfer::quantize::app;

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

app::QuantizeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return app::parse_quantize_options(static_cast<int>(argv.size()), argv.data());
}

std::filesystem::path scratch(const char* name) {
    const auto dir = std::filesystem::temp_directory_path() /
                     (std::string("ninfer-exl3-quantize-") + name + "-" +
                      std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void test_options() {
    const app::QuantizeOptions options = parse({"ninfer-quantize", "m.ninfer", "--out", "o",
                                                "--hessians", "h", "--bits", "7", "--head-bits",
                                                "12", "--hq", "--seed", "3", "--out-scales",
                                                "always", "--limit", "2"});
    expect(options.artifact == "m.ninfer" && options.output == "o" && options.hessians == "h",
           "required paths were not parsed");
    expect(options.bits_half == 7 && options.head_bits_half == 12 && options.hq &&
               options.seed == 3 && options.limit == 2 &&
               options.out_scales == app::OutScaleMode::Always,
           "optional values were not parsed");
    expect(parse({"ninfer-quantize", "m.ninfer", "--list"}).list_only,
           "--list did not select the listing mode");
    const app::QuantizeOptions experts =
        parse({"ninfer-quantize", "m.ninfer", "--out", "o", "--hessians", "h", "--experts-only",
               "--only", "text/layers/0/", "--calibrate-only"});
    expect(experts.experts_only && experts.calibrate_only && experts.only == "text/layers/0/",
           "--experts-only, --only and --calibrate-only were not parsed");
    expect(parse({"ninfer-quantize", "--help"}).help, "--help did not select help");
    expect_throws<std::invalid_argument>(
        [] { (void)parse({"ninfer-quantize", "m.ninfer", "--hessians", "h"}); },
        "a missing --out was accepted");
    expect_throws<std::invalid_argument>(
        [] { (void)parse({"ninfer-quantize", "m.ninfer", "--out", "o"}); },
        "a missing --hessians was accepted");
    expect_throws<std::invalid_argument>(
        [] { (void)parse({"ninfer-quantize", "m.ninfer", "--out", "o", "--bits", "1"}); },
        "an out-of-range --bits was accepted");
    expect_throws<std::invalid_argument>(
        [] { (void)parse({"ninfer-quantize", "m.ninfer", "--out", "o", "--unknown"}); },
        "an unknown option was accepted");

    expect(app::sanitize_parameter_name("text/layers/0/mlp/gate") == "text_layers_0_mlp_gate",
           "parameter name sanitization changed the expected characters");

    const app::QuantizeOptions base = parse({"ninfer-quantize", "m.ninfer", "--out", "o",
                                             "--hessians", "h", "--bits", "8", "--head-bits", "12"});
    expect(app::bitrate_for_parameter(base, "text/layers/0/mlp/gate") == 8,
           "ordinary projections must use --bits");
    expect(app::bitrate_for_parameter(base, "text/output_head") == 12,
           "the output head must use --head-bits");

    const app::QuantizeOptions hq = parse({"ninfer-quantize", "m.ninfer", "--out", "o",
                                           "--hessians", "h", "--bits", "8", "--hq"});
    expect(app::bitrate_for_parameter(hq, "text/layers/0/attention/query") == 10,
           "--hq did not promote attention");
    expect(app::bitrate_for_parameter(hq, "text/layers/0/gdn/a_projection") == 10,
           "--hq did not promote the GDN control projections");
    expect(app::bitrate_for_parameter(hq, "text/layers/0/mlp/gate") == 8,
           "--hq promoted an unlisted projection");
}

void test_hessian_round_trip() {
    const std::filesystem::path dir = scratch("hessian");
    const std::vector<float> h{1.0F, 2.0F, 3.0F, 4.0F};
    app::write_f32_file(dir / "x.h.f32", h);
    expect(app::read_hessian_f32(dir / "x.h.f32", 2) == h, "Hessian round trip changed values");
    expect_throws<std::runtime_error>(
        [&] { (void)app::read_hessian_f32(dir / "x.h.f32", 3); },
        "a short Hessian was accepted");
    app::write_f32_file(dir / "long.h.f32", std::vector<float>(5, 0.0F));
    expect_throws<std::runtime_error>(
        [&] { (void)app::read_hessian_f32(dir / "long.h.f32", 2); },
        "a Hessian with trailing bytes was accepted");
    std::filesystem::remove_all(dir);
}

void test_safetensors_writer() {
    const std::filesystem::path dir = scratch("safetensors");
    std::vector<app::SafetensorsTensor> tensors;
    tensors.push_back({"p.trellis", "U8", {1, 1, 6}, {std::byte{1}, std::byte{2}, std::byte{3},
                                                      std::byte{4}, std::byte{5}, std::byte{6}}});
    tensors.push_back({"p.su", "F32", {2}, {std::byte{0}, std::byte{0}, std::byte{0x80},
                                            std::byte{0x3f}, std::byte{0}, std::byte{0}, std::byte{0},
                                            std::byte{0x40}}});
    app::write_safetensors(dir / "exl3.safetensors", tensors);

    std::ifstream in(dir / "exl3.safetensors", std::ios::binary);
    std::uint64_t header_size = 0;
    in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    std::string header_text(header_size, '\0');
    in.read(header_text.data(), static_cast<std::streamsize>(header_size));
    const auto header = nlohmann::json::parse(header_text);
    expect(header.at("p.trellis").at("dtype") == "U8" &&
               header.at("p.trellis").at("shape") == std::vector<int>{1, 1, 6} &&
               header.at("p.trellis").at("data_offsets") == std::vector<int>{0, 6},
           "trellis header entry is wrong");
    expect(header.at("p.su").at("dtype") == "F32" &&
               header.at("p.su").at("data_offsets") == std::vector<int>{6, 14},
           "scale header entry is wrong");
    const std::uint64_t data_size = 6 + 8;
    expect(std::filesystem::file_size(dir / "exl3.safetensors") ==
               sizeof(std::uint64_t) + header_size + data_size,
           "safetensors file size does not match its header and data");
    std::vector<std::byte> data(data_size);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    expect(data[0] == std::byte{1} && data[6] == std::byte{0} && data[13] == std::byte{0x40},
           "safetensors data section is wrong");
    std::filesystem::remove_all(dir);
}

void test_trellis_packing() {
    std::mt19937 generator(1234);
    for (const int half_bits : {2, 3, 5, 8, 11, 16}) {
        std::vector<std::byte> tile(static_cast<std::size_t>(16 * half_bits));
        for (auto& byte : tile) { byte = static_cast<std::byte>(generator() & 0xFF); }
        std::array<std::uint16_t, ninfer::artifact::kExl3WeightsPerTile> states{};
        ninfer::artifact::exl3_unpack_trellis_tile(tile, half_bits, states);
        expect(app::pack_trellis_states(states.data(), half_bits) == tile,
               "pack_trellis_states did not reproduce the source tile");
    }
}

void test_transpose() {
    const std::vector<float> nk{1, 2, 3, 4, 5, 6}; // [2][3]
    expect(app::transpose_to_kn(nk, 2, 3) == std::vector<float>{1, 4, 2, 5, 3, 6},
           "transpose_to_kn is wrong");
    expect_throws<std::invalid_argument>(
        [] { (void)app::transpose_to_kn({1, 2}, 2, 3); },
        "a size-mismatched transpose was accepted");
}

} // namespace

int main() {
    test_options();
    test_hessian_round_trip();
    test_safetensors_writer();
    test_trellis_packing();
    test_transpose();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
