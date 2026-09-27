#include "options.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::quantize::app {
namespace {

int parse_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    char* end = nullptr;
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

OutScaleMode parse_out_scales(const char* text) {
    const std::string value(text);
    if (value == "auto") { return OutScaleMode::Auto; }
    if (value == "always") { return OutScaleMode::Always; }
    if (value == "never") { return OutScaleMode::Never; }
    throw std::invalid_argument("invalid out-scales mode: " + value);
}

void require_rate(int half_bits, const char* label) {
    if (half_bits < 2 || half_bits > 16) {
        throw std::invalid_argument(std::string(label) + " must be in [2, 16] half bits");
    }
}

} // namespace

std::string quantize_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> --out DIR --hessians DIR [--trace FILE --activation-model PATH]\n"
           "       --device N              CUDA device index (default 0)\n"
           "       --hessians DIR          per-parameter Hessian files; also their output dir\n"
           "       --trace FILE            packed calibration trace to build Hessians from\n"
           "       --activation-model PATH servable artifact whose activations are observed\n"
           "       --bits N                projection rate in half bits, 2..16 (default 8)\n"
           "       --head-bits N           output-head rate in half bits (default 12)\n"
           "       --hq                    promote attention and GDN control projections by 1 bit\n"
           "       --seed N                quantizer RNG seed (default 0)\n"
           "       --out-scales auto|always|never  (default auto)\n"
           "       --limit N               quantize at most N eligible parameters (debug)\n"
           "       --list                  list eligible parameters and exit without a device\n"
           "       --help                  print this message\n"
           "       quantizes each eligible BF16 projection and writes an exl3_mul1 source store\n"
           "       (one safetensors file of <name>.trellis/.su/.sv) plus DIR/report.json\n";
}

std::string sanitize_parameter_name(std::string_view name) {
    std::string out(name);
    for (char& ch : out) {
        if (ch == '/' || ch == '\\' || ch == ':') { ch = '_'; }
    }
    if (out.empty()) { throw std::invalid_argument("empty parameter name"); }
    return out;
}

int bitrate_for_parameter(const QuantizeOptions& options, std::string_view name) {
    int half_bits = options.bits_half;
    if (name == "text/output_head") {
        half_bits = options.head_bits_half;
    } else if (options.hq &&
               (name.find("/attention/") != std::string_view::npos ||
                name.ends_with("/gdn/a_projection") || name.ends_with("/gdn/b_projection"))) {
        half_bits += 2;
    }
    if (half_bits < 2) { half_bits = 2; }
    if (half_bits > 16) { half_bits = 16; }
    return half_bits;
}

QuantizeOptions parse_quantize_options(int argc, char** argv) {
    QuantizeOptions options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" ||
                      std::string_view(argv[1]) == "-h")) {
        options.help = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact = argv[1];
    bool output_set  = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--out") {
            options.output = require_value("--out");
            output_set     = true;
        } else if (arg == "--hessians") {
            options.hessians = require_value("--hessians");
        } else if (arg == "--trace") {
            options.trace = require_value("--trace");
        } else if (arg == "--activation-model") {
            options.activation_model = require_value("--activation-model");
        } else if (arg == "--calibration-rows") {
            options.calibration_rows =
                static_cast<std::uint32_t>(parse_int(require_value("--calibration-rows"),
                                                     "calibration-rows"));
        } else if (arg == "--device") {
            options.device = parse_int(require_value("--device"), "device");
        } else if (arg == "--bits") {
            options.bits_half = parse_int(require_value("--bits"), "bits");
            require_rate(options.bits_half, "--bits");
        } else if (arg == "--head-bits") {
            options.head_bits_half = parse_int(require_value("--head-bits"), "head-bits");
            require_rate(options.head_bits_half, "--head-bits");
        } else if (arg == "--hq") {
            options.hq = true;
        } else if (arg == "--seed") {
            options.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--out-scales") {
            options.out_scales = parse_out_scales(require_value("--out-scales"));
        } else if (arg == "--limit") {
            options.limit = parse_int(require_value("--limit"), "limit");
        } else if (arg == "--list") {
            options.list_only = true;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.help) { return options; }
    if (options.artifact.empty()) { throw std::invalid_argument("artifact path is required"); }
    if (options.list_only) { return options; }
    if (!output_set || options.output.empty()) {
        throw std::invalid_argument("--out is required");
    }
    if (options.hessians.empty()) {
        throw std::invalid_argument("--hessians is required in this version");
    }
    if (!options.trace.empty() && options.activation_model.empty()) {
        throw std::invalid_argument("--trace requires --activation-model");
    }
    return options;
}

} // namespace ninfer::quantize::app
