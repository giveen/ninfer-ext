#pragma once

// Command line for the offline EXL3 quantizer. Parsing is pure host code so it is unit tested
// without a device.

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

namespace ninfer::quantize::app {

enum class OutScaleMode {
    Auto,
    Always,
    Never,
};

struct QuantizeOptions {
    std::filesystem::path artifact;
    std::filesystem::path output;
    // Per-parameter Hessian files, named after the sanitized logical parameter (K*K FP32 words).
    std::filesystem::path hessians;
    // When set, run the activation model over the packed trace first and write the Hessians.
    std::filesystem::path trace;
    std::filesystem::path activation_model;
    std::uint32_t calibration_rows = 0; // 0 selects every trace row
    int device                  = 0;
    int bits_half               = 8;  // 4.0 bpw for ordinary projections
    int head_bits_half          = 12; // 6.0 bpw for the output head
    bool hq                     = false;
    std::uint64_t seed          = 0;
    bool list_only              = false;
    bool calibrate_only         = false;
    bool experts_only           = false; // quantize routed experts only (dense stays as converted)
    std::string only;                // quantize only parents whose name contains this text
    std::int64_t limit          = 0; // 0 selects every eligible parameter
    OutScaleMode out_scales     = OutScaleMode::Auto;
    bool help                   = false;
    // Per-tensor rates in half bits from the M5 allocation, keyed by logical parameter name. A name
    // present here overrides --bits/--head-bits/--hq; `rates` is the JSON the caller loads into it.
    std::filesystem::path rates;
    std::map<std::string, int> rate_overrides;
};

[[nodiscard]] QuantizeOptions parse_quantize_options(int argc, char** argv);
[[nodiscard]] std::string quantize_usage_text(const char* argv0);

// Logical parameter names carry '/', which is fine for safetensors keys but not for one flat
// Hessian filename per parameter.
[[nodiscard]] std::string sanitize_parameter_name(std::string_view name);

// Rate in half bits for one logical parameter: the output head uses head_bits_half, and --hq
// promotes attention and the two small GDN control projections by one bit per weight (two half
// bits), matching the plan's promotion set.
[[nodiscard]] int bitrate_for_parameter(const QuantizeOptions& options, std::string_view name);

} // namespace ninfer::quantize::app
