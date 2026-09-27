#pragma once

// Reads the packed calibration trace `tools/exl3/sample_traces.py` writes: a single safetensors
// file with `input_ids` I64 [rows, row_tokens] and `lengths` I64 [rows]. Pure host code.

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ninfer::quantize::app {

struct CalibrationTrace {
    std::uint32_t rows       = 0;
    std::uint32_t row_tokens = 0;
    // Right-padded rows; `lengths[row]` is the number of real tokens, so the padding is inert.
    std::vector<std::int32_t> input_ids; // rows * row_tokens
    std::vector<std::int32_t> lengths;   // rows
};

[[nodiscard]] CalibrationTrace read_calibration_trace(const std::filesystem::path& path);

} // namespace ninfer::quantize::app
