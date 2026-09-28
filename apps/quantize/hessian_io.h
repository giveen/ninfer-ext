#pragma once

// Per-parameter Hessian interchange: raw little-endian FP32 words, one file per sanitized logical
// parameter name. The calibration stage writes them; the quantizer reads them. Host code only.

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::quantize::app {

// Reads exactly k * k FP32 words.
[[nodiscard]] std::vector<float> read_hessian_f32(const std::filesystem::path& path,
                                                  std::uint64_t k);

// A k x k identity: the uncalibrated Hessian, which makes the quantizer minimize the plain
// reconstruction error. Used for a projection the calibration cannot observe.
[[nodiscard]] std::vector<float> identity_hessian(std::uint64_t k);

void write_f32_file(const std::filesystem::path& path, std::span<const float> values);

} // namespace ninfer::quantize::app
