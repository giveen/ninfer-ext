#pragma once

// Offline Hessian calibration: run the model over the packed trace and accumulate E[XᵀX] for every
// eligible projection, writing one raw FP32 file per logical parameter name. The activation model
// is a servable artifact (for example the NVFP4 or Q4 build) whose weights fit on the device; the
// weight artifact supplies the eligible parameter names and is later quantized.

#include <cstdint>
#include <filesystem>

namespace ninfer::quantize::app {

struct CalibrationTrace;
struct LinearParameter;

struct CalibrationResult {
    std::uint32_t rows          = 0;
    std::uint64_t tokens        = 0;
    std::uint64_t hessians      = 0;
    std::uint64_t skipped_names = 0; // eligible names with no observed site (for example MTP)
};

CalibrationResult calibrate(const std::filesystem::path& activation_model,
                            const std::filesystem::path& weights_artifact,
                            const CalibrationTrace& trace,
                            const std::filesystem::path& hessians, int device,
                            std::uint32_t max_rows = 0,
                            std::uint64_t device_budget_bytes = 8ULL << 30);

} // namespace ninfer::quantize::app
