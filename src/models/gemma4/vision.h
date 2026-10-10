#pragma once

// The Gemma 4 image encoder: one image's patches in, its soft tokens in the text width out.
//
//   patch embed  x = W_patch * pixels + table_x[column] + table_y[row]
//   27 layers    sandwich-norm attention (q/k RMS-normed per head with weights, v without, 2-D RoPE,
//                scale 1, every patch sees every patch) and a sandwich-norm GeGLU MLP; no layer scalar
//   pooling      3x3 cell means times sqrt(hidden), standardized by the stored bias and scale
//   projection   a weightless RMSNorm, then W_embed into the text width
//
// One image is encoded per call with exactly its own patches, so there is no padding to mask.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/model.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::gemma4 {

struct ImagePatches {
    // The patch grid: `grid_width` patches across, `grid_height` down, both multiples of 3.
    std::int32_t grid_width  = 0;
    std::int32_t grid_height = 0;
    // BF16 [patches, 3 * 16 * 16]: patch p = row * grid_width + column, each patch's pixels in (row,
    // column, channel) order, already mapped to 2 * (u / 255) - 1.
    std::span<const std::uint16_t> pixels;

    [[nodiscard]] std::int32_t patches() const noexcept { return grid_width * grid_height; }
};

// Arena bytes encode_image needs for an image of `patches` patches.
[[nodiscard]] std::size_t vision_workspace_bytes(const VisionConfig& config, std::int32_t patches);

// Encodes `image` into `out`, BF16 [text hidden, patches / 9], on the execution stream. The model must
// have its vision tower bound. The arena is used as a scoped frame.
void encode_image(const Model& model, const ImagePatches& image, DeviceArena& arena, Tensor& out,
                  DeviceExecutionView execution);

} // namespace ninfer::models::gemma4
