#pragma once

// Gemma 4's image preprocessing, as transformers' Gemma4ImageProcessor does it:
//
//   resize     to the largest aspect-preserving size whose sides are multiples of 48 (a 3x3 cell of
//              16-pixel patches) and whose patches fit the 280-soft-token budget, bicubic with
//              antialiasing; also upscaling, and the zero-side edge cases of the reference
//   pixels     u8 * (1/255), then 2 * (x - 0.5), rounded to BF16 (the patch embedder's own mapping)
//   patches    row-major over the patch grid, each patch's pixels in (row, column, channel) order
//
// One image becomes `soft_tokens` = patches / 9 positions of the prompt.

#include "media/decode/decode.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

struct PreparedImage {
    std::int32_t grid_width  = 0; // patches across
    std::int32_t grid_height = 0; // patches down
    // BF16 [patches, 768].
    std::shared_ptr<const std::vector<std::uint16_t>> pixels;

    [[nodiscard]] std::int32_t soft_tokens() const noexcept { return grid_width * grid_height / 9; }
};

// The resized (height, width) for an image of `height` x `width` pixels. Throws
// std::invalid_argument for an image the reference cannot resize (both sides rounding to zero).
struct ImageSize {
    int height = 0;
    int width  = 0;
};

[[nodiscard]] ImageSize gemma_image_size(int height, int width);

// Pixels the budget holds: a larger image is downscaled.
inline constexpr std::uint64_t kGemmaImageBudgetPixels = 280ULL * 9ULL * 16ULL * 16ULL;

// Decodes, resizes and patchifies one image. `checkpoint` runs between stages and inside the
// resize.
[[nodiscard]] PreparedImage prepare_gemma_image(std::span<const std::uint8_t> bytes,
                                                const media::decode::Policy& policy,
                                                const std::function<void()>& checkpoint = {});

// Patchifies an already-resized image whose sides are multiples of 48.
[[nodiscard]] PreparedImage patchify_gemma_image(const media::decode::Image& image);

} // namespace ninfer::models::gemma4
