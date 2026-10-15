#pragma once

#include "media/decode/decode.h"

#include <functional>

namespace ninfer::media::decode {

// Resizes an RGB image in place with torchvision's antialiased bicubic (the Keys/Pillow a = -0.5
// kernel, support widened by the downscale factor), horizontal pass first, each pass rounded half
// to even and clamped to uint8, which is what torchvision does on uint8 input. A no-op when the
// size already matches. `width` and `height` must be positive and `image` must carry a
// size-matched RGB buffer, otherwise std::invalid_argument is thrown. `checkpoint`, when set, runs
// every 16 rows so a caller can cancel a long resize; a checkpoint that throws aborts the resize
// and leaves `image` unmodified.
void resize_bicubic(Image& image, int width, int height,
                    const std::function<void()>& checkpoint = {});

} // namespace ninfer::media::decode
