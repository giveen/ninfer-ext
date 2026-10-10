#pragma once

#include "media/decode/decode.h"

#include <functional>

namespace ninfer::media::decode {

// Resizes an RGB image in place with torchvision's antialiased bicubic (the Keys/Pillow a = -0.5
// kernel, support widened by the downscale factor), horizontal pass first, each pass rounded half to
// even and clamped to uint8, which is what torchvision does on uint8 input. A no-op when the size
// already matches. `checkpoint`, when set, runs every 16 rows so a caller can cancel a long resize.
void resize_bicubic(Image& image, int width, int height,
                    const std::function<void()>& checkpoint = {});

} // namespace ninfer::media::decode
