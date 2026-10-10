#include "models/gemma4/image_processor.h"

#include "media/decode/resize.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::gemma4 {
namespace {

constexpr int kPatch       = 16;
constexpr int kPool        = 3;
constexpr int kSide        = kPatch * kPool; // 48
constexpr int kMaxPatches  = 280 * kPool * kPool;
constexpr int kPatchValues = kPatch * kPatch * 3;

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

} // namespace

ImageSize gemma_image_size(int height, int width) {
    if (height <= 0 || width <= 0) throw std::invalid_argument("image has no pixels");
    // get_aspect_ratio_preserving_size, in the reference's double precision.
    const double total_px  = static_cast<double>(height) * width;
    const double target_px = static_cast<double>(kMaxPatches) * kPatch * kPatch;
    const double factor    = std::sqrt(target_px / total_px);
    int target_height = static_cast<int>(std::floor(factor * height / kSide)) * kSide;
    int target_width  = static_cast<int>(std::floor(factor * width / kSide)) * kSide;
    if (target_height == 0 && target_width == 0) {
        throw std::invalid_argument("image is too small to resize to a 48-pixel multiple");
    }
    const int max_side = (kMaxPatches / (kPool * kPool)) * kSide;
    if (target_height == 0) {
        target_height = kSide;
        target_width  = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * kSide,
                                 max_side);
    } else if (target_width == 0) {
        target_width  = kSide;
        target_height = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * kSide,
                                 max_side);
    }
    if (static_cast<double>(target_height) * target_width > target_px) {
        throw std::invalid_argument("image resize exceeds the patch budget");
    }
    return {target_height, target_width};
}

PreparedImage patchify_gemma_image(const media::decode::Image& image) {
    if (image.width % kSide != 0 || image.height % kSide != 0 || image.width == 0 ||
        image.height == 0) {
        throw std::invalid_argument("patchify_gemma_image: sides must be positive multiples of 48");
    }
    PreparedImage out;
    out.grid_width  = image.width / kPatch;
    out.grid_height = image.height / kPatch;
    auto pixels     = std::make_shared<std::vector<std::uint16_t>>(
        static_cast<std::size_t>(out.grid_width) * out.grid_height * kPatchValues);
    // rescale: the uint8 value times float32(1/255), then the embedder's 2 * (x - 0.5), in float32.
    constexpr float kRescale = static_cast<float>(1.0 / 255.0);
    std::array<std::uint16_t, 256> lut{};
    for (int value = 0; value < 256; ++value) {
        lut[static_cast<std::size_t>(value)] =
            to_bf16(2.0F * (static_cast<float>(value) * kRescale - 0.5F));
    }
    std::uint16_t* cursor = pixels->data();
    for (int row = 0; row < out.grid_height; ++row) {
        for (int column = 0; column < out.grid_width; ++column) {
            for (int y = 0; y < kPatch; ++y) {
                const std::uint8_t* source =
                    image.rgb.data() + (static_cast<std::size_t>(row * kPatch + y) * image.width +
                                        static_cast<std::size_t>(column) * kPatch) *
                                           3;
                for (int x = 0; x < kPatch * 3; ++x) *cursor++ = lut[source[x]];
            }
        }
    }
    out.pixels = std::move(pixels);
    return out;
}

PreparedImage prepare_gemma_image(std::span<const std::uint8_t> bytes,
                                  const media::decode::Policy& policy,
                                  const std::function<void()>& checkpoint) {
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    if (checkpoint) checkpoint();
    const ImageSize size = gemma_image_size(image.height, image.width);
    media::decode::resize_bicubic(image, size.width, size.height, checkpoint);
    if (checkpoint) checkpoint();
    return patchify_gemma_image(image);
}

} // namespace ninfer::models::gemma4
