// Gemma 4's image preprocessing against transformers' processor.
//
// tools/verify/gemma4_vision_reference.py prepare writes the processor's resize rule over a table of
// sizes (sizes.txt: "height width target_height target_width" or "height width error") and, per image,
// the source path (imageN.source) and the processor's BF16 patches (imageN.bin, imageN.grid). The rule
// must agree exactly. For a lossless source the patches must agree within one uint8 step (Pillow's
// bicubic against NInfer's torchvision-matching one, and the BF16 rounding of the mapped value); a
// JPEG's are reported only, because FFmpeg and Pillow decode it differently. NInfer's patches are
// written beside the reference's (imageN.ninfer.bin) so the effect of a difference can be measured.
//
// NINFER_GEMMA_VISION_DUMP  the directory prepare wrote

#include "models/gemma4/image_processor.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;

float decode_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

int main() {
    const char* dump = std::getenv("NINFER_GEMMA_VISION_DUMP");
    if (dump == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_VISION_DUMP must be set\n";
        return 77;
    }
    int failures = 0;
    std::ifstream sizes(std::filesystem::path(dump) / "sizes.txt");
    std::string line;
    int checked = 0;
    while (std::getline(sizes, line)) {
        std::istringstream fields(line);
        int height = 0, width = 0;
        std::string target_height;
        fields >> height >> width >> target_height;
        std::string got;
        try {
            const gemma::ImageSize size = gemma::gemma_image_size(height, width);
            got = std::to_string(size.height) + " " + std::to_string(size.width);
        } catch (const std::invalid_argument&) {
            got = "error";
        }
        std::string expected = target_height;
        if (target_height != "error") {
            int target_width = 0;
            fields >> target_width;
            expected += " " + std::to_string(target_width);
        }
        if (got != expected) {
            std::cerr << "size " << height << "x" << width << ": got " << got << ", expected "
                      << expected << "\n";
            ++failures;
        }
        ++checked;
    }
    if (checked == 0) {
        std::cerr << "no sizes.txt in the dump directory\n";
        return 1;
    }
    std::cout << "resize rule: " << checked << " sizes checked\n";

    for (int index = 0;; ++index) {
        const std::filesystem::path base =
            std::filesystem::path(dump) / ("image" + std::to_string(index));
        std::ifstream source_file(base.string() + ".source");
        if (!source_file) break;
        std::string source;
        std::getline(source_file, source);
        int grid_width = 0, grid_height = 0;
        std::ifstream(base.string() + ".grid") >> grid_width >> grid_height;
        const std::vector<std::uint8_t> bytes = read_bytes(source);
        const gemma::PreparedImage image =
            gemma::prepare_gemma_image(bytes, media::decode::Policy{});
        if (image.grid_width != grid_width || image.grid_height != grid_height) {
            std::cerr << "image" << index << ": grid " << image.grid_width << "x" << image.grid_height
                      << ", expected " << grid_width << "x" << grid_height << "\n";
            ++failures;
            continue;
        }
        // NInfer's own patches beside the reference's, so the effect of a difference can be measured.
        std::ofstream(base.string() + ".ninfer.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(image.pixels->data()),
                   static_cast<std::streamsize>(image.pixels->size() * 2));
        const std::vector<std::uint8_t> reference_bytes = read_bytes(base.string() + ".bin");
        std::vector<std::uint16_t> reference(reference_bytes.size() / 2);
        std::memcpy(reference.data(), reference_bytes.data(), reference.size() * 2);
        if (reference.size() != image.pixels->size()) {
            std::cerr << "image" << index << ": patch count differs\n";
            ++failures;
            continue;
        }
        // One uint8 step is 2/255 after the 2x - 1 mapping.
        std::size_t differing = 0;
        double worst          = 0.0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const double delta = std::abs(static_cast<double>(decode_bf16((*image.pixels)[i])) -
                                          decode_bf16(reference[i]));
            differing += delta != 0.0;
            worst = std::max(worst, delta * 255.0 / 2.0);
        }
        std::cout << "image" << index << ": " << grid_width << "x" << grid_height << " patches, "
                  << differing << " of " << reference.size() << " values differ, worst "
                  << worst << " uint8 steps\n";
        // A lossy source is decoded differently: FFmpeg upsamples 4:2:0 chroma by nearest neighbour
        // where Pillow's libjpeg interpolates, and their IDCTs differ, so a JPEG is reported, not judged.
        // Lossless sources must agree within the BF16 rounding of the mapped value (under one step).
        const bool lossy = source.ends_with(".jpg") || source.ends_with(".jpeg");
        if (lossy) {
            std::cout << "image" << index << ": lossy source, decoder difference reported only\n";
        } else if (worst > 1.0) {
            ++failures;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 image processor\n";
    return failures ? 1 : 0;
}
