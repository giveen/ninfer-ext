// The image encoder against its reference.
//
// tools/verify/gemma4_vision_reference.py prepare writes transformers' preprocessed patches for a few
// images (imageN.bin, BF16 [patches, 768], and imageN.grid, "width height" in patches). This test
// encodes each with the artifact's vision tower and writes its soft tokens to imageN.out.bin (BF16
// [tokens, 5376]); the script's compare mode runs transformers' tower in FP32 on the same images.
//
// NINFER_GEMMA_ARTIFACT     an artifact converted with --components text,vision
// NINFER_GEMMA_VISION_DUMP  the directory prepare wrote

#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/load.h"
#include "models/gemma4/vision.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
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

} // namespace

int main() {
    const char* path = std::getenv("NINFER_GEMMA_ARTIFACT");
    const char* dump = std::getenv("NINFER_GEMMA_VISION_DUMP");
    if (path == nullptr || dump == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA_ARTIFACT and NINFER_GEMMA_VISION_DUMP must be set\n";
        return 77;
    }
    try {
        DeviceContext device;
        models::LoadOptions options;
        options.vision = true;
        auto model     = gemma::load_model(path, options, device);
        const auto& vision = *model->vision_config();
        DeviceArena arena(gemma::vision_workspace_bytes(vision, vision.max_patches()));
        const DeviceExecutionView execution = device.execution_view();
        const cudaStream_t stream           = execution.stream;

        int failures = 0;
        for (int index = 0;; ++index) {
            const std::filesystem::path base =
                std::filesystem::path(dump) / ("image" + std::to_string(index));
            std::ifstream grid(base.string() + ".grid");
            if (!grid) {
                if (index == 0) throw std::runtime_error("no image0.grid in the dump directory");
                break;
            }
            gemma::ImagePatches image;
            grid >> image.grid_width >> image.grid_height;
            std::ifstream in(base.string() + ".bin", std::ios::binary);
            std::vector<std::uint16_t> pixels(static_cast<std::size_t>(image.patches()) * 768);
            if (!in.read(reinterpret_cast<char*>(pixels.data()),
                         static_cast<std::streamsize>(pixels.size() * 2))) {
                throw std::runtime_error(base.string() + ".bin is shorter than its grid");
            }
            image.pixels = pixels;
            const std::int32_t tokens = image.patches() / 9;
            void* out_storage = nullptr;
            CUDA_CHECK(cudaMalloc(&out_storage,
                                  static_cast<std::size_t>(vision.output_hidden_size) * tokens * 2));
            Tensor out(out_storage, DType::BF16,
                       {static_cast<std::int32_t>(vision.output_hidden_size), tokens});
            gemma::encode_image(*model, image, arena, out, execution);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> host(static_cast<std::size_t>(out.numel()));
            CUDA_CHECK(cudaMemcpy(host.data(), out.data, host.size() * 2, cudaMemcpyDeviceToHost));
            std::ofstream(base.string() + ".out.bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(host.data()),
                       static_cast<std::streamsize>(host.size() * 2));
            std::size_t non_finite = 0;
            for (std::uint16_t value : host) non_finite += !std::isfinite(decode_bf16(value));
            std::cout << "image" << index << ": " << image.grid_width << "x" << image.grid_height
                      << " patches -> " << tokens << " soft tokens, non-finite " << non_finite << "\n";
            failures += non_finite != 0;
            CUDA_CHECK(cudaFree(out_storage));
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
