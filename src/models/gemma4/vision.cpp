#include "models/gemma4/vision.h"

#include "core/device.h"
#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/vision_pool.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

// Slices the arena allocates per image: hidden-sized and intermediate-sized activations, the patch
// input, and the I32/FP32 controls. Each allocation is rounded up to the arena's 256-byte alignment.
constexpr std::size_t kHiddenSlices       = 11;
constexpr std::size_t kIntermediateSlices = 3;
constexpr std::size_t kAllocations       = kHiddenSlices + kIntermediateSlices + 8;

} // namespace

std::size_t vision_workspace_bytes(const VisionConfig& config, std::int32_t patches) {
    if (patches < 1) throw std::invalid_argument("vision_workspace_bytes: patches must be positive");
    const std::size_t p      = static_cast<std::size_t>(patches);
    const std::size_t cells  = p / (config.pooling_kernel_size * config.pooling_kernel_size) + 1;
    const std::size_t h      = config.hidden_size;
    const std::size_t pixels = 3ULL * config.patch_size * config.patch_size;
    const std::size_t bf16   = 2;
    std::size_t bytes = (kHiddenSlices * h + kIntermediateSlices * config.intermediate_size + pixels) *
                            p * bf16 +
                        2 * h * cells * bf16 + (2 + 4) * p * 4 + 4 * p * 4;
    bytes += kAllocations * 256;
    // The dense attention's single segment needs no workspace of its own.
    return bytes;
}

void encode_image(const Model& model, const ImagePatches& image, DeviceArena& arena, Tensor& out,
                  DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    if (!model.vision_config() || !model.weights().vision) {
        throw std::logic_error("encode_image: the vision tower is not bound");
    }
    const VisionConfig& config   = *model.vision_config();
    const VisionWeights& weights = *model.weights().vision;
    const std::int32_t k         = static_cast<std::int32_t>(config.pooling_kernel_size);
    const std::int32_t patches   = image.patches();
    const std::int32_t h         = static_cast<std::int32_t>(config.hidden_size);
    const std::int32_t i         = static_cast<std::int32_t>(config.intermediate_size);
    const std::int32_t d         = static_cast<std::int32_t>(config.head_dim);
    const std::int32_t heads     = static_cast<std::int32_t>(config.num_attention_heads);
    const std::int32_t pixels    = static_cast<std::int32_t>(3 * config.patch_size * config.patch_size);
    const std::int32_t positions = static_cast<std::int32_t>(config.position_embedding_size);
    if (image.grid_width < k || image.grid_height < k || image.grid_width % k != 0 ||
        image.grid_height % k != 0 || patches > static_cast<std::int32_t>(config.max_patches()) ||
        image.grid_width > positions || image.grid_height > positions) {
        throw std::invalid_argument("encode_image: the patch grid is outside the tower's range");
    }
    if (image.pixels.size() != static_cast<std::size_t>(patches) * static_cast<std::size_t>(pixels)) {
        throw std::invalid_argument("encode_image: the pixels do not cover the patch grid");
    }
    const std::int32_t cells = (image.grid_width / k) * (image.grid_height / k);
    if (out.dtype != DType::BF16 || out.ne[0] != static_cast<std::int32_t>(config.output_hidden_size) ||
        out.ne[1] != cells || !out.is_contiguous()) {
        throw std::invalid_argument("encode_image: out must be contiguous BF16 [text hidden, cells]");
    }

    DeviceArena::Scope scope = arena.scope();
    const auto bf16          = [&](std::int32_t rows, std::int32_t columns) {
        return arena.alloc(DType::BF16, {rows, columns});
    };
    Tensor input    = bf16(pixels, patches);
    Tensor hidden   = bf16(h, patches);
    Tensor normed   = bf16(h, patches);
    Tensor branch   = bf16(h, patches);
    Tensor q        = bf16(h, patches);
    Tensor k_raw    = bf16(h, patches);
    Tensor v        = bf16(h, patches);
    Tensor q_normed = arena.alloc(DType::BF16, {d, heads, patches});
    Tensor k_normed = arena.alloc(DType::BF16, {d, heads, patches});
    Tensor v_normed = arena.alloc(DType::BF16, {d, heads, patches});
    Tensor attended = arena.alloc(DType::BF16, {d, heads, patches});
    Tensor gate     = bf16(i, patches);
    Tensor up       = bf16(i, patches);
    Tensor product  = bf16(i, patches);
    Tensor rope_positions  = arena.alloc(DType::I32, {patches, 2});
    Tensor table_rows      = arena.alloc(DType::I32, {4, patches});
    Tensor table_weights   = arena.alloc(DType::FP32, {4, patches});
    Tensor pooled          = bf16(h, cells);
    Tensor pooled_normed   = bf16(h, cells);

    // Controls: RoPE axis 0 is the column and axis 1 the row (axis-major), and the position table is
    // read at the column's x row and the row's y row with the other two corners weighted zero.
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(patches) * 2);
    std::vector<std::int32_t> host_rows(static_cast<std::size_t>(patches) * 4);
    std::vector<float> host_weights(static_cast<std::size_t>(patches) * 4);
    for (std::int32_t p = 0; p < patches; ++p) {
        const std::int32_t column = p % image.grid_width;
        const std::int32_t row    = p / image.grid_width;
        host_positions[static_cast<std::size_t>(p)]           = column;
        host_positions[static_cast<std::size_t>(patches + p)] = row;
        const std::size_t c = static_cast<std::size_t>(p) * 4;
        host_rows[c + 0] = column;
        host_rows[c + 1] = positions + row;
        host_rows[c + 2] = column;
        host_rows[c + 3] = column;
        host_weights[c + 0] = 1.0F;
        host_weights[c + 1] = 1.0F;
    }
    CUDA_CHECK(cudaMemcpyAsync(input.data, image.pixels.data(), image.pixels.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(rope_positions.data, host_positions.data(),
                               host_positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(table_rows.data, host_rows.data(),
                               host_rows.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(table_weights.data, host_weights.data(),
                               host_weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    const auto weight = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)).weight; };
    const float eps   = config.rms_norm_eps;

    ops::linear(input, weight(weights.patch_embedding), hidden, stream);
    ops::vision_pos_embed_add(model.tensor(weights.position_embedding), table_rows, table_weights,
                              hidden, stream);

    const ops::AttentionHeadGeometry geometry{d, heads, heads};
    for (const VisionLayerWeights& layer : weights.layers) {
        ops::rmsnorm(hidden, model.tensor(layer.input_norm), eps, false, normed, stream);
        ops::linear(normed, weight(layer.query), q, stream);
        ops::linear(normed, weight(layer.key), k_raw, stream);
        ops::linear(normed, weight(layer.value), v, stream);
        ops::rmsnorm(q.view({d, heads, patches}), model.tensor(layer.query_norm), eps, false,
                     q_normed, stream);
        ops::rmsnorm(k_raw.view({d, heads, patches}), model.tensor(layer.key_norm), eps, false,
                     k_normed, stream);
        ops::rmsnorm(v.view({d, heads, patches}), eps, v_normed, stream);
        ops::rope(rope_positions, d, d / 2, config.rope_theta, q_normed, k_normed, execution);
        ops::softmax_attention(q_normed, k_normed, v_normed, geometry, 1.0F, arena, attended,
                               stream);
        Tensor attended_flat = attended.view({h, patches});
        ops::linear(attended_flat, weight(layer.output), branch, stream);
        ops::rmsnorm(branch, model.tensor(layer.post_attention_norm), eps, false, normed, stream);
        ops::residual_add(normed, hidden, stream);

        ops::rmsnorm(hidden, model.tensor(layer.pre_feedforward_norm), eps, false, normed, stream);
        ops::linear(normed, weight(layer.mlp.gate), gate, stream);
        ops::linear(normed, weight(layer.mlp.up), up, stream);
        ops::gelu_mul(gate, up, product, stream);
        ops::linear(product, weight(layer.mlp.down), branch, stream);
        ops::rmsnorm(branch, model.tensor(layer.post_feedforward_norm), eps, false, normed, stream);
        ops::residual_add(normed, hidden, stream);
    }

    ops::vision_pool_standardize(hidden, image.grid_width, image.grid_height, k,
                                 std::sqrt(static_cast<float>(h)), model.tensor(weights.std_bias),
                                 model.tensor(weights.std_scale), pooled, stream);
    ops::rmsnorm(pooled, eps, pooled_normed, stream);
    ops::linear(pooled_normed, weight(weights.embedding_projection), out, stream);
}

} // namespace ninfer::models::gemma4
