// ninfer::ops - nemotron_h_moe_experts wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/nemotron_h_moe_experts.h"

#include "ops/launcher/nemotron_h_moe_experts.h" // detail::nemotron_h_moe_experts_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("nemotron_h_moe_experts: " + message);
}

// A contiguous 2-D tensor: the leading two extents carry the shape and the trailing two are one,
// and the layout really is row-major rather than inferred from the extents. A tensor whose shape
// matches but whose strides do not -- a transposed view, say -- would otherwise be read as if it
// were contiguous, which is a wrong answer rather than a refusal.
void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        fail(std::string(label) + " must be a contiguous matrix of the geometry's shape");
    }
    if (!tensor.is_contiguous()) { fail(std::string(label) + " must be contiguous"); }
}

// One bank at its logical shape. The row count is the geometry's expert count times the width of
// that bank's own matrices, so a bank that holds fewer experts than the routing can name -- or one
// bound at a different latent width -- is refused here rather than read out of range by a kernel.
detail::NemotronHMoeBankLaunch prepare_bank(const Weight& weight, std::int64_t rows,
                                            std::int64_t columns, const char* label) {
    if (weight.payload == nullptr) { fail(std::string(label) + " must be bound"); }
    if (weight.ndim != 2 || static_cast<std::int64_t>(weight.shape[0]) != rows ||
        static_cast<std::int64_t>(weight.shape[1]) != columns) {
        fail(std::string(label) + " must be [" + std::to_string(rows) + ", " +
             std::to_string(columns) + "]");
    }
    if (weight.qtype != QType::BF16) { fail(std::string(label) + " must be BF16"); }

    detail::NemotronHMoeBankLaunch bank;
    bank.qtype   = weight.qtype;
    bank.payload = weight.payload;
    bank.columns = static_cast<std::int32_t>(columns);
    return bank;
}

} // namespace

std::size_t nemotron_h_moe_experts_workspace_bytes(const NemotronHMoeExpertGeometry& geometry,
                                                   QType qtype, std::int32_t tokens) {
    if (geometry.latent_size <= 0 || geometry.num_experts <= 0 || geometry.intermediate_size <= 0 ||
        geometry.top_k <= 0) {
        fail("the geometry must be positive");
    }
    if (tokens <= 0) { fail("T must be positive"); }
    return detail::nemotron_h_moe_experts_workspace(
        qtype, geometry.latent_size, geometry.num_experts, geometry.intermediate_size,
        geometry.top_k, tokens);
}

void nemotron_h_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                            const NemotronHMoeExpertWeights& weights,
                            const NemotronHMoeExpertGeometry& geometry, Tensor& out,
                            Tensor& workspace, cudaStream_t stream) {
    const std::int32_t latent       = geometry.latent_size;
    const std::int32_t experts      = geometry.num_experts;
    const std::int32_t top_k        = geometry.top_k;
    const std::int32_t intermediate = geometry.intermediate_size;
    if (latent <= 0 || experts <= 0 || intermediate <= 0) { fail("the geometry must be positive"); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, latent, tokens, "x");
    if (out.dtype != DType::BF16) { fail("out must be BF16"); }
    require_matrix(out, latent, tokens, "out");
    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    const std::int64_t up_rows   = static_cast<std::int64_t>(intermediate) * experts;
    const std::int64_t down_rows = static_cast<std::int64_t>(latent) * experts;
    const detail::NemotronHMoeBankLaunch up =
        prepare_bank(weights.up, up_rows, latent, "nemotron_h_moe_experts up");
    const detail::NemotronHMoeBankLaunch down =
        prepare_bank(weights.down, down_rows, intermediate, "nemotron_h_moe_experts down");
    if (up.qtype != down.qtype) { fail("the two banks must share one format"); }

    const std::size_t required = nemotron_h_moe_experts_workspace_bytes(geometry, up.qtype, tokens);
    if (required > 0 &&
        (workspace.data == nullptr || !workspace.is_contiguous() || workspace.bytes() < required ||
         reinterpret_cast<std::uintptr_t>(workspace.data) % 16 != 0)) {
        fail("workspace must be a contiguous, 16-byte aligned buffer of at least " +
             std::to_string(required) + " bytes");
    }

    detail::nemotron_h_moe_experts_launch(x.data, expert_ids.data, expert_weights.data, up, down,
                                          latent, experts, intermediate, top_k, tokens,
                                          workspace.data, out.data, stream);
}

void nemotron_h_moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                            const ExpertWeights& bank, ExpertBankSpan source,
                            const NemotronHMoeExpertGeometry& geometry, Tensor& out,
                            Tensor& workspace, cudaStream_t stream) {
    const std::int32_t latent       = geometry.latent_size;
    const std::int32_t experts      = geometry.num_experts;
    const std::int32_t top_k        = geometry.top_k;
    const std::int32_t intermediate = geometry.intermediate_size;
    if (latent <= 0 || experts <= 0 || intermediate <= 0) { fail("the geometry must be positive"); }
    if (top_k <= 0 || top_k > experts) { fail("top_k must lie in [1, num_experts]"); }
    // The bank's own expert count, not the geometry's: the two are required to be the same number,
    // so a caller that took one layer's count and another layer's width -- or a global count --
    // is refused here rather than by a kernel reading past the bank.
    if (source.experts != experts || source.first_expert < 0) {
        fail("the bank span must be this layer's own: " + std::to_string(source.experts) +
             " experts against a geometry of " + std::to_string(experts));
    }
    if (source.experts > kMaxBankExperts) {
        fail("a bank of more than " + std::to_string(kMaxBankExperts) +
             " experts has no resolve table");
    }
    // One expert's rows are read as a contiguous run of both planes, which the K16M128x4 scale
    // tiling only allows when a run covers whole 128-row blocks: a bank whose latent width or
    // intermediate width is not a multiple of 128 would have an expert sharing a scale block with
    // its neighbour, and its planes would not be addressable as `base + e*stride`.
    if (latent % 128 != 0 || intermediate % 128 != 0) {
        fail("the plane-addressed route needs a latent and an intermediate width that are both "
             "multiples of 128, where a bank of " +
             std::to_string(experts) + " experts at " + std::to_string(latent) + "x" +
             std::to_string(intermediate) + " is neither");
    }

    if (x.dtype != DType::BF16) { fail("x must be BF16"); }
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, latent, tokens, "x");
    if (out.dtype != DType::BF16) { fail("out must be BF16"); }
    require_matrix(out, latent, tokens, "out");
    if (expert_ids.dtype != DType::I32) { fail("expert_ids must be I32"); }
    require_matrix(expert_ids, top_k, tokens, "expert_ids");
    if (expert_weights.dtype != DType::FP32) { fail("expert_weights must be FP32"); }
    require_matrix(expert_weights, top_k, tokens, "expert_weights");

    // The layout this bank must be: one expert of this layer, stated once by the same helper the
    // planner and the pager size the pool from. A bank bound with another layer's width, another
    // format or a plane count of its own cannot match it.
    const ExpertBankShape expected = nvfp4_expert_bank(experts, latent, intermediate, 1);
    const ExpertLayout& want       = expected.layout;
    if (bank.layout.format != ExpertFormat::Nvfp4 || bank.layout.planes != want.planes) {
        fail("the bank must be block-scaled NVFP4 with four planes");
    }
    // One expert's planes are read at `base[p] + e*stride[p]`. A stored or staged bank strides by
    // the layer's own per-expert plane size; the expert cache's slot pool strides every plane by
    // the pool's own -- one expert per slot, as wide as the pool's widest bank -- which is how a
    // gathered call reads the expert it routed to by slot id. Both are this layer's addressing,
    // and the layout check beside them pins the plane sizes either way, so a stride of either form
    // is admitted. Anything else is another layer's or another pool's geometry and is refused.
    for (std::int32_t plane = 0; plane < want.planes; ++plane) {
        if (bank.base[plane] == nullptr) {
            fail("bank plane " + std::to_string(plane) + " is unbound");
        }
        const std::int64_t stride = bank.stride[plane];
        const bool own_plane      = stride == want.plane_bytes[plane];
        const bool pool_slot      = stride == bank.stride[0] && stride >= want.slot_bytes;
        if (bank.layout.plane_bytes[plane] != want.plane_bytes[plane] || stride % 16 != 0 ||
            !(own_plane || pool_slot)) {
            fail("bank plane " + std::to_string(plane) +
                 " must stride by this layer's own per-expert plane size, or by a slot pool that "
                 "holds one of its experts, 16-byte aligned");
        }
    }
    if (bank.layout.slot_bytes != want.slot_bytes) {
        fail("the bank's expert size is not this layer's");
    }
    if (bank.gate_up_divisors == nullptr || bank.down_divisors == nullptr ||
        bank.gate_up_divisor_rows <= 0 || bank.down_divisor_rows <= 0 ||
        expected.gate_up_rows % bank.gate_up_divisor_rows != 0 ||
        expected.down_rows % bank.down_divisor_rows != 0) {
        fail("the bank's weight divisors must cover its rows");
    }
    // This conversion is W4A16: the *activation* divisors select the A4 route, which is a different
    // entry point. Conflating the two would run A16 as if the inputs had been quantized.
    if (bank.gate_up_input_divisor != 0.0F || bank.down_input_divisor != 0.0F) {
        fail("this entry point is the A16 route; a bank with activation divisors takes the A4 one");
    }

    // The same check the whole-matrix entry point makes, against this format's own query: a call
    // the row-tiled route takes needs the two buffers the query reports -- one token tile's, at any
    // width -- and the per-token route a wider call takes needs nothing.
    const std::size_t required =
        nemotron_h_moe_experts_workspace_bytes(geometry, QType::NVFP4, tokens);
    if (required > 0 &&
        (workspace.data == nullptr || !workspace.is_contiguous() || workspace.bytes() < required ||
         reinterpret_cast<std::uintptr_t>(workspace.data) % 16 != 0)) {
        fail("workspace must be a contiguous, 16-byte aligned buffer of at least " +
             std::to_string(required) + " bytes");
    }

    detail::nemotron_h_moe_experts_planar_launch(x.data, expert_ids.data, expert_weights.data, bank,
                                                 source, latent, intermediate, top_k, tokens,
                                                 workspace.data, out.data, stream);
}

} // namespace ninfer::ops
