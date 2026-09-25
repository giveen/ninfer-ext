// ninfer::ops - offloaded MoE wrappers: validate the public contract, carve workspace, launch.
#include "ninfer/ops/offload_moe.h"

#include "ops/offload_moe/launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// Token columns per internal pass; bounds the per-assignment intermediate and partial buffers.
constexpr std::int32_t kChunkColumns = 1024;

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

void require_routed(const Tensor& t, DType dtype, std::int32_t columns, const char* op,
                    const char* label) {
    require_dense(t, dtype, op, label);
    require(t.ne[0] == kOffloadMoeTopK && t.ne[1] == columns, op,
            std::string(label) + " must be [10,T]");
}

constexpr std::size_t align(std::size_t value) { return (value + 255) / 256 * 256; }

struct ChunkLayout {
    std::size_t counts, sorted_assign, sorted_slot, jobs, job_count, act, partial, total;
};

ChunkLayout chunk_layout(std::int32_t columns, std::int32_t slots) {
    const std::size_t assignments = static_cast<std::size_t>(columns) * kOffloadMoeTopK;
    ChunkLayout out{};
    std::size_t cursor = 0;
    auto take          = [&](std::size_t bytes) {
        const std::size_t at = cursor;
        cursor += align(bytes);
        return at;
    };
    out.counts        = take(static_cast<std::size_t>(std::max(slots, 1)) * 4);
    out.sorted_assign = take(assignments * 4);
    out.sorted_slot   = take(assignments * 4);
    out.jobs          = take(assignments * 4);
    out.job_count     = take(4);
    out.act           = take(assignments * kOffloadMoeIntermediate * 2);
    out.partial       = take(assignments * kOffloadMoeHidden * 4);
    out.total         = cursor;
    return out;
}

struct A4Layout {
    std::size_t input_codes, input_scales, counts, offsets, rank, packed_token, packed_index,
        job_experts, job_columns, job_count, middle_codes, middle_scales, grouped, total;
};

A4Layout a4_layout(std::int32_t columns) {
    const std::size_t tokens      = static_cast<std::size_t>(columns);
    const std::size_t assignments = tokens * kOffloadMoeTopK;
    const auto jobs               = static_cast<std::size_t>(detail::a4_max_jobs(columns));
    A4Layout out{};
    std::size_t cursor = 0;
    auto take          = [&](std::size_t bytes) {
        const std::size_t at = cursor;
        cursor += align(bytes);
        return at;
    };
    out.input_codes   = take(tokens * kOffloadMoeHidden / 2);
    out.input_scales  = take(tokens * kOffloadMoeHidden / 16);
    out.counts        = take(static_cast<std::size_t>(kOffloadMoeExperts) * 4);
    out.offsets       = take(static_cast<std::size_t>(kOffloadMoeExperts + 1) * 4);
    out.rank          = take(assignments * 4);
    out.packed_token  = take(assignments * 4);
    out.packed_index  = take(assignments * 4);
    out.job_experts   = take(jobs * 4);
    out.job_columns   = take(jobs * 4);
    out.job_count     = take(4);
    out.middle_codes  = take(assignments * kOffloadMoeIntermediate / 2);
    out.middle_scales = take(assignments * kOffloadMoeIntermediate / 16);
    out.grouped       = take(assignments * kOffloadMoeHidden * 2);
    out.total         = cursor;
    return out;
}

} // namespace

std::size_t moe_route_workspace_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("moe_route_workspace_bytes: T <= 0"); }
    return align(static_cast<std::size_t>(tokens) * (kOffloadMoeExperts + 1) * sizeof(float)) + 256;
}

void moe_route(const Tensor& x, const Tensor& router, WorkspaceArena& workspace, Tensor& ids,
               Tensor& weights, Tensor& shared_gate, cudaStream_t stream) {
    constexpr const char* op = "moe_route";
    require_dense(x, DType::BF16, op, "x");
    require(x.ne[0] == kOffloadMoeHidden && x.ne[2] == 1 && x.ne[3] == 1, op, "x must be [2560,T]");
    require_dense(router, DType::BF16, op, "router");
    require(router.numel() == static_cast<std::int64_t>(kOffloadMoeExperts + 1) * kOffloadMoeHidden,
            op, "router must be BF16 [513,2560]");
    require_routed(ids, DType::I32, x.ne[1], op, "ids");
    require_routed(weights, DType::FP32, x.ne[1], op, "weights");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    require(shared_gate.numel() == x.ne[1], op, "shared gate must be [T]");
    require(x.data != nullptr && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(router.data) % 16 == 0,
            op, "x and router must be 16-byte aligned");
    auto scope   = workspace.scope();
    auto* logits = static_cast<float*>(workspace
                                           .alloc_bytes(static_cast<std::size_t>(x.ne[1]) *
                                                        (kOffloadMoeExperts + 1) * sizeof(float))
                                           .data);
    detail::moe_route_launch(x, router, logits, ids, weights, shared_gate, stream);
}

void expert_cache_resolve(const Tensor& ids, std::int32_t layer, const ExpertCacheState& cache,
                          Tensor& slot_ids, Tensor& misses, cudaStream_t stream) {
    constexpr const char* op = "expert_cache_resolve";
    require_routed(ids, DType::I32, ids.ne[1], op, "ids");
    require_routed(slot_ids, DType::I32, ids.ne[1], op, "slot ids");
    require_dense(misses, DType::I32, op, "misses");
    require(misses.numel() >= 2 * ids.numel() + 1, op, "misses must hold 2*(10*T)+1 words");
    require(layer >= 0 && layer < cache.layers, op, "layer outside the cache");
    require(cache.slots >= ids.numel(), op, "the cache must hold every assignment of one call");
    require(cache.slot_of && cache.owner && cache.stamp && cache.clock && cache.statistics &&
                cache.pool,
            op, "cache state is incomplete");
    detail::expert_cache_resolve_launch(ids, layer, cache, slot_ids, misses, stream);
}

void expert_cache_fetch(const ExpertWeights& bank, const Tensor& misses, std::int32_t max_misses,
                        const ExpertCacheState& cache, cudaStream_t stream) {
    constexpr const char* op = "expert_cache_fetch";
    require_dense(misses, DType::I32, op, "misses");
    require(max_misses > 0 && misses.numel() >= 2 * max_misses + 1, op, "invalid miss capacity");
    for (const auto* base : bank.base) { require(base != nullptr, op, "bank planes are required"); }
    require(cache.pool != nullptr, op, "cache pool is required");
    detail::expert_cache_fetch_launch(bank, misses, max_misses, cache, stream);
}

ExpertWeights expert_cache_weights(const ExpertCacheState& cache, const ExpertWeights& bank) {
    ExpertWeights out             = bank;
    const std::int64_t offsets[4] = {
        0, kExpertGateUpCodeBytes, kExpertGateUpCodeBytes + kExpertGateUpScaleBytes,
        kExpertGateUpCodeBytes + kExpertGateUpScaleBytes + kExpertDownCodeBytes};
    for (int p = 0; p < 4; ++p) {
        out.base[p]   = cache.pool + offsets[p];
        out.stride[p] = kExpertSlotBytes;
    }
    return out;
}

std::size_t moe_experts_workspace_bytes(std::int32_t tokens, std::int32_t slots) {
    if (tokens <= 0) { throw std::invalid_argument("moe_experts_workspace_bytes: T <= 0"); }
    return chunk_layout(std::min(tokens, kChunkColumns), slots).total + 256;
}

void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& slot_ids,
                 const Tensor& weights, const Tensor& shared_gate, const Tensor& shared,
                 const ExpertWeights& weights_source, std::int32_t slots, WorkspaceArena& workspace,
                 Tensor& y, cudaStream_t stream) {
    constexpr const char* op = "moe_experts";
    require_dense(x, DType::BF16, op, "x");
    const std::int32_t columns = x.ne[1];
    require(x.ne[0] == kOffloadMoeHidden, op, "x must be [2560,T]");
    require_routed(expert_ids, DType::I32, columns, op, "expert ids");
    require_routed(slot_ids, DType::I32, columns, op, "slot ids");
    require_routed(weights, DType::FP32, columns, op, "weights");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    require_dense(shared, DType::BF16, op, "shared");
    require_dense(y, DType::BF16, op, "y");
    require(shared.numel() == x.numel() && y.numel() == x.numel(), op, "shared/y must match x");
    require(weights_source.gate_up_divisors != nullptr && weights_source.down_divisors != nullptr &&
                weights_source.gate_up_divisor_rows > 0 && weights_source.down_divisor_rows > 0,
            op, "expert divisors are required");
    const std::int32_t chunk = std::min(columns, kChunkColumns);
    require(chunk * kOffloadMoeTopK <= 1024 || slots <= 4096, op,
            "large token chunks require a staged bank indexed by expert id");
    const ChunkLayout layout = chunk_layout(chunk, slots);
    auto scope               = workspace.scope();
    auto* base               = static_cast<std::byte*>(workspace.alloc_bytes(layout.total).data);
    for (std::int32_t begin = 0; begin < columns; begin += chunk) {
        const std::int32_t count = std::min(chunk, columns - begin);
        const std::int64_t k0    = static_cast<std::int64_t>(begin) * kOffloadMoeTopK;
        detail::MoeChunk c{};
        c.x = static_cast<const __nv_bfloat16*>(x.data) +
              static_cast<std::int64_t>(begin) * kOffloadMoeHidden;
        c.expert_ids  = static_cast<const std::int32_t*>(expert_ids.data) + k0;
        c.slot_ids    = static_cast<const std::int32_t*>(slot_ids.data) + k0;
        c.weights     = static_cast<const float*>(weights.data) + k0;
        c.shared_gate = static_cast<const float*>(shared_gate.data) + begin;
        c.shared      = static_cast<const __nv_bfloat16*>(shared.data) +
                   static_cast<std::int64_t>(begin) * kOffloadMoeHidden;
        c.y = static_cast<__nv_bfloat16*>(y.data) +
              static_cast<std::int64_t>(begin) * kOffloadMoeHidden;
        c.columns       = count;
        c.slots         = slots;
        c.source        = weights_source;
        c.counts        = reinterpret_cast<std::int32_t*>(base + layout.counts);
        c.sorted_assign = reinterpret_cast<std::int32_t*>(base + layout.sorted_assign);
        c.sorted_slot   = reinterpret_cast<std::int32_t*>(base + layout.sorted_slot);
        c.jobs          = reinterpret_cast<std::int32_t*>(base + layout.jobs);
        c.job_count     = reinterpret_cast<std::int32_t*>(base + layout.job_count);
        c.act           = reinterpret_cast<__nv_bfloat16*>(base + layout.act);
        c.partial       = reinterpret_cast<float*>(base + layout.partial);
        detail::moe_experts_chunk_launch(c, stream);
    }
}

std::size_t moe_experts_a4_workspace_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("moe_experts_a4_workspace_bytes: T <= 0"); }
    return a4_layout(std::min(tokens, kChunkColumns)).total + 256;
}

void moe_experts_a4(const Tensor& x, const Tensor& expert_ids, const Tensor& weights,
                    const Tensor& shared_gate, const Tensor& shared, const ExpertWeights& staged,
                    WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    constexpr const char* op = "moe_experts_a4";
    require_dense(x, DType::BF16, op, "x");
    const std::int32_t columns = x.ne[1];
    require(x.ne[0] == kOffloadMoeHidden, op, "x must be [2560,T]");
    require_routed(expert_ids, DType::I32, columns, op, "expert ids");
    require_routed(weights, DType::FP32, columns, op, "weights");
    require_dense(shared_gate, DType::FP32, op, "shared gate");
    require(shared_gate.numel() == columns, op, "shared gate must be [T]");
    require_dense(shared, DType::BF16, op, "shared");
    require_dense(y, DType::BF16, op, "y");
    require(shared.numel() == x.numel() && y.numel() == x.numel(), op, "shared/y must match x");
    require(staged.stride[0] == kExpertGateUpCodeBytes &&
                staged.stride[1] == kExpertGateUpScaleBytes &&
                staged.stride[2] == kExpertDownCodeBytes &&
                staged.stride[3] == kExpertDownScaleBytes,
            op, "the staged bank must use the plane layout");
    require(staged.gate_up_divisors != nullptr && staged.down_divisors != nullptr &&
                staged.gate_up_divisor_rows == kOffloadMoeIntermediate &&
                staged.down_divisor_rows == kOffloadMoeHidden,
            op, "one weight divisor per expert gate, up and down is required");
    require(staged.gate_up_input_divisor > 0.0F && staged.down_input_divisor > 0.0F, op,
            "A4 activation divisors must be positive");
    const std::int32_t chunk = std::min(columns, kChunkColumns);
    const A4Layout layout    = a4_layout(chunk);
    auto scope               = workspace.scope();
    auto* base               = static_cast<std::byte*>(workspace.alloc_bytes(layout.total).data);
    for (std::int32_t begin = 0; begin < columns; begin += chunk) {
        const std::int64_t k0 = static_cast<std::int64_t>(begin) * kOffloadMoeTopK;
        const std::int64_t h0 = static_cast<std::int64_t>(begin) * kOffloadMoeHidden;
        detail::MoeA4Chunk c{};
        c.x             = static_cast<const __nv_bfloat16*>(x.data) + h0;
        c.expert_ids    = static_cast<const std::int32_t*>(expert_ids.data) + k0;
        c.weights       = static_cast<const float*>(weights.data) + k0;
        c.shared_gate   = static_cast<const float*>(shared_gate.data) + begin;
        c.shared        = static_cast<const __nv_bfloat16*>(shared.data) + h0;
        c.y             = static_cast<__nv_bfloat16*>(y.data) + h0;
        c.columns       = std::min(chunk, columns - begin);
        c.source        = staged;
        c.input_codes   = reinterpret_cast<std::uint8_t*>(base + layout.input_codes);
        c.input_scales  = reinterpret_cast<std::uint8_t*>(base + layout.input_scales);
        c.counts        = reinterpret_cast<std::int32_t*>(base + layout.counts);
        c.offsets       = reinterpret_cast<std::int32_t*>(base + layout.offsets);
        c.rank          = reinterpret_cast<std::int32_t*>(base + layout.rank);
        c.packed_token  = reinterpret_cast<std::int32_t*>(base + layout.packed_token);
        c.packed_index  = reinterpret_cast<std::int32_t*>(base + layout.packed_index);
        c.job_experts   = reinterpret_cast<std::int32_t*>(base + layout.job_experts);
        c.job_columns   = reinterpret_cast<std::int32_t*>(base + layout.job_columns);
        c.job_count     = reinterpret_cast<std::int32_t*>(base + layout.job_count);
        c.middle_codes  = reinterpret_cast<std::uint8_t*>(base + layout.middle_codes);
        c.middle_scales = reinterpret_cast<std::uint8_t*>(base + layout.middle_scales);
        c.grouped       = reinterpret_cast<__nv_bfloat16*>(base + layout.grouped);
        detail::moe_experts_a4_launch(c, stream);
    }
}

} // namespace ninfer::ops
