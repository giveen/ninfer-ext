// Cold-cache benchmark for the public qsa_select contract (Qwen Sparse Attention block selection).
//
// The selection cost is quadratic in the prompt: every query column scores every visible 4-token
// block. This bench reproduces the production call shape of Qwen3.8-Flash-Next (indexer 4x128,
// budget 2048 in blocks of 4, one lane) without loading a model, and prints a checksum of the
// selection so a faster kernel can be shown to select exactly the same blocks.
//
// Examples:
//   ./build/bench/ninfer_qsa_select_bench
//   ./build/bench/ninfer_qsa_select_bench --tokens 65536 --columns 16384
//   ./build/bench/ninfer_qsa_select_bench --sweep 8192,16384,32768,65536
//   ./build/bench/ninfer_qsa_select_bench --tokens 65536 --columns 16384 --profile

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ninfer/ops/sparse_attention.h"

#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kQueryHeads   = 24;
constexpr std::int32_t kKvHeads      = 2;
constexpr std::int32_t kHeadDim      = 256;
constexpr std::int32_t kIndexHeads   = 4;
constexpr std::int32_t kIndexDim     = ops::kQsaIndexKeyDim;
constexpr std::int32_t kRatio        = 4;
constexpr std::int32_t kBudget       = 2048;
constexpr std::int32_t kRotary       = 64;
constexpr float kTheta               = 1.0e7F;
constexpr float kEps                 = 1.0e-6F;
constexpr std::int32_t kPage         = kPagedKVPageSize;
constexpr std::int32_t kRecordWords  = ops::kQsaIndexRecordWords;
constexpr std::int32_t kDefaultTokens  = 65536;
constexpr std::int32_t kDefaultColumns = 16384;
constexpr std::int32_t kWarmup         = 2;
constexpr std::int32_t kRepeat         = 10;

ops::QsaGeometry geometry() {
    ops::QsaGeometry g;
    g.query_heads  = kQueryHeads;
    g.kv_heads     = kKvHeads;
    g.head_dim     = kHeadDim;
    g.index_heads  = kIndexHeads;
    g.index_dim    = kIndexDim;
    g.budget       = kBudget;
    g.ratio        = kRatio;
    g.rotary_dim   = kRotary;
    g.rope_theta   = kTheta;
    g.rms_norm_eps = kEps;
    for (int i = 0; i < kRotary / 2; ++i) { g.pair_axes[i] = static_cast<std::uint8_t>(i % 3); }
    return g;
}

std::int32_t axis_position(std::int32_t token, int axis) { return token / (axis + 1) + 3 * axis; }

std::uint16_t f32_to_bf16_bits(float value) {
    const std::uint32_t b = *reinterpret_cast<const std::uint32_t*>(&value);
    const std::uint32_t rounding = 0x7FFFU + ((b >> 16) & 1U);
    return static_cast<std::uint16_t>((b + rounding) >> 16);
}

// Deterministic, cheap pseudo-random BF16 in [-1, 1): the score kernel's arithmetic, not the data,
// is what this bench measures.
std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> out(count);
    std::uint32_t state = seed * 1664525U + 1013904223U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
        out[i]        = f32_to_bf16_bits((2.0F * u - 1.0F) * scale);
    }
    return out;
}

template <class T>
DeviceBuffer to_device(const std::vector<T>& host) {
    DeviceBuffer buffer(host.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return buffer;
}

DeviceBuffer to_device_i32(const std::vector<std::int32_t>& host) {
    DeviceBuffer buffer(host.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice));
    return buffer;
}

template <class T>
std::vector<T> from_device(const void* data, std::size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

struct Result {
    double ms    = 0.0;
    double flops = 0.0;
    std::uint64_t hash = 0;
    std::int64_t selected_total = 0;
    // Optional second measurement of the attention over the selection the run produced.
    double attention_ms = 0.0;
    double attention_flops = 0.0;
};

// FNV-1a over the selection: identical bytes for an identical block choice.
std::uint64_t hash_i32(const std::vector<std::int32_t>& values, std::uint64_t seed) {
    std::uint64_t h = seed;
    for (const std::int32_t v : values) {
        std::uint64_t x = static_cast<std::uint64_t>(v);
        for (int byte = 0; byte < 4; ++byte) {
            h ^= (x >> (8 * byte)) & 0xFFU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

// Blocks scored per query column at `tokens`: sum of the causal block counts over all `columns`
// columns ending at `tokens` (the last column sees every token).
double score_flops(std::int32_t tokens, std::int32_t columns) {
    const double blocks = static_cast<double>(tokens) / kRatio;
    double pairs        = 0.0;
    for (std::int32_t i = 0; i < columns; ++i) {
        pairs += (static_cast<double>(tokens - columns + 1 + i)) / kRatio;
    }
    (void) blocks;
    return pairs * kIndexHeads * kIndexDim * 2.0;
}

Result run_case(std::int32_t tokens, std::int32_t columns, bool profile, bool attention,
                KvCacheStorage storage) {
    const ops::QsaGeometry g = geometry();
    const std::int32_t pages = (tokens + kPage - 1) / kPage;

    // Identity paging: logical page p maps to physical page p.
    std::vector<std::int32_t> tables(static_cast<std::size_t>(pages), 0);
    std::iota(tables.begin(), tables.end(), 0);
    std::vector<std::int32_t> rows{0};

    // Index records: the kernel pools these block keys once per call, then scores them.
    const std::size_t record_words = static_cast<std::size_t>(kRecordWords) * kPage * pages;
    std::vector<std::uint16_t> records = random_bf16(record_words, 0x5aU, 1.0F);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token / kPage) * kPage * kRecordWords +
                                 (token % kPage) * kRecordWords;
        auto* words = reinterpret_cast<std::uint16_t*>(records.data() + base + kIndexDim);
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint32_t p = static_cast<std::uint32_t>(axis_position(token, axis));
            words[2 * axis]       = static_cast<std::uint16_t>(p & 0xFFFFU);
            words[2 * axis + 1]   = static_cast<std::uint16_t>(p >> 16);
        }
    }

    std::vector<std::int32_t> cache_positions(columns);
    std::vector<std::int32_t> rope(static_cast<std::size_t>(columns) * 3);
    for (std::int32_t c = 0; c < columns; ++c) {
        const std::int32_t position = tokens - columns + c;
        cache_positions[c]          = position;
        for (int axis = 0; axis < 3; ++axis) {
            rope[static_cast<std::size_t>(axis) * columns + c] = axis_position(position, axis);
        }
    }

    const std::size_t query_words =
        static_cast<std::size_t>(kIndexHeads) * kIndexDim * columns;
    const std::vector<std::uint16_t> index_query = random_bf16(query_words, 0x21U, 2.0F);
    const std::vector<std::uint16_t> query_norm  = random_bf16(kIndexDim, 0x22U, 0.3F);
    const std::vector<std::uint16_t> key_norm    = random_bf16(kIndexDim, 0x23U, 0.3F);

    DeviceBuffer drecords = to_device(records);
    DeviceBuffer dtables  = to_device_i32(tables);
    DeviceBuffer drows    = to_device_i32(rows);
    DeviceBuffer dcache   = to_device_i32(cache_positions);
    DeviceBuffer drope    = to_device_i32(rope);
    DeviceBuffer diq      = to_device(index_query);
    DeviceBuffer dqn      = to_device(query_norm);
    DeviceBuffer dkn      = to_device(key_norm);
    DeviceBuffer dvalid   = to_device_i32(std::vector<std::int32_t>{columns});

    Tensor tiq(diq.p, DType::BF16, {kIndexHeads * kIndexDim, columns, 1});
    Tensor tqn(dqn.p, DType::BF16, {kIndexDim});
    Tensor tkn(dkn.p, DType::BF16, {kIndexDim});
    Tensor tqc(dcache.p, DType::I32, {columns, 1});
    Tensor tqr(drope.p, DType::I32, {columns, 3});
    Tensor tvalid(dvalid.p, DType::I32, {1});
    Tensor trows(drows.p, DType::I32, {1});
    Tensor ttables(dtables.p, DType::I32, {pages, 1});
    ops::QsaIndexPlane plane{
        .pages = Tensor(drecords.p, DType::BF16, {kRecordWords, kPage, 1, pages}),
        .block_tables = ttables};

    DeviceBuffer selected(static_cast<std::size_t>(g.max_selected()) * columns * 4);
    DeviceBuffer counts(static_cast<std::size_t>(columns) * 4);
    Tensor tsel(selected.p, DType::I32, {g.max_selected(), columns, 1});
    Tensor tcnt(counts.p, DType::I32, {columns, 1});

    WorkspaceArena workspace(ops::qsa_select_workspace_bytes(g, tokens, columns));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < kWarmup; ++i) {
        ops::qsa_select(tiq, tqr, tqc, &tvalid, trows, tqn, tkn, plane, g, tokens, workspace, tsel,
                        tcnt, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (profile) { CUDA_CHECK(cudaProfilerStart()); }
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kRepeat; ++i) {
        // A fresh workspace scope per call keeps the scratch allocation identical to production.
        ops::qsa_select(tiq, tqr, tqc, &tvalid, trows, tqn, tkn, plane, g, tokens, workspace, tsel,
                        tcnt, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (profile) { CUDA_CHECK(cudaProfilerStop()); }
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    Result result;
    result.ms    = static_cast<double>(elapsed_ms) / kRepeat;
    result.flops = score_flops(tokens, columns);

    const auto sel_host = from_device<std::int32_t>(
        selected.p, static_cast<std::size_t>(g.max_selected()) * columns);
    const auto cnt_host =
        from_device<std::int32_t>(counts.p, static_cast<std::size_t>(columns));
    result.hash           = hash_i32(sel_host, 1469598103934665603ULL);
    result.hash           = hash_i32(cnt_host, result.hash);
    result.selected_total = std::accumulate(cnt_host.begin(), cnt_host.end(), std::int64_t{0});
    if (attention) {
        // The attention reads the selection above from a paged cache of the given profile. The
        // pages are zero: only the addressing and the widening run.
        const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
        const auto plane_bytes = [&](const PagedKVVectorLayout& vector, bool scale) {
            return static_cast<std::size_t>(kPage) * kKvHeads * pages *
                   (scale ? vector.scale_leading_extent : vector.data_leading_extent) *
                   dtype_size(scale ? vector.scale_dtype : vector.data_dtype);
        };
        DeviceBuffer k_pages(plane_bytes(layout.key, false));
        DeviceBuffer v_pages(plane_bytes(layout.value, false));
        DeviceBuffer k_scales(plane_bytes(layout.key, true));
        DeviceBuffer v_scales(plane_bytes(layout.value, true));
        CUDA_CHECK(cudaMemset(k_pages.p, 0, plane_bytes(layout.key, false)));
        CUDA_CHECK(cudaMemset(v_pages.p, 0, plane_bytes(layout.value, false)));
        CUDA_CHECK(cudaMemset(k_scales.p, 0, plane_bytes(layout.key, true)));
        CUDA_CHECK(cudaMemset(v_scales.p, 0, plane_bytes(layout.value, true)));
        const auto plane_view = [&](DeviceBuffer& buffer, DType dtype, std::int32_t extent) {
            return Tensor(buffer.p, dtype, {extent, kPage, kKvHeads, pages});
        };
        PagedKVBatchLayerView kv{
            .k_pages = plane_view(k_pages, layout.key.data_dtype, layout.key.data_leading_extent),
            .v_pages = plane_view(v_pages, layout.value.data_dtype, layout.value.data_leading_extent),
            .k_scale_pages =
                plane_view(k_scales, layout.key.scale_dtype, layout.key.scale_leading_extent),
            .v_scale_pages =
                plane_view(v_scales, layout.value.scale_dtype, layout.value.scale_leading_extent),
            .block_tables = ttables,
            .head_dim     = kHeadDim,
            .num_kv_heads = kKvHeads,
            .storage      = storage};
        const std::size_t q_words = static_cast<std::size_t>(kHeadDim) * kQueryHeads * columns;
        const std::vector<std::uint16_t> q_host = random_bf16(q_words, 0x31U, 1.0F);
        DeviceBuffer dq  = to_device(q_host);
        DeviceBuffer dout(q_words * sizeof(std::uint16_t));
        Tensor tq(dq.p, DType::BF16, {kHeadDim, kQueryHeads, columns, 1});
        Tensor tout(dout.p, DType::BF16, {kHeadDim, kQueryHeads, columns, 1});
        const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
        WorkspaceArena attn_workspace(
            ops::qsa_attention_workspace_bytes(g, columns));
        for (int i = 0; i < kWarmup; ++i) {
            ops::qsa_attention(tq, tsel, tcnt, trows, kv, scale, attn_workspace, tout, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (profile) { CUDA_CHECK(cudaProfilerStart()); }
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < kRepeat; ++i) {
            ops::qsa_attention(tq, tsel, tcnt, trows, kv, scale, attn_workspace, tout, nullptr);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        if (profile) { CUDA_CHECK(cudaProfilerStop()); }
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        // 2 (QK and PV) * head_dim products per kept key per query head.
        result.attention_flops =
            2.0 * 2.0 * kHeadDim * kQueryHeads *
            (static_cast<double>(result.selected_total) / kQueryHeads);
        result.attention_ms = static_cast<double>(elapsed_ms) / kRepeat;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return result;
}

void print_case(std::int32_t tokens, std::int32_t columns, const Result& r) {
    const double tflops = r.ms > 0.0 ? r.flops / (r.ms * 1.0e9) : 0.0;
    std::printf("tokens %6d  columns %6d  selection %8.3f ms  %7.2f TFLOP/s  kept %10lld  hash %016llx\n",
                tokens, columns, r.ms, tflops, static_cast<long long>(r.selected_total),
                static_cast<unsigned long long>(r.hash));
    if (r.attention_ms > 0.0) {
        std::printf("%44s attention %8.3f ms  %7.2f TFLOP/s\n", "",
                    r.attention_ms, r.attention_flops / (r.attention_ms * 1.0e9));
    }
}

std::vector<std::int32_t> parse_list(const char* text) {
    std::vector<std::int32_t> out;
    std::string list(text);
    std::size_t begin = 0;
    while (begin <= list.size()) {
        const std::size_t end = list.find(',', begin);
        const std::string item =
            list.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!item.empty()) { out.push_back(std::atoi(item.c_str())); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::int32_t tokens  = kDefaultTokens;
    std::int32_t columns = kDefaultColumns;
    std::vector<std::int32_t> sweep;
    bool profile   = false;
    bool attention = false;
    KvCacheStorage storage = KvCacheStorage::Fp8E4M3Row256;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next       = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "qsa_select_bench: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--tokens") {
            tokens = std::atoi(next("--tokens"));
        } else if (arg == "--columns") {
            columns = std::atoi(next("--columns"));
        } else if (arg == "--sweep") {
            sweep = parse_list(next("--sweep"));
        } else if (arg == "--attention") {
            attention = true;
        } else if (arg == "--kv") {
            const std::string mode = next("--kv");
            if (mode == "bf16") {
                storage = KvCacheStorage::BFloat16;
            } else if (mode == "fp8") {
                storage = KvCacheStorage::Fp8E4M3Row256;
            } else if (mode == "int8") {
                storage = KvCacheStorage::Int8Group64;
            } else if (mode == "nvfp4") {
                storage = KvCacheStorage::Nvfp4Group16;
            } else if (mode == "k8v4") {
                storage = KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                std::fprintf(stderr, "qsa_select_bench: unknown --kv %s\n", mode.c_str());
                return 2;
            }
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("usage: ninfer_qsa_select_bench [--tokens N] [--columns N] [--sweep a,b,c] "
                        "[--attention] [--kv bf16|fp8|int8|nvfp4|k8v4] [--profile]\n");
            return 0;
        } else {
            std::fprintf(stderr, "qsa_select_bench: unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (tokens <= 0 || columns <= 0 || columns > tokens) {
        std::fprintf(stderr, "qsa_select_bench: need 0 < columns <= tokens\n");
        return 2;
    }

    std::printf("qsa_select: indexer %dx%d, ratio %d, budget %d, one lane\n", kIndexHeads, kIndexDim,
                kRatio, kBudget);
    if (!sweep.empty()) {
        for (const std::int32_t depth : sweep) {
            if (depth <= 0) { continue; }
            // The production prompt reads a chunk whose last column sees the whole prompt; keep the
            // column count at the default chunk unless the depth is smaller.
            const std::int32_t cols = std::min(columns, depth);
            print_case(depth, cols, run_case(depth, cols, profile, attention, storage));
        }
        return 0;
    }
    print_case(tokens, columns, run_case(tokens, columns, profile, attention, storage));
    return 0;
}
