// Qwen Sparse Attention Ops at the Qwen3.8-Flash-Next geometry (24/2 heads of 256, indexer 4x128,
// budget 2048 in blocks of 4) over permuted physical pages. qsa_append and qsa_attention run over
// every KV profile: BF16, and the quantized FP8, int8 G64, NVFP4 G16 and K8V4 caches.
//
// qsa_append is exact. qsa_select is compared with an FP64 oracle of the block scores; a block may
// differ from the oracle only when its oracle score lies within a near-tie allowance of the
// selection cut. qsa_attention is compared with FP64 softmax attention over the selection the
// device produced, reading K/V at their storage boundary.
#include "ninfer/ops/sparse_attention.h"

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/kv_page_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <optional>
#include <tuple>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <cuda_fp16.h>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kQueryHeads   = 24;
constexpr std::int32_t kKvHeads      = 2;
constexpr std::int32_t kHeadDim      = 256;
constexpr std::int32_t kIndexHeads   = 4;
constexpr std::int32_t kIndexDim     = 128;
constexpr std::int32_t kRatio        = 4;
constexpr std::int32_t kBudget       = 2048;
constexpr std::int32_t kRotary       = 64;
constexpr float kTheta               = 1.0e7F;
constexpr float kEps                 = 1.0e-6F;
constexpr std::int32_t kPage         = kPagedKVPageSize;
constexpr std::int32_t kLanes        = 2;
constexpr std::int32_t kTokens       = 2200; // cached per lane
constexpr std::int32_t kPagesPerLane = (kTokens + kPage - 1) / kPage;
constexpr std::int32_t kPages        = kPagesPerLane * kLanes;
constexpr double kNearTie            = 2.0e-3; // relative to the largest block score

constexpr ReductionCriterion kAttentionOutput{4.0e-3, 1.0e-2, 1.0e-2};

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

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

// Three-axis positions of a token: distinct per axis so the axis mapping matters.
std::int32_t axis_position(std::int32_t token, int axis) { return token / (axis + 1) + 3 * axis; }

struct Lane {
    std::int32_t table_row;
    std::vector<int> pages; // logical page -> physical page
};

std::vector<double> rotate(std::vector<double> v, const std::int32_t position[3]) {
    for (int i = 0; i < kRotary / 2; ++i) {
        const double angle = position[i % 3] * std::pow(double(kTheta), -2.0 * i / double(kRotary));
        const double x0 = v[i], x1 = v[i + kRotary / 2];
        v[i]               = x0 * std::cos(angle) - x1 * std::sin(angle);
        v[i + kRotary / 2] = x1 * std::cos(angle) + x0 * std::sin(angle);
    }
    return v;
}

double e4m3_value(std::uint8_t code) {
    const int e    = (code >> 3) & 0xF;
    const int m    = code & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return (code & 0x80) != 0 ? -v : v;
}

double half_value(std::uint16_t bits_value) {
    __half h;
    std::memcpy(&h, &bits_value, 2);
    return double(__half2float(h));
}

// Normalized natural-order Sylvester transform, H[i][j] = (-1)^popcount(i & j) / 16: the rotation
// FP8 K rows are stored in.
std::vector<double> hadamard256(std::vector<double> v) {
    for (std::size_t span = 1; span < v.size(); span <<= 1) {
        for (std::size_t base = 0; base < v.size(); base += 2 * span) {
            for (std::size_t i = base; i < base + span; ++i) {
                const double low = v[i], high = v[i + span];
                v[i]        = low + high;
                v[i + span] = low - high;
            }
        }
    }
    for (double& x : v) { x /= 16.0; }
    return v;
}

double e2m1_value(std::uint8_t nibble) {
    static constexpr double kMagnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v                        = kMagnitude[nibble & 7];
    return (nibble & 8) != 0 ? -v : v;
}

// The device planes of one quantized profile and their host copies, decoded with the stored
// scales: FP8 codes times the row scale, int8 codes times their G64 scale, E2M1 nibbles (feature
// 2i in the low nibble of byte i) times their G16 E4M3 scale.
struct QuantizedCache {
    std::string name;
    bool v_rotated;
    PagedKVStorageLayout layout;
    DeviceBuffer k_pages, v_pages, k_scales, v_scales;
    PagedKVBatchLayerView view;
    std::vector<std::uint8_t> k_host, v_host, k_scale_host, v_scale_host;

    static std::size_t data_bytes(const PagedKVVectorLayout& vector) {
        return static_cast<std::size_t>(kPage) * kKvHeads * kPages * vector.data_leading_extent *
               dtype_size(vector.data_dtype);
    }
    static std::size_t scale_bytes(const PagedKVVectorLayout& vector) {
        return static_cast<std::size_t>(kPage) * kKvHeads * kPages * vector.scale_leading_extent *
               dtype_size(vector.scale_dtype);
    }
    static Tensor pages(const DeviceBuffer& buffer, DType dtype, std::int32_t extent) {
        return Tensor(buffer.p, dtype, {extent, kPage, kKvHeads, kPages});
    }

    QuantizedCache(KvCacheStorage storage, const char* label, const Tensor& tables)
        : name(label), v_rotated(storage == KvCacheStorage::Nvfp4Group16 ||
                                 storage == KvCacheStorage::Fp8KeyNvfp4Value),
          layout(paged_kv_storage_layout(storage, kHeadDim)),
          k_pages(to_device(std::vector<std::uint8_t>(data_bytes(layout.key), 0))),
          v_pages(to_device(std::vector<std::uint8_t>(data_bytes(layout.value), 0))),
          k_scales(to_device(std::vector<std::uint8_t>(scale_bytes(layout.key), 0))),
          v_scales(to_device(std::vector<std::uint8_t>(scale_bytes(layout.value), 0))) {
        view = PagedKVBatchLayerView{
            .k_pages = pages(k_pages, layout.key.data_dtype, layout.key.data_leading_extent),
            .v_pages = pages(v_pages, layout.value.data_dtype, layout.value.data_leading_extent),
            .k_scale_pages =
                pages(k_scales, layout.key.scale_dtype, layout.key.scale_leading_extent),
            .v_scale_pages =
                pages(v_scales, layout.value.scale_dtype, layout.value.scale_leading_extent),
            .block_tables = tables,
            .head_dim     = kHeadDim,
            .num_kv_heads = kKvHeads,
            .storage      = storage};
    }

    void download() {
        k_host       = from_device<std::uint8_t>(k_pages, data_bytes(layout.key));
        v_host       = from_device<std::uint8_t>(v_pages, data_bytes(layout.value));
        k_scale_host = from_device<std::uint8_t>(k_scales, scale_bytes(layout.key));
        v_scale_host = from_device<std::uint8_t>(v_scales, scale_bytes(layout.value));
    }

    static double decode(const PagedKVVectorLayout& vector, const std::vector<std::uint8_t>& codes,
                         const std::vector<std::uint8_t>& scales, std::size_t row, int d) {
        const auto half_at = [&](std::size_t i) {
            return half_value(static_cast<std::uint16_t>(scales[2 * i] | (scales[2 * i + 1] << 8)));
        };
        switch (vector.data_dtype) {
        case DType::FP8_E4M3FN: return e4m3_value(codes[row * kHeadDim + d]) * half_at(row);
        case DType::I8:
            return double(static_cast<std::int8_t>(codes[row * kHeadDim + d])) *
                   half_at(row * 4 + d / 64);
        default: {
            const std::uint8_t byte = codes[row * (kHeadDim / 2) + d / 2];
            return e2m1_value(d % 2 != 0 ? byte >> 4 : byte & 0xF) *
                   e4m3_value(scales[row * 16 + d / 16]);
        }
        }
    }

    [[nodiscard]] double key(std::size_t row, int d) const {
        return decode(layout.key, k_host, k_scale_host, row, d);
    }
    [[nodiscard]] double value(std::size_t row, int d) const {
        return decode(layout.value, v_host, v_scale_host, row, d);
    }
};

std::vector<double> offset_norm(const std::vector<double>& x, const std::vector<float>& weight) {
    double sum = 0.0;
    for (const double v : x) { sum += v * v; }
    const double inv = 1.0 / std::sqrt(sum / double(x.size()) + double(kEps));
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { out[i] = x[i] * inv * (1.0 + double(weight[i])); }
    return out;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures             = 0;
    const ops::QsaGeometry g = geometry();

    // Paging: lane b uses table row 1-b; physical pages are interleaved across lanes and reversed.
    std::vector<Lane> lanes(kLanes);
    std::vector<int> tables(static_cast<std::size_t>(kPagesPerLane) * kLanes, -1);
    for (int b = 0; b < kLanes; ++b) {
        lanes[b].table_row = kLanes - 1 - b;
        for (int p = 0; p < kPagesPerLane; ++p) {
            const int physical = (kPagesPerLane - 1 - p) * kLanes + b;
            lanes[b].pages.push_back(physical);
            tables[static_cast<std::size_t>(lanes[b].table_row) * kPagesPerLane + p] = physical;
        }
    }

    // ---- append: all cached tokens of both lanes in one call ----
    const std::size_t columns = static_cast<std::size_t>(kTokens) * kLanes;
    const auto k              = random_bf16(columns * kKvHeads * kHeadDim, 0x11U, -2.0F, 2.0F);
    const auto v              = random_bf16(columns * kKvHeads * kHeadDim, 0x12U, -2.0F, 2.0F);
    const auto index_keys     = random_bf16(columns * kIndexDim, 0x13U, -2.0F, 2.0F);
    std::vector<int> cache_positions(columns), rope(columns * 3);
    for (int b = 0; b < kLanes; ++b) {
        for (int t = 0; t < kTokens; ++t) {
            const std::size_t c = static_cast<std::size_t>(b) * kTokens + t;
            cache_positions[c]  = t;
            for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(t, a); }
        }
    }
    DeviceBuffer dk = to_device(bits(k)), dv = to_device(bits(v));
    DeviceBuffer dindex = to_device(bits(index_keys));
    DeviceBuffer dcache = to_device_i32(cache_positions), drope = to_device_i32(rope);
    DeviceBuffer dtables = to_device_i32(tables);
    std::vector<int> rows_host(kLanes);
    for (int b = 0; b < kLanes; ++b) { rows_host[b] = lanes[b].table_row; }
    DeviceBuffer drows = to_device_i32(rows_host);
    const std::size_t page_elements =
        static_cast<std::size_t>(kHeadDim) * kPage * kKvHeads * kPages;
    DeviceBuffer k_pages = to_device(std::vector<std::uint16_t>(page_elements, 0));
    DeviceBuffer v_pages = to_device(std::vector<std::uint16_t>(page_elements, 0));
    const std::size_t record_elements =
        static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage * kPages;
    DeviceBuffer records = to_device(std::vector<std::uint16_t>(record_elements, 0));

    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, kLanes});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKvHeads, kTokens, kLanes});
    Tensor tindex(dindex.p, DType::BF16, {kIndexDim, kTokens, kLanes});
    Tensor tcache(dcache.p, DType::I32, {kTokens, kLanes});
    Tensor trope(drope.p, DType::I32, {static_cast<std::int32_t>(columns), 3});
    Tensor ttables(dtables.p, DType::I32, {kPagesPerLane, kLanes});
    Tensor trows(drows.p, DType::I32, {kLanes});
    PagedKVBatchLayerView kv{
        .k_pages      = Tensor(k_pages.p, DType::BF16, {kHeadDim, kPage, kKvHeads, kPages}),
        .v_pages      = Tensor(v_pages.p, DType::FP16, {kHeadDim, kPage, kKvHeads, kPages}),
        .block_tables = ttables,
        .head_dim     = kHeadDim,
        .num_kv_heads = kKvHeads,
        .storage      = KvCacheStorage::BFloat16};
    ops::QsaIndexPlane plane{
        .pages = Tensor(records.p, DType::BF16, {ops::kQsaIndexRecordWords, kPage, 1, kPages}),
        .block_tables = ttables};
    ops::qsa_append(tk, tv, tindex, trope, tcache, nullptr, trows, kv, plane, nullptr);
    cuda_synchronize();

    const auto k_host = from_device<std::uint16_t>(k_pages, page_elements);
    const auto v_host = from_device<std::uint16_t>(v_pages, page_elements);
    const auto r_host = from_device<std::uint16_t>(records, record_elements);
    const auto key_at = [&](int lane, int token, int head, int d) {
        const std::size_t row =
            (static_cast<std::size_t>(lanes[lane].pages[token / kPage]) * kKvHeads + head) * kPage +
            token % kPage;
        return row * kHeadDim + d;
    };
    const auto record_at = [&](int lane, int token) {
        return (static_cast<std::size_t>(lanes[lane].pages[token / kPage]) * kPage +
                token % kPage) *
               ops::kQsaIndexRecordWords;
    };
    {
        int mismatches = 0;
        const auto kb = bits(k), vb = bits(v), ib = bits(index_keys);
        for (int b = 0; b < kLanes && mismatches == 0; ++b) {
            for (int t = 0; t < kTokens && mismatches == 0; ++t) {
                const std::size_t c = static_cast<std::size_t>(b) * kTokens + t;
                for (int h = 0; h < kKvHeads; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t source = (c * kKvHeads + h) * kHeadDim + d;
                        const __half fp16        = __float2half_rn(bf16_to_f32(vb[source]));
                        std::uint16_t fp16_bits;
                        std::memcpy(&fp16_bits, &fp16, 2);
                        mismatches += k_host[key_at(b, t, h, d)] != kb[source];
                        mismatches += v_host[key_at(b, t, h, d)] != fp16_bits;
                    }
                }
                const std::size_t base = record_at(b, t);
                for (int d = 0; d < kIndexDim; ++d) {
                    mismatches += r_host[base + d] != ib[c * kIndexDim + d];
                }
                for (int a = 0; a < 3; ++a) {
                    const std::uint32_t p = static_cast<std::uint32_t>(axis_position(t, a));
                    mismatches += r_host[base + kIndexDim + 2 * a] != (p & 0xFFFFU);
                    mismatches += r_host[base + kIndexDim + 2 * a + 1] != (p >> 16);
                }
            }
        }
        if (mismatches != 0) {
            std::cerr << "qsa_append: stored K/V/index records differ from their inputs\n";
            ++failures;
        }
    }

    // The quantized profiles over the same tokens, stored with the dense cache's codecs: K is
    // Hadamard-rotated in every profile, V where it is NVFP4. Each stored row, decoded with its
    // stored scales, must match the (rotated) input within its codec's error: a relative RMS bound
    // that a wrong rotation, group or nibble order exceeds many times over.
    std::deque<QuantizedCache> quantized;
    for (const auto& [storage, name, k_rms, v_rms] :
         {std::tuple{KvCacheStorage::Fp8E4M3Row256, "FP8", 0.05, 0.05},
          std::tuple{KvCacheStorage::Int8Group64, "INT8", 0.02, 0.02},
          std::tuple{KvCacheStorage::Nvfp4Group16, "NVFP4", 0.2, 0.2},
          std::tuple{KvCacheStorage::Fp8KeyNvfp4Value, "K8V4", 0.05, 0.2}}) {
        QuantizedCache& cache = quantized.emplace_back(storage, name, ttables);
        ops::qsa_append(tk, tv, tindex, trope, tcache, nullptr, trows, cache.view, plane, nullptr);
        cuda_synchronize();
        cache.download();
        double k_error = 0.0, k_norm = 0.0, v_error = 0.0, v_norm = 0.0;
        for (int b = 0; b < kLanes; ++b) {
            for (int t = 0; t < kTokens; t += 7) {
                const std::size_t c = static_cast<std::size_t>(b) * kTokens + t;
                for (int h = 0; h < kKvHeads; ++h) {
                    const std::size_t source = (c * kKvHeads + h) * kHeadDim;
                    std::vector<double> kr(k.begin() + source, k.begin() + source + kHeadDim);
                    std::vector<double> vr(v.begin() + source, v.begin() + source + kHeadDim);
                    kr = hadamard256(std::move(kr));
                    if (cache.v_rotated) { vr = hadamard256(std::move(vr)); }
                    const std::size_t row = key_at(b, t, h, 0) / kHeadDim;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const double ke = cache.key(row, d) - kr[d];
                        const double ve = cache.value(row, d) - vr[d];
                        k_error += ke * ke;
                        k_norm += kr[d] * kr[d];
                        v_error += ve * ve;
                        v_norm += vr[d] * vr[d];
                    }
                }
            }
        }
        const double k_rel = std::sqrt(k_error / k_norm), v_rel = std::sqrt(v_error / v_norm);
        if (!(k_rel <= k_rms) || !(v_rel <= v_rms)) {
            std::cerr << "qsa_append " << name << ": decoded rows differ from their inputs (K rms "
                      << k_rel << " > " << k_rms << " or V rms " << v_rel << " > " << v_rms
                      << ")\n";
            ++failures;
        }
    }

    // ---- select: lane 0 is sparse (n > 2051), lane 1 dense; lane 1's last column is invalid ----
    constexpr std::int32_t W            = 3;
    const int query_position[kLanes][W] = {{2195, 2197, 2199}, {900, 901, 902}};
    const std::vector<int> valid_host   = {W, W - 1};
    const auto index_query              = random_bf16(
        static_cast<std::size_t>(kIndexHeads) * kIndexDim * W * kLanes, 0x21U, -2.0F, 2.0F);
    const auto query_norm = random_bf16(kIndexDim, 0x22U, -0.3F, 0.3F);
    const auto key_norm   = random_bf16(kIndexDim, 0x23U, -0.3F, 0.3F);
    std::vector<int> qcache(W * kLanes), qrope(3 * W * kLanes);
    for (int b = 0; b < kLanes; ++b) {
        for (int w = 0; w < W; ++w) {
            const int c = b * W + w;
            qcache[c]   = query_position[b][w];
            for (int a = 0; a < 3; ++a) { qrope[a * W * kLanes + c] = axis_position(qcache[c], a); }
        }
    }
    DeviceBuffer diq = to_device(bits(index_query));
    DeviceBuffer dqn = to_device(bits(query_norm)), dkn = to_device(bits(key_norm));
    DeviceBuffer dqc = to_device_i32(qcache), dqr = to_device_i32(qrope);
    DeviceBuffer dvalid = to_device_i32(valid_host);
    GuardedDeviceBuffer selected(static_cast<std::size_t>(g.max_selected()) * W * kLanes * 4);
    GuardedDeviceBuffer counts(static_cast<std::size_t>(W) * kLanes * 4);
    Tensor tiq(diq.p, DType::BF16, {kIndexHeads * kIndexDim, W, kLanes});
    Tensor tqn(dqn.p, DType::BF16, {kIndexDim});
    Tensor tkn(dkn.p, DType::BF16, {kIndexDim});
    Tensor tqc(dqc.p, DType::I32, {W, kLanes});
    Tensor tqr(dqr.p, DType::I32, {W * kLanes, 3});
    Tensor tvalid(dvalid.p, DType::I32, {kLanes});
    Tensor tsel(selected.data(), DType::I32, {g.max_selected(), W, kLanes});
    Tensor tcnt(counts.data(), DType::I32, {W, kLanes});
    {
        WorkspaceArena workspace(ops::qsa_select_workspace_bytes(g, kTokens, W * kLanes));
        ops::qsa_select(tiq, tqr, tqc, &tvalid, trows, tqn, tkn, plane, g, kTokens, workspace, tsel,
                        tcnt, nullptr);
        cuda_synchronize();
    }
    failures += selected.verify_guards("qsa_select selected") + counts.verify_guards("qsa_select");
    const auto sel_host =
        from_device<int>(selected.data(), static_cast<std::size_t>(g.max_selected()) * W * kLanes);
    const auto cnt_host = from_device<int>(counts.data(), static_cast<std::size_t>(W) * kLanes);

    for (int b = 0; b < kLanes; ++b) {
        for (int w = 0; w < W; ++w) {
            const int c = b * W + w;
            const std::string label =
                "qsa_select lane=" + std::to_string(b) + " column=" + std::to_string(w);
            if (w >= valid_host[b]) {
                if (cnt_host[c] != 0) {
                    std::cerr << label << ": invalid column has a selection\n";
                    ++failures;
                }
                continue;
            }
            const int n      = qcache[c] + 1;
            const int blocks = n / kRatio;
            std::vector<int> expected;
            std::vector<double> scores(blocks, 0.0);
            if (blocks <= g.block_topk()) {
                expected.resize(n);
                std::iota(expected.begin(), expected.end(), 0);
            } else {
                std::int32_t qpos[3];
                for (int a = 0; a < 3; ++a) { qpos[a] = axis_position(qcache[c], a); }
                std::vector<std::vector<double>> queries(kIndexHeads);
                for (int h = 0; h < kIndexHeads; ++h) {
                    std::vector<double> raw(kIndexDim);
                    for (int d = 0; d < kIndexDim; ++d) {
                        raw[d] = index_query[(static_cast<std::size_t>(c) * kIndexHeads + h) *
                                                 kIndexDim +
                                             d];
                    }
                    queries[h] = rotate(offset_norm(raw, query_norm), qpos);
                }
                for (int blk = 0; blk < blocks; ++blk) {
                    std::vector<double> mean(kIndexDim, 0.0);
                    for (int t = 0; t < kRatio; ++t) {
                        const std::size_t source =
                            (static_cast<std::size_t>(b) * kTokens + blk * kRatio + t) * kIndexDim;
                        for (int d = 0; d < kIndexDim; ++d) { mean[d] += index_keys[source + d]; }
                    }
                    // Upstream pools raw keys in the activation dtype (BF16).
                    for (double& m : mean) {
                        m = bf16_to_f32(f32_to_bf16(static_cast<float>(m / kRatio)));
                    }
                    std::int32_t kpos[3];
                    for (int a = 0; a < 3; ++a) { kpos[a] = axis_position(blk * kRatio, a); }
                    const auto key = rotate(offset_norm(mean, key_norm), kpos);
                    double score   = 0.0;
                    for (const auto& q : queries) {
                        double dot = 0.0;
                        for (int d = 0; d < kIndexDim; ++d) { dot += q[d] * key[d]; }
                        score += std::max(dot, 0.0);
                    }
                    scores[blk] = score / std::sqrt(double(kIndexDim));
                }
                std::vector<int> order(blocks);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int x, int y) { return scores[x] > scores[y]; });
                std::vector<int> kept(order.begin(), order.begin() + g.block_topk());
                std::sort(kept.begin(), kept.end());
                for (const int blk : kept) {
                    for (int t = 0; t < kRatio; ++t) { expected.push_back(blk * kRatio + t); }
                }
                for (int t = blocks * kRatio; t < n; ++t) { expected.push_back(t); }
            }
            const std::vector<int> got(
                sel_host.begin() + static_cast<std::ptrdiff_t>(c) * g.max_selected(),
                sel_host.begin() + static_cast<std::ptrdiff_t>(c) * g.max_selected() + cnt_host[c]);
            if (static_cast<int>(got.size()) != static_cast<int>(expected.size()) ||
                !std::is_sorted(got.begin(), got.end())) {
                std::cerr << label << ": count " << got.size() << " expected " << expected.size()
                          << " (or unsorted)\n";
                ++failures;
                continue;
            }
            if (got != expected) {
                // Every differing block must be a near tie at the selection cut.
                std::vector<double> sorted = scores;
                std::sort(sorted.rbegin(), sorted.rend());
                const double cut   = sorted[g.block_topk() - 1];
                const double scale = sorted.front();
                std::set<int> a(got.begin(), got.end()), e(expected.begin(), expected.end());
                bool near = true;
                for (int t = 0; t < n; ++t) {
                    if (a.count(t) != e.count(t)) {
                        const int blk = t / kRatio;
                        near =
                            near && blk < blocks && std::abs(scores[blk] - cut) <= kNearTie * scale;
                    }
                }
                if (!near) {
                    std::cerr << label << ": selection differs beyond near ties\n";
                    ++failures;
                }
            }
        }
    }

    // ---- attention: FP64 softmax over each column's selection ----
    // Narrow calls split every selection across CTAs; a call of 256+ (column, KV head) pairs
    // runs one CTA per pair. Both routes are compared with the same oracle.
    // A quantized profile reads each stored code times its stored scale, rotates the query like the
    // stored keys, and rotates the output of a rotated V back.
    const auto attend = [&](const std::string& label, std::int32_t width,
                            const std::vector<int>& sel, const std::vector<int>& cnt,
                            std::uint32_t seed, const QuantizedCache* cache) {
        const auto key = [&](int b, int token, int kvh, int d) {
            const std::size_t at = key_at(b, token, kvh, d);
            return cache != nullptr ? cache->key(at / kHeadDim, d) : double(bf16_to_f32(k_host[at]));
        };
        const auto value = [&](int b, int token, int kvh, int d) {
            const std::size_t at = key_at(b, token, kvh, d);
            return cache != nullptr ? cache->value(at / kHeadDim, d) : half_value(v_host[at]);
        };
        const auto q = random_bf16(
            static_cast<std::size_t>(kHeadDim) * kQueryHeads * width * kLanes, seed, -1.0F, 1.0F);
        DeviceBuffer dq = to_device(bits(q)), dsel = to_device_i32(sel), dcnt = to_device_i32(cnt);
        GuardedDeviceBuffer out(q.size() * 2);
        Tensor tq(dq.p, DType::BF16, {kHeadDim, kQueryHeads, width, kLanes});
        Tensor ts(dsel.p, DType::I32, {g.max_selected(), width, kLanes});
        Tensor tc(dcnt.p, DType::I32, {width, kLanes});
        Tensor tout(out.data(), DType::BF16, {kHeadDim, kQueryHeads, width, kLanes});
        const float scale = 1.0F / 16.0F;
        WorkspaceArena workspace(ops::qsa_attention_workspace_bytes(g, width * kLanes));
        ops::qsa_attention(tq, ts, tc, trows, cache != nullptr ? cache->view : kv, scale, workspace,
                           tout, nullptr);
        cuda_synchronize();
        std::vector<double> reference(q.size(), 0.0);
        for (int b = 0; b < kLanes; ++b) {
            for (int w = 0; w < width; ++w) {
                const int c     = b * width + w;
                const int count = cnt[c];
                const int* list = sel.data() + static_cast<std::ptrdiff_t>(c) * g.max_selected();
                for (int h = 0; h < kQueryHeads; ++h) {
                    const int kvh = h / (kQueryHeads / kKvHeads);
                    const float* qrow =
                        q.data() + (static_cast<std::size_t>(c) * kQueryHeads + h) * kHeadDim;
                    std::vector<double> query(qrow, qrow + kHeadDim);
                    if (cache != nullptr) { query = hadamard256(std::move(query)); }
                    std::vector<double> logits(count);
                    double max_logit = -1e300;
                    for (int j = 0; j < count; ++j) {
                        double dot = 0.0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            dot += query[d] * key(b, list[j], kvh, d);
                        }
                        logits[j] = dot * scale;
                        max_logit = std::max(max_logit, logits[j]);
                    }
                    double denominator = 0.0;
                    for (double& l : logits) {
                        l = std::exp(l - max_logit);
                        denominator += l;
                    }
                    double* o = reference.data() +
                                (static_cast<std::size_t>(c) * kQueryHeads + h) * kHeadDim;
                    std::vector<double> row(kHeadDim, 0.0);
                    for (int j = 0; j < count; ++j) {
                        for (int d = 0; d < kHeadDim; ++d) {
                            row[d] += logits[j] / denominator * value(b, list[j], kvh, d);
                        }
                    }
                    if (cache != nullptr && cache->v_rotated) { row = hadamard256(std::move(row)); }
                    std::copy(row.begin(), row.end(), o);
                }
            }
        }
        return verify_reduction(label, from_device_bf16(out.data(), q.size()), reference,
                                kAttentionOutput) +
               out.verify_guards(label);
    };
    // A sorted, duplicate-free selection of `count` cached tokens.
    const auto pick = [&](int count, std::uint32_t seed) {
        std::vector<int> all(kTokens);
        std::iota(all.begin(), all.end(), 0);
        std::uint32_t state = seed;
        for (int i = kTokens - 1; i > 0; --i) {
            state = state * 1664525U + 1013904223U;
            std::swap(all[i], all[state % static_cast<std::uint32_t>(i + 1)]);
        }
        std::vector<int> out(all.begin(), all.begin() + count);
        std::sort(out.begin(), out.end());
        return out;
    };
    const auto synthetic = [&](std::int32_t width, const std::vector<int>& counts) {
        std::vector<int> sel(static_cast<std::size_t>(g.max_selected()) * width * kLanes, 0);
        for (std::size_t c = 0; c < counts.size(); ++c) {
            const auto list = pick(counts[c], 0x40U + static_cast<std::uint32_t>(c));
            std::copy(list.begin(), list.end(),
                      sel.begin() + static_cast<std::ptrdiff_t>(c) * g.max_selected());
        }
        return sel;
    };

    std::vector<const QuantizedCache*> caches = {nullptr};
    for (const QuantizedCache& cache : quantized) { caches.push_back(&cache); }
    for (const QuantizedCache* cache : caches) {
        const std::string profile =
            std::string("qsa_attention ") + (cache != nullptr ? cache->name + " " : "");
        // The device selection: 6 columns split every selection.
        failures += attend(profile + "select", W, sel_host, cnt_host, 0x31U, cache);
        // Decode width: a full selection beside a single key, so later splits of the short column
        // are empty.
        {
            const std::vector<int> counts = {g.max_selected(), 1};
            failures += attend(profile + "decode", 1, synthetic(1, counts), counts, 0x32U, cache);
        }
        // 128 columns x 2 KV heads fill the GPU: one CTA per pair, counts from 0 to the maximum.
        {
            constexpr std::int32_t kWide = 64;
            std::vector<int> counts(static_cast<std::size_t>(kWide) * kLanes);
            for (std::size_t c = 0; c < counts.size(); ++c) {
                counts[c] =
                    static_cast<int>((c * 977U) % static_cast<std::size_t>(g.max_selected() + 1));
            }
            counts[1] = 0;
            counts[2] = g.max_selected();
            failures +=
                attend(profile + "wide", kWide, synthetic(kWide, counts), counts, 0x33U, cache);
        }
    }

    // ---- Host arm: odd logical pages of both lanes read in place from Host records ----
    // Records use the production HostKVPageLayout on 2 MiB pages; the abandoned Device copies are
    // 0xFF-poisoned (NaN codes/scales), so output must be bit-identical to the resident call.
    const auto run_bits = [&](const PagedKVBatchLayerView& view, const Tensor& rows,
                              std::int32_t width, const std::vector<int>& sel,
                              const std::vector<int>& cnt, std::uint32_t seed) {
        const auto q = random_bf16(
            static_cast<std::size_t>(kHeadDim) * kQueryHeads * width * kLanes, seed, -1.0F, 1.0F);
        DeviceBuffer dq = to_device(bits(q)), dsel = to_device_i32(sel), dcnt = to_device_i32(cnt);
        GuardedDeviceBuffer out(q.size() * 2);
        Tensor tq(dq.p, DType::BF16, {kHeadDim, kQueryHeads, width, kLanes});
        Tensor ts(dsel.p, DType::I32, {g.max_selected(), width, kLanes});
        Tensor tc(dcnt.p, DType::I32, {width, kLanes});
        Tensor tout(out.data(), DType::BF16, {kHeadDim, kQueryHeads, width, kLanes});
        WorkspaceArena workspace(ops::qsa_attention_workspace_bytes(g, width * kLanes));
        ops::qsa_attention(tq, ts, tc, rows, view, 1.0F / 16.0F, workspace, tout, nullptr);
        cuda_synchronize();
        return from_device<std::uint16_t>(out.data(), q.size());
    };
    for (const QuantizedCache* cache : caches) {
        const PagedKVBatchLayerView& resident = cache != nullptr ? cache->view : kv;
        const std::string label = std::string("qsa_attention host-arm ") +
                                  (cache != nullptr ? cache->name : std::string("BF16"));
        std::vector<const Tensor*> planes{&resident.k_pages, &resident.v_pages};
        if (resident.k_scale_pages.data != nullptr) planes.push_back(&resident.k_scale_pages);
        if (resident.v_scale_pages.data != nullptr) planes.push_back(&resident.v_scale_pages);
        KVPageGeometry page_geometry;
        for (const Tensor* plane : planes) {
            page_geometry.planes.push_back({plane->dtype, plane->ne[0], kKvHeads, 256});
        }
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(page_geometry);

        std::vector<DeviceBuffer> copies;
        for (const Tensor* plane : planes) {
            const std::size_t bytes = static_cast<std::size_t>(plane->ne[0]) * kPage * kKvHeads *
                                      kPages * dtype_size(plane->dtype);
            copies.push_back(to_device(std::vector<std::uint8_t>(bytes, 0)));
            cuda_check(cudaMemcpy(copies.back().p, plane->data, bytes, cudaMemcpyDeviceToDevice),
                       "copy KV plane");
        }
        std::vector<int> streamed_tables = tables;
        std::vector<std::size_t> streamed;
        for (std::size_t entry = 0; entry < tables.size(); ++entry) {
            if ((entry % kPagesPerLane) % 2 == 1) streamed.push_back(entry);
        }
        PinnedHostBuffer records_host((streamed.size() + 1) * host_layout.page_stride,
                                      PinnedHostPages::Huge);
        auto* records_base = static_cast<std::byte*>(records_host.data());
        for (std::size_t index = 0; index < streamed.size(); ++index) {
            const int physical = tables[streamed[index]];
            // Record 0 is left unused so the words exercise a nonzero record offset.
            const std::size_t record = (index + 1) * host_layout.page_stride;
            for (std::size_t plane = 0; plane < planes.size(); ++plane) {
                const std::size_t bytes = host_layout.planes[plane].page_payload_bytes;
                auto* page = static_cast<std::byte*>(copies[plane].p) +
                             static_cast<std::size_t>(physical) * bytes;
                cuda_check(cudaMemcpy(records_base + record + host_layout.planes[plane].offset,
                                      page, bytes, cudaMemcpyDeviceToHost),
                           "stream KV page");
                cuda_check(cudaMemset(page, 0xff, bytes), "poison KV page");
            }
            streamed_tables[streamed[index]] =
                KVPageRef(HostPageUnit{static_cast<std::uint32_t>(record / kHostKVPageUnitBytes)})
                    .word();
        }
        DeviceBuffer dstreamed = to_device_i32(streamed_tables);
        PagedKVBatchLayerView host_view = resident;
        host_view.block_tables          = Tensor(dstreamed.p, DType::I32, {kPagesPerLane, kLanes});
        const auto base                 = [&](std::size_t plane) -> const std::byte* {
            return records_base + host_layout.planes[plane].offset;
        };
        host_view.k_pages.data = copies[0].p;
        host_view.v_pages.data = copies[1].p;
        host_view.host         = PagedKVHostPlanes{.k = base(0), .v = base(1)};
        std::size_t next       = 2;
        if (resident.k_scale_pages.data != nullptr) {
            host_view.k_scale_pages.data = copies[next].p;
            host_view.host.k_scale       = base(next++);
        }
        if (resident.v_scale_pages.data != nullptr) {
            host_view.v_scale_pages.data = copies[next].p;
            host_view.host.v_scale       = base(next++);
        }

        const std::vector<int> decode_counts = {g.max_selected(), 37};
        const auto decode_sel                = synthetic(1, decode_counts);
        failures += verify_exact((label + " select").c_str(),
                                 run_bits(host_view, trows, W, sel_host, cnt_host, 0x51U),
                                 run_bits(resident, trows, W, sel_host, cnt_host, 0x51U));
        failures += verify_exact((label + " decode").c_str(),
                                 run_bits(host_view, trows, 1, decode_sel, decode_counts, 0x52U),
                                 run_bits(resident, trows, 1, decode_sel, decode_counts, 0x52U));
    }

    // ---- Host arm of qsa_select: odd logical pages' index records read from Host records ----
    // The abandoned Device records are 0xFF-poisoned (NaN keys), so selections must be identical
    // to the resident call at the verify width and at decode width.
    {
        const std::size_t page_bytes = static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage *
                                       sizeof(std::uint16_t);
        DeviceBuffer poisoned = to_device(std::vector<std::uint8_t>(page_bytes * kPages, 0));
        cuda_check(cudaMemcpy(poisoned.p, records.p, page_bytes * kPages, cudaMemcpyDeviceToDevice),
                   "copy index records");
        std::vector<int> streamed_tables = tables;
        // Host records are spaced two units past the payload so the words exercise a stride that
        // differs from the Device page group.
        const std::size_t record_stride =
            (page_bytes / kHostKVPageUnitBytes + 2) * kHostKVPageUnitBytes;
        PinnedHostBuffer host_records((tables.size() + 1) * record_stride, PinnedHostPages::Huge);
        auto* host_base = static_cast<std::byte*>(host_records.data());
        std::size_t next = 1; // record 0 unused: words carry a nonzero offset
        for (std::size_t entry = 0; entry < tables.size(); ++entry) {
            if ((entry % kPagesPerLane) % 2 == 0) continue;
            auto* page = static_cast<std::byte*>(poisoned.p) +
                         static_cast<std::size_t>(tables[entry]) * page_bytes;
            const std::size_t record = next++ * record_stride;
            cuda_check(cudaMemcpy(host_base + record, page, page_bytes, cudaMemcpyDeviceToHost),
                       "stream index page");
            cuda_check(cudaMemset(page, 0xff, page_bytes), "poison index page");
            streamed_tables[entry] =
                KVPageRef(HostPageUnit{static_cast<std::uint32_t>(record / kHostKVPageUnitBytes)})
                    .word();
        }
        DeviceBuffer dstreamed = to_device_i32(streamed_tables);
        const ops::QsaIndexPlane host_plane{
            .pages = Tensor(poisoned.p, DType::BF16, {ops::kQsaIndexRecordWords, kPage, 1, kPages}),
            .block_tables = Tensor(dstreamed.p, DType::I32, {kPagesPerLane, kLanes}),
            .host         = host_base,
        };
        const auto select = [&](const ops::QsaIndexPlane& index, std::int32_t width) {
            std::vector<int> rope(3 * width * kLanes);
            for (int c = 0; c < width * kLanes; ++c) {
                const int position = qcache[(c / width) * W + W - width + c % width];
                for (int a = 0; a < 3; ++a) { rope[a * width * kLanes + c] = axis_position(position, a); }
            }
            DeviceBuffer drope = to_device_i32(rope);
            DeviceBuffer dqc_w = to_device_i32([&] {
                std::vector<int> positions(width * kLanes);
                for (int c = 0; c < width * kLanes; ++c) {
                    positions[c] = qcache[(c / width) * W + W - width + c % width];
                }
                return positions;
            }());
            DeviceBuffer diq_w = to_device(bits([&] {
                std::vector<float> q;
                const std::size_t column = static_cast<std::size_t>(kIndexHeads) * kIndexDim;
                for (int b = 0; b < kLanes; ++b) {
                    for (int w = W - width; w < W; ++w) {
                        const auto first = index_query.begin() +
                                           static_cast<std::ptrdiff_t>((b * W + w) * column);
                        q.insert(q.end(), first, first + static_cast<std::ptrdiff_t>(column));
                    }
                }
                return q;
            }()));
            const std::vector<int> valid = {width, width};
            DeviceBuffer dvalid_w = to_device_i32(valid);
            DeviceBuffer out_sel  = to_device_i32(std::vector<int>(g.max_selected() * width * kLanes, 0));
            DeviceBuffer out_cnt  = to_device_i32(std::vector<int>(width * kLanes, 0));
            WorkspaceArena workspace(ops::qsa_select_workspace_bytes(g, kTokens, width * kLanes));
            Tensor sel(out_sel.p, DType::I32, {g.max_selected(), width, kLanes});
            Tensor cnt(out_cnt.p, DType::I32, {width, kLanes});
            Tensor valid_t(dvalid_w.p, DType::I32, {kLanes});
            ops::qsa_select(Tensor(diq_w.p, DType::BF16, {kIndexHeads * kIndexDim, width, kLanes}),
                            Tensor(drope.p, DType::I32, {width * kLanes, 3}),
                            Tensor(dqc_w.p, DType::I32, {width, kLanes}), &valid_t, trows, tqn,
                            tkn, index, g, kTokens, workspace, sel, cnt, nullptr);
            cuda_synchronize();
            auto result = from_device<int>(out_sel.p, g.max_selected() * width * kLanes);
            const auto counts_out = from_device<int>(out_cnt.p, width * kLanes);
            result.insert(result.end(), counts_out.begin(), counts_out.end());
            return result;
        };
        for (const std::int32_t width : {W, 1}) {
            const auto resident = select(plane, width);
            const auto streamed = select(host_plane, width);
            if (resident != streamed) {
                std::cerr << "qsa_select host-arm width " << width
                          << ": selections differ from the resident call\n";
                ++failures;
            }
        }
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_attention correctness\n";
    return failures == 0 ? 0 : 1;
}
