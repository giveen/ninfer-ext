// Qwen Sparse Attention Ops at the Qwen3.8-Flash-Next geometry (24/2 heads of 256, indexer 4x128,
// budget 2048 in blocks of 4) with the BF16 KV profile over permuted physical pages.
//
// qsa_append is exact. qsa_select is compared with an FP64 oracle of the block scores; a block may
// differ from the oracle only when its oracle score lies within a near-tie allowance of the
// selection cut. qsa_attention is compared with FP64 softmax attention over the selection the
// device produced, reading K/V at their storage boundary.
#include "ninfer/ops/sparse_attention.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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
    const auto attend = [&](const std::string& label, std::int32_t width,
                            const std::vector<int>& sel, const std::vector<int>& cnt,
                            std::uint32_t seed) {
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
        ops::qsa_attention(tq, ts, tc, trows, kv, scale, workspace, tout, nullptr);
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
                    std::vector<double> logits(count);
                    double max_logit = -1e300;
                    for (int j = 0; j < count; ++j) {
                        double dot = 0.0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            dot +=
                                double(qrow[d]) * bf16_to_f32(k_host[key_at(b, list[j], kvh, d)]);
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
                    for (int j = 0; j < count; ++j) {
                        for (int d = 0; d < kHeadDim; ++d) {
                            __half hv;
                            const std::uint16_t word = v_host[key_at(b, list[j], kvh, d)];
                            std::memcpy(&hv, &word, 2);
                            o[d] += logits[j] / denominator * double(__half2float(hv));
                        }
                    }
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

    // The device selection: 6 columns split every selection.
    failures += attend("qsa_attention select", W, sel_host, cnt_host, 0x31U);
    // Decode width: a full selection beside a single key, so later splits of the short column
    // are empty.
    {
        const std::vector<int> counts = {g.max_selected(), 1};
        failures += attend("qsa_attention decode", 1, synthetic(1, counts), counts, 0x32U);
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
        failures += attend("qsa_attention wide", kWide, synthetic(kWide, counts), counts, 0x33U);
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_attention correctness\n";
    return failures == 0 ? 0 : 1;
}
