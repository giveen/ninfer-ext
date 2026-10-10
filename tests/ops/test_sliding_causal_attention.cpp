#include "ninfer/ops/sliding_causal_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One storage rounding of a weighted average of represented BF16 values, with an absolute floor so a
// value that cancels to nothing is not judged by a relative rule.
constexpr PointwiseCriterion sliding_causal_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.5e-3};
}

struct Case {
    const char* label;
    std::int32_t head_dim, query_heads, kv_heads;
    std::int32_t key_tokens, query_tokens, batch;
    std::int32_t first_key_position, first_query_position, key_step;
    std::uint32_t window;
    // Rotates which slot holds which position, the way the sliding ring stores them.
    std::int32_t ring = 0;
    // Queries [block_begin, block_end) form an image block: each sees keys up to the block's last
    // position. Empty when block_end == 0.
    std::int32_t block_begin = 0, block_end = 0;
};

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::size_t count(const Case& c, std::int32_t heads, std::int32_t tokens) {
    return static_cast<std::size_t>(c.head_dim) * heads * tokens * c.batch;
}

std::size_t index(const Case& c, std::int32_t d, std::int32_t head, std::int32_t token,
                  std::int32_t batch) {
    return (static_cast<std::size_t>(head) +
            static_cast<std::size_t>(c.query_heads) *
                (token + static_cast<std::size_t>(c.query_tokens) * batch)) *
               c.head_dim +
           d;
}

// Key/value tensors have their own head count and token count, so they index differently.
std::size_t kv_index(const Case& c, std::int32_t d, std::int32_t kv_head, std::int32_t key,
                     std::int32_t batch) {
    return (static_cast<std::size_t>(kv_head) +
            static_cast<std::size_t>(c.kv_heads) *
                (key + static_cast<std::size_t>(c.key_tokens) * batch)) *
               c.head_dim +
           d;
}

int run_case(const Case& c, std::uint32_t seed) {
    const std::size_t q_count = count(c, c.query_heads, c.query_tokens);
    const std::size_t k_count = count(c, c.kv_heads, c.key_tokens);
    std::vector<float> q(q_count), k(k_count), v(k_count);
    fill_uniform(q, seed, -0.35f, 0.35f);
    fill_uniform(k, seed + 1U, -0.4f, 0.4f);
    fill_uniform(v, seed + 2U, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<std::int32_t> position_q(static_cast<std::size_t>(c.query_tokens) * c.batch);
    std::vector<std::int32_t> position_k(static_cast<std::size_t>(c.key_tokens) * c.batch);
    std::vector<std::int32_t> position_high(position_q.size());
    const bool blocked = c.block_end > 0;
    for (std::int32_t b = 0; b < c.batch; ++b) {
        for (std::int32_t t = 0; t < c.query_tokens; ++t) {
            position_q[t + c.query_tokens * b] = c.first_query_position + t;
            const bool in_block                = t >= c.block_begin && t < c.block_end;
            position_high[t + c.query_tokens * b] =
                c.first_query_position + (blocked && in_block ? c.block_end - 1 : t);
        }
        for (std::int32_t s = 0; s < c.key_tokens; ++s) {
            const std::int32_t slot = (s + c.ring) % c.key_tokens;
            position_k[s + c.key_tokens * b] = c.first_key_position + slot * c.key_step + b;
        }
    }

    const ops::AttentionHeadGeometry geometry{c.head_dim, c.query_heads, c.kv_heads};
    std::vector<double> expected(q_count, 0.0);
    for (std::int32_t b = 0; b < c.batch; ++b) {
        naive_dense_softmax_attention(
            geometry, c.query_tokens, c.key_tokens, 1.0,
            [&](int d, int head, int token) {
                return static_cast<double>(q[index(c, d, head, token, b)]);
            },
            [&](int d, int kv_head, int key) {
                return static_cast<double>(k[kv_index(c, d, kv_head, key, b)]);
            },
            [&](int d, int kv_head, int key) {
                return static_cast<double>(v[kv_index(c, d, kv_head, key, b)]);
            },
            [&](int query, int key) {
                const std::int32_t at = query + c.query_tokens * b;
                const std::int32_t pk = position_k[key + c.key_tokens * b];
                return pk <= position_high[at] &&
                       position_q[at] - pk < static_cast<std::int32_t>(c.window);
            },
            [&](int d, int head, int token, double value) {
                expected[index(c, d, head, token, b)] = value;
            });
    }

    // A key no query sees stands for an unwritten cache slot, which may hold any bits: NaN there must
    // not reach the output.
    for (std::int32_t b = 0; b < c.batch; ++b) {
        for (std::int32_t key = 0; key < c.key_tokens; ++key) {
            bool seen = false;
            for (std::int32_t t = 0; t < c.query_tokens; ++t) {
                const std::int32_t at = t + c.query_tokens * b;
                const std::int32_t pk = position_k[key + c.key_tokens * b];
                seen = seen || (pk <= position_high[at] &&
                                position_q[at] - pk < static_cast<std::int32_t>(c.window));
            }
            if (seen) continue;
            for (std::int32_t h = 0; h < c.kv_heads; ++h) {
                for (std::int32_t d = 0; d < c.head_dim; ++d) {
                    k[kv_index(c, d, h, key, b)] = std::numeric_limits<float>::quiet_NaN();
                    v[kv_index(c, d, h, key, b)] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }

    // Encode to the BF16 bits the tensors declare; uploading the raw floats would hand the kernel
    // 4-byte values where it reads 2-byte ones.
    const auto q_bits = encode_bf16(q);
    const auto k_bits = encode_bf16(k);
    const auto v_bits = encode_bf16(v);

    GuardedDeviceBuffer device_q(q_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_k(k_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_v(k_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pq(position_q.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_pk(position_k.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_ph(position_high.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(q_count * sizeof(std::uint16_t));
    device_q.copy_from_host(q_bits.data(), device_q.bytes());
    device_k.copy_from_host(k_bits.data(), device_k.bytes());
    device_v.copy_from_host(v_bits.data(), device_v.bytes());
    device_pq.copy_from_host(position_q.data(), device_pq.bytes());
    device_pk.copy_from_host(position_k.data(), device_pk.bytes());
    device_ph.copy_from_host(position_high.data(), device_ph.bytes());
    device_out.fill(0x7d);

    Tensor q_tensor(device_q.data(), DType::BF16, {c.head_dim, c.query_heads, c.query_tokens, c.batch});
    Tensor k_tensor(device_k.data(), DType::BF16, {c.head_dim, c.kv_heads, c.key_tokens, c.batch});
    Tensor v_tensor(device_v.data(), DType::BF16, {c.head_dim, c.kv_heads, c.key_tokens, c.batch});
    Tensor pq_tensor(device_pq.data(), DType::I32, {c.query_tokens, c.batch});
    Tensor pk_tensor(device_pk.data(), DType::I32, {c.key_tokens, c.batch});
    const Tensor ph_tensor =
        blocked ? Tensor(device_ph.data(), DType::I32, {c.query_tokens, c.batch}) : Tensor{};
    Tensor out_tensor(device_out.data(), DType::BF16, {c.head_dim, c.query_heads, c.query_tokens, c.batch});

    // The same keys as two sets: the first `head` keys of every batch, then the rest as the extra
    // set, the way a pass's own keys sit beside a ring.
    const std::int32_t head  = c.key_tokens > 1 ? c.key_tokens / 3 + 1 : c.key_tokens;
    const std::int32_t extra = c.key_tokens - head;
    const auto take = [&](const std::vector<std::uint16_t>& bits, std::int32_t from, std::int32_t n) {
        std::vector<std::uint16_t> part;
        const std::size_t row = static_cast<std::size_t>(c.head_dim) * c.kv_heads;
        for (std::int32_t b = 0; b < c.batch; ++b) {
            const auto* base = bits.data() + (static_cast<std::size_t>(b) * c.key_tokens + from) * row;
            part.insert(part.end(), base, base + static_cast<std::size_t>(n) * row);
        }
        return part;
    };
    const auto take_positions = [&](std::int32_t from, std::int32_t n) {
        std::vector<std::int32_t> part;
        for (std::int32_t b = 0; b < c.batch; ++b) {
            const auto* base = position_k.data() + static_cast<std::size_t>(b) * c.key_tokens + from;
            part.insert(part.end(), base, base + n);
        }
        return part;
    };
    const std::size_t row_bytes = static_cast<std::size_t>(c.head_dim) * c.kv_heads * 2;
    GuardedDeviceBuffer device_k1(std::max<std::size_t>(16, row_bytes * head * c.batch));
    GuardedDeviceBuffer device_v1(std::max<std::size_t>(16, row_bytes * head * c.batch));
    GuardedDeviceBuffer device_p1(std::max<std::size_t>(16, 4u * head * c.batch));
    GuardedDeviceBuffer device_k2(std::max<std::size_t>(16, row_bytes * extra * c.batch));
    GuardedDeviceBuffer device_v2(std::max<std::size_t>(16, row_bytes * extra * c.batch));
    GuardedDeviceBuffer device_p2(std::max<std::size_t>(16, 4u * extra * c.batch));
    if (extra > 0) {
        const auto k1 = take(k_bits, 0, head), v1 = take(v_bits, 0, head);
        const auto k2 = take(k_bits, head, extra), v2 = take(v_bits, head, extra);
        const auto p1 = take_positions(0, head), p2 = take_positions(head, extra);
        device_k1.copy_from_host(k1.data(), k1.size() * 2);
        device_v1.copy_from_host(v1.data(), v1.size() * 2);
        device_p1.copy_from_host(p1.data(), p1.size() * 4);
        device_k2.copy_from_host(k2.data(), k2.size() * 2);
        device_v2.copy_from_host(v2.data(), v2.size() * 2);
        device_p2.copy_from_host(p2.data(), p2.size() * 4);
    }

    // Each route is checked: one pass, the key split a workspace allows, and two key sets.
    const std::size_t split_bytes = ops::sliding_causal_attention_workspace_bytes(
        geometry, c.query_tokens, c.key_tokens, c.batch);
    GuardedDeviceBuffer device_workspace(std::max<std::size_t>(split_bytes, 16));
    int failures = 0;
    std::vector<double> got;
    const Tensor none;
    for (int route = 0; route < 3; ++route) {
        if (route == 2 && extra == 0) continue;
        Tensor workspace;
        if (route >= 1 && split_bytes > 0) {
            workspace = Tensor(device_workspace.data(), DType::U8,
                               {static_cast<std::int32_t>(split_bytes)});
        }
        device_out.fill(0x7d);
        if (route < 2) {
            ops::sliding_causal_attention(q_tensor, k_tensor, v_tensor, pq_tensor, ph_tensor,
                                          pk_tensor, none, none, none, geometry, c.window, 1.0F,
                                          out_tensor, workspace, nullptr);
        } else {
            Tensor k1(device_k1.data(), DType::BF16, {c.head_dim, c.kv_heads, head, c.batch});
            Tensor v1(device_v1.data(), DType::BF16, {c.head_dim, c.kv_heads, head, c.batch});
            Tensor p1(device_p1.data(), DType::I32, {head, c.batch});
            Tensor k2(device_k2.data(), DType::BF16, {c.head_dim, c.kv_heads, extra, c.batch});
            Tensor v2(device_v2.data(), DType::BF16, {c.head_dim, c.kv_heads, extra, c.batch});
            Tensor p2(device_p2.data(), DType::I32, {extra, c.batch});
            ops::sliding_causal_attention(q_tensor, k1, v1, pq_tensor, ph_tensor, p1, k2, v2, p2,
                                          geometry, c.window, 1.0F, out_tensor, workspace, nullptr);
        }
        cuda_synchronize();
        const std::string label = std::string(c.label) +
                                  (route == 0 ? " (one pass)" : route == 1 ? " (split)" : " (two key sets)");
        got = from_device_bf16(device_out.data(), q_count);
        failures += verify_pointwise(label.c_str(), got, expected, sliding_causal_criterion());
    }

    // Where no key at all is visible the contract is exact zero, not a Softmax of nothing.
    if (c.first_query_position - (c.first_key_position + (c.key_tokens - 1) * c.key_step) >=
        static_cast<std::int32_t>(c.window)) {
        for (std::size_t i = 0; i < q_count; ++i) {
            if (got[i] != 0.0) {
                std::cerr << c.label << ": a query with no visible key is not zero\n";
                ++failures;
                break;
            }
        }
    }
    failures += device_out.verify_guards("sliding_causal_attention");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Gemma 4's sliding geometry, with the window wide enough that it does not bind, so the causal
    // rule alone decides the visible set.
    failures += run_case({"gemma sliding window 1024", 256, 32, 16, 8, 4, 1, 1000, 1005, 1, 1024},
                         7101U);
    // A window that does bind, plus the boundary pair: distance window-1 visible, distance window not.
    failures += run_case({"narrow window", 64, 8, 4, 8, 4, 1, 0, 4, 1, 5}, 7102U);
    failures += run_case({"window boundary", 32, 4, 2, 6, 6, 1, 0, 0, 1, 3}, 7103U);
    // A gap where nothing is visible: the output must be exact zero.
    failures += run_case({"no visible key", 32, 4, 2, 4, 2, 1, 0, 100, 1, 8}, 7104U);
    // Two batches, to check the batch strides of q/k/v and of both position vectors.
    failures += run_case({"two batches", 128, 16, 8, 6, 3, 2, 40, 44, 1, 16}, 7105U);
    // Future keys only: every distance is negative, so again nothing is visible.
    failures += run_case({"future keys only", 32, 4, 2, 4, 2, 1, 500, 0, 1, 64}, 7106U);

    // A full Gemma ring: 1152 slots in rotated order behind a 128-token pass, so the visible set
    // spans many key tiles, starts mid-ring, and some tiles are invisible to every row.
    failures += run_case({"gemma ring pass", 256, 32, 16, 1152, 128, 1, 1000, 2024, 1, 1024, 300},
                         7107U);
    // More query rows than one block, with a window that cuts through a tile.
    failures += run_case({"many rows", 64, 8, 4, 200, 70, 2, 0, 150, 1, 37, 17}, 7108U);
    // Decode over a full rotated ring, which is the case the key split serves.
    failures += run_case({"gemma ring decode", 256, 32, 16, 1152, 1, 1, 1000, 2151, 1, 1024, 77},
                         7109U);

    // Gemma 4 image blocks. Keys ahead of a query inside its block are visible, the window's lower
    // edge stays at each query, and the block crosses key tiles in rotated ring order. Keys past
    // the block stay invisible and NaN-poisoned. "image past the window" puts the block's late keys
    // within reach while its early ones fall out of the window for the block's last queries.
    failures += run_case(
        {"image block in a ring pass", 256, 32, 16, 1152, 128, 1, 1000, 2024, 1, 1024, 300, 30, 90},
        7110U);
    failures += run_case(
        {"image block, many rows", 64, 8, 4, 200, 70, 2, 0, 130, 1, 1024, 17, 3, 61}, 7111U);
    failures +=
        run_case({"image past the window", 64, 8, 4, 300, 300, 1, 0, 0, 1, 40, 0, 20, 120}, 7112U);

    std::cout << (failures ? "FAIL" : "OK") << " sliding_causal_attention\n";
    return failures ? 1 : 0;
}
