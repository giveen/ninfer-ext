#include "ninfer/ops/sliding_causal_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cstdint>
#include <iostream>
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
    for (std::int32_t b = 0; b < c.batch; ++b) {
        for (std::int32_t t = 0; t < c.query_tokens; ++t) {
            position_q[t + c.query_tokens * b] = c.first_query_position + t;
        }
        for (std::int32_t s = 0; s < c.key_tokens; ++s) {
            position_k[s + c.key_tokens * b] = c.first_key_position + s * c.key_step + b;
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
                const std::int32_t distance =
                    position_q[query + c.query_tokens * b] - position_k[key + c.key_tokens * b];
                return distance >= 0 && distance < static_cast<std::int32_t>(c.window);
            },
            [&](int d, int head, int token, double value) {
                expected[index(c, d, head, token, b)] = value;
            });
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
    GuardedDeviceBuffer device_out(q_count * sizeof(std::uint16_t));
    device_q.copy_from_host(q_bits.data(), device_q.bytes());
    device_k.copy_from_host(k_bits.data(), device_k.bytes());
    device_v.copy_from_host(v_bits.data(), device_v.bytes());
    device_pq.copy_from_host(position_q.data(), device_pq.bytes());
    device_pk.copy_from_host(position_k.data(), device_pk.bytes());
    device_out.fill(0x7d);

    Tensor q_tensor(device_q.data(), DType::BF16, {c.head_dim, c.query_heads, c.query_tokens, c.batch});
    Tensor k_tensor(device_k.data(), DType::BF16, {c.head_dim, c.kv_heads, c.key_tokens, c.batch});
    Tensor v_tensor(device_v.data(), DType::BF16, {c.head_dim, c.kv_heads, c.key_tokens, c.batch});
    Tensor pq_tensor(device_pq.data(), DType::I32, {c.query_tokens, c.batch});
    Tensor pk_tensor(device_pk.data(), DType::I32, {c.key_tokens, c.batch});
    Tensor out_tensor(device_out.data(), DType::BF16, {c.head_dim, c.query_heads, c.query_tokens, c.batch});

    ops::sliding_causal_attention(q_tensor, k_tensor, v_tensor, pq_tensor, pk_tensor, geometry,
                                  c.window, 1.0F, out_tensor, nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(device_out.data(), q_count);
    int failures =
        verify_pointwise(c.label, got, expected, sliding_causal_criterion());

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

    std::cout << (failures ? "FAIL" : "OK") << " sliding_causal_attention\n";
    return failures ? 1 : 0;
}
