#include "ninfer/ops/causal_compact_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion causal_compact_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.5e-3};
}

struct Case {
    const char* label;
    std::int32_t rotary_dim, rope_angles, query_heads, kv_heads;
    std::int32_t key_tokens, query_tokens, batch, first_query_position;
    std::uint32_t seed;
};

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

bool rotated(const Case& c, std::int32_t d) {
    return d < c.rope_angles || (d >= c.rotary_dim / 2 && d < c.rotary_dim / 2 + c.rope_angles);
}

// Where a dimension lives in a stored compact row: the value vector first, then the rotated key dims.
std::int32_t stored(const Case& c, std::int32_t d) {
    if (d < c.rope_angles) { return c.rotary_dim + d; }
    return c.rotary_dim + (d - c.rotary_dim / 2 + c.rope_angles);
}

std::size_t q_index(const Case& c, std::int32_t d, std::int32_t head, std::int32_t token,
                    std::int32_t batch) {
    return (static_cast<std::size_t>(head) +
            static_cast<std::size_t>(c.query_heads) *
                (token + static_cast<std::size_t>(c.query_tokens) * batch)) *
               c.rotary_dim +
           d;
}

std::size_t kv_index(const Case& c, std::int32_t d, std::int32_t kv_head, std::int32_t key,
                     std::int32_t batch) {
    const std::int32_t width = c.rotary_dim + 2 * c.rope_angles;
    return (static_cast<std::size_t>(kv_head) +
            static_cast<std::size_t>(c.kv_heads) *
                (key + static_cast<std::size_t>(c.key_tokens) * batch)) *
               width +
           d;
}

int run_case(const Case& c) {
    const std::int32_t width      = c.rotary_dim + 2 * c.rope_angles;
    const std::size_t q_count     = static_cast<std::size_t>(c.rotary_dim) * c.query_heads *
                                c.query_tokens * c.batch;
    const std::size_t kv_count = static_cast<std::size_t>(width) * c.kv_heads * c.key_tokens * c.batch;

    std::vector<float> q(q_count), kv(kv_count);
    fill_uniform(q, c.seed, -0.35f, 0.35f);
    fill_uniform(kv, c.seed + 1U, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(kv);

    std::vector<std::int32_t> position_q(static_cast<std::size_t>(c.query_tokens) * c.batch);
    std::vector<std::int32_t> position_k(static_cast<std::size_t>(c.key_tokens) * c.batch);
    for (std::int32_t b = 0; b < c.batch; ++b) {
        for (std::int32_t t = 0; t < c.query_tokens; ++t) {
            position_q[t + c.query_tokens * b] = c.first_query_position + t;
        }
        for (std::int32_t s = 0; s < c.key_tokens; ++s) {
            position_k[s + c.key_tokens * b] = s;
        }
    }

    const ops::AttentionHeadGeometry geometry{c.rotary_dim, c.query_heads, c.kv_heads};
    std::vector<double> expected(q_count, 0.0);
    for (std::int32_t b = 0; b < c.batch; ++b) {
        naive_dense_softmax_attention(
            geometry, c.query_tokens, c.key_tokens, 1.0,
            [&](int d, int head, int token) {
                return static_cast<double>(q[q_index(c, d, head, token, b)]);
            },
            [&](int d, int kv_head, int key) {
                const std::int32_t slot = rotated(c, d) ? stored(c, d) : d;
                return static_cast<double>(kv[kv_index(c, slot, kv_head, key, b)]);
            },
            [&](int d, int kv_head, int key) {
                return static_cast<double>(kv[kv_index(c, d, kv_head, key, b)]);
            },
            [&](int query, int key) {
                return position_k[key + c.key_tokens * b] <= position_q[query + c.query_tokens * b];
            },
            [&](int d, int head, int token, double value) {
                expected[q_index(c, d, head, token, b)] = value;
            });
    }

    const auto q_bits = encode_bf16(q);
    GuardedDeviceBuffer device_q(q_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_kv(kv_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_pq(position_q.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_pk(position_k.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(q_count * sizeof(std::uint16_t));
    device_q.copy_from_host(q_bits.data(), device_q.bytes());
    device_kv.copy_from_host(encode_bf16(kv).data(), device_kv.bytes());
    device_pq.copy_from_host(position_q.data(), device_pq.bytes());
    device_pk.copy_from_host(position_k.data(), device_pk.bytes());
    device_out.fill(0x7d);

    Tensor q_tensor(device_q.data(), DType::BF16, {c.rotary_dim, c.query_heads, c.query_tokens, c.batch});
    Tensor kv_tensor(device_kv.data(), DType::BF16, {width, c.kv_heads, c.key_tokens, c.batch});
    Tensor pq_tensor(device_pq.data(), DType::I32, {c.query_tokens, c.batch});
    Tensor pk_tensor(device_pk.data(), DType::I32, {c.key_tokens, c.batch});
    Tensor out_tensor(device_out.data(), DType::BF16, {c.rotary_dim, c.query_heads, c.query_tokens, c.batch});

    ops::causal_compact_attention(q_tensor, kv_tensor, pq_tensor, pk_tensor, geometry,
                                  c.rotary_dim, c.rope_angles, 1.0F, out_tensor, nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(device_out.data(), q_count);
    int failures = verify_pointwise(c.label, got, expected, causal_compact_criterion());
    failures += device_out.verify_guards("causal_compact_attention");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Gemma 4's global geometry: D 512, 32 query heads, 4 KV heads, 64 rotated pairs.
    failures += run_case({"gemma global 512/64", 512, 64, 32, 4, 6, 3, 1, 3, 8101U});
    // Two batches: the batch strides of q, kv and both position vectors.
    failures += run_case({"gemma global two batches", 512, 64, 32, 4, 5, 2, 2, 4, 8102U});
    // A narrower rotation, so the stored run is not half the head.
    failures += run_case({"narrow rotation", 128, 16, 8, 2, 5, 3, 1, 2, 8103U});
    // Every key is ahead of the query, so nothing is visible and the output is exact zero.
    failures += run_case({"no visible key", 64, 8, 4, 2, 4, 2, 1, 100, 8104U});

    std::cout << (failures ? "FAIL" : "OK") << " causal_compact_attention\n";
    return failures ? 1 : 0;
}
