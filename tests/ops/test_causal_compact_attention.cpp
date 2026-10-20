#include "ninfer/ops/causal_compact_attention.h"
#include "ninfer/ops/compact_kv_rows_quantize.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cmath>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The legacy default stream with the device's SM count, which the key split policy reads.
DeviceExecutionView test_execution() {
    int device = 0, count = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return {nullptr, count};
}

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

// The represented value of an E4M3 code and an FP16 scale, evaluated in double: the same decode the
// kernel applies when it stages a coded row, so the oracle below sees the stored bytes themselves.
double decode_e4m3fn(std::uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    double magnitude   = 0.0;
    if (exponent == 0) {
        magnitude = std::ldexp(static_cast<double>(mantissa) / 8.0, -6);
    } else if (!(exponent == 15 && mantissa == 7)) {
        magnitude = std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    }
    return (code & 0x80U) != 0 ? -magnitude : magnitude;
}

double decode_f16(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 31U;
    const std::uint32_t mantissa = bits & 1023U;
    std::uint32_t value          = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            int shift       = 0;
            std::uint32_t m = mantissa;
            while ((m & 1024U) == 0) {
                m <<= 1;
                ++shift;
            }
            value |= static_cast<std::uint32_t>(127 - 15 - shift) << 23;
            value |= (m & 1023U) << 13;
        }
    } else {
        value |= (exponent + 127 - 15) << 23;
        value |= mantissa << 13;
    }
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return static_cast<double>(out);
}

bool rotated(const Case& c, std::int32_t d) {
    return d < c.rope_angles || (d >= c.rotary_dim / 2 && d < c.rotary_dim / 2 + c.rope_angles);
}

// Where a dimension lives in a stored compact row: the value vector first, then the rotated key
// dims.
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
    const std::int32_t width = c.rotary_dim + 2 * c.rope_angles;
    const std::size_t q_count =
        static_cast<std::size_t>(c.rotary_dim) * c.query_heads * c.query_tokens * c.batch;
    const std::size_t kv_count =
        static_cast<std::size_t>(width) * c.kv_heads * c.key_tokens * c.batch;

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
        for (std::int32_t s = 0; s < c.key_tokens; ++s) { position_k[s + c.key_tokens * b] = s; }
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

    Tensor q_tensor(device_q.data(), DType::BF16,
                    {c.rotary_dim, c.query_heads, c.query_tokens, c.batch});
    Tensor kv_tensor(device_kv.data(), DType::BF16, {width, c.kv_heads, c.key_tokens, c.batch});
    Tensor pq_tensor(device_pq.data(), DType::I32, {c.query_tokens, c.batch});
    Tensor pk_tensor(device_pk.data(), DType::I32, {c.key_tokens, c.batch});
    Tensor out_tensor(device_out.data(), DType::BF16,
                      {c.rotary_dim, c.query_heads, c.query_tokens, c.batch});

    // The keys here hold their own index as position, so `indexed_keys` is a true statement and
    // must not change the result. Each route is checked: one pass, and the key split a workspace
    // allows.
    const std::size_t split_bytes = ops::causal_compact_attention_workspace_bytes(
        geometry, c.rotary_dim, c.query_tokens, c.key_tokens, c.batch);
    GuardedDeviceBuffer device_workspace(std::max<std::size_t>(split_bytes, 16));
    int failures = 0;
    for (int route = 0; route < 3; ++route) {
        const bool split   = route > 0;
        const bool indexed = route == 2;
        Tensor workspace;
        if (split && split_bytes > 0) {
            workspace = Tensor(device_workspace.data(), DType::U8,
                               {static_cast<std::int32_t>(split_bytes)});
        }
        device_out.fill(0x7d);
        ops::causal_compact_attention(q_tensor, kv_tensor, pq_tensor, pk_tensor, geometry,
                                      c.rotary_dim, c.rope_angles, 1.0F, indexed, out_tensor,
                                      workspace, test_execution());
        cuda_synchronize();
        const std::string label = std::string(c.label) + (route == 0   ? " (one pass)"
                                                          : route == 1 ? " (split)"
                                                                       : " (split, indexed)");
        const auto got          = from_device_bf16(device_out.data(), q_count);
        failures += verify_pointwise(label.c_str(), got, expected, causal_compact_criterion());
        failures += device_out.verify_guards("causal_compact_attention");
    }

    // The coded route: the same rows through `compact_kv_rows_quantize`, then attended from the
    // stored FP8 planes. Only the 640-wide Gemma geometry has a registered codec.
    if (width == 640) {
        const std::size_t row_width = static_cast<std::size_t>(width) * c.kv_heads;
        const std::size_t block     = row_width * c.key_tokens;
        GuardedDeviceBuffer device_codes(kv_count);
        GuardedDeviceBuffer device_scales(static_cast<std::size_t>(c.kv_heads) * c.key_tokens *
                                          c.batch * sizeof(std::uint16_t));
        device_codes.fill(0);
        device_scales.fill(0);
        for (std::int32_t b = 0; b < c.batch; ++b) {
            Tensor rows(static_cast<std::uint8_t*>(device_kv.data()) +
                            b * block * sizeof(std::uint16_t),
                        DType::BF16, {static_cast<std::int32_t>(row_width), c.key_tokens});
            Tensor positions(static_cast<std::uint8_t*>(device_pk.data()) +
                                 b * c.key_tokens * static_cast<std::int32_t>(sizeof(std::int32_t)),
                             DType::I32, {c.key_tokens, 1, 1, 1});
            Tensor codes(static_cast<std::uint8_t*>(device_codes.data()) + b * block,
                         DType::FP8_E4M3FN, {width, c.kv_heads, c.key_tokens, 1});
            Tensor scales(static_cast<std::uint8_t*>(device_scales.data()) +
                              b * static_cast<std::size_t>(c.kv_heads) * c.key_tokens *
                                  sizeof(std::uint16_t),
                          DType::FP16, {c.kv_heads, c.key_tokens, 1, 1});
            ops::compact_kv_rows_quantize(rows, positions, codes, scales, nullptr);
        }
        cuda_synchronize();

        std::vector<std::uint8_t> code_host(kv_count);
        std::vector<std::uint8_t> scale_host(static_cast<std::size_t>(c.kv_heads) * c.key_tokens *
                                             c.batch * sizeof(std::uint16_t));
        device_codes.copy_to_host(code_host.data(), code_host.size());
        device_scales.copy_to_host(scale_host.data(), scale_host.size());
        const auto fp16_at = [&](std::size_t index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, scale_host.data() + index * sizeof(bits), sizeof(bits));
            return bits;
        };
        // The represented value of every stored element, in the stored row's own basis.
        std::vector<double> coded_kv(kv_count);
        for (std::int32_t b = 0; b < c.batch; ++b) {
            for (std::int32_t head = 0; head < c.kv_heads; ++head) {
                for (std::int32_t key = 0; key < c.key_tokens; ++key) {
                    const std::size_t scale_index =
                        static_cast<std::size_t>(head) +
                        static_cast<std::size_t>(c.kv_heads) *
                            (key + static_cast<std::size_t>(c.key_tokens) * b);
                    const double scale = decode_f16(fp16_at(scale_index));
                    for (std::int32_t d = 0; d < width; ++d) {
                        const std::size_t at = kv_index(c, d, head, key, b);
                        // The kernel stages `bf16(code * scale)`; the oracle reads that same value,
                        // so only the attention's own BF16 rounding is left to compare.
                        const auto represented =
                            static_cast<float>(decode_e4m3fn(code_host[at]) * scale);
                        coded_kv[at] = bf16_to_f32(f32_to_bf16(represented));
                    }
                }
            }
        }
        const std::vector<double> original(kv.begin(), kv.end());
        const ReductionCriterion fp8_round_trip{/*relative_l2*/ 3.0e-2, /*gross_absolute*/ 5.0e-3,
                                                /*gross_relative_to_max*/ 8.0e-2};
        failures += verify_reduction(std::string(c.label) + " coded round trip", coded_kv, original,
                                     fp8_round_trip);

        // The oracle over the represented values: only the route's BF16 staging is left.
        std::vector<double> coded_expected(q_count, 0.0);
        for (std::int32_t b = 0; b < c.batch; ++b) {
            naive_dense_softmax_attention(
                geometry, c.query_tokens, c.key_tokens, 1.0,
                [&](int d, int head, int token) {
                    return static_cast<double>(q[q_index(c, d, head, token, b)]);
                },
                [&](int d, int kv_head, int key) {
                    const std::int32_t slot = rotated(c, d) ? stored(c, d) : d;
                    return coded_kv[kv_index(c, slot, kv_head, key, b)];
                },
                [&](int d, int kv_head, int key) {
                    return coded_kv[kv_index(c, d, kv_head, key, b)];
                },
                [&](int query, int key) {
                    return position_k[key + c.key_tokens * b] <=
                           position_q[query + c.query_tokens * b];
                },
                [&](int d, int head, int token, double value) {
                    coded_expected[q_index(c, d, head, token, b)] = value;
                });
        }

        GuardedDeviceBuffer coded_workspace(std::max<std::size_t>(split_bytes, 16));
        Tensor codes_tensor(device_codes.data(), DType::FP8_E4M3FN,
                            {width, c.kv_heads, c.key_tokens, c.batch});
        Tensor scales_tensor(device_scales.data(), DType::FP16,
                             {c.kv_heads, c.key_tokens, c.batch, 1});
        for (int route = 0; route < 3; ++route) {
            const bool split   = route > 0;
            const bool indexed = route == 2;
            Tensor workspace;
            if (split && split_bytes > 0) {
                workspace = Tensor(coded_workspace.data(), DType::U8,
                                   {static_cast<std::int32_t>(split_bytes)});
            }
            device_out.fill(0x7d);
            ops::causal_compact_attention_codes(
                q_tensor, codes_tensor, scales_tensor, pq_tensor, pk_tensor, geometry, c.rotary_dim,
                c.rope_angles, 1.0F, indexed, out_tensor, workspace, test_execution());
            cuda_synchronize();
            const std::string label = std::string(c.label) + " coded" +
                                      (route == 0   ? " (one pass)"
                                       : route == 1 ? " (split)"
                                                    : " (split, indexed)");
            const auto got          = from_device_bf16(device_out.data(), q_count);
            failures +=
                verify_pointwise(label.c_str(), got, coded_expected, causal_compact_criterion());
            failures += device_out.verify_guards("causal_compact_attention_codes");
        }
        failures += device_codes.verify_guards("compact_kv_rows_quantize codes");
        failures += device_scales.verify_guards("compact_kv_rows_quantize scales");
    }
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
    // Gemma's geometry over many key tiles, with more query rows than one block.
    failures += run_case({"gemma global long", 512, 64, 32, 4, 700, 40, 1, 660, 8105U});
    // Decode: one query row per head over a long cache, which is the case the key split serves.
    failures += run_case({"gemma global decode", 512, 64, 32, 4, 2000, 1, 1, 1500, 8106U});
    // Decode at an 8K depth over a larger cache: more than 32 key splits, so the combine weighs
    // splits in both of its lanes, and the indexed scan stops at the query.
    failures += run_case({"gemma global decode at 8K", 512, 64, 32, 4, 8300, 1, 1, 8100, 8110U});

    // Prefill-sized passes take the tiled kernel: one late in a long cache, one at an 8K prompt's
    // depth, and two batches.
    failures += run_case({"gemma global prefill", 512, 64, 32, 4, 1200, 256, 1, 944, 8107U});
    failures += run_case({"gemma global prefill at 8K", 512, 64, 32, 4, 8000, 64, 1, 7936, 8109U});
    failures +=
        run_case({"gemma global prefill two batches", 512, 64, 32, 4, 300, 40, 2, 260, 8108U});

    std::cout << (failures ? "FAIL" : "OK") << " causal_compact_attention\n";
    return failures ? 1 : 0;
}
