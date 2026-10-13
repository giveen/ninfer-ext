// Qualifies the dense D256 KV ring codec the 26B's sliding layers keep behind `--kv-dtype`.
//
// Two checks per profile. The round trip writes random BF16 rows into a ring's quantized planes
// through `ops::dense_kv_ring_encode`, decodes them back through `ops::dense_kv_ring_decode`, and
// compares the decoded rows against the rows they came from. The attention oracle builds a full
// ring, runs the same encode and decode, attends at the model's real geometry (D256, 16 query
// heads, 8 KV heads, window 1024, scale 1.0), and compares against an independent FP64 reference
// that decodes the stored plane bytes itself and runs the Softmax Attention at that geometry.
#include "core/paged_kv_storage.h"
#include "ninfer/ops/dense_kv_ring.h"
#include "ninfer/ops/sliding_causal_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kHeads      = 8; // the model's sliding KV heads
constexpr std::int32_t kQueryHeads = 16;
constexpr float kAttentionScale    = 1.0F;
constexpr std::uint32_t kWindow    = 1024;

struct Profile {
    KvCacheStorage storage;
    const char* name;
    // The decoded row against the BF16 row it came from, one criterion per side because a mixed
    // storage (K8V4) quantizes its two sides with two formats.
    ReductionCriterion round_trip_key;
    ReductionCriterion round_trip_value;
    // The whole route against the FP64 reference at the real geometry.
    ReductionCriterion attention;
};

// Every storage's attention route lands within 5% of the same number, because what the comparison
// sees at this key count is the route's own BF16 rounding of the decoded rows rather than the
// format's precision: 1024 keys average the quantization noise down by about sqrt(1024). One
// criterion covers all four, and it is tight enough that a wrong rotation, scale or group would
// miss it by two orders of magnitude.
// Measured relative L2: fp8 0.00300/0.00316, int8 0.00309/0.00304, nvfp4 0.00316/0.00304,
// k8v4 0.00304/0.00313.
constexpr ReductionCriterion kAttentionCriterion{
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 1.5e-3,
    /*gross_relative_to_max_reference*/ 4.0e-3,
};

// Each format's own floor, from the round trip over 524,288 decoded elements. An independent FP64
// emulation of the same codecs over the same distribution reproduces every one of them: FP8-row
// 0.0261 (measured 0.02610 key / 0.02542 value), INT8-G64 0.0060 (0.00614 / 0.00437), NVFP4-G16
// 0.0960 key and 0.1019 value (0.09529 / 0.09518).
constexpr ReductionCriterion kRoundTripFp8{/*relative_l2*/ 3.0e-2, /*gross_absolute*/ 5.0e-3,
                                           /*gross_relative_to_max*/ 8.0e-2};
constexpr ReductionCriterion kRoundTripInt8{/*relative_l2*/ 8.0e-3, /*gross_absolute*/ 3.0e-3,
                                            /*gross_relative_to_max*/ 1.5e-2};
constexpr ReductionCriterion kRoundTripNvfp4{/*relative_l2*/ 1.2e-1, /*gross_absolute*/ 1.0e-2,
                                             /*gross_relative_to_max*/ 3.5e-1};

constexpr Profile kProfiles[] = {
    {KvCacheStorage::Fp8E4M3Row256, "fp8", kRoundTripFp8, kRoundTripFp8, kAttentionCriterion},
    {KvCacheStorage::Int8Group64, "int8", kRoundTripInt8, kRoundTripInt8, kAttentionCriterion},
    {KvCacheStorage::Nvfp4Group16, "nvfp4", kRoundTripNvfp4, kRoundTripNvfp4, kAttentionCriterion},
    {KvCacheStorage::Fp8KeyNvfp4Value, "k8v4", kRoundTripFp8, kRoundTripNvfp4, kAttentionCriterion},
};

// One ring shape and one pass over it. `wrap` puts the pass's own row at slot 0 and its position
// one past the ring, which is what a narrow pass that writes first does.
struct RingCase {
    const char* label;
    std::int32_t slots;
    std::int32_t pass_tokens;
    bool wrap;
};

// The fixed normalized D256 Sylvester transform the D256 KV codecs rotate a row with. The kernel
// applies it as a butterfly over the low five dimension bits and one over the high three; a flat
// butterfly over all eight is the same matrix and the same 2^-4 normalization. It is an involution,
// which is why the decode is expressed in the encoder's own basis.
void normalized_hadamard_d256(std::array<double, kHeadDim>& row) {
    for (int span = 1; span < kHeadDim; span <<= 1) {
        for (int base = 0; base < kHeadDim; base += 2 * span) {
            for (int i = 0; i < span; ++i) {
                const double low  = row[static_cast<std::size_t>(base + i)];
                const double high = row[static_cast<std::size_t>(base + i + span)];
                row[static_cast<std::size_t>(base + i)]        = low + high;
                row[static_cast<std::size_t>(base + i + span)] = low - high;
            }
        }
    }
    for (double& value : row) value /= 16.0;
}

double decode_e4m3_positive(std::uint8_t code) {
    if (code == 0) return 0.0;
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exponent == 0) return std::ldexp(static_cast<double>(mantissa) / 8.0, -6);
    return std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
}

double decode_e4m3fn(std::uint8_t code) {
    const double magnitude = decode_e4m3_positive(code & 0x7fU);
    return (code & 0x80U) != 0 ? -magnitude : magnitude;
}

double decode_e2m1(std::uint8_t code) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude             = kValues[code & 7U];
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

double decode_f16(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 31U;
    const std::uint32_t mantissa = bits & 1023U;
    std::uint32_t value          = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            // A subnormal half: normalize it into a float.
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

[[nodiscard]] bool storage_rotates_value(KvCacheStorage storage) {
    return storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value;
}

// The decoded value of one stored element, in the basis the encoder read: the format's represented
// value, then its row rotation undone.
double decoded_element(KvCacheStorage storage, bool key, const std::uint8_t* codes,
                       const std::uint8_t* scales, std::int32_t head, std::int32_t slot,
                       std::int32_t d) {
    const auto fp16 = [](const std::uint8_t* at, std::size_t index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, at + index * sizeof(std::uint16_t), sizeof(bits));
        return bits;
    };
    const bool key_fp8    = storage == KvCacheStorage::Fp8E4M3Row256 ||
                            (key && storage == KvCacheStorage::Fp8KeyNvfp4Value);
    const std::size_t row = static_cast<std::size_t>(head + kHeads * slot);
    if (key_fp8) {
        return decode_e4m3fn(codes[row * kHeadDim + static_cast<std::size_t>(d)]) *
               decode_f16(fp16(scales, row));
    }
    if (storage == KvCacheStorage::Int8Group64) {
        const auto code =
            static_cast<std::int8_t>(codes[row * kHeadDim + static_cast<std::size_t>(d)]);
        return static_cast<double>(code) *
               decode_f16(fp16(scales, row * 4 + static_cast<std::size_t>(d) / 64));
    }
    const std::uint8_t byte = codes[row * (kHeadDim / 2) + static_cast<std::size_t>(d) / 2];
    const std::uint8_t code =
        (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4) : static_cast<std::uint8_t>(byte & 15U);
    return decode_e2m1(code) *
           decode_e4m3fn(scales[row * (kHeadDim / 16) + static_cast<std::size_t>(d) / 16]);
}

// One whole side of a ring, decoded in FP64: [d, head, slot] with d fastest.
std::vector<double> decode_side(KvCacheStorage storage, bool key, const std::uint8_t* codes,
                                const std::uint8_t* scales, std::int32_t slots) {
    const bool rotated = key || storage_rotates_value(storage);
    std::vector<double> out(static_cast<std::size_t>(kHeadDim) * kHeads * slots);
    std::array<double, kHeadDim> row{};
    for (std::int32_t head = 0; head < kHeads; ++head) {
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                row[static_cast<std::size_t>(d)] =
                    decoded_element(storage, key, codes, scales, head, slot, d);
            }
            if (rotated) normalized_hadamard_d256(row);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                out[static_cast<std::size_t>(d) +
                    static_cast<std::size_t>(kHeadDim) *
                        static_cast<std::size_t>(head + kHeads * slot)] =
                    row[static_cast<std::size_t>(d)];
            }
        }
    }
    return out;
}

std::size_t plane_bytes(const PagedKVVectorLayout& vector, std::int32_t slots, bool codes) {
    const std::int32_t extent = codes ? vector.data_leading_extent : vector.scale_leading_extent;
    const DType dtype         = codes ? vector.data_dtype : vector.scale_dtype;
    return static_cast<std::size_t>(extent) * kHeads * slots * dtype_size(dtype);
}

DeviceExecutionView test_execution() {
    int device = 0, count = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device),
               "cudaDeviceGetAttribute");
    return {nullptr, count};
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Every row of a quantized ring is decoded back to the BF16 row it came from, at the slots the rows
// were written to; a slot nothing was written to decodes to exact zero.
int round_trip(const Profile& profile, std::uint32_t seed) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(profile.storage, kHeadDim);
    const std::int32_t slots          = 256;
    const std::int32_t rows           = 64;
    const std::size_t row_width       = static_cast<std::size_t>(kHeadDim) * kHeads;

    std::vector<float> keys(row_width * rows), values(row_width * rows);
    fill_uniform(keys, seed, -0.5F, 0.5F);
    fill_uniform(values, seed + 1U, -0.8F, 0.8F);
    round_to_bf16(keys);
    round_to_bf16(values);

    // Distinct slots, in a shuffled order so the write is not a copy.
    std::vector<std::int32_t> where(static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) where[static_cast<std::size_t>(slot)] = slot;
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::int32_t slot = slots - 1; slot > 0; --slot) {
        state           = state * 1664525U + 1013904223U;
        const auto pick = static_cast<std::int32_t>(state % static_cast<std::uint32_t>(slot + 1));
        std::swap(where[static_cast<std::size_t>(slot)], where[static_cast<std::size_t>(pick)]);
    }
    const std::vector<std::int32_t> slots_of_row(where.begin(), where.begin() + rows);
    std::vector<std::int32_t> row_of_slot(static_cast<std::size_t>(slots), -1);
    for (std::int32_t row = 0; row < rows; ++row) {
        row_of_slot[static_cast<std::size_t>(slots_of_row[static_cast<std::size_t>(row)])] = row;
    }

    const auto key_bits = to_bf16_bits(keys), value_bits = to_bf16_bits(values);
    GuardedDeviceBuffer device_keys(row_width * rows * 2);
    GuardedDeviceBuffer device_values(row_width * rows * 2);
    GuardedDeviceBuffer device_slots(rows * sizeof(std::int32_t));
    device_keys.copy_from_host(key_bits.data(), device_keys.bytes());
    device_values.copy_from_host(value_bits.data(), device_values.bytes());
    device_slots.copy_from_host(slots_of_row.data(), device_slots.bytes());

    const std::size_t key_code_bytes    = plane_bytes(layout.key, slots, true);
    const std::size_t key_scale_bytes   = plane_bytes(layout.key, slots, false);
    const std::size_t value_code_bytes  = plane_bytes(layout.value, slots, true);
    const std::size_t value_scale_bytes = plane_bytes(layout.value, slots, false);
    GuardedDeviceBuffer device_key_codes(key_code_bytes);
    GuardedDeviceBuffer device_key_scales(key_scale_bytes);
    GuardedDeviceBuffer device_value_codes(value_code_bytes);
    GuardedDeviceBuffer device_value_scales(value_scale_bytes);
    // A ring's planes are cleared when the cache allocates them, so a slot no token has written
    // decodes to exact zero; the round trip checks exactly that below.
    device_key_codes.fill(0);
    device_key_scales.fill(0);
    device_value_codes.fill(0);
    device_value_scales.fill(0);

    Tensor key_rows(device_keys.data(), DType::BF16, {kHeadDim, kHeads, rows});
    Tensor value_rows(device_values.data(), DType::BF16, {kHeadDim, kHeads, rows});
    Tensor row_slots(device_slots.data(), DType::I32, {rows, 1, 1, 1});
    Tensor key_codes(device_key_codes.data(), layout.key.data_dtype,
                     {layout.key.data_leading_extent, kHeads, slots});
    Tensor key_scales(device_key_scales.data(), layout.key.scale_dtype,
                      {layout.key.scale_leading_extent, kHeads, slots});
    Tensor value_codes(device_value_codes.data(), layout.value.data_dtype,
                       {layout.value.data_leading_extent, kHeads, slots});
    Tensor value_scales(device_value_scales.data(), layout.value.scale_dtype,
                        {layout.value.scale_leading_extent, kHeads, slots});
    ops::dense_kv_ring_encode(key_rows, value_rows, row_slots, profile.storage, key_codes,
                              key_scales, value_codes, value_scales, nullptr);

    GuardedDeviceBuffer device_ring_keys(row_width * slots * 2);
    GuardedDeviceBuffer device_ring_values(row_width * slots * 2);
    Tensor ring_keys(device_ring_keys.data(), DType::BF16, {kHeadDim, kHeads, slots});
    Tensor ring_values(device_ring_values.data(), DType::BF16, {kHeadDim, kHeads, slots});
    ops::dense_kv_ring_decode(key_codes, key_scales, value_codes, value_scales, profile.storage,
                              ring_keys, ring_values, nullptr);
    cuda_synchronize();

    const auto got_keys   = from_device_bf16(device_ring_keys.data(), row_width * slots);
    const auto got_values = from_device_bf16(device_ring_values.data(), row_width * slots);
    std::vector<double> expected(static_cast<std::size_t>(row_width) * slots, 0.0);
    std::vector<double> expected_values(expected.size(), 0.0);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto d           = static_cast<std::int32_t>(index % kHeadDim);
        const auto head        = static_cast<std::int32_t>((index / kHeadDim) % kHeads);
        const auto slot        = static_cast<std::int32_t>(index / (kHeadDim * kHeads));
        const std::int32_t row = row_of_slot[static_cast<std::size_t>(slot)];
        if (row < 0) continue; // never written: the decode must leave exact zero
        const std::size_t at   = static_cast<std::size_t>(d) +
                                 static_cast<std::size_t>(kHeadDim) * (head + kHeads * row);
        expected[index]        = static_cast<double>(keys[at]);
        expected_values[index] = static_cast<double>(values[at]);
    }
    int failures = 0;
    failures += verify_reduction(std::string(profile.name) + " round trip keys", got_keys, expected,
                                 profile.round_trip_key);
    failures += verify_reduction(std::string(profile.name) + " round trip values", got_values,
                                 expected_values, profile.round_trip_value);
    failures += device_keys.verify_guards("dense_kv_ring_encode keys");
    failures += device_ring_keys.verify_guards("dense_kv_ring_decode keys");
    return failures;
}

// A full ring, the same encode and decode, and the sliding attention at the model's geometry.
int attention_oracle(const Profile& profile, const RingCase& ring, std::uint32_t seed) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(profile.storage, kHeadDim);
    const std::int32_t slots          = ring.slots;
    const std::int32_t tokens         = ring.pass_tokens;
    const std::int32_t extra          = ring.wrap ? 0 : tokens;
    const std::size_t row_width       = static_cast<std::size_t>(kHeadDim) * kHeads;
    const ops::AttentionHeadGeometry geometry{kHeadDim, kQueryHeads, kHeads};

    // The ring's rows at their slots, and the pass's own rows beside them.
    std::vector<float> ring_keys(row_width * slots), ring_values(row_width * slots);
    fill_uniform(ring_keys, seed, -0.5F, 0.5F);
    fill_uniform(ring_values, seed + 1U, -0.8F, 0.8F);
    std::vector<float> pass_keys(row_width * tokens), pass_values(row_width * tokens);
    fill_uniform(pass_keys, seed + 2U, -0.5F, 0.5F);
    fill_uniform(pass_values, seed + 3U, -0.8F, 0.8F);
    round_to_bf16(ring_keys);
    round_to_bf16(ring_values);
    round_to_bf16(pass_keys);
    round_to_bf16(pass_values);
    std::vector<std::int32_t> slot_position(static_cast<std::size_t>(slots));
    std::vector<std::int32_t> row_slot(static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        slot_position[static_cast<std::size_t>(slot)] = ring.wrap && slot == 0 ? slots : slot;
        row_slot[static_cast<std::size_t>(slot)]      = slot;
    }
    if (ring.wrap) {
        // The pass wrote its own row first, over the oldest slot.
        std::copy(pass_keys.begin(), pass_keys.end(), ring_keys.begin());
        std::copy(pass_values.begin(), pass_values.end(), ring_values.begin());
    }

    std::vector<float> q(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    fill_uniform(q, seed + 4U, -0.5F, 0.5F);
    round_to_bf16(q);
    const std::vector<std::int32_t> query_position = [&] {
        std::vector<std::int32_t> out(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            out[static_cast<std::size_t>(token)] = slots + token;
        }
        return out;
    }();

    const auto ring_key_bits = to_bf16_bits(ring_keys), ring_value_bits = to_bf16_bits(ring_values);
    GuardedDeviceBuffer device_rows(row_width * slots * 2);
    GuardedDeviceBuffer device_row_values(row_width * slots * 2);
    GuardedDeviceBuffer device_row_slots(slots * sizeof(std::int32_t));
    GuardedDeviceBuffer device_slot_position(slots * sizeof(std::int32_t));
    device_rows.copy_from_host(ring_key_bits.data(), device_rows.bytes());
    device_row_values.copy_from_host(ring_value_bits.data(), device_row_values.bytes());
    device_row_slots.copy_from_host(row_slot.data(), device_row_slots.bytes());
    device_slot_position.copy_from_host(slot_position.data(), device_slot_position.bytes());

    const std::size_t key_code_bytes    = plane_bytes(layout.key, slots, true);
    const std::size_t key_scale_bytes   = plane_bytes(layout.key, slots, false);
    const std::size_t value_code_bytes  = plane_bytes(layout.value, slots, true);
    const std::size_t value_scale_bytes = plane_bytes(layout.value, slots, false);
    GuardedDeviceBuffer device_key_codes(key_code_bytes);
    GuardedDeviceBuffer device_key_scales(key_scale_bytes);
    GuardedDeviceBuffer device_value_codes(value_code_bytes);
    GuardedDeviceBuffer device_value_scales(value_scale_bytes);
    GuardedDeviceBuffer device_ring_keys(row_width * slots * 2);
    GuardedDeviceBuffer device_ring_values(row_width * slots * 2);

    Tensor row_keys(device_rows.data(), DType::BF16, {kHeadDim, kHeads, slots});
    Tensor row_values(device_row_values.data(), DType::BF16, {kHeadDim, kHeads, slots});
    Tensor row_slots(device_row_slots.data(), DType::I32, {slots, 1, 1, 1});
    Tensor key_codes(device_key_codes.data(), layout.key.data_dtype,
                     {layout.key.data_leading_extent, kHeads, slots});
    Tensor key_scales(device_key_scales.data(), layout.key.scale_dtype,
                      {layout.key.scale_leading_extent, kHeads, slots});
    Tensor value_codes(device_value_codes.data(), layout.value.data_dtype,
                       {layout.value.data_leading_extent, kHeads, slots});
    Tensor value_scales(device_value_scales.data(), layout.value.scale_dtype,
                        {layout.value.scale_leading_extent, kHeads, slots});
    ops::dense_kv_ring_encode(row_keys, row_values, row_slots, profile.storage, key_codes,
                              key_scales, value_codes, value_scales, nullptr);

    Tensor ring_keys_out(device_ring_keys.data(), DType::BF16, {kHeadDim, kHeads, slots});
    Tensor ring_values_out(device_ring_values.data(), DType::BF16, {kHeadDim, kHeads, slots});
    ops::dense_kv_ring_decode(key_codes, key_scales, value_codes, value_scales, profile.storage,
                              ring_keys_out, ring_values_out, nullptr);

    // The pass's own rows, when the ring does not already hold them.
    GuardedDeviceBuffer device_pass_keys(std::max<std::size_t>(16, row_width * tokens * 2));
    GuardedDeviceBuffer device_pass_values(std::max<std::size_t>(16, row_width * tokens * 2));
    GuardedDeviceBuffer device_query(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * 2);
    GuardedDeviceBuffer device_query_position(tokens * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * 2);
    if (extra > 0) {
        const auto bits = to_bf16_bits(pass_keys);
        device_pass_keys.copy_from_host(bits.data(), row_width * tokens * 2);
        const auto value_bits = to_bf16_bits(pass_values);
        device_pass_values.copy_from_host(value_bits.data(), row_width * tokens * 2);
    }
    {
        const auto bits = to_bf16_bits(q);
        device_query.copy_from_host(bits.data(), device_query.bytes());
        device_query_position.copy_from_host(query_position.data(), device_query_position.bytes());
    }

    const std::size_t split =
        ops::sliding_causal_attention_workspace_bytes(geometry, tokens, slots + extra, 1);
    GuardedDeviceBuffer device_workspace(std::max<std::size_t>(split, 16));
    const Tensor none;
    const Tensor position_pass(device_query_position.data(), DType::I32, {tokens, 1, 1, 1});
    const Tensor position_ring(device_slot_position.data(), DType::I32, {slots, 1, 1, 1});
    Tensor k_ring(device_ring_keys.data(), DType::BF16, {kHeadDim, kHeads, slots, 1});
    Tensor v_ring(device_ring_values.data(), DType::BF16, {kHeadDim, kHeads, slots, 1});
    Tensor k_pass(device_pass_keys.data(), DType::BF16, {kHeadDim, kHeads, tokens, 1});
    Tensor v_pass(device_pass_values.data(), DType::BF16, {kHeadDim, kHeads, tokens, 1});
    Tensor out(device_out.data(), DType::BF16, {kHeadDim, kQueryHeads, tokens, 1});
    Tensor q_tensor(device_query.data(), DType::BF16, {kHeadDim, kQueryHeads, tokens, 1});
    const Tensor workspace =
        split > 0 ? Tensor(device_workspace.data(), DType::U8, {static_cast<std::int32_t>(split)})
                  : none;
    device_out.fill(0x7d);
    ops::sliding_causal_attention(q_tensor, k_ring, v_ring, position_pass, none, position_ring,
                                  extra > 0 ? k_pass : none, extra > 0 ? v_pass : none,
                                  extra > 0 ? position_pass : none, geometry, kWindow,
                                  kAttentionScale, out, workspace, test_execution());
    cuda_synchronize();

    // The reference: the same stored bytes, decoded independently, against the same queries.
    std::vector<std::uint8_t> key_code_host(key_code_bytes), key_scale_host(key_scale_bytes),
        value_code_host(value_code_bytes), value_scale_host(value_scale_bytes);
    device_key_codes.copy_to_host(key_code_host.data(), key_code_bytes);
    device_key_scales.copy_to_host(key_scale_host.data(), key_scale_bytes);
    device_value_codes.copy_to_host(value_code_host.data(), value_code_bytes);
    device_value_scales.copy_to_host(value_scale_host.data(), value_scale_bytes);
    const std::vector<double> decoded_keys =
        decode_side(profile.storage, true, key_code_host.data(), key_scale_host.data(), slots);
    const std::vector<double> decoded_values =
        decode_side(profile.storage, false, value_code_host.data(), value_scale_host.data(), slots);

    const std::int32_t key_count = slots + extra;
    const auto key_position      = [&](int key) {
        return key < slots ? slot_position[static_cast<std::size_t>(key)] : slots + (key - slots);
    };
    const auto bf16_at = [](const std::vector<float>& rows, int d, int head, int token) {
        return static_cast<double>(
            rows[static_cast<std::size_t>(d) +
                 static_cast<std::size_t>(kHeadDim) * (head + kHeads * token)]);
    };
    std::vector<double> expected(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens, 0.0);
    naive_dense_softmax_attention(
        geometry, tokens, key_count, static_cast<double>(kAttentionScale),
        [&](int d, int head, int token) {
            return static_cast<double>(
                q[static_cast<std::size_t>(d) +
                  static_cast<std::size_t>(kHeadDim) * (head + kQueryHeads * token)]);
        },
        [&](int d, int head, int key) {
            return key < slots ? decoded_keys[static_cast<std::size_t>(d) +
                                              static_cast<std::size_t>(kHeadDim) *
                                                  static_cast<std::size_t>(head + kHeads * key)]
                               : bf16_at(pass_keys, d, head, key - slots);
        },
        [&](int d, int head, int key) {
            return key < slots ? decoded_values[static_cast<std::size_t>(d) +
                                                static_cast<std::size_t>(kHeadDim) *
                                                    static_cast<std::size_t>(head + kHeads * key)]
                               : bf16_at(pass_values, d, head, key - slots);
        },
        [&](int query, int key) {
            const std::int32_t pq = query_position[static_cast<std::size_t>(query)];
            const std::int32_t pk = key_position(key);
            return pk <= pq && static_cast<std::int64_t>(pq) - pk < kWindow;
        },
        [&](int d, int head, int token, double value) {
            expected[static_cast<std::size_t>(d) +
                     static_cast<std::size_t>(kHeadDim) *
                         static_cast<std::size_t>(head + kQueryHeads * token)] = value;
        });

    const auto got = from_device_bf16(device_out.data(),
                                      static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    int failures   = 0;
    failures += verify_reduction(std::string(profile.name) + " attention " + ring.label, got,
                                 expected, profile.attention);
    failures += device_key_codes.verify_guards("dense_kv_ring_encode key codes");
    failures += device_key_scales.verify_guards("dense_kv_ring_encode key scales");
    failures += device_value_codes.verify_guards("dense_kv_ring_encode value codes");
    failures += device_value_scales.verify_guards("dense_kv_ring_encode value scales");
    failures += device_ring_keys.verify_guards("dense_kv_ring_decode attention");
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
    for (const Profile& profile : kProfiles) {
        failures += round_trip(profile, 8301U);
        // A full window behind a wide pass: the ring's oldest slot falls just outside the window
        // while the pass's own keys are its second key set.
        failures +=
            attention_oracle(profile, {"full ring behind a wide pass", 1024, 4, false}, 8302U);
        // A narrow pass that writes first: its own row lands at the oldest slot and the ring's
        // positions wrap.
        failures +=
            attention_oracle(profile, {"wrapped ring behind a narrow pass", 1024, 1, true}, 8303U);
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_kv_quant\n";
    return failures ? 1 : 0;
}
