#pragma once

// An int4 group-32 row matrix (q4_g32_fp16_rows in row_group_v1) built on the host: its codes,
// binary16 scales, the payload exactly as the layout stores it, an FP64 reading of each value that
// shares no decoder with the Ops, and a device Weight view.
#include "core/weight.h"
#include "ops/quantized_weight.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ninfer::test {

// An int4 group-32 row bank in row_group_v1: codes in [-8,7] stored as code + 8, one binary16
// scale per 32 elements of a row. The oracle reads it through its own decode below.
struct Q4RowsBank {
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::vector<std::int8_t> codes;    // [n, k]
    std::vector<std::uint16_t> scales; // [n, k / 32], binary16

    double value(std::int32_t row, std::int32_t column) const {
        return static_cast<double>(codes[static_cast<std::size_t>(row) * k + column]) *
               static_cast<double>(quantized_weight::detail::f16_to_f32(
                   scales[static_cast<std::size_t>(row) * (k / 32) + column / 32]));
    }

    std::size_t scale_offset() const {
        return (static_cast<std::size_t>(n) * k / 2 + 255) / 256 * 256;
    }

    std::vector<std::uint8_t> payload() const {
        std::vector<std::uint8_t> bytes(scale_offset() + scales.size() * 2, 0);
        for (std::size_t i = 0; i < codes.size(); i += 2) {
            bytes[i / 2] = static_cast<std::uint8_t>((codes[i] + 8) | ((codes[i + 1] + 8) << 4));
        }
        std::memcpy(bytes.data() + scale_offset(), scales.data(), scales.size() * 2);
        return bytes;
    }

    Weight device_weight(void* device_payload, std::size_t bytes) const {
        Weight w;
        w.payload       = device_payload;
        w.payload_bytes = bytes;
        w.qtype         = QType::Q4_G32_FP16_ROWS;
        w.layout        = QuantLayout::RowGroup;
        w.group_size    = 32;
        w.group         = 32;
        w.scale_dtype   = DType::FP16;
        w.ndim          = 2;
        w.shape[0] = w.padded_shape[0] = n;
        w.shape[1] = w.padded_shape[1] = k;
        w.n                            = n;
        w.k                            = k;
        w.qdata                        = device_payload;
        w.scales = static_cast<std::uint8_t*>(device_payload) + scale_offset();
        return w;
    }
};

Q4RowsBank make_q4_rows_bank(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    Q4RowsBank bank;
    bank.n = n;
    bank.k = k;
    bank.codes.resize(static_cast<std::size_t>(n) * k);
    bank.scales.resize(static_cast<std::size_t>(n) * (k / 32));
    std::uint32_t state = seed * 2654435761U + 1U;
    auto next           = [&] {
        state = state * 1664525U + 1013904223U;
        return state >> 8;
    };
    for (auto& code : bank.codes) { code = static_cast<std::int8_t>(static_cast<int>(next() % 16) - 8); }
    for (auto& scale : bank.scales) {
        // Learned-looking steps across two octaves, and some that are not powers of two.
        scale = quantized_weight::detail::f32_to_f16(0.004F + 0.0001F * static_cast<float>(next() % 80));
    }
    return bank;
}


} // namespace ninfer::test
