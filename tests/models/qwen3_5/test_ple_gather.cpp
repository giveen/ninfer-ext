// The n-gram (PLE) gather must produce the same BF16 embedding whether rows are faulted through
// the mapping or streamed with direct I/O, for both table formats (row-scaled FP8 and 4-bit group
// codes). The synthetic table spans two volume files at unaligned offsets, so rows straddle 4 KiB
// pages and the second segment starts mid-page. Mapped decode is checked against an independent
// reconstruction of the stored words.
#include "models/qwen3_5/execution/ple_gather.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_5;

namespace {

constexpr std::uint32_t kRows = 5000;
constexpr std::uint32_t kEos  = 7;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

void write_file(const std::filesystem::path& path, std::uint64_t leading, const std::byte* data,
                std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::vector<char> pad(leading, 'x');
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    // Trailing bytes so the last direct page read is not short.
    const std::vector<char> tail(8192, 'y');
    out.write(tail.data(), static_cast<std::streamsize>(tail.size()));
}

std::uint16_t to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

float from_half(std::uint16_t word) {
    const float magnitude =
        ((word >> 10) & 0x1F) == 0
            ? std::ldexp(static_cast<float>(word & 0x3FF), -24)
            : std::ldexp(1.0F + static_cast<float>(word & 0x3FF) / 1024.0F,
                         static_cast<int>((word >> 10) & 0x1F) - 15);
    return (word & 0x8000) ? -magnitude : magnitude;
}

float e4m3(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0F, -6) : std::ldexp(1.0F + m / 8.0F, e - 7);
    return (c & 0x80) ? -v : v;
}

// One scenario: FP8 rows of 16 values or 4-bit rows of 32 values (one group).
int run(bool grouped) {
    const std::uint32_t row_width = grouped ? 32 : 16;
    // Two 2-gram heads and two 3-gram heads: a 4-head embedding.
    PleConfig config;
    config.layer           = 1;
    config.ngram_size      = 3;
    config.heads_per_ngram = 2;
    config.embed_dim       = 4 * row_width;
    config.conv_kernel     = 4;
    config.eos_token_id    = kEos;
    config.table_rows      = kRows;
    config.multipliers     = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    config.head_moduli     = {1201, 1213, 1217, 1223};
    std::uint64_t offset   = 0;
    for (const auto modulus : config.head_moduli) {
        config.head_offsets.push_back(offset);
        offset += modulus;
    }

    // Object: codes [rows, code_bytes] then the scale plane at a 256-byte aligned offset. FP8 stores
    // one BF16 row multiplier per row; the 4-bit table one binary16 scale per row here (one group).
    const std::uint64_t code_bytes  = grouped ? row_width / 2 : row_width;
    const std::uint64_t scale_bytes = 2;
    const std::uint64_t scale_plane = (kRows * code_bytes + 255) / 256 * 256;
    std::vector<std::byte> object(scale_plane + kRows * scale_bytes);
    std::mt19937 rng(grouped ? 4321 : 1234);
    for (std::uint64_t i = 0; i < kRows * code_bytes; ++i) {
        std::uint8_t code = static_cast<std::uint8_t>(rng());
        // FP8 must be finite E4M3FN (exclude the 0x7F / 0xFF NaN words); nibbles take any value.
        if (!grouped && (code & 0x7F) == 0x7F) { code ^= 1; }
        object[i] = std::byte{code};
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const std::uint16_t scale = grouped ? static_cast<std::uint16_t>(0x2000 + (rng() % 0x1800))
                                            : static_cast<std::uint16_t>(0x3C00 + (rng() % 0x0300));
        std::memcpy(object.data() + scale_plane + r * scale_bytes, &scale, 2);
    }

    // Split mid-table at a row boundary; place both pieces at unaligned file offsets.
    const std::uint64_t split = 1777 * code_bytes;
    const auto dir = std::filesystem::current_path() / (grouped ? "ple_gather_q4_volumes" : "ple_gather_test_volumes");
    std::filesystem::create_directories(dir);
    write_file(dir / "a.bin", 1234, object.data(), split);
    write_file(dir / "b.bin", 777, object.data() + split, object.size() - split);

    execution::PleTable table;
    table.rows            = kRows;
    table.width           = static_cast<std::int32_t>(row_width);
    table.row_bytes       = code_bytes;
    table.scale_plane     = scale_plane;
    table.scale_row_bytes = scale_bytes;
    table.format          = grouped ? QType::Q4_G32_FP16_ROWS : QType::FP8_E4M3FN_ROW_BF16;
    table.segments        = {
        {0, split, object.data(), dir / "a.bin", 1234},
        {split, object.size() - split, object.data() + split, dir / "b.bin", 777},
    };

    std::vector<std::int32_t> tokens(300);
    for (auto& token : tokens) { token = static_cast<std::int32_t>(rng() % 50000); }
    tokens[5] = tokens[140] = static_cast<std::int32_t>(kEos);

    int failures = 0;
    try {
        execution::PleGather mapped(table, config, false);
        execution::PleGather streamed(table, config, true);
        failures += check(!mapped.streamed() && streamed.streamed(), "residency flag mismatch");
        // A decode-sized call, a column after EOS, and a prefill-sized call.
        for (const auto [first, count] :
             {std::pair<std::size_t, std::size_t>{0, 1}, {6, 3}, {141, 1}, {10, 290}}) {
            std::vector<std::uint16_t> a(count * config.embed_dim), b(a.size());
            mapped.gather(tokens, first, count, a);
            streamed.gather(tokens, first, count, b);
            failures += check(a == b, "streamed n-gram rows differ from mapped rows");
        }
        // Mapped decode is independently the stored words times their scale.
        std::vector<std::uint16_t> column(config.embed_dim);
        mapped.gather(tokens, 20, 1, column);
        std::vector<std::uint64_t> rows(config.heads());
        const std::int64_t context[3] = {tokens[20], tokens[19], tokens[18]};
        mapped.rows(context, rows);
        for (std::uint32_t h = 0; h < config.heads(); ++h) {
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, object.data() + scale_plane + rows[h] * scale_bytes, 2);
            float scale;
            if (grouped) {
                scale = from_half(scale_bits);
            } else {
                const std::uint32_t scale_word = static_cast<std::uint32_t>(scale_bits) << 16;
                std::memcpy(&scale, &scale_word, 4);
            }
            const auto* codes = reinterpret_cast<const std::uint8_t*>(object.data() + rows[h] * code_bytes);
            for (std::uint32_t i = 0; i < row_width; ++i) {
                const float value =
                    grouped ? static_cast<float>(((i & 1) ? codes[i / 2] >> 4 : codes[i / 2] & 0xF) - 8) * scale
                            : e4m3(codes[i]) * scale;
                if (column[h * row_width + i] != to_bf16(value)) {
                    std::cerr << (grouped ? "4-bit" : "FP8") << " n-gram row decode mismatch at head " << h
                              << '\n';
                    ++failures;
                    break;
                }
            }
        }
    } catch (const std::runtime_error& error) {
        // O_DIRECT is unsupported on some filesystems (tmpfs); that is an environment limit.
        std::cerr << "SKIP: " << error.what() << '\n';
        std::filesystem::remove_all(dir);
        return -1;
    }
    std::filesystem::remove_all(dir);
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    for (const bool grouped : {false, true}) {
        const int result = run(grouped);
        if (result < 0) { return 77; }
        failures += result;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " PLE gather mapped/stream equivalence\n";
    return failures == 0 ? 0 : 1;
}
