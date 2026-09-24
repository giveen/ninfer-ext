// The n-gram (PLE) gather must produce the same BF16 embedding whether rows are faulted through
// the mapping or streamed with direct I/O. The synthetic table spans two volume files at unaligned
// offsets, so rows straddle 4 KiB pages and the second segment starts mid-page.
#include "models/qwen3_5/execution/ple_gather.h"

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

constexpr std::uint32_t kRows     = 5000;
constexpr std::uint32_t kRowWidth = 16;
constexpr std::uint32_t kEos      = 7;

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

} // namespace

int main() {
    // Two 2-gram heads and two 3-gram heads of 16 columns: a 64-wide embedding.
    PleConfig config;
    config.layer           = 1;
    config.ngram_size      = 3;
    config.heads_per_ngram = 2;
    config.embed_dim       = 4 * kRowWidth;
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

    // Object: FP8 codes [rows, 16], then BF16 row multipliers at a 256-byte aligned plane.
    const std::uint64_t scale_plane = (kRows * kRowWidth + 255) / 256 * 256;
    std::vector<std::byte> object(scale_plane + kRows * 2);
    std::mt19937 rng(1234);
    for (std::uint64_t i = 0; i < kRows * kRowWidth; ++i) {
        // Finite E4M3FN codes only (exclude the 0x7F / 0xFF NaN words).
        std::uint8_t code = static_cast<std::uint8_t>(rng());
        if ((code & 0x7F) == 0x7F) { code ^= 1; }
        object[i] = std::byte{code};
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const std::uint16_t scale = static_cast<std::uint16_t>(0x3C00 + (rng() % 0x0300));
        std::memcpy(object.data() + scale_plane + r * 2, &scale, 2);
    }

    // Split mid-table at a row boundary; place both pieces at unaligned file offsets.
    const std::uint64_t split = 1777 * kRowWidth;
    const auto dir            = std::filesystem::current_path() / "ple_gather_test_volumes";
    std::filesystem::create_directories(dir);
    write_file(dir / "a.bin", 1234, object.data(), split);
    write_file(dir / "b.bin", 777, object.data() + split, object.size() - split);

    execution::PleTable table;
    table.rows        = kRows;
    table.width       = kRowWidth;
    table.row_bytes   = kRowWidth;
    table.scale_plane = scale_plane;
    table.segments    = {
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
        // Mapped decode is independently the E4M3 row times its BF16 multiplier.
        std::vector<std::uint16_t> column(config.embed_dim);
        mapped.gather(tokens, 20, 1, column);
        std::vector<std::uint64_t> rows(config.heads());
        const std::int64_t context[3] = {tokens[20], tokens[19], tokens[18]};
        mapped.rows(context, rows);
        const auto e4m3 = [](std::uint8_t c) {
            const int e = (c >> 3) & 15, m = c & 7;
            const float v = e == 0 ? std::ldexp(m / 8.0F, -6) : std::ldexp(1.0F + m / 8.0F, e - 7);
            return (c & 0x80) ? -v : v;
        };
        for (std::uint32_t h = 0; h < config.heads(); ++h) {
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, object.data() + scale_plane + rows[h] * 2, 2);
            float scale;
            const std::uint32_t scale_word = static_cast<std::uint32_t>(scale_bits) << 16;
            std::memcpy(&scale, &scale_word, 4);
            for (std::uint32_t i = 0; i < kRowWidth; ++i) {
                const float value =
                    e4m3(static_cast<std::uint8_t>(object[rows[h] * kRowWidth + i])) * scale;
                std::uint32_t bits;
                std::memcpy(&bits, &value, 4);
                const auto bf16 =
                    static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
                if (column[h * kRowWidth + i] != bf16) {
                    std::cerr << "n-gram row decode mismatch at head " << h << '\n';
                    ++failures;
                    break;
                }
            }
        }
    } catch (const std::runtime_error& error) {
        // O_DIRECT is unsupported on some filesystems (tmpfs); that is an environment limit.
        std::cerr << "SKIP: " << error.what() << '\n';
        std::filesystem::remove_all(dir);
        return 77;
    }
    std::filesystem::remove_all(dir);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " PLE gather mapped/stream equivalence\n";
    return failures == 0 ? 0 : 1;
}
