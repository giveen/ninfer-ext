#include "models/qwen3_5/execution/ple_gather.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

// Columns per host worker before the gather fans out; page faults dominate a cold table.
constexpr std::size_t kColumnsPerWorker = 64;
constexpr std::uint32_t kMaximumNgram   = 8;

float e4m3fn(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xF;
    const int mantissa = code & 0x7;
    float magnitude    = 0.0F;
    if (exponent == 0xF && mantissa == 0x7) {
        magnitude = std::nanf("");
    } else if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    } else {
        magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    }
    return sign != 0 ? -magnitude : magnitude;
}

float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if (std::isnan(value)) { return static_cast<std::uint16_t>((bits >> 16) | 0x40U); }
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

} // namespace

PleGather::PleGather(const PleTable& table, const PleConfig& config)
    : table_(table), config_(config), width_(config.embed_dim), row_width_(config.row_width()) {
    if (config.ngram_size < 2 || config.ngram_size > kMaximumNgram ||
        config.multipliers.size() != config.ngram_size ||
        config.head_moduli.size() != config.heads() ||
        config.head_offsets.size() != config.heads() || row_width_ == 0 ||
        static_cast<std::uint64_t>(row_width_) * config.heads() != width_) {
        throw std::invalid_argument("PLE gather configuration is inconsistent");
    }
    if (table.rows != config.table_rows || table.width != static_cast<std::int32_t>(row_width_) ||
        table.row_bytes < row_width_ || table.segments.empty()) {
        throw std::invalid_argument("PLE table does not match its configuration");
    }
    for (std::uint32_t head = 0; head < config.heads(); ++head) {
        if (config.head_offsets[head] + config.head_moduli[head] > table.rows) {
            throw std::invalid_argument("PLE hash head exceeds the table");
        }
    }
    for (int code = 0; code < 256; ++code) {
        e4m3_[static_cast<std::size_t>(code)] = e4m3fn(static_cast<std::uint8_t>(code));
    }
}

const std::byte* PleGather::locate(std::uint64_t object_offset, std::uint64_t bytes) const {
    const auto it =
        std::upper_bound(table_.segments.begin(), table_.segments.end(), object_offset,
                         [](std::uint64_t offset, const artifact::MappedObjectSegment& segment) {
                             return offset < segment.object_offset;
                         });
    if (it == table_.segments.begin()) { throw std::out_of_range("PLE table offset is unmapped"); }
    const auto& segment        = *std::prev(it);
    const std::uint64_t within = object_offset - segment.object_offset;
    if (within + bytes > segment.bytes) {
        throw std::out_of_range("PLE table row crosses a mapped segment");
    }
    return segment.data + within;
}

void PleGather::rows(std::span<const std::int64_t> context, std::span<std::uint64_t> out) const {
    if (context.size() != config_.ngram_size || out.size() != config_.heads()) {
        throw std::invalid_argument("PLE row query has the wrong extent");
    }
    std::uint32_t head = 0;
    for (std::uint32_t n = 2; n <= config_.ngram_size; ++n) {
        // Token ids and multipliers keep every product below 2^63, so the XOR is non-negative.
        std::uint64_t mixed = 0;
        for (std::uint32_t k = 0; k < n; ++k) {
            mixed ^= static_cast<std::uint64_t>(context[k]) * config_.multipliers[k];
        }
        for (std::uint32_t j = 0; j < config_.heads_per_ngram; ++j, ++head) {
            out[head] = mixed % config_.head_moduli[head] + config_.head_offsets[head];
        }
    }
}

void PleGather::column(std::span<const std::int32_t> tokens, std::size_t position,
                       std::uint16_t* out) const {
    const auto eos = static_cast<std::int64_t>(config_.eos_token_id);
    std::array<std::int64_t, kMaximumNgram> context{};
    context[0]  = tokens[position];
    bool closed = false;
    for (std::uint32_t s = 1; s < config_.ngram_size; ++s) {
        if (!closed && s <= position) {
            const std::int64_t token = tokens[position - s];
            if (token == eos) { closed = true; }
            context[s] = closed ? eos : token;
        } else {
            context[s] = eos;
        }
    }
    std::array<std::uint64_t, 32> rows_buffer{};
    const std::span<std::uint64_t> row_ids(rows_buffer.data(), config_.heads());
    rows(std::span<const std::int64_t>(context.data(), config_.ngram_size), row_ids);
    for (std::uint32_t head = 0; head < config_.heads(); ++head) {
        const std::uint64_t row = row_ids[head];
        const std::byte* codes  = locate(row * table_.row_bytes, row_width_);
        std::uint16_t scale_word;
        std::memcpy(&scale_word, locate(table_.scale_plane + row * 2U, 2), 2);
        const float scale          = bf16_to_float(scale_word);
        std::uint16_t* destination = out + static_cast<std::size_t>(head) * row_width_;
        for (std::uint32_t i = 0; i < row_width_; ++i) {
            destination[i] = float_to_bf16(e4m3_[static_cast<std::uint8_t>(codes[i])] * scale);
        }
    }
}

void PleGather::gather(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count,
                       std::span<std::uint16_t> out) const {
    if (first > tokens.size() || count > tokens.size() - first ||
        out.size() < static_cast<std::size_t>(width_) * count) {
        throw std::invalid_argument("PLE gather range is invalid");
    }
    if (config_.heads() > 32) { throw std::invalid_argument("PLE gather supports 32 heads"); }
    const auto run = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            column(tokens, first + i, out.data() + i * width_);
        }
    };
    const std::size_t hardware = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t workers =
        std::min<std::size_t>(std::min<std::size_t>(hardware, 16), count / kColumnsPerWorker);
    if (workers <= 1) {
        run(0, count);
        return;
    }
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    const std::size_t share = (count + workers - 1) / workers;
    for (std::size_t w = 1; w < workers; ++w) {
        const std::size_t begin = std::min(count, w * share);
        const std::size_t end   = std::min(count, begin + share);
        threads.emplace_back([&, begin, end] { run(begin, end); });
    }
    run(0, std::min(count, share));
}

} // namespace ninfer::models::qwen3_5::execution
