// ninfer::ops - PLE wrappers: validate the public contract, then launch.
#include "ninfer/ops/ple.h"

#include "ops/ple/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous non-empty " +
                (dtype == DType::I32    ? "I32"
                 : dtype == DType::FP32 ? "FP32"
                                        : "BF16"));
}

void require_lanes(const Tensor& slots, std::int32_t lanes, const char* op, const char* label) {
    require_dense(slots, DType::I32, op, label);
    require(slots.numel() == lanes, op, std::string(label) + " must hold one slot per lane");
}

} // namespace

void ngram_hash_rows(const Tensor& tokens, const Tensor& history, const Tensor& source_slots,
                     const NgramHashTable& table, Tensor& rows, cudaStream_t stream) {
    constexpr const char* op = "ngram_hash_rows";
    require(table.ngram_size >= 2 && table.ngram_size <= kNgramMaximumSize &&
                table.heads_per_ngram > 0 && table.heads() <= kNgramMaximumHeads,
            op, "n-gram geometry exceeds the supported table");
    require_dense(tokens, DType::I32, op, "tokens");
    require(tokens.ne[2] == 1 && tokens.ne[3] == 1, op, "tokens must be [W,B]");
    require_dense(history, DType::I32, op, "history");
    require(history.ne[0] == table.ngram_size - 1, op, "history rows must be ngram_size-1");
    require_lanes(source_slots, tokens.ne[1], op, "source_slots");
    require_dense(rows, DType::I32, op, "rows");
    require(rows.ne[0] == table.heads() && rows.numel() == tokens.numel() * table.heads(), op,
            "rows must be [heads,W,B]");
    detail::ngram_hash_rows_launch(tokens, history, source_slots, table, rows, stream);
}

void ngram_history_advance(const Tensor& tokens, const Tensor* valid_columns, Tensor& history,
                           const Tensor& source_slots, const Tensor& destination_slots,
                           cudaStream_t stream) {
    constexpr const char* op = "ngram_history_advance";
    require_dense(tokens, DType::I32, op, "tokens");
    require_dense(history, DType::I32, op, "history");
    require(history.ne[0] >= 1 && history.ne[0] < kNgramMaximumSize, op, "invalid history rows");
    require_lanes(source_slots, tokens.ne[1], op, "source_slots");
    require_lanes(destination_slots, tokens.ne[1], op, "destination_slots");
    if (valid_columns != nullptr) { require_lanes(*valid_columns, tokens.ne[1], op, "valid"); }
    detail::ngram_history_advance_launch(tokens, valid_columns, history, source_slots,
                                         destination_slots, stream);
}

void gather_scaled_fp8_rows(const Weight& table, const Tensor& rows, Tensor& out,
                            cudaStream_t stream) {
    constexpr const char* op = "gather_scaled_fp8_rows";
    require(table.qtype == QType::FP8_E4M3FN_ROW_BF16 && table.qdata != nullptr &&
                table.scales != nullptr && table.n > 0 && table.k > 0,
            op, "table must be a complete row-scaled FP8 weight");
    require_dense(rows, DType::I32, op, "rows");
    require(rows.ne[2] == 1 && rows.ne[3] == 1, op, "rows must be [heads,T]");
    require_dense(out, DType::BF16, op, "out");
    require(out.ne[0] == rows.ne[0] * table.k && out.ne[1] == rows.ne[1], op,
            "out must be [heads*width,T]");
    detail::gather_scaled_fp8_rows_launch(table, rows, out, stream);
}

void ple_gate(const Tensor& key, const Tensor& query, const Tensor& value, std::int32_t streams,
              Tensor& gated, cudaStream_t stream) {
    constexpr const char* op = "ple_gate";
    require(streams > 0 && streams <= 16, op, "streams must be in [1,16]");
    for (const Tensor* t : {&key, &query, static_cast<const Tensor*>(&gated)}) {
        require_dense(*t, DType::BF16, op, "wide operand");
        require(t->ne[0] == value.ne[0] * streams && t->ne[1] == value.ne[1], op,
                "wide operands must be [S*H,T]");
    }
    require_dense(value, DType::BF16, op, "value");
    detail::ple_gate_launch(key, query, value, streams, gated, stream);
}

void ple_dilated_conv(const Tensor& normed, const Tensor& gated, const Tensor& weight,
                      std::int32_t dilation, const Tensor& states, const Tensor& source_slots,
                      Tensor& residual, cudaStream_t stream) {
    constexpr const char* op = "ple_dilated_conv";
    for (const Tensor* t : {&normed, &gated, static_cast<const Tensor*>(&residual)}) {
        require_dense(*t, DType::BF16, op, "column operand");
        require(t->ne[0] == normed.ne[0] && t->ne[1] == normed.ne[1] && t->ne[2] == normed.ne[2],
                op, "column operands must be [C,W,B]");
    }
    require_dense(weight, DType::BF16, op, "weight");
    require(weight.ne[0] == normed.ne[0] && weight.ne[1] >= 2 && weight.ne[2] == 1, op,
            "weight must be [K,C] stored as ne={C,K}");
    const std::int32_t history = (weight.ne[1] - 1) * dilation;
    require(dilation > 0 && history <= 32, op, "history must be in [1,32] columns");
    require_dense(states, DType::BF16, op, "states");
    require(states.ne[0] == history * normed.ne[0], op, "states must be [history*C, S]");
    require_lanes(source_slots, normed.ne[2], op, "source_slots");
    detail::ple_dilated_conv_launch(normed, gated, weight, dilation, states, source_slots, residual,
                                    stream);
}

void ple_conv_advance(const Tensor& normed, const Tensor* valid_columns, std::int32_t history,
                      Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                      cudaStream_t stream) {
    constexpr const char* op = "ple_conv_advance";
    require_dense(normed, DType::BF16, op, "normed");
    require_dense(states, DType::BF16, op, "states");
    require(history > 0 && history <= 32 && states.ne[0] == history * normed.ne[0], op,
            "states must be [history*C, S]");
    require_lanes(source_slots, normed.ne[2], op, "source_slots");
    require_lanes(destination_slots, normed.ne[2], op, "destination_slots");
    if (valid_columns != nullptr) { require_lanes(*valid_columns, normed.ne[2], op, "valid"); }
    detail::ple_conv_advance_launch(normed, valid_columns, history, states, source_slots,
                                    destination_slots, stream);
}

} // namespace ninfer::ops
