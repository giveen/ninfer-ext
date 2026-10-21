// ninfer::ops - gather_ids wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/gather_ids.h"

#include "ops/launcher/gather_ids.h" // detail::gather_ids_launch

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_row(const Tensor& t, const char* label) {
    if (t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("gather_ids: ") + label + " must be rank-1");
    }
}

} // namespace

void gather_ids(const Tensor& index, const Tensor& table, Tensor& out, std::int32_t count,
                std::int32_t table_rows, cudaStream_t stream) {
    if (index.dtype != DType::I32 || table.dtype != DType::I32 || out.dtype != DType::I32) {
        throw std::invalid_argument("gather_ids: index/table/out must be I32");
    }
    require_row(index, "index");
    require_row(table, "table");
    require_row(out, "out");
    if (count < 0) { throw std::invalid_argument("gather_ids: count must be nonnegative"); }
    if (count == 0) { return; }
    if (index.ne[0] != count || out.ne[0] != count) {
        throw std::invalid_argument("gather_ids: index/out extent must equal count");
    }
    if (table_rows <= 0 || table.ne[0] != table_rows) {
        throw std::invalid_argument("gather_ids: table rows must equal table_rows");
    }
    if (!index.is_contiguous() || !table.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("gather_ids: index/table/out must be contiguous");
    }
    if (index.data == nullptr || table.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("gather_ids: data must be non-null");
    }
    if (overlaps(index, out) || overlaps(table, out)) {
        throw std::invalid_argument("gather_ids: out must not overlap its inputs");
    }
    detail::gather_ids_launch(index, table, out, count, stream);
}

} // namespace ninfer::ops
