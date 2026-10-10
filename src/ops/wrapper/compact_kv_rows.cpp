// ninfer::ops - compact_kv_rows wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/compact_kv_rows.h"

#include "ops/launcher/compact_kv_rows.h" // detail::compact_kv_rows_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("compact_kv_rows: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("compact_kv_rows: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void compact_kv_rows(const Tensor& v, const Tensor& k, Tensor& out, std::int32_t rotary_dim,
                     std::int32_t rotary_pairs, cudaStream_t stream) {
    if (v.dtype != DType::BF16 || k.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("compact_kv_rows: v/k/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (v.ne[d] != k.ne[d]) {
            throw std::invalid_argument("compact_kv_rows: v/k shapes must match");
        }
    }
    if (v.ne[0] != rotary_dim) {
        throw std::invalid_argument("compact_kv_rows: rotary_dim must equal v.ne[0]");
    }
    if (rotary_pairs <= 0 || rotary_pairs > rotary_dim / 2) {
        throw std::invalid_argument("compact_kv_rows: rotary_pairs must be in [1,rotary_dim/2]");
    }
    if (out.ne[0] != rotary_dim + 2 * rotary_pairs || out.ne[1] != v.ne[1] ||
        out.ne[2] != v.ne[2] || out.ne[3] != v.ne[3]) {
        throw std::invalid_argument(
            "compact_kv_rows: out must be [rotary_dim+2*rotary_pairs,v.ne[1],v.ne[2],v.ne[3]]");
    }
    if (numel_allow_zero(out, "out") == 0) { return; }
    (void)numel_allow_zero(v, "v");
    (void)numel_allow_zero(k, "k");
    if (!v.is_contiguous() || !k.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("compact_kv_rows: v/k/out must be contiguous");
    }
    if (v.data == nullptr || k.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("compact_kv_rows: v/k/out data must be non-null");
    }

    detail::compact_kv_rows_launch(v, k, out, rotary_dim, rotary_pairs, stream);
}

} // namespace ninfer::ops
