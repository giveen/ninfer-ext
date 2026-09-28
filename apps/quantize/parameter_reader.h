#pragma once

// Enumerate the artifact's executable BF16 projections and read one at a time as FP32 host
// matrices. Pure host code: it never touches a device, so it is unit tested without one.

#include "artifact/reader.h"
#include "artifact/schema.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::quantize::app {

struct LinearParameter {
    std::string name;
    std::uint64_t n = 0; // stored rows = output channels
    std::uint64_t k = 0; // stored columns = input channels
    artifact::Binding binding;
};

// Every binding that is a rank-two contiguous BF16 matrix with both dimensions 128-aligned and at
// least one Use. That is exactly the set the EXL3 format can represent and the recipe will assign.
[[nodiscard]] std::vector<LinearParameter> enumerate_linear_parameters(const artifact::Reader& reader);

// The model executes a shared-input group as one contiguous parent. A LinearParent is one stored
// parent object whose every bound row range belongs to an eligible projection, with its members in
// row order. `key` is the first member's logical name, which names the quantizer's source entry.
struct LinearParent {
    std::string key;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    artifact::Binding binding; // the whole object, for reading
    std::vector<std::string> members;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges; // per member, element range
};

[[nodiscard]] std::vector<LinearParent> enumerate_linear_parents(const artifact::Reader& reader);

// Stored values in logical order, reshaped to [n][k] FP32 (row-major, n = shape[0]).
[[nodiscard]] std::vector<float> read_parameter_matrix(const artifact::Reader& reader,
                                                       const LinearParameter& parameter);

// [n][k] -> [k][n]: the quantizer wants rows on input channels.
[[nodiscard]] std::vector<float> transpose_to_kn(const std::vector<float>& nk, std::uint64_t n,
                                                 std::uint64_t k);

} // namespace ninfer::quantize::app
