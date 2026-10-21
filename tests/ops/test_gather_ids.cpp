// gather_ids: out[t] = table[index[t]]. The oracle is a host array read, and the test also checks
// that the Op leaves its inputs untouched and stays inside its buffers.
#include "ninfer/ops/gather_ids.h"

#include "core/device.h"

#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(std::int32_t table_rows, std::int32_t count, std::uint32_t seed, bool reject) {
    std::mt19937 rng(seed);
    std::vector<std::int32_t> table(static_cast<std::size_t>(table_rows));
    for (std::int32_t row = 0; row < table_rows; ++row) {
        table[static_cast<std::size_t>(row)] = row * 7 - 1000;
    }
    std::vector<std::int32_t> index(static_cast<std::size_t>(count));
    std::vector<std::int32_t> expected(static_cast<std::size_t>(count));
    for (std::int32_t t = 0; t < count; ++t) {
        const std::int32_t row                = static_cast<std::int32_t>(rng() % table_rows);
        index[static_cast<std::size_t>(t)]    = row;
        expected[static_cast<std::size_t>(t)] = table[static_cast<std::size_t>(row)];
    }

    GuardedDeviceBuffer device_table(table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_index(index.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(count) * sizeof(std::int32_t));
    device_table.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
    device_index.copy_from_host(index.data(), index.size() * sizeof(std::int32_t));
    device_out.fill(0xcd);

    Tensor table_tensor(device_table.data(), DType::I32, {table_rows});
    Tensor index_tensor(device_index.data(), DType::I32, {count});
    Tensor out_tensor(device_out.data(), DType::I32, {count});

    const std::string label =
        "gather_ids table=" + std::to_string(table_rows) + " count=" + std::to_string(count);
    int failures = 0;
    if (reject) {
        Tensor bad_index(device_index.data(), DType::BF16, {count});
        try {
            ops::gather_ids(bad_index, table_tensor, out_tensor, count, table_rows, nullptr);
            std::cout << "FAIL: " << label << " accepted a non-I32 index\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        return failures;
    }

    ops::gather_ids(index_tensor, table_tensor, out_tensor, count, table_rows, nullptr);
    cuda_synchronize();

    const auto actual =
        from_device<std::int32_t>(device_out.data(), static_cast<std::size_t>(count));
    const auto table_after =
        from_device<std::int32_t>(device_table.data(), static_cast<std::size_t>(table_rows));
    const auto index_after =
        from_device<std::int32_t>(device_index.data(), static_cast<std::size_t>(count));

    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves table").c_str(), table_after, table);
    failures += verify_exact((label + " preserves index").c_str(), index_after, index);
    failures += device_table.verify_guards((label + " table").c_str());
    failures += device_index.verify_guards((label + " index").c_str());
    failures += device_out.verify_guards((label + " out").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    failures += run_case(30057, 1, 1, false);
    failures += run_case(30057, 3, 2, false);
    failures += run_case(1, 4, 3, false);
    failures += run_case(248077, 2, 4, false);
    failures += run_case(30057, 3, 5, true);
    if (failures != 0) {
        std::cout << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gather_ids: all cases passed\n";
    return 0;
}
