// CPU-only test for the packed calibration-trace reader.

#include "source_writer.h"
#include "trace_reader.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace app = ninfer::quantize::app;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

template <typename Exception, typename Operation>
void expect_throws(Operation operation, const char* message) {
    try {
        operation();
    } catch (const Exception&) { return; }
    expect(false, message);
}

std::filesystem::path scratch() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("ninfer-exl3-trace-" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

std::vector<std::byte> i64_bytes(const std::vector<std::int64_t>& values) {
    std::vector<std::byte> out(values.size() * sizeof(std::int64_t));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

void test_trace_round_trip() {
    const std::filesystem::path dir = scratch();
    std::vector<app::SafetensorsTensor> tensors;
    tensors.push_back({"input_ids", "I64", {2, 4}, i64_bytes({1, 2, 3, 4, 5, 6, 0, 0})});
    tensors.push_back({"lengths", "I64", {2}, i64_bytes({4, 2})});
    app::write_safetensors(dir / "trace.safetensors", tensors);

    const app::CalibrationTrace trace = app::read_calibration_trace(dir / "trace.safetensors");
    expect(trace.rows == 2 && trace.row_tokens == 4, "trace shape is wrong");
    expect(trace.input_ids == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 0, 0},
           "trace token ids are wrong");
    expect(trace.lengths == std::vector<std::int32_t>{4, 2}, "trace lengths are wrong");

    std::vector<app::SafetensorsTensor> bad;
    bad.push_back({"input_ids", "I64", {2, 4}, i64_bytes({1, 2, 3, 4, 5, 6, 0, 0})});
    bad.push_back({"lengths", "I64", {2}, i64_bytes({4, 9})});
    app::write_safetensors(dir / "bad.safetensors", bad);
    expect_throws<std::invalid_argument>(
        [&] { (void)app::read_calibration_trace(dir / "bad.safetensors"); },
        "a length past the row width was accepted");

    app::write_safetensors(dir / "missing.safetensors",
                           {{"lengths", "I64", {2}, i64_bytes({1, 1})}});
    expect_throws<std::invalid_argument>(
        [&] { (void)app::read_calibration_trace(dir / "missing.safetensors"); },
        "a trace without input_ids was accepted");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    test_trace_round_trip();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
