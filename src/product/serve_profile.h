#pragma once

// Readers for a `ninfer_serve_profile` document written by ninfer-optimizer. Shared by the serving
// product and the one-shot CLI so `--profile` means the same file in both.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::product {

// One option from a profile: a bare flag name (no leading dashes) and, for valueless flags such as
// `--vision`, no value. Boolean false options are omitted entirely.
struct ProfileOption {
    std::string flag;
    std::string value;
    bool has_value = false;
};

// Read a profile. Throws std::invalid_argument with a human-readable reason on any problem.
[[nodiscard]] inline std::vector<ProfileOption> read_serve_profile(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) { throw std::invalid_argument("cannot read --profile file: " + path); }
    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("--profile is not valid JSON: " + std::string(error.what()));
    }
    if (document.value("artifact_type", std::string()) != "ninfer_serve_profile") {
        throw std::invalid_argument("--profile is not a ninfer_serve_profile: " + path);
    }
    if (document.value("schema_version", 0) != 1) {
        throw std::invalid_argument("--profile has an unsupported schema_version: " + path);
    }
    const auto options = document.find("options");
    if (options == document.end() || !options->is_object()) {
        throw std::invalid_argument("--profile has no options object: " + path);
    }
    std::vector<ProfileOption> result;
    for (const auto& [key, value] : options->items()) {
        if (key.empty() || key.rfind("--", 0) == 0) {
            throw std::invalid_argument(
                "--profile option names omit the leading dashes, got: " + key);
        }
        ProfileOption option;
        option.flag = key;
        if (value.is_boolean()) {
            if (value.get<bool>()) { result.push_back(std::move(option)); }
        } else if (value.is_string()) {
            option.value     = value.get<std::string>();
            option.has_value = true;
            result.push_back(std::move(option));
        } else if (value.is_number_integer()) {
            option.value     = std::to_string(value.get<std::int64_t>());
            option.has_value = true;
            result.push_back(std::move(option));
        } else if (value.is_number_unsigned()) {
            option.value     = std::to_string(value.get<std::uint64_t>());
            option.has_value = true;
            result.push_back(std::move(option));
        } else {
            throw std::invalid_argument(
                "--profile option " + key + " has an unsupported value type");
        }
    }
    return result;
}

} // namespace ninfer::product
