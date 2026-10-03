// Offline EXL3 quantizer. Reads a BF16 `.ninfer` artifact, quantizes each eligible projection with
// `quantize_tensor`, and writes an `exl3_mul1` source store the converter imports:
//
//   <out>/exl3.safetensors   <name>.trellis U8[n/16,K/16,16*rate], <name>.su F32[K], <name>.sv F32[N]
//   <out>/report.json        per-parameter rate, proxy error and time
//
// This version takes per-parameter Hessians from `--hessians` (K*K raw FP32). The calibration
// stage that produces them runs the model forward; it is the next step. The app is offline: it
// owns its device allocations and is not linked into the Engine or serving.

#include "calibration.h"
#include "hessian_io.h"
#include "options.h"
#include "parameter_reader.h"
#include "source_writer.h"
#include "trace_reader.h"

#include "artifact/reader.h"
#include "quantize/exl3/pipeline.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace q    = ninfer::quantize::exl3;
namespace app  = ninfer::quantize::app;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

q::OutScales out_scales_mode(app::OutScaleMode mode) {
    switch (mode) {
    case app::OutScaleMode::Always: return q::OutScales::Always;
    case app::OutScaleMode::Never: return q::OutScales::Never;
    default: return q::OutScales::Auto;
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void resize(std::size_t count) {
        reset();
        if (count == 0) { return; }
        check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
        count_ = count;
    }
    void reset() noexcept {
        if (data_ != nullptr) { (void)cudaFree(data_); }
        data_  = nullptr;
        count_ = 0;
    }
    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    T* data_          = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
std::vector<std::byte> to_bytes(const std::vector<T>& values) {
    std::vector<std::byte> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

struct MatrixResult {
    std::vector<app::SafetensorsTensor> tensors; // <name>.trellis, .su, .sv
    nlohmann::json report;
};

// Quantize one [n][k] matrix (row-major, rows are output channels) against its Hessian.
MatrixResult quantize_matrix(const std::string& name, const std::vector<float>& nk,
                             std::uint64_t n, std::uint64_t k, const std::vector<float>& hessian,
                             int half_bits, std::uint64_t seed, q::OutScales out_scales,
                             const std::vector<std::string>& members) {
    const std::vector<float> kn = app::transpose_to_kn(nk, n, k);
    const std::uint64_t tile_states = (n / 16) * (k / 16) * 256;
    DeviceBuffer<float> d_w;
    DeviceBuffer<float> d_h;
    DeviceBuffer<float> d_su;
    DeviceBuffer<float> d_sv;
    DeviceBuffer<float> d_wq;
    std::vector<std::uint16_t> states(tile_states);
    d_w.resize(kn.size());
    d_h.resize(hessian.size());
    d_su.resize(k);
    d_sv.resize(n);
    d_wq.resize(kn.size());
    check(cudaMemcpy(d_w.get(), kn.data(), kn.size() * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy(w)");
    check(cudaMemcpy(d_h.get(), hessian.data(), hessian.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy(h)");

    q::TensorOptions tensor_options;
    tensor_options.bitrate_half_bits = half_bits;
    tensor_options.seed              = seed;
    tensor_options.out_scales        = out_scales;

    const auto start = std::chrono::steady_clock::now();
    const q::TensorReport tensor_report =
        q::quantize_tensor(d_w.get(), d_h.get(), static_cast<std::int64_t>(k),
                           static_cast<std::int64_t>(n), tensor_options, states.data(), d_su.get(),
                           d_sv.get(), d_wq.get(), nullptr);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::vector<float> su(k);
    std::vector<float> sv(n);
    check(cudaMemcpy(su.data(), d_su.get(), su.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy(su)");
    check(cudaMemcpy(sv.data(), d_sv.get(), sv.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy(sv)");

    std::vector<std::byte> trellis;
    trellis.reserve(static_cast<std::size_t>(tile_states / 256) * 16 * half_bits);
    for (std::size_t tile = 0; tile < tile_states / 256; ++tile) {
        const std::vector<std::byte> packed =
            app::pack_trellis_states(states.data() + tile * 256, half_bits);
        trellis.insert(trellis.end(), packed.begin(), packed.end());
    }
    MatrixResult result;
    result.tensors.push_back(app::SafetensorsTensor{
        name + ".trellis", "U8", {n / 16, k / 16, static_cast<std::uint64_t>(16 * half_bits)},
        std::move(trellis)});
    result.tensors.push_back(app::SafetensorsTensor{name + ".su", "F32", {k}, to_bytes(su)});
    result.tensors.push_back(app::SafetensorsTensor{name + ".sv", "F32", {n}, to_bytes(sv)});
    result.report = {
        {"name", name},
        {"members", members},
        {"half_bits", half_bits},
        {"proxy_error", tensor_report.proxy_error},
        {"proxy_error_before_refit", tensor_report.proxy_error_before_refit},
        {"global_scale", tensor_report.global_scale},
        {"out_scales", tensor_report.out_scales},
        {"damping_retries", tensor_report.damping_retries},
        {"refit_input_skipped", tensor_report.refit_input_skipped},
        {"seconds", seconds},
    };
    return result;
}

} // namespace

int main(int argc, char** argv) {
    app::QuantizeOptions options;
    try {
        options = app::parse_quantize_options(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-quantize: %s\n", error.what());
        std::fprintf(stderr, "%s", app::quantize_usage_text(argv[0]).c_str());
        return 1;
    }
    if (options.help) {
        std::fputs(app::quantize_usage_text(argv[0]).c_str(), stdout);
        return 0;
    }
    if (!options.rates.empty()) {
        try {
            std::ifstream rates_file(options.rates);
            if (!rates_file) {
                throw std::runtime_error("cannot read " + options.rates.string());
            }
            const nlohmann::json rates = nlohmann::json::parse(rates_file);
            for (const auto& [name, value] : rates.items()) {
                options.rate_overrides.emplace(name, value.get<int>());
            }
            std::fprintf(stderr, "loaded %zu per-tensor rates from %s\n",
                         options.rate_overrides.size(), options.rates.string().c_str());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "ninfer-quantize: rates: %s\n", error.what());
            return 1;
        }
    }

    try {
        const ninfer::artifact::Reader reader(options.artifact);
        const std::vector<app::LinearParent> parents = app::enumerate_linear_parents(reader);
        if (parents.empty()) {
            std::fprintf(stderr, "ninfer-quantize: no eligible BF16 projections\n");
            return 1;
        }
        if (options.list_only) {
            std::size_t members = 0;
            for (const auto& parent : parents) { members += parent.members.size(); }
            std::printf("eligible parents: %zu (%zu projections)\n", parents.size(), members);
            for (const auto& parent : parents) {
                std::printf("  %s [%llu, %llu] rate %d half bits (%zu members)\n",
                            parent.key.c_str(), static_cast<unsigned long long>(parent.n),
                            static_cast<unsigned long long>(parent.k),
                            app::bitrate_for_parameter(options, parent.key),
                            parent.members.size());
            }
            return 0;
        }

        std::filesystem::create_directories(options.output);
        check(cudaSetDevice(options.device), "cudaSetDevice");

        if (!options.trace.empty()) {
            const app::CalibrationTrace trace = app::read_calibration_trace(options.trace);
            const app::CalibrationResult calibration = app::calibrate(
                options.activation_model, options.artifact, trace, options.hessians, options.device,
                options.calibration_rows);
            std::fprintf(stderr,
                         "calibrated %u rows (%llu tokens): %llu Hessians, %llu names skipped\n",
                         calibration.rows,
                         static_cast<unsigned long long>(calibration.tokens),
                         static_cast<unsigned long long>(calibration.hessians),
                         static_cast<unsigned long long>(calibration.skipped_names));
        }

        if (options.calibrate_only) { return 0; }

        nlohmann::json report = {
            {"artifact", options.artifact.string()},
            {"bits_half", options.bits_half},
            {"head_bits_half", options.head_bits_half},
            {"hq", options.hq},
            {"tensors", nlohmann::json::array()},
        };

        std::vector<app::SafetensorsTensor> tensors;
        std::size_t done    = 0;
        std::size_t skipped = 0;
        for (const auto& parent : parents) {
            if (options.limit > 0 && done >= static_cast<std::size_t>(options.limit)) { break; }
            // Routed-expert banks are split into per-expert matrices below.
            if (parent.key.find("/moe/experts/") != std::string::npos) { continue; }
            if (!options.only.empty() && parent.key.find(options.only) == std::string::npos) { continue; }
            if (options.experts_only) { continue; }
            const int half_bits = app::bitrate_for_parameter(options, parent.key);
            const std::uint64_t n = parent.n;
            const std::uint64_t k = parent.k;
            const app::LinearParameter parameter{parent.key, n, k, parent.binding};
            const std::string stem = app::sanitize_parameter_name(parent.key);

            const std::vector<float> nk = app::read_parameter_matrix(reader, parameter);
            const std::filesystem::path hessian_path =
                options.hessians / (stem + ".h.f32");
            std::vector<float> hessian;
            if (std::filesystem::is_regular_file(hessian_path)) {
                hessian = app::read_hessian_f32(hessian_path, k);
            } else if (parent.key.starts_with("mtp/")) {
                // An MTP projection the calibration could not observe still has to be quantized:
                // leaving it at its stored precision makes the artifact unloadable, because BF16 has
                // no MTP geometry. Fall back to the uncalibrated identity Hessian.
                hessian = app::identity_hessian(k);
            } else {
                // No calibration Hessian for a Text projection: leave the parameter at its stored
                // precision rather than quantizing it without calibration.
                ++skipped;
                continue;
            }
            MatrixResult result =
                quantize_matrix(parent.key, nk, n, k, hessian, half_bits, options.seed + done,
                                out_scales_mode(options.out_scales), parent.members);
            for (auto& tensor : result.tensors) { tensors.push_back(std::move(tensor)); }
            const double proxy   = result.report["proxy_error"].get<double>();
            const double seconds = result.report["seconds"].get<double>();
            report["tensors"].push_back(std::move(result.report));
            ++done;
            std::fprintf(stderr, "[%zu/%zu] %s rate %d: proxy %g (%.1f s)\n", done, parents.size(),
                         parent.key.c_str(), half_bits, proxy, seconds);
        }

        // Routed experts: every expert gate, up and down is its own matrix (own su/sv), quantized
        // against the layer's shared block-input Hessian (gate, up) or the expert's own down-input
        // Hessian. One shard per stored bank keeps memory bounded and makes the run resumable.
        std::size_t expert_matrices = 0;
        double expert_proxy_sum     = 0.0;
        double expert_proxy_max     = 0.0;
        nlohmann::json expert_files = nlohmann::json::object();
        {
            std::string shared_prefix;
            std::vector<float> shared_hessian;
            for (const auto& bank : parents) {
                if (bank.key.find("/moe/experts/") == std::string::npos) { continue; }
                if (!options.only.empty() && bank.key.find(options.only) == std::string::npos) { continue; }
                if (options.limit > 0 && done + expert_matrices >=
                                             static_cast<std::size_t>(options.limit)) {
                    break;
                }
                const std::string shard_name =
                    "exl3-" + app::sanitize_parameter_name(bank.key) + ".safetensors";
                const std::filesystem::path shard   = options.output / shard_name;
                const std::filesystem::path sidecar = options.output / (shard_name + ".json");
                if (std::filesystem::is_regular_file(shard) &&
                    std::filesystem::is_regular_file(sidecar)) {
                    std::ifstream in(sidecar);
                    const nlohmann::json prior = nlohmann::json::parse(in);
                    for (const auto& name : prior["names"]) { expert_files[name.get<std::string>()] = shard_name; }
                    expert_matrices += prior["names"].size() / 3;
                    for (const auto& entry : prior["tensors"]) {
                        const double proxy = entry["proxy_error"].get<double>();
                        expert_proxy_sum += proxy;
                        expert_proxy_max = std::max(expert_proxy_max, proxy);
                    }
                    std::fprintf(stderr, "resume: %s already written\n", shard_name.c_str());
                    continue;
                }
                const auto& part = bank.binding.parts.front();
                const std::vector<std::byte> payload = reader.read_object(part.object);
                if (bank.binding.parts.size() != 1 || part.end * 2 > payload.size()) {
                    throw std::runtime_error(bank.key + ": expert bank is not one BF16 object");
                }
                std::vector<app::SafetensorsTensor> shard_tensors;
                nlohmann::json shard_reports = nlohmann::json::array();
                const std::uint64_t k        = bank.k;
                bool bank_complete           = true;
                for (std::size_t member = 0; member < bank.members.size(); ++member) {
                    if (options.limit > 0 && done + expert_matrices >=
                                                 static_cast<std::size_t>(options.limit)) {
                        bank_complete = false;
                        break;
                    }
                    const std::string& name = bank.members[member];
                    const auto [begin, end] = bank.ranges[member];
                    const std::uint64_t n   = (end - begin) / k;
                    if (n == 0 || n % 128 != 0 || k % 128 != 0 || (end - begin) % k != 0) {
                        throw std::runtime_error(name + ": expert matrix is not 128-aligned");
                    }
                    const bool is_down = name.ends_with("/down");
                    const std::string input_name =
                        is_down ? name
                                : name.substr(0, name.find("/moe/experts/")) + "/moe/input";
                    const std::filesystem::path hessian_path =
                        options.hessians / (app::sanitize_parameter_name(input_name) + ".h.f32");
                    std::vector<float> own_hessian;
                    const std::vector<float>* hessian = nullptr;
                    if (!std::filesystem::is_regular_file(hessian_path)) {
                        if (!name.starts_with("mtp/")) {
                            throw std::runtime_error(name + ": no calibration Hessian " +
                                                     hessian_path.string());
                        }
                        // MTP experts the calibration could not observe: uncalibrated identity.
                        own_hessian = app::identity_hessian(k);
                        hessian     = &own_hessian;
                    } else if (is_down) {
                        own_hessian = app::read_hessian_f32(hessian_path, k);
                        hessian     = &own_hessian;
                    } else {
                        if (shared_prefix != input_name) {
                            shared_hessian = app::read_hessian_f32(hessian_path, k);
                            shared_prefix  = input_name;
                        }
                        hessian = &shared_hessian;
                    }
                    std::vector<float> nk(static_cast<std::size_t>(end - begin));
                    for (std::uint64_t i = 0; i < end - begin; ++i) {
                        const std::size_t at = static_cast<std::size_t>((begin + i) * 2);
                        const std::uint32_t bits =
                            static_cast<std::uint32_t>(
                                std::to_integer<std::uint8_t>(payload[at]) |
                                (std::to_integer<std::uint8_t>(payload[at + 1]) << 8))
                            << 16;
                        std::memcpy(&nk[static_cast<std::size_t>(i)], &bits, sizeof(float));
                    }
                    const int half_bits = app::bitrate_for_parameter(options, name);
                    MatrixResult result = quantize_matrix(
                        name, nk, n, k, *hessian, half_bits,
                        options.seed + static_cast<std::uint64_t>(&bank - parents.data()) * 1000003ULL + member,
                        out_scales_mode(options.out_scales), {name});
                    for (auto& tensor : result.tensors) {
                        expert_files[tensor.name] = shard_name;
                        shard_tensors.push_back(std::move(tensor));
                    }
                    const double proxy = result.report["proxy_error"].get<double>();
                    const double result_seconds = result.report["seconds"].get<double>();
                    expert_proxy_sum += proxy;
                    expert_proxy_max = std::max(expert_proxy_max, proxy);
                    shard_reports.push_back(std::move(result.report));
                    ++expert_matrices;
                    if (expert_matrices % 64 == 0 || options.limit > 0) {
                        std::fprintf(stderr, "[experts %zu] %s rate %d: proxy %g (%.2f s)\n", expert_matrices,
                                     name.c_str(), half_bits, proxy, result_seconds);
                    }
                }
                if (shard_tensors.empty()) { continue; }
                app::write_safetensors(shard, shard_tensors);
                if (bank_complete) {
                    nlohmann::json names = nlohmann::json::array();
                    for (const auto& tensor : shard_tensors) { names.push_back(tensor.name); }
                    std::ofstream out(sidecar, std::ios::binary | std::ios::trunc);
                    out << nlohmann::json{{"names", names}, {"tensors", shard_reports}}.dump() << '\n';
                }
                std::fprintf(stderr, "wrote %s (%zu matrices)\n", shard_name.c_str(),
                             shard_tensors.size() / 3);
            }
        }

        if (!tensors.empty()) {
            app::write_safetensors(options.output / "exl3.safetensors", tensors);
            for (const auto& tensor : tensors) { expert_files[tensor.name] = "exl3.safetensors"; }
        }
        if (expert_matrices > 0) {
            // A directory of shards is one safetensors source through its index.
            std::ofstream index(options.output / "model.safetensors.index.json",
                                std::ios::binary | std::ios::trunc);
            index << nlohmann::json{{"metadata", nlohmann::json::object()},
                                    {"weight_map", expert_files}}
                         .dump(2)
                  << '\n';
            report["expert_matrices"]  = expert_matrices;
            report["expert_proxy_mean"] = expert_proxy_sum / static_cast<double>(expert_matrices);
            report["expert_proxy_max"]  = expert_proxy_max;
        }
        report["quantized"] = done;
        report["skipped"]   = skipped;
        std::ofstream out(options.output / "report.json", std::ios::binary | std::ios::trunc);
        out << report.dump(2) << '\n';
        std::fprintf(stderr, "wrote %s (%zu dense tensors, %zu expert matrices)\n",
                     options.output.string().c_str(), done, expert_matrices);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-quantize: %s\n", error.what());
        return 1;
    }
}
