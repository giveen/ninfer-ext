// Offline quantization sensitivity measurement for the EXL3 bit allocation.
//
// Scores a packed trace clean, then perturbs one EXL3 projection at a time with seeded Gaussian noise
// of a known relative Frobenius norm and re-scores. The loss is the token NLL from
// `Engine::score_tokens`, which is the cross-entropy H(data, model); its increase over the clean model
// is exactly the KLD increase, because the data entropy is constant. That is the quantity the
// reference measures over the full logits, and it needs no logits API here.
//
// The weight's per-element rms is read from the unquantized source: the noise is added to the decoded
// (rotated) values, so `sigma = rfn * rms` makes the perturbation's relative norm exactly `rfn`.

#include "artifact/reader.h"
#include "models/qwen3_5/execution/probe_registry.h"
#include "ninfer/engine.h"
#include "ops/linear/exl3/exl3_dispatch.h"
#include "parameter_reader.h"
#include "trace_reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace app = ninfer::quantize::app;

struct Options {
    std::filesystem::path artifact;
    std::filesystem::path source;
    std::filesystem::path trace;
    std::filesystem::path output;
    float rfn        = 0.29F;
    std::uint32_t rows = 0; // 0 selects every trace row
    int device       = 0;
    bool help        = false;
};

const char* usage() {
    return "usage: ninfer-sensitivity <model.ninfer> --source <bf16.ninfer> --trace FILE --out FILE\n"
           "       --source PATH   unquantized artifact, for each weight's rms\n"
           "       --trace FILE    packed calibration trace (input_ids/lengths safetensors)\n"
           "       --out FILE      JSON sensitivities\n"
           "       --rfn F         relative Frobenius norm of the injected noise (default 0.29)\n"
           "       --rows N        trace rows to score (default: every row)\n"
           "       --device N      CUDA device index (default 0)\n";
}

Options parse(int argc, char** argv) {
    Options options;
    const auto value = [&](int& index) {
        if (index + 1 >= argc) { throw std::invalid_argument("missing value"); }
        return std::string(argv[++index]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--source") {
            options.source = value(i);
        } else if (arg == "--trace") {
            options.trace = value(i);
        } else if (arg == "--out") {
            options.output = value(i);
        } else if (arg == "--rfn") {
            options.rfn = std::stof(value(i));
        } else if (arg == "--rows") {
            options.rows = static_cast<std::uint32_t>(std::stoul(value(i)));
        } else if (arg == "--device") {
            options.device = std::stoi(value(i));
        } else if (arg.starts_with("--")) {
            throw std::invalid_argument("unknown option " + arg);
        } else {
            options.artifact = arg;
        }
    }
    return options;
}

// Sums the token negative log-likelihood over the first `rows` trace rows.
double score_nll(ninfer::Engine& engine, const app::CalibrationTrace& trace, std::uint32_t rows) {
    double total = 0.0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::int32_t length = trace.lengths[row];
        if (length < 2) { continue; }
        std::vector<ninfer::TokenId> tokens(
            trace.input_ids.begin() + static_cast<std::ptrdiff_t>(row) * trace.row_tokens,
            trace.input_ids.begin() + static_cast<std::ptrdiff_t>(row) * trace.row_tokens + length);
        const std::vector<float> logprobs = engine.score_tokens(std::move(tokens), 1);
        for (const float logprob : logprobs) { total -= static_cast<double>(logprob); }
    }
    return total;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-sensitivity: %s\n%s", error.what(), usage());
        return 1;
    }
    if (options.help) {
        std::fputs(usage(), stdout);
        return 0;
    }
    if (options.artifact.empty() || options.source.empty() || options.trace.empty() ||
        options.output.empty()) {
        std::fputs(usage(), stderr);
        return 1;
    }

    try {
        const app::CalibrationTrace trace = app::read_calibration_trace(options.trace);
        const std::uint32_t rows =
            options.rows == 0 ? trace.rows : std::min(options.rows, trace.rows);
        if (rows == 0) { throw std::invalid_argument("trace is empty"); }

        // Per-tensor weight rms from the unquantized source.
        std::map<std::string, double> rms;
        {
            const ninfer::artifact::Reader reader(options.source);
            for (const app::LinearParameter& parameter : app::enumerate_linear_parameters(reader)) {
                const std::vector<float> matrix = app::read_parameter_matrix(reader, parameter);
                double square = 0.0;
                for (const float value : matrix) { square += static_cast<double>(value) * value; }
                rms[parameter.name] =
                    matrix.empty() ? 0.0 : std::sqrt(square / static_cast<double>(matrix.size()));
            }
        }

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = options.artifact;
        engine_options.purpose       = ninfer::EnginePurpose::CausalScoring;
        engine_options.device        = options.device;
        engine_options.max_context   = trace.row_tokens;
        ninfer::Engine engine(std::move(engine_options));

        const auto& targets = ninfer::models::qwen3_5::execution::probe_targets();
        if (targets.empty()) { throw std::runtime_error("artifact has no EXL3 projections"); }
        std::fprintf(stderr, "sensitivity: %zu targets, %u rows, rfn %.4g\n", targets.size(), rows,
                     static_cast<double>(options.rfn));

        const double clean = score_nll(engine, trace, rows);
        std::fprintf(stderr, "clean NLL %.6f over %u rows\n", clean, rows);

        nlohmann::json report = {
            {"artifact", options.artifact.string()},
            {"source", options.source.string()},
            {"rfn", options.rfn},
            {"rows", rows},
            {"clean_nll", clean},
            {"targets", nlohmann::json::array()},
        };
        std::size_t done = 0;
        for (const auto& target : targets) {
            const auto rms_it = rms.find(target.name);
            if (rms_it == rms.end() || rms_it->second <= 0.0) {
                std::fprintf(stderr, "  skip %s (no source rms)\n", target.name.c_str());
                continue;
            }
            auto& probe        = ninfer::ops::detail::exl3_weight_probe();
            probe.target       = target.qdata;
            probe.rfn          = options.rfn;
            probe.rms          = static_cast<float>(rms_it->second);
            probe.seed         = 0x5eed1234U;
            const double perturbed = score_nll(engine, trace, rows);
            probe              = {};
            const double kld   = perturbed - clean;
            report["targets"].push_back({
                {"name", target.name},
                {"n", target.n},
                {"k", target.k},
                {"numel", static_cast<std::uint64_t>(target.n) * target.k},
                {"rms", rms_it->second},
                {"rfn", options.rfn},
                {"nll", perturbed},
                {"kld", kld},
            });
            ++done;
            std::fprintf(stderr, "  [%zu/%zu] %s kld %.6g\n", done, targets.size(),
                         target.name.c_str(), kld);
        }

        std::ofstream out(options.output, std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot write " + options.output.string()); }
        out << report.dump(2) << "\n";
        std::fprintf(stderr, "wrote %s\n", options.output.string().c_str());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-sensitivity: %s\n", error.what());
        return 1;
    }
    return 0;
}
