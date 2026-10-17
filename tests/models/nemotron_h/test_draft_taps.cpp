// The draft's conditioning, on the real artifact.
//
// A LiLiCorr drafter cannot compute its own context: its `fc` consumes the target's residual stream
// at six tapped blocks, so whether this family can carry that draft at all is a question about what
// its forward publishes. This test runs the target over real text, captures those taps for every
// position, and dumps them beside the target's own greedy decisions -- which is what an acceptance
// probe needs, and what makes that probe possible before any draft code exists.
//
// Two things a dump is worthless without are checked here rather than assumed. The one-pass capture
// and the incremental one must agree bit for bit, which is what makes a tap the residual of the
// position it is filed under: the incremental arm is a normal decode, one token per pass, and the
// one-pass arm re-reads the whole sequence from a zero state. And every armed block must have
// published its residual, because a capture that silently misses a layer would look like a drafter
// that accepts nothing.
//
//   NINFER_NEMOTRON_ARTIFACT  the 45 GB artifact; unset skips (77)
//   NINFER_TAP_DUMP           a directory to write the dump into; unset runs the checks only
//   NINFER_TAP_TEXT           a UTF-8 file to use as the prompt; unset uses the built-in paragraph
//   NINFER_TAP_LAYERS         the tapped block ids, ascending (default 1,18,35,51,68,85)
//   NINFER_TAP_PROMPT_TOKENS  prompt length in tokens (default 512)
//   NINFER_TAP_GENERATED      tokens to decode greedily (default 128)

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/nemotron_h/frontend.h"
#include "models/nemotron_h/load.h"
#include "models/nemotron_h/program.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace nemotron = ninfer::models::nemotron_h;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) { return; }
    ++failures;
    std::cout << "FAIL: " << message << '\n';
}

// Long enough to fill the draft's 1024-position window with real text rather than a repeated
// sentence, and reading as ordinary prose so the target's own next-token distribution is not
// degenerate.
constexpr const char* kDefaultText =
    "The Nemotron hybrid stack that this engine serves is a mixture of three mixers rather than "
    "one "
    "attention: most of its blocks are Mamba-2 selective scans, most of the rest are latent "
    "mixture "
    "of experts layers whose routed experts live in host memory, and only eight of its eighty "
    "eight "
    "blocks attend over the sequence at all. That composition is what a speculative drafter has to "
    "condition on, and it is why the drafter does not read the target's attention cache but the "
    "residual stream itself, at six points along the way, projected into its own width. A drafter "
    "that reads the right records and still accepts almost nothing is not a drafter with a bug in "
    "its arithmetic; it is usually a drafter trained against a different post-trained checkpoint, "
    "or against a different quantization of the same one, and the only way to tell the difference "
    "is to measure the accepted length on the real pair rather than to trust the mechanism.";

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

void write_file(const std::filesystem::path& path, const void* bytes, std::size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot write " + path.string()); }
    output.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    if (!output) { throw std::runtime_error("cannot write " + path.string()); }
}

std::int32_t env_int(const char* name, std::int32_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    const int value = std::stoi(text);
    if (value <= 0) { throw std::runtime_error(std::string(name) + " must be positive"); }
    return value;
}

std::vector<std::uint32_t> parse_layers(const char* text) {
    std::vector<std::uint32_t> layers;
    std::istringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (field.empty()) { continue; }
        layers.push_back(static_cast<std::uint32_t>(std::stoul(field)));
    }
    if (layers.empty()) { throw std::runtime_error("NINFER_TAP_LAYERS names no block"); }
    return layers;
}

// One tap's columns out of the capture. `features` is `[layers * hidden, tokens]` with the token
// axis strided -- the composition's own residual order, channels contiguous -- so one tap's block
// is a 2D copy whose destination is packed: the result is `[tokens][hidden]`, token-major.
std::vector<std::uint16_t> tap_block(const ninfer::Tensor& features, std::int32_t index,
                                     std::int32_t hidden, std::int32_t tokens) {
    const std::size_t width = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    std::vector<std::uint16_t> block(static_cast<std::size_t>(hidden) *
                                     static_cast<std::size_t>(tokens));
    const auto* source =
        static_cast<const std::byte*>(features.data) + static_cast<std::size_t>(index) * width;
    const cudaError_t status =
        cudaMemcpy2D(block.data(), width, source, static_cast<std::size_t>(features.nb[1]), width,
                     static_cast<std::size_t>(tokens), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("reading a tap block: ") + cudaGetErrorString(status));
    }
    return block;
}

// The capture of one pass, appended at the position it belongs to: a later pass must never
// overwrite an earlier one's columns, which is what filing by `first_position` means.
void append_pass(const nemotron::Program& program, std::vector<std::vector<std::uint16_t>>& into,
                 std::int32_t hidden, std::vector<std::int32_t>& covered) {
    const nemotron::PassTaps& taps = program.taps();
    const std::int32_t tokens      = taps.tokens;
    if (tokens <= 0) { throw std::runtime_error("an armed pass captured no column"); }
    if (into.size() != taps.layers.size()) {
        throw std::runtime_error("the capture's layer count changed between passes");
    }
    for (std::size_t index = 0; index < taps.layers.size(); ++index) {
        const std::vector<std::uint16_t> block =
            tap_block(taps.features, static_cast<std::int32_t>(index), hidden, tokens);
        const std::size_t offset =
            static_cast<std::size_t>(taps.first_position) * static_cast<std::size_t>(hidden);
        if (into[index].size() < offset + block.size()) {
            throw std::runtime_error("a pass captured positions past the sequence's end");
        }
        std::copy(block.begin(), block.end(),
                  into[index].begin() + static_cast<std::ptrdiff_t>(offset));
    }
    for (std::int32_t column = 0; column < tokens; ++column) {
        const std::size_t at = static_cast<std::size_t>(taps.first_position + column);
        if (at >= covered.size()) {
            throw std::runtime_error("a pass captured positions past the sequence's end");
        }
        ++covered[at];
    }
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    if (artifact_path == nullptr || *artifact_path == '\0') {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const char* tap_text                    = std::getenv("NINFER_TAP_TEXT");
        const std::string text                  = tap_text != nullptr && *tap_text != '\0'
                                                      ? read_file(tap_text)
                                                      : std::string(kDefaultText);
        const char* layers_text                 = std::getenv("NINFER_TAP_LAYERS");
        const std::vector<std::uint32_t> layers = parse_layers(
            layers_text != nullptr && *layers_text != '\0' ? layers_text : "1,18,35,51,68,85");
        const std::int32_t prompt_tokens = env_int("NINFER_TAP_PROMPT_TOKENS", 512);
        const std::int32_t generated     = env_int("NINFER_TAP_GENERATED", 128);

        const std::int32_t capacity = prompt_tokens + generated + 8;

        ninfer::EngineOptions options;
        options.device          = 0;
        options.max_context     = static_cast<std::uint32_t>(capacity);
        options.max_concurrency = 1;
        options.prefill_chunk   = static_cast<std::uint32_t>(capacity);
        options.kv_cache        = ninfer::KvCacheStorage::BFloat16;
        options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(capacity));

        // The resident banks' budget is everything the card has *after* the state, one pass's
        // storage, the expert pool and the runtime's own reservation -- the same arithmetic the
        // Engine does, and without it the loader spends the whole card on resident banks and the
        // Program has nothing left to allocate. The pool is inside the budget the plan spends, so
        // it comes out of the reservation rather than being counted in both places.
        ninfer::artifact::Reader reader(artifact_path);
        const nemotron::Config artifact_config = nemotron::parse_config(reader.directory());
        nemotron::ExpertResidencyRequest residency;
        residency.automatic      = options.expert_cache.mode == ninfer::ExpertCacheMode::Automatic;
        residency.explicit_bytes = options.expert_cache.explicit_bytes;
        residency.cache_slots    = options.expert_cache.cache_slots;
        residency.reserve_bytes =
            nemotron::Program::device_bytes(artifact_config.text, options) -
            nemotron::Program::expert_pool_device_bytes(artifact_config.text, options) +
            nemotron::kRuntimeReservationBytes +
            (options.kv_capacity.mode == ninfer::KvCapacityMode::Automatic
                 ? options.kv_capacity.automatic_headroom_bytes
                 : 0);
        auto plan = nemotron::plan_load(reader, ninfer::models::load_options(options), residency);
        // Copied before the plan is consumed by the materializer: the frontend owns its own
        // strings.
        const nemotron::TextResources& resources = plan.resources();
        const nemotron::FrontendResources frontend_resources{
            .tokenizer_json         = std::string(resources.tokenizer_json),
            .tokenizer_config_json  = std::string(resources.tokenizer_config_json),
            .generation_config_json = std::string(resources.generation_config_json),
            .chat_template_jinja    = std::string(resources.chat_template_jinja),
        };

        ninfer::DeviceContext device;
        auto model                         = nemotron::materialize_model(std::move(plan), device);
        const nemotron::TextConfig& config = model->config();
        const std::int32_t hidden          = static_cast<std::int32_t>(config.hidden_size);

        nemotron::Frontend frontend(frontend_resources, static_cast<std::uint32_t>(capacity));
        const std::vector<ninfer::TokenId> ids = frontend.tokenize_text(text);
        if (static_cast<std::int32_t>(ids.size()) < 64) {
            throw std::runtime_error("the prompt text tokenized to fewer than 64 tokens, which is "
                                     "no context for a drafter to condition on");
        }
        // The built-in paragraph is shorter than the default prompt length, so the length is what
        // the text affords rather than what was asked: a longer NINFER_TAP_TEXT is the way to ask
        // for more. The capacity above is an upper bound either way.
        const std::int32_t actual_prompt =
            std::min(prompt_tokens, static_cast<std::int32_t>(ids.size()));
        const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + actual_prompt);

        nemotron::Program program(*model, options, device);
        const ninfer::DeviceExecutionView execution = device.execution_view();

        // ---- The incremental arm: a normal decode, one token per pass, capturing as it goes ----
        const std::int32_t total = capacity - 8;
        std::vector<std::vector<std::uint16_t>> incremental(
            layers.size(), std::vector<std::uint16_t>(static_cast<std::size_t>(total) *
                                                      static_cast<std::size_t>(hidden)));
        std::vector<std::int32_t> covered(static_cast<std::size_t>(total), 0);
        std::vector<std::int32_t> stream(prompt);
        program.capture_taps(layers);
        program.create(0, execution);
        program.prefill(0, prompt, execution);
        append_pass(program, incremental, hidden, covered);
        std::int32_t stopped = -1;
        for (std::int32_t step = 0; step < generated; ++step) {
            const std::int32_t drawn = program.sample(0, execution);
            stream.push_back(drawn);
            if (drawn == 2 || drawn == 11) {
                stopped = step;
                break;
            }
            program.decode(0, drawn, execution);
            append_pass(program, incremental, hidden, covered);
        }
        const std::int32_t positions = static_cast<std::int32_t>(stream.size());
        const std::int32_t captured =
            static_cast<std::int32_t>(std::count(covered.begin(), covered.begin() + positions, 1));
        expect(captured == positions, "the incremental capture covered " +
                                          std::to_string(captured) + " of " +
                                          std::to_string(positions) + " positions exactly once");

        // ---- The one-pass arm: the same sequence from a zero state, in one pass ----
        std::vector<std::vector<std::uint16_t>> one_pass(
            layers.size(), std::vector<std::uint16_t>(static_cast<std::size_t>(positions) *
                                                      static_cast<std::size_t>(hidden)));
        std::vector<std::int32_t> one_covered(static_cast<std::size_t>(total), 0);
        program.finish(0);
        program.create(0, execution);
        program.prefill(0, stream, execution);
        const nemotron::PassTaps& pass = program.taps();
        expect(pass.tokens == positions && pass.first_position == 0,
               "the one-pass prefill captured " + std::to_string(pass.tokens) + " columns from " +
                   std::to_string(pass.first_position));
        append_pass(program, one_pass, hidden, one_covered);

        // A second capture of the same pass must be the same bytes: what is compared here is a
        // copy the Program makes on the pass's own stream, and anything nondeterministic in it
        // would poison every measurement built on the dump.
        std::vector<std::vector<std::uint16_t>> repeat(
            layers.size(), std::vector<std::uint16_t>(static_cast<std::size_t>(positions) *
                                                      static_cast<std::size_t>(hidden)));
        std::vector<std::int32_t> repeat_covered(static_cast<std::size_t>(total), 0);
        program.finish(0);
        program.create(0, execution);
        program.prefill(0, stream, execution);
        append_pass(program, repeat, hidden, repeat_covered);
        std::size_t unstable = 0;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            for (std::size_t at = 0; at < one_pass[index].size(); ++at) {
                if (one_pass[index][at] != repeat[index][at]) { ++unstable; }
            }
        }
        expect(unstable == 0, std::to_string(unstable) +
                                  " tap values differ between two captures of the same pass");
        program.stop_capturing_taps();
        program.finish(0);

        // The arms run the same tokens through different pass widths, and this family's Ops are
        // route-dependent at the last bit -- a one-column call reaches the narrow expert route and
        // a wide one the row-tiled route, and each format's GEMV and MMA routes reduce K in a
        // different order. So the arms are asserted equal only where their widths agree, which is
        // the prompt: both arms prefilled its 512 tokens as one wide pass. That agreement is what
        // says the taps are filed under the right block and the right position, and it covers every
        // tap over the whole prompt. The generated tail is where the widths part -- one column per
        // pass against one 640-wide pass -- so it is *reported* rather than asserted: a
        // route-dependent last bit that reaches a router near-tie is a property of the engine, not
        // of the capture, and no tolerance bounds what it amplifies to.
        const std::size_t prompt_values =
            static_cast<std::size_t>(actual_prompt) * static_cast<std::size_t>(hidden);
        std::size_t differing = 0;
        std::string first_difference;
        double worst_rel_l2                   = 0.0;
        double worst_max_abs                  = 0.0;
        int worst_block                       = -1;
        std::int32_t first_differing_position = -1;
        std::vector<std::int32_t> first_at(layers.size(), -1);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            double tail_diff_sq = 0.0;
            double tail_ref_sq  = 0.0;
            double tail_max_abs = 0.0;
            for (std::size_t at = 0; at < one_pass[index].size(); ++at) {
                if (one_pass[index][at] == incremental[index][at]) { continue; }
                ++differing;
                const std::int32_t position =
                    static_cast<std::int32_t>(at / static_cast<std::size_t>(hidden));
                if (first_differing_position < 0) { first_differing_position = position; }
                if (first_at[index] < 0) { first_at[index] = position; }
                if (first_difference.empty()) {
                    first_difference = "block " + std::to_string(layers[index]) + ", position " +
                                       std::to_string(position) + ", channel " +
                                       std::to_string(at % static_cast<std::size_t>(hidden));
                }
                if (at < prompt_values) { continue; }
                const double actual = static_cast<double>(
                    std::bit_cast<float>(static_cast<std::uint32_t>(one_pass[index][at]) << 16));
                const double reference = static_cast<double>(
                    std::bit_cast<float>(static_cast<std::uint32_t>(incremental[index][at]) << 16));
                tail_diff_sq += (actual - reference) * (actual - reference);
                tail_ref_sq += reference * reference;
                tail_max_abs = std::max(tail_max_abs, std::abs(actual - reference));
            }
            const double rel_l2 = tail_ref_sq > 0.0 ? std::sqrt(tail_diff_sq / tail_ref_sq) : 0.0;
            std::cout << "  block " << layers[index] << ": first difference at position "
                      << (first_at[index] < 0 ? -1 : first_at[index]) << " | tail rel L2 " << rel_l2
                      << " | max abs " << tail_max_abs << '\n';
            if (rel_l2 > worst_rel_l2 || (rel_l2 == worst_rel_l2 && tail_max_abs > worst_max_abs)) {
                worst_rel_l2  = rel_l2;
                worst_max_abs = tail_max_abs;
                worst_block   = static_cast<int>(layers[index]);
            }
        }
        expect(first_differing_position < 0 || first_differing_position >= actual_prompt,
               "the arms differ before the first position whose pass width differs, at position " +
                   std::to_string(first_differing_position));
        // No bar on the tail's magnitude, deliberately. The two arms differ in the *routes* their
        // passes select -- a one-column call reaches the narrow expert route and a wide one the
        // row-tiled route, and each format's GEMV and MMA reduce K in a different order -- so the
        // last bit of every per-token result is route-dependent, and a mixture-of-experts router
        // turns that into a discrete expert-selection difference that no tolerance bounds. Measured
        // here: a Mamba block's residual differs by at most 2^-9 between the arms at the first
        // decode column, and block 85 reaches max abs 30 later in the state. The numbers are
        // reported below and belong beside the plan's decode-equivalence claim, which holds for
        // equal routes and not across widths.
        if (worst_rel_l2 > 0.0) {
            std::cout << "note: the arms' tail diverges by construction (route selection, not a "
                         "capture failure): rel L2 "
                      << worst_rel_l2 << " on block " << worst_block << ", max abs "
                      << worst_max_abs << '\n';
        }
        std::cout << "arms: " << differing << " of "
                  << one_pass.size() * static_cast<std::size_t>(positions) *
                         static_cast<std::size_t>(hidden)
                  << " values differ; the first is at position "
                  << (first_differing_position < 0 ? -1 : first_differing_position)
                  << " (the prompt is " << actual_prompt
                  << " wide in both arms), worst tail rel L2 " << worst_rel_l2 << " on block "
                  << worst_block << ", max abs " << worst_max_abs
                  << (first_difference.empty() ? "" : "; " + first_difference) << '\n';

        std::cout << "taps: " << one_pass.size() << " blocks | " << positions << " positions | "
                  << hidden << " channels | prompt " << actual_prompt << " | generated "
                  << (positions - actual_prompt) << (stopped >= 0 ? " | stopped early" : "")
                  << '\n';
        for (std::size_t index = 0; index < layers.size(); ++index) {
            std::uint64_t sum  = 0;
            std::uint16_t peak = 0;
            bool finite        = true;
            for (const std::uint16_t word : one_pass[index]) {
                sum += word;
                peak = std::max(peak, static_cast<std::uint16_t>(word & 0x7FFFU));
                finite =
                    finite &&
                    std::isfinite(std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16));
            }
            expect(finite, "block " + std::to_string(layers[index]) + "'s taps are not finite");
            std::cout << "  block " << layers[index] << ": bf16 word sum " << sum << " | peak word "
                      << peak << '\n';
        }

        // ---- The dump: what the acceptance probe reads ----
        // The incremental arm's capture: those are the taps the engine conditions on, because a
        // request's context is its prompt's pass followed by one column per committed token, never
        // one wide pass over everything.
        const char* dump = std::getenv("NINFER_TAP_DUMP");
        if (dump != nullptr && *dump != '\0') {
            const std::filesystem::path directory(dump);
            std::filesystem::create_directories(directory);
            for (std::size_t index = 0; index < layers.size(); ++index) {
                write_file(directory / ("tap_" + std::to_string(layers[index]) + ".bf16"),
                           incremental[index].data(),
                           incremental[index].size() * sizeof(std::uint16_t));
            }
            write_file(directory / "tokens.i32", stream.data(),
                       stream.size() * sizeof(std::int32_t));
            std::ostringstream meta;
            meta << "{\n  \"artifact\": \"" << artifact_path << "\",\n  \"layers\": [";
            for (std::size_t index = 0; index < layers.size(); ++index) {
                meta << (index == 0 ? "" : ", ") << layers[index];
            }
            meta << "],\n  \"hidden\": " << hidden << ",\n  \"positions\": " << positions
                 << ",\n  \"prompt_tokens\": " << actual_prompt
                 << ",\n  \"generated_tokens\": " << (positions - actual_prompt)
                 << ",\n  \"eos_ids\": [2, 11]"
                 << ",\n  \"tap_file\": \"tap_<layer>.bf16: bf16 [positions, hidden], token-major"
                    " (the composition's own residual order); captured along the path the engine"
                    " runs -- the prompt as one pass, then one column per generated token\""
                 << ",\n  \"tokens_file\": \"tokens.i32: i32 [positions], the prompt followed by"
                    " the greedily drawn stream; every position at or past prompt_tokens is the"
                    " target's own argmax decision\"\n}\n";
            write_file(directory / "meta.json", meta.str().data(), meta.str().size());
            std::cout << "dump: " << directory.string() << " (" << layers.size() << " tap files, "
                      << positions << " positions)\n";
        }
        if (failures != 0) {
            std::cout << "nemotron_h draft taps: " << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "nemotron_h draft taps: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "nemotron_h draft taps: " << error.what() << '\n';
        return 1;
    }
}
