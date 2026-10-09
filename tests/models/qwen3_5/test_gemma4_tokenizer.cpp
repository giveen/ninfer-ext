// Gemma 4 tokenizer: exact token ids against the reference, and the pipeline family's refusal.
//
// The expected ids come from `tokenizers` (Rust) reading the checkpoint's own tokenizer.json —
// `tests/fixtures/gemma4/tokenizer_ids.json` records the source digest and the oracle that
// produced them. Point NINFER_GEMMA_CHECKPOINT at a Gemma 4 checkpoint directory to run.
#include "models/qwen3_5/frontend/tokenizer.h"

#include "artifact/schema.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using ninfer::models::qwen3_5::frontend::EncodeOptions;
using ninfer::models::qwen3_5::frontend::Tokenizer;
using ninfer::models::qwen3_5::frontend::TokenizerFamily;
using ninfer::models::qwen3_5::frontend::TokenizerResources;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    const char* checkpoint = std::getenv("NINFER_GEMMA_CHECKPOINT");
    if (checkpoint == nullptr || *checkpoint == '\0') {
        std::cout << "skip: NINFER_GEMMA_CHECKPOINT is not set\n";
        return 77;
    }
    const std::string dir(checkpoint);
    const std::string tokenizer_json = read_file(dir + "/tokenizer.json");
    const std::string config_json    = read_file(dir + "/tokenizer_config.json");
    const std::string generation     = read_file(dir + "/generation_config.json");

    const std::string fixture = read_file("tests/fixtures/gemma4/tokenizer_ids.json");
    const nlohmann::json cases = ninfer::artifact::parse_json(fixture, "gemma4 tokenizer fixture");

    // The Gemma family loads the checkpoint's pipeline; the Qwen family must refuse it rather than
    // tokenize it as something else.
    const Tokenizer gemma(TokenizerResources{tokenizer_json, config_json, generation,
                                             TokenizerFamily::Gemma});
    bool refused_as_qwen = false;
    try {
        const Tokenizer qwen(TokenizerResources{tokenizer_json, config_json, generation});
        (void)qwen;
    } catch (const std::exception&) { refused_as_qwen = true; }
    expect(refused_as_qwen, "the Qwen family must refuse Gemma's tokenizer.json");

    EncodeOptions options;
    options.parse_added_tokens = false;
    std::size_t index          = 0;
    for (const auto& item : cases.at("cases")) {
        const std::string text = item.at("text").get<std::string>();
        const auto got         = gemma.encode(text, options);
        std::vector<int> want;
        for (const auto& id : item.at("ids")) { want.push_back(id.get<int>()); }
        if (got != want) {
            std::ostringstream message;
            message << "case " << index << " " << item.at("text").dump() << " -> [";
            for (std::size_t i = 0; i < got.size(); ++i) {
                message << (i ? ", " : "") << got[i];
            }
            message << "] expected [";
            for (std::size_t i = 0; i < want.size(); ++i) {
                message << (i ? ", " : "") << want[i];
            }
            message << ']';
            message << " | got bytes";
            for (const int id : got) { message << ' ' << std::string(gemma.decode_token_bytes(id)); }
            message << " | want bytes";
            for (const int id : want) { message << ' ' << std::string(gemma.decode_token_bytes(id)); }
            expect(false, message.str());
        }
        ++index;
    }
    if (failures == 0) {
        std::cout << "OK: " << index << " Gemma 4 tokenizer cases match the reference exactly\n";
    }
    return failures == 0 ? 0 : 1;
}
