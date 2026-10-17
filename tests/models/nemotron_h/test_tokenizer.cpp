// The Nemotron-H tokenizer: the checkpoint's own pipeline is accepted and every other family's is
// refused, and -- with NINFER_NEMOTRON_ARTIFACT set -- the family's word boundaries and token ids
// match the reference `tokenizers` pipeline exactly, case by case.
//
// The pipeline arms need no artifact: the description is written by hand here, so the acceptance
// and each refusal run wherever the tests do. The reference cases come from
// `tools/verify/nemotron_h_frontend_fixture.py`, which reads the artifact's own tokenizer.json.
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tokenizer_test_access.h"

#include "artifact/schema.h"
#include "artifact_resources.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ninfer::models::qwen3_5::frontend::EncodeOptions;
using ninfer::models::qwen3_5::frontend::Tokenizer;
using ninfer::models::qwen3_5::frontend::TokenizerFamily;
using ninfer::models::qwen3_5::frontend::TokenizerResources;

// The Nanotron-v2 split, written out here as the reference pipeline's own description of it. It has
// to stay byte-identical to `NEMOTRON_SPLIT_PATTERN` in tools/convert/resources.py and to the
// tokenizer the artifact carries; the family's acceptance arm below is what checks that.
constexpr std::string_view kSplit =
    R"split([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)split";

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// A minimal but complete tokenizer.json: the pipeline under test, one vocabulary entry per verdict,
// and the empty merge table the validation arms never reach.
Json tokenizer_json(std::string_view pattern, std::string_view behavior, bool invert,
                    bool add_prefix_space, bool use_regex, bool byte_fallback, bool ignore_merges,
                    bool with_normalizer, std::string_view decoder = "ByteLevel") {
    Json split{{"type", "Split"},
               {"pattern", {{"Regex", pattern}}},
               {"behavior", behavior},
               {"invert", invert}};
    Json bytes{
        {"type", "ByteLevel"}, {"add_prefix_space", add_prefix_space}, {"use_regex", use_regex}};
    Json root{
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", Json::array({Json{{"id", 0},
                                           {"content", "<unk>"},
                                           {"single_word", false},
                                           {"lstrip", false},
                                           {"rstrip", false},
                                           {"normalized", false},
                                           {"special", true}}})},
        {"normalizer", with_normalizer ? Json{{"type", "NFC"}} : Json(nullptr)},
        {"pre_tokenizer",
         Json{{"type", "Sequence"}, {"pretokenizers", Json::array({split, bytes})}}},
        {"decoder", Json{{"type", decoder}}},
        {"post_processor", Json{{"type", "ByteLevel"}}},
        {"model", Json{{"type", "BPE"},
                       {"dropout", nullptr},
                       {"unk_token", "<unk>"},
                       {"continuing_subword_prefix", nullptr},
                       {"end_of_word_suffix", nullptr},
                       {"fuse_unk", false},
                       {"byte_fallback", byte_fallback},
                       {"ignore_merges", ignore_merges},
                       {"vocab", Json{{"<unk>", 0}, {"a", 1}, {"b", 2}}},
                       {"merges", Json::array()}}},
    };
    return root;
}

std::string tokenizer_config_json() {
    Json decoder{};
    decoder["0"] = Json{{"content", "<unk>"}, {"single_word", false}, {"lstrip", false},
                        {"rstrip", false},    {"normalized", false},  {"special", true}};
    return Json{{"add_bos_token", false},
                {"add_prefix_space", false},
                {"pad_token", "<|im_end|>"},
                {"added_tokens_decoder", decoder}}
        .dump();
}

constexpr std::string_view kGenerationConfig = R"({"eos_token_id": [2, 11]})";

// True when the family refuses the description, with the field the refusal names.
bool refused(TokenizerFamily family, const Json& root, std::string& message) {
    const std::string tokenizer = root.dump();
    const std::string config    = tokenizer_config_json();
    const std::string generation(kGenerationConfig);
    try {
        const Tokenizer loaded(TokenizerResources{tokenizer, config, generation, family});
        (void)loaded;
        return false;
    } catch (const std::exception& error) {
        message = error.what();
        return true;
    }
}

void check_pipeline_arms() {
    const Json declared =
        tokenizer_json(kSplit, "Isolated", false, false, false, false, false, false);
    std::string message;
    expect(!refused(TokenizerFamily::Nemotron, declared, message),
           "the checkpoint's own pipeline must be accepted by the Nemotron family: " + message);

    struct Refusal {
        const char* what;
        TokenizerFamily family;
        Json root;
    };

    const Json qwen = tokenizer_json(
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
        "Isolated", false, false, false, false, false, false);
    const std::vector<Refusal> refusals{
        {"another family's split pattern", TokenizerFamily::Nemotron,
         tokenizer_json(R"([^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|\s+)", "Isolated", false, false,
                        false, false, false, false)},
        {"a normalizer", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, false, false, false, false, true)},
        {"a merged-with-previous split", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "MergedWithPrevious", false, false, false, false, false, false)},
        {"an inverted split", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", true, false, false, false, false, false)},
        {"a byte-level step that adds a prefix space", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, true, false, false, false, false)},
        {"a byte-level step with its own regex", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, false, true, false, false, false)},
        {"byte fallback", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, false, false, true, false, false)},
        {"ignore_merges", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, false, false, false, true, false)},
        {"a decoder that is not byte-level", TokenizerFamily::Nemotron,
         tokenizer_json(kSplit, "Isolated", false, false, false, false, false, false, "Fuse")},
        {"the Qwen family", TokenizerFamily::Qwen, declared},
        {"the Gemma family", TokenizerFamily::Gemma, declared},
        {"the Nemotron family on the Qwen pipeline", TokenizerFamily::Nemotron, qwen},
    };
    for (const Refusal& refusal : refusals) {
        std::string text;
        expect(refused(refusal.family, refusal.root, text),
               std::string("must refuse ") + refusal.what);
        std::cout << "refused " << refusal.what << ": " << text << '\n';
    }
}

// The family's word boundaries, walked exactly as the tokenizer walks them, against the reference
// pipeline's own pieces.
void check_split_cases(const Json& cases) {
    for (const Json& item : cases) {
        const std::string text = item.at("text").get<std::string>();
        std::vector<std::pair<std::size_t, std::size_t>> want;
        for (const Json& piece : item.at("pieces")) {
            want.emplace_back(piece.at(0).get<std::size_t>(), piece.at(1).get<std::size_t>());
        }
        std::vector<std::pair<std::size_t, std::size_t>> got;
        for (std::size_t begin = 0; begin < text.size();) {
            const std::size_t end = ninfer::models::qwen3_5::frontend::tokenizer_word_end(
                text, begin, TokenizerFamily::Nemotron);
            got.emplace_back(begin, end);
            begin = end;
        }
        if (got != want) {
            std::ostringstream message;
            message << "split of " << Json(text).dump() << "\n  got ";
            for (const auto& [begin, end] : got) {
                message << Json(text.substr(begin, end - begin)).dump() << ' ';
            }
            message << "\n  want ";
            for (const auto& [begin, end] : want) {
                message << Json(text.substr(begin, end - begin)).dump() << ' ';
            }
            expect(false, message.str());
        }
    }
}

void check_id_cases(const Tokenizer& tokenizer, const Json& cases) {
    for (const Json& item : cases) {
        const std::string text = item.at("text").get<std::string>();
        std::vector<int> want;
        for (const Json& id : item.at("ids")) { want.push_back(id.get<int>()); }
        const std::vector<int> got = tokenizer.encode(text);
        if (got != want) {
            std::ostringstream message;
            message << "ids of " << Json(text).dump() << "\n  got  [";
            for (std::size_t index = 0; index < got.size(); ++index) {
                message << (index ? ", " : "") << got[index];
            }
            message << "]\n  want [";
            for (std::size_t index = 0; index < want.size(); ++index) {
                message << (index ? ", " : "") << want[index];
            }
            message << ']';
            expect(false, message.str());
            continue;
        }
        const std::string decoded = tokenizer.decode(want);
        expect(decoded == text,
               "case " + Json(text).dump() + " decodes to " + Json(decoded).dump());
    }
}

// The probe the reference test drives: the ids this tokenizer produces for one corpus.
int print_ids(const std::string& corpus_path, const std::string& tokenizer_out) {
    const char* artifact = ninfer::test::nemotron_artifact_path();
    if (artifact == nullptr) {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }
    const ninfer::test::NemotronResources resources(artifact);
    if (!tokenizer_out.empty()) {
        std::ofstream out(tokenizer_out, std::ios::binary | std::ios::trunc);
        out << resources.frontend().tokenizer_json;
    }
    const Tokenizer tokenizer(TokenizerResources{
        resources.frontend().tokenizer_json, resources.frontend().tokenizer_config_json,
        resources.frontend().generation_config_json, TokenizerFamily::Nemotron});
    const std::string corpus   = read_file(corpus_path);
    const std::vector<int> ids = tokenizer.encode(corpus);
    Json out{{"ids", ids}};
    std::cout << out.dump() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--print-ids") {
        if (argc < 3) {
            std::cerr << "usage: --print-ids <corpus> [<tokenizer.json out>]\n";
            return 2;
        }
        try {
            return print_ids(argv[2], argc > 3 ? argv[3] : std::string{});
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }

    check_pipeline_arms();
    if (failures != 0) {
        std::cerr << "the pipeline arms failed\n";
        return 1;
    }

    const char* artifact = ninfer::test::nemotron_artifact_path();
    if (artifact == nullptr) {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set (pipeline arms passed)\n";
        return 77;
    }
    try {
        const ninfer::test::NemotronResources resources(artifact);
        const Tokenizer tokenizer(TokenizerResources{
            resources.frontend().tokenizer_json, resources.frontend().tokenizer_config_json,
            resources.frontend().generation_config_json, TokenizerFamily::Nemotron});
        const Json fixture =
            ninfer::artifact::parse_json(read_file(std::string(NINFER_SOURCE_DIR) +
                                                   "/tests/fixtures/nemotron_h/tokenizer_ids.json"),
                                         "nemotron_h tokenizer fixture");
        expect(static_cast<std::size_t>(fixture.at("vocab_size").get<int>()) ==
                   tokenizer.vocab_size(),
               "the fixture was generated from another vocabulary (" +
                   std::to_string(fixture.at("vocab_size").get<int>()) + " against " +
                   std::to_string(tokenizer.vocab_size()) + ")");
        check_split_cases(fixture.at("cases"));
        check_id_cases(tokenizer, fixture.at("cases"));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "the reference cases failed\n";
        return 1;
    }
    std::cout << "the Nemotron-H tokenizer matches the reference split and ids\n";
    return 0;
}
