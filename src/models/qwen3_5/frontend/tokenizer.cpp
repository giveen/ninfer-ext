#include "models/qwen3_5/frontend/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json    = nlohmann::json;
namespace uni = ninfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

// U+2581, the marker Gemma's normalizer writes in place of a space.
constexpr std::string_view kGemmaSpaceMarker = "\xe2\x96\x81";

constexpr std::string_view kQwenSplitPattern =
    R"qwen((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)qwen";

void validate_pipeline(const Json& root, const Json& model, TokenizerFamily family) {
    const char* family_name = family == TokenizerFamily::Gemma ? "Gemma" : "Qwen";
    const auto require = [family_name](bool valid, const char* field) {
        if (!valid) {
            throw std::invalid_argument(std::string("tokenizer.json ") + field +
                                        " is not supported by the " + family_name +
                                        " tokenizer");
        }
    };
    const auto type = [](const Json& value, const char* name) {
        return value.is_object() && value.contains("type") && value["type"] == name;
    };
    // Omitted pipeline descriptions use the architecture's fixed tokenizer semantics.
    // An explicit description must agree with the implemented transformations.
    if (family == TokenizerFamily::Gemma) {
        // Gemma's normalizer replaces a space with the U+2581 marker and leaves the text
        // otherwise alone; its pre-tokenizer splits on newlines only, because a BPE merge may not
        // cross a newline, and declares no byte-level step -- the symbols are raw UTF-8.
        if (root.contains("normalizer")) {
            const auto& normalizer = root["normalizer"];
            require(type(normalizer, "Replace") &&
                        normalizer.contains("pattern") && normalizer["pattern"].is_object() &&
                        normalizer["pattern"].value("String", std::string{}) == " " &&
                        normalizer.value("content", std::string{}) == kGemmaSpaceMarker,
                    "normalizer");
        }
        if (root.contains("pre_tokenizer")) {
            const auto& pre = root["pre_tokenizer"];
            require(type(pre, "Split") && pre.contains("pattern") && pre["pattern"].is_object() &&
                        pre["pattern"].value("String", std::string{}) == " " &&
                        pre.value("behavior", std::string{}) == "MergedWithPrevious" &&
                        pre.value("invert", Json(false)) == false,
                    "pre_tokenizer");
        }
    } else if (root.contains("normalizer")) {
        require(type(root["normalizer"], "NFC"), "normalizer");
    }
    if (family == TokenizerFamily::Qwen && root.contains("pre_tokenizer")) {
        const auto& pre = root["pre_tokenizer"];
        require(type(pre, "Sequence") && pre.contains("pretokenizers") &&
                    pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 2,
                "pre_tokenizer");
        const auto& split = pre["pretokenizers"][0];
        const auto& bytes = pre["pretokenizers"][1];
        require(type(split, "Split") && split.contains("pattern") && split["pattern"].is_object() &&
                    split["pattern"].value("Regex", std::string{}) == kQwenSplitPattern &&
                    split.value("behavior", std::string{}) == "Isolated" &&
                    split.value("invert", Json(false)) == false,
                "pre_tokenizer.Split");
        require(type(bytes, "ByteLevel") && bytes.value("add_prefix_space", Json(true)) == false &&
                    bytes.value("use_regex", Json(true)) == false,
                "pre_tokenizer.ByteLevel");
    }
    if (family == TokenizerFamily::Gemma) {
        // The decoder unwinds the alphabet: the marker becomes a space again, an unmergeable
        // character is written from its `<0xXX>` bytes, and the pieces are fused into text.
        if (root.contains("decoder")) {
            const auto& decoder = root["decoder"];
            const auto& steps   = decoder.is_object() && decoder.contains("decoders")
                                      ? decoder["decoders"]
                                      : Json();
            require(type(decoder, "Sequence") && steps.is_array() && steps.size() == 3 &&
                        type(steps[0], "Replace") && steps[0].value("content", std::string{}) == " " &&
                        steps[0].contains("pattern") && steps[0]["pattern"].is_object() &&
                        steps[0]["pattern"].value("String", std::string{}) == kGemmaSpaceMarker &&
                        type(steps[1], "ByteFallback") && type(steps[2], "Fuse"),
                    "decoder");
        }
        if (root.contains("post_processor") && !root["post_processor"].is_null()) {
            // A template that adds nothing: the runtime applies the chat protocol itself.
            const auto& post = root["post_processor"];
            const auto& specials = post.is_object() && post.contains("special_tokens")
                                       ? post["special_tokens"]
                                       : Json();
            const auto& single = post.is_object() && post.contains("single") ? post["single"] : Json();
            require(type(post, "TemplateProcessing") && specials.is_object() &&
                        specials.empty() && single.is_array() && single.size() == 1,
                    "post_processor");
        }
        require(model.value("byte_fallback", Json(false)) == true, "model.byte_fallback");
    } else {
        if (root.contains("decoder")) { require(type(root["decoder"], "ByteLevel"), "decoder"); }
        if (root.contains("post_processor") && !root["post_processor"].is_null()) {
            require(type(root["post_processor"], "ByteLevel"), "post_processor");
        }
    }
    if (model.contains("dropout") && !model["dropout"].is_null()) {
        require(model["dropout"].is_number() && model["dropout"] == 0, "model.dropout");
    }
    for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        require(!model.contains(field) || model[field].is_null() || model[field] == "", field);
    }
    require(model.value("ignore_merges", Json(false)) == false, "model.ignore_merges");
}

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        std::string(label));
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    std::string(label));
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + std::string(label));
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", label);
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

AddedToken parse_added_token_decoder_entry(int id, const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens_decoder item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    token.id          = id;
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

void validate_supported_added_token(const AddedToken& token, std::string_view label) {
    if (token.content.empty()) {
        throw std::invalid_argument("added token content must not be empty in " +
                                    std::string(label));
    }
    if (token.single_word || token.lstrip || token.rstrip || token.normalized) {
        throw std::invalid_argument("Tokenizer only supports added tokens with single_word=false, "
                                    "lstrip=false, rstrip=false, and normalized=false in " +
                                    std::string(label));
    }
}

bool same_added_token(const AddedToken& lhs, const AddedToken& rhs) noexcept {
    return lhs.id == rhs.id && lhs.content == rhs.content && lhs.single_word == rhs.single_word &&
           lhs.lstrip == rhs.lstrip && lhs.rstrip == rhs.rstrip &&
           lhs.normalized == rhs.normalized && lhs.special == rhs.special;
}

int parse_added_token_decoder_id(std::string_view key, std::string_view label) {
    std::int64_t parsed     = -1;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), parsed);
    if (error != std::errc{} || end != key.data() + key.size() || parsed < 0 ||
        parsed > kMaxTokenId || std::to_string(parsed) != key) {
        throw std::invalid_argument("added_tokens_decoder key must be a nonnegative token id in " +
                                    std::string(label));
    }
    return static_cast<int>(parsed);
}

std::vector<AddedToken>
load_added_tokens(const Json& root, std::string_view label, std::vector<std::string>& id_to_token,
                  const std::unordered_set<int>& occupied_vocab_ids,
                  const std::unordered_map<std::string, int>& occupied_vocab_tokens) {
    const Json& added = require_array_field(root, "added_tokens", label);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    std::unordered_map<std::string, int> seen_added_contents;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, label);
        validate_supported_added_token(token, label);
        const auto index = static_cast<std::size_t>(token.id);
        // A checkpoint may list a special token in the vocabulary as well as in added_tokens: some
        // write the specials with the base vocabulary (Gemma) and some do not (Qwen). That is the
        // same token twice, so it is accepted only when the two definitions agree.
        const bool in_vocab_id      = occupied_vocab_ids.contains(token.id);
        const bool in_vocab_content = occupied_vocab_tokens.contains(token.content);
        if (in_vocab_id || in_vocab_content) {
            const bool agrees = in_vocab_id && in_vocab_content &&
                                index < id_to_token.size() &&
                                id_to_token[index] == token.content &&
                                occupied_vocab_tokens.at(token.content) == token.id;
            if (!agrees) {
                throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                            std::string(label));
            }
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " +
                                        std::string(label));
        }
        if (!seen_added_contents.emplace(token.content, token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate content mapping in " +
                                        std::string(label));
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

void merge_added_tokens_decoder(const Json& root, std::string_view label,
                                std::vector<std::string>& id_to_token,
                                const std::unordered_set<int>& occupied_vocab_ids,
                                const std::unordered_map<std::string, int>& occupied_vocab_tokens,
                                std::vector<AddedToken>& tokens, bool required) {
    if (!root.contains("added_tokens_decoder") || root["added_tokens_decoder"].is_null()) {
        // Some checkpoints describe their added tokens once, in tokenizer.json alone (Gemma);
        // others repeat them here so the two can be cross-checked (Qwen).
        if (required) {
            throw std::invalid_argument("missing field added_tokens_decoder in " +
                                        std::string(label));
        }
        return;
    }
    const Json& decoder = root["added_tokens_decoder"];
    std::unordered_map<int, std::size_t> token_by_id;
    std::unordered_map<std::string, int> token_by_content;
    std::unordered_set<int> decoder_ids;
    token_by_id.reserve(tokens.size() + decoder.size());
    token_by_content.reserve(tokens.size() + decoder.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        token_by_id.emplace(tokens[index].id, index);
        token_by_content.emplace(tokens[index].content, tokens[index].id);
    }

    for (const auto& item : decoder.items()) {
        const int id = parse_added_token_decoder_id(item.key(), label);
        if (!decoder_ids.insert(id).second) {
            throw std::invalid_argument("added_tokens_decoder has duplicate id mapping in " +
                                        std::string(label));
        }
        AddedToken token = parse_added_token_decoder_entry(id, item.value(), label);
        validate_supported_added_token(token, label);

        const auto existing_id = token_by_id.find(id);
        if (existing_id != token_by_id.end()) {
            if (!same_added_token(tokens.at(existing_id->second), token)) {
                throw std::invalid_argument("conflicting added-token definition for id " +
                                            std::to_string(id) +
                                            " between tokenizer.json and tokenizer_config.json");
            }
            continue;
        }
        const auto existing_content = occupied_vocab_tokens.find(token.content);
        if (occupied_vocab_ids.contains(id) &&
            !(existing_content != occupied_vocab_tokens.end() &&
              existing_content->second == id &&
              static_cast<std::size_t>(id) < id_to_token.size() &&
              id_to_token[static_cast<std::size_t>(id)] == token.content)) {
            throw std::invalid_argument("added_tokens_decoder overlaps vocabulary id " +
                                        std::to_string(id));
        }
        if (token_by_content.contains(token.content) ||
            occupied_vocab_tokens.contains(token.content)) {
            throw std::invalid_argument("conflicting added-token content mapping for " +
                                        token.content);
        }

        const auto index = static_cast<std::size_t>(id);
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        if (!id_to_token[index].empty()) {
            throw std::invalid_argument("duplicate tokenizer mapping for id " + std::to_string(id));
        }
        id_to_token[index] = token.content;
        token_by_id.emplace(id, tokens.size());
        token_by_content.emplace(token.content, id);
        tokens.push_back(std::move(token));
    }
    std::ranges::sort(tokens,
                      [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument(
                "field eos_token_id must not be empty in generation_config.json");
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or array in generation_config.json");
}

std::uint64_t merge_pair_key(int left, int right) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
           static_cast<std::uint32_t>(right);
}

BpeMergeTable load_bpe_merge_rules(const Json& model, std::string_view label,
                                   const std::unordered_map<std::string, int>& token_to_id) {
    const Json& merges = require_array_field(model, "merges", label);
    BpeMergeTable rules;
    rules.reset(merges.size());
    // Rank order matters here and is not incidental: linear probing lets the first claimant keep
    // a slot, and a low rank is a merge the encoder performs often, so walking model.merges in
    // order hands the frequent rules their home slot. Filling the table out of an unordered
    // container instead is measurably slower.
    int rank = 0;
    for (const Json& item : merges) {
        std::string left;
        std::string right;
        if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string()) {
            left  = item[0].get<std::string>();
            right = item[1].get<std::string>();
        } else if (item.is_string()) {
            const std::string pair  = item.get<std::string>();
            const std::size_t space = pair.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 >= pair.size() ||
                pair.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("malformed model.merges entry in " +
                                            std::string(label));
            }
            left  = pair.substr(0, space);
            right = pair.substr(space + 1);
        } else {
            throw std::invalid_argument("field model.merges must contain symbol pairs in " +
                                        std::string(label));
        }
        const auto left_id   = token_to_id.find(left);
        const auto right_id  = token_to_id.find(right);
        const auto result_id = token_to_id.find(left + right);
        if (left_id == token_to_id.end() || right_id == token_to_id.end() ||
            result_id == token_to_id.end()) {
            throw std::invalid_argument("model.merges references a symbol outside model.vocab in " +
                                        std::string(label));
        }
        const bool inserted =
            rules.insert(merge_pair_key(left_id->second, right_id->second),
                         BpeMergeRule{.rank = rank++, .result = result_id->second});
        if (!inserted) { throw std::invalid_argument("duplicate merge pair in model.merges"); }
    }
    return rules;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(codepoint, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

// Qwen's pipeline requires NFC. Gemma's normalizer only rewrites a space to the U+2581 marker,
// which the symbol builder applies, so its text is used exactly as it arrives.
std::string normalize_for_family(std::string_view text, TokenizerFamily family) {
    return family == TokenizerFamily::Gemma ? std::string(text) : uni::normalize_nfc(text);
}

std::string decode_byte_level_token(std::string_view token, int id) {
    static const std::unordered_map<std::uint32_t, char> byte_decoder = build_byte_level_decoder();
    std::string bytes;
    const std::vector<uni::CodepointSpan> codepoints =
        uni::utf8_codepoints(token, "Tokenizer::decode token id " + std::to_string(id));
    bytes.reserve(codepoints.size());
    for (const uni::CodepointSpan& codepoint : codepoints) {
        const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
        if (byte == byte_decoder.end()) {
            throw std::invalid_argument("Tokenizer::decode token id " + std::to_string(id) +
                                        " contains a character outside the byte-level alphabet");
        }
        bytes.push_back(byte->second);
    }
    return bytes;
}

// Gemma's tokens are text: the U+2581 marker stands for a space, a `<0xXX>` token stands for the
// byte it names (the byte fallback), and every other token is already its own characters.
std::string decode_gemma_token(std::string_view token, int id) {
    if (token == kGemmaSpaceMarker) { return " "; }
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' &&
        token[5] == '>') {
        const auto digit = [&](char value) -> int {
            if (value >= '0' && value <= '9') { return value - '0'; }
            if (value >= 'A' && value <= 'F') { return value - 'A' + 10; }
            if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
            throw std::invalid_argument("Tokenizer::decode token id " + std::to_string(id) +
                                        " has a malformed byte fallback");
        };
        return std::string(1, static_cast<char>((digit(token[3]) << 4) | digit(token[4])));
    }
    return std::string(token);
}

std::array<std::string, 256> build_byte_level_encoder() {
    std::array<std::string, 256> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder[static_cast<unsigned char>(byte)] =
            uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoint));
    }
    return encoder;
}

bool is_newline(std::int32_t codepoint) noexcept { return codepoint == '\r' || codepoint == '\n'; }

bool is_letter_or_mark(std::int32_t codepoint) noexcept {
    return uni::is_letter(codepoint) || uni::is_mark(codepoint);
}

bool is_non_newline_non_letter_non_number(std::int32_t codepoint) noexcept {
    return !is_newline(codepoint) && !uni::is_letter(codepoint) && !uni::is_number(codepoint);
}

bool is_non_space_non_letter_mark_number(std::int32_t codepoint) noexcept {
    return !uni::is_whitespace(codepoint) && !uni::is_letter(codepoint) &&
           !uni::is_mark(codepoint) && !uni::is_number(codepoint);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
        const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
    }
    return true;
}

uni::CodepointSpan qwen_codepoint_at(std::string_view text, std::size_t offset) {
    return uni::utf8_codepoint_at(text, offset, "Tokenizer::encode input");
}

std::size_t qwen_word_end(std::string_view text, std::size_t begin) {
    const uni::CodepointSpan first = qwen_codepoint_at(text, begin);
    const std::size_t after_first  = begin + first.length;
    const std::int32_t cp          = first.value;

    if (cp == '\'') {
        constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (std::string_view suffix : suffixes) {
            if (ascii_ci_matches(text, after_first, suffix)) { return after_first + suffix.size(); }
        }
    }

    const bool has_next     = after_first < text.size();
    const std::int32_t next = has_next ? qwen_codepoint_at(text, after_first).value : 0;
    if (is_letter_or_mark(cp) ||
        (is_non_newline_non_letter_non_number(cp) && has_next && is_letter_or_mark(next))) {
        std::size_t end = is_letter_or_mark(cp) ? begin : after_first;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_letter_or_mark(value.value)) { break; }
            end += value.length;
        }
        return end;
    }

    if (uni::is_number(cp)) { return after_first; }

    if ((cp == ' ' && has_next && is_non_space_non_letter_mark_number(next)) ||
        is_non_space_non_letter_mark_number(cp)) {
        std::size_t end = cp == ' ' ? after_first : begin;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_non_space_non_letter_mark_number(value.value)) { break; }
            end += value.length;
        }
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_newline(value.value)) { break; }
            end += value.length;
        }
        return end;
    }

    if (uni::is_whitespace(cp)) {
        std::size_t end              = begin;
        std::size_t last_begin       = begin;
        std::size_t last_newline_end = std::string_view::npos;
        std::size_t count            = 0;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!uni::is_whitespace(value.value)) { break; }
            last_begin = end;
            end += value.length;
            ++count;
            if (is_newline(value.value)) { last_newline_end = end; }
        }
        if (last_newline_end != std::string_view::npos) { return last_newline_end; }
        if (end == text.size()) { return end; }
        return count >= 2 ? last_begin : end;
    }

    return after_first;
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::ranges::find(stop_token_ids, id) != stop_token_ids.end();
}

struct BpeNode {
    int symbol               = -1;
    int previous             = -1;
    int next                 = -1;
    std::size_t end          = 0;
    std::uint32_t generation = 0;
    bool live                = true;
};

struct BpeCandidate {
    int rank                       = 0;
    int left                       = -1;
    int right                      = -1;
    int result                     = -1;
    std::uint32_t left_generation  = 0;
    std::uint32_t right_generation = 0;
};

struct BpeWordEnd {
    std::size_t normalized_offset = 0;
    std::size_t token_frontier    = 0;
};

struct LaterBpeCandidate {
    bool operator()(const BpeCandidate& lhs, const BpeCandidate& rhs) const noexcept {
        if (lhs.rank != rhs.rank) { return lhs.rank > rhs.rank; }
        return lhs.left > rhs.left;
    }
};

std::array<int, 256> load_byte_token_ids(const std::unordered_map<std::string, int>& token_to_id) {
    static const std::array<std::string, 256> byte_encoder = build_byte_level_encoder();
    std::array<int, 256> ids;
    ids.fill(-1);
    for (std::size_t byte = 0; byte < ids.size(); ++byte) {
        const auto token = token_to_id.find(byte_encoder[byte]);
        if (token != token_to_id.end()) { ids[byte] = token->second; }
    }
    return ids;
}

// The BPE input alphabet: one symbol per position of a word, with the source byte offset just past
// it. Qwen's alphabet is one byte per symbol through the GPT-2 byte table; Gemma's is one UTF-8
// character, with a space rendered as the U+2581 marker and any character the vocabulary lacks
// falling back to its bytes' `<0xXX>` tokens.
struct BpeSymbol {
    int token              = -1;
    std::size_t source_end = 0;
};

struct SymbolContext {
    TokenizerFamily family                    = TokenizerFamily::Qwen;
    const std::array<int, 256>* byte_token_ids = nullptr;
    const std::unordered_map<std::string, int>* vocab = nullptr;
    int marker_token                          = -1;
    const std::array<int, 256>* byte_fallback  = nullptr;
};

// Bytes of a Gemma-family tokenizer that hold the `<0xXX>` fallback for each byte value.
std::array<int, 256> load_byte_fallback_ids(const std::unordered_map<std::string, int>& token_to_id) {
    static const char* digits = "0123456789ABCDEF";
    std::array<int, 256> ids;
    ids.fill(-1);
    for (std::size_t byte = 0; byte < ids.size(); ++byte) {
        std::string name = "<0x00>";
        name[3]          = digits[(byte >> 4) & 0xF];
        name[4]          = digits[byte & 0xF];
        const auto token = token_to_id.find(name);
        if (token != token_to_id.end()) { ids[byte] = token->second; }
    }
    return ids;
}

// A merge may not cross a newline, so Gemma's words are the runs of newlines and of everything
// else; every other boundary is decided by the merges themselves.
std::size_t gemma_word_end(std::string_view text, std::size_t begin) {
    const bool newline = text[begin] == '\n';
    std::size_t end    = begin;
    while (end < text.size() && (text[end] == '\n') == newline) { ++end; }
    return end;
}

void build_word_symbols(std::string_view text, std::size_t begin, std::size_t end,
                        const SymbolContext& context, std::vector<BpeSymbol>& symbols) {
    symbols.clear();
    if (context.family == TokenizerFamily::Qwen) {
        for (std::size_t at = begin; at < end; ++at) {
            const int token = (*context.byte_token_ids)[static_cast<unsigned char>(text[at])];
            if (token < 0) {
                throw std::invalid_argument(
                    "Tokenizer::encode produced byte symbol outside vocabulary");
            }
            symbols.push_back(BpeSymbol{token, at + 1});
        }
        return;
    }
    for (std::size_t at = begin; at < end;) {
        const uni::CodepointSpan span = qwen_codepoint_at(text, at);
        if (span.value == ' ' && context.marker_token >= 0) {
            symbols.push_back(BpeSymbol{context.marker_token, at + span.length});
            at += span.length;
            continue;
        }
        const auto found = context.vocab->find(std::string(text.substr(at, span.length)));
        if (found != context.vocab->end()) {
            symbols.push_back(BpeSymbol{found->second, at + span.length});
        } else {
            for (std::size_t byte = at; byte < at + span.length; ++byte) {
                const int token = (*context.byte_fallback)[static_cast<unsigned char>(text[byte])];
                if (token < 0) {
                    throw std::invalid_argument(
                        "Tokenizer::encode has no byte fallback for a character outside the "
                        "vocabulary");
                }
                symbols.push_back(BpeSymbol{token, byte + 1});
            }
        }
        at += span.length;
    }
}

bool append_normalized_bpe_ids(std::vector<int>& ids, std::string_view normalized,
                               const BpeMergeTable& merge_rules, const SymbolContext& context,
                               std::size_t max_tokens,
                               std::vector<std::size_t>* token_ends = nullptr,
                               std::vector<BpeWordEnd>* word_ends   = nullptr) {
    if (normalized.empty()) { return true; }
    if (ids.size() == max_tokens) { return false; }

    std::vector<BpeSymbol> symbols;
    for (std::size_t begin = 0; begin < normalized.size();) {
        const std::size_t end = context.family == TokenizerFamily::Gemma
                                    ? gemma_word_end(normalized, begin)
                                    : qwen_word_end(normalized, begin);
        build_word_symbols(normalized, begin, end, context, symbols);
        std::vector<BpeNode> nodes(symbols.size());
        for (std::size_t index = 0; index < symbols.size(); ++index) {
            nodes[index] =
                BpeNode{.symbol   = symbols[index].token,
                        .previous = index == 0 ? -1 : static_cast<int>(index - 1),
                        .next     = index + 1 == symbols.size() ? -1 : static_cast<int>(index + 1),
                        // Word-local, as the token boundary mapping adds the word's own begin.
                        .end      = symbols[index].source_end - begin};
        }

        std::priority_queue<BpeCandidate, std::vector<BpeCandidate>, LaterBpeCandidate> queue;
        const auto push_candidate = [&](int left) {
            if (left < 0 || !nodes[static_cast<std::size_t>(left)].live) { return; }
            const int right = nodes[static_cast<std::size_t>(left)].next;
            if (right < 0) { return; }
            const BpeMergeEntry* rule =
                merge_rules.find(merge_pair_key(nodes[static_cast<std::size_t>(left)].symbol,
                                                nodes[static_cast<std::size_t>(right)].symbol));
            if (rule == nullptr) { return; }
            queue.push(BpeCandidate{
                .rank             = rule->rank,
                .left             = left,
                .right            = right,
                .result           = rule->result,
                .left_generation  = nodes[static_cast<std::size_t>(left)].generation,
                .right_generation = nodes[static_cast<std::size_t>(right)].generation,
            });
        };
        for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
            push_candidate(static_cast<int>(index));
        }
        while (!queue.empty()) {
            const BpeCandidate candidate = queue.top();
            queue.pop();
            BpeNode& left  = nodes[static_cast<std::size_t>(candidate.left)];
            BpeNode& right = nodes[static_cast<std::size_t>(candidate.right)];
            if (!left.live || !right.live || left.next != candidate.right ||
                left.generation != candidate.left_generation ||
                right.generation != candidate.right_generation) {
                continue;
            }
            left.symbol = candidate.result;
            left.end    = right.end;
            ++left.generation;
            left.next  = right.next;
            right.live = false;
            ++right.generation;
            if (left.next >= 0) {
                nodes[static_cast<std::size_t>(left.next)].previous = candidate.left;
            }
            push_candidate(left.previous);
            push_candidate(candidate.left);
        }
        for (int node = nodes.empty() ? -1 : 0; node >= 0;
             node     = nodes[static_cast<std::size_t>(node)].next) {
            ids.push_back(nodes[static_cast<std::size_t>(node)].symbol);
            if (token_ends != nullptr) {
                token_ends->push_back(begin + nodes[static_cast<std::size_t>(node)].end);
            }
            if (ids.size() == max_tokens) { return false; }
        }
        if (word_ends != nullptr) {
            word_ends->push_back(
                BpeWordEnd{.normalized_offset = end, .token_frontier = ids.size()});
        }
        begin = end;
    }
    return true;
}

struct IndexedByteBoundary {
    std::size_t offset = 0;
    std::size_t index  = 0;
};

bool append_ordinary_text(BoundaryEncodedText& encoded, std::string_view text,
                          std::size_t text_offset, std::span<const IndexedByteBoundary> boundaries,
                          const BpeMergeTable& merge_rules, const SymbolContext& symbols,
                          std::size_t max_tokens) {
    const std::size_t token_base = encoded.input_ids.size();
    if (text.empty()) {
        for (const IndexedByteBoundary boundary : boundaries) {
            encoded.boundaries[boundary.index] =
                TokenBoundaryResult{.exact_frontier = token_base, .stable_frontier = token_base};
        }
        return true;
    }

    const std::string normalized = normalize_for_family(text, symbols.family);
    const bool has_internal_boundary =
        std::ranges::any_of(boundaries, [&](IndexedByteBoundary boundary) {
            const std::size_t local = boundary.offset - text_offset;
            return local != 0 && local != text.size();
        });
    std::vector<std::size_t> token_ends;
    std::vector<BpeWordEnd> word_ends;
    if (has_internal_boundary) { token_ends.reserve(normalized.size()); }
    if (!append_normalized_bpe_ids(encoded.input_ids, normalized, merge_rules, symbols,
                                   max_tokens, has_internal_boundary ? &token_ends : nullptr,
                                   has_internal_boundary ? &word_ends : nullptr)) {
        return false;
    }
    const std::size_t token_end = encoded.input_ids.size();
    if (!has_internal_boundary) {
        for (const IndexedByteBoundary boundary : boundaries) {
            const std::size_t frontier = boundary.offset == text_offset ? token_base : token_end;
            encoded.boundaries[boundary.index] =
                TokenBoundaryResult{.exact_frontier = frontier, .stable_frontier = frontier};
        }
        return true;
    }

    std::vector<std::size_t> normalized_boundaries(boundaries.size());
    std::string independently_normalized;
    independently_normalized.reserve(normalized.size());
    std::size_t raw_begin = 0;
    for (std::size_t request = 0; request < boundaries.size(); ++request) {
        const std::size_t raw_end = boundaries[request].offset - text_offset;
        if (raw_end != raw_begin) {
            independently_normalized += normalize_for_family(
                text.substr(raw_begin, raw_end - raw_begin), symbols.family);
            raw_begin = raw_end;
        }
        normalized_boundaries[request] = independently_normalized.size();
    }
    if (raw_begin != text.size()) {
        independently_normalized += normalize_for_family(text.substr(raw_begin), symbols.family);
    }

    if (independently_normalized != normalized) {
        // A requested marker splits a canonical-composition sequence. The full token stream stays
        // authoritative; conservatively retain only the preceding added-token/ordinary boundary.
        for (const IndexedByteBoundary boundary : boundaries) {
            const std::size_t local = boundary.offset - text_offset;
            if (local == 0 || local == text.size()) {
                const std::size_t frontier = local == 0 ? token_base : token_end;
                encoded.boundaries[boundary.index] =
                    TokenBoundaryResult{.exact_frontier = frontier, .stable_frontier = frontier};
            } else {
                encoded.boundaries[boundary.index] = TokenBoundaryResult{
                    .exact_frontier = std::nullopt, .stable_frontier = token_base};
            }
        }
        return true;
    }

    std::size_t token_cursor    = 0;
    std::size_t word_cursor     = 0;
    std::size_t stable_frontier = token_base;
    for (std::size_t request = 0; request < boundaries.size(); ++request) {
        const std::size_t normalized_boundary = normalized_boundaries[request];
        while (token_cursor < token_ends.size() &&
               token_ends[token_cursor] <= normalized_boundary) {
            ++token_cursor;
        }
        while (word_cursor < word_ends.size() &&
               word_ends[word_cursor].normalized_offset <= normalized_boundary) {
            stable_frontier = word_ends[word_cursor].token_frontier;
            ++word_cursor;
        }
        const bool exact =
            normalized_boundary == 0 ||
            (token_cursor != 0 && token_ends[token_cursor - 1] == normalized_boundary);
        const std::size_t frontier                    = token_base + token_cursor;
        encoded.boundaries[boundaries[request].index] = TokenBoundaryResult{
            .exact_frontier  = exact ? std::optional<std::size_t>(frontier) : std::nullopt,
            .stable_frontier = stable_frontier};
    }
    return true;
}

} // namespace

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("embedded tokenizer resources are empty");
    }
    family_ = resources.family;
    constexpr std::string_view tokenizer_label        = "tokenizer.json";
    constexpr std::string_view tokenizer_config_label = "tokenizer_config.json";
    const Json root = read_json_asset(resources.tokenizer_json, tokenizer_label);
    const Json tokenizer_config =
        read_json_asset(resources.tokenizer_config_json, tokenizer_config_label);
    const Json& model = require_object_field(root, "model", tokenizer_label);
    validate_pipeline(root, model, resources.family);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_label);
    decoded_token_bytes_         = std::move(vocab_metadata.id_to_token);
    vocab_token_to_id_           = std::move(vocab_metadata.token_to_id);
    valid_token_ids_.resize(decoded_token_bytes_.size());
    for (const int id : vocab_metadata.occupied_ids) {
        valid_token_ids_.at(static_cast<std::size_t>(id)) = true;
    }
    added_tokens_ = load_added_tokens(root, tokenizer_label, decoded_token_bytes_,
                                      vocab_metadata.occupied_ids, vocab_token_to_id_);
    merge_added_tokens_decoder(tokenizer_config, tokenizer_config_label, decoded_token_bytes_,
                               vocab_metadata.occupied_ids, vocab_token_to_id_, added_tokens_,
                               resources.family != TokenizerFamily::Gemma);
    for (std::size_t index = 0; index < added_tokens_.size(); ++index) {
        const std::string& content = added_tokens_[index].content;
        if (!content.empty()) {
            added_token_candidates_[static_cast<unsigned char>(content.front())].push_back(index);
        }
    }
    if (valid_token_ids_.size() < decoded_token_bytes_.size()) {
        valid_token_ids_.resize(decoded_token_bytes_.size());
    }
    std::vector<bool> added_token_ids(decoded_token_bytes_.size());
    special_token_ids_.resize(decoded_token_bytes_.size());
    for (const AddedToken& token : added_tokens_) {
        const auto index             = static_cast<std::size_t>(token.id);
        valid_token_ids_.at(index)   = true;
        added_token_ids.at(index)    = true;
        special_token_ids_.at(index) = token.special;
    }
    for (std::size_t index = 0; index < decoded_token_bytes_.size(); ++index) {
        if (family_ == TokenizerFamily::Gemma) {
            decoded_token_bytes_[index] =
                decode_gemma_token(decoded_token_bytes_[index], static_cast<int>(index));
        } else if (valid_token_ids_[index] && !added_token_ids[index]) {
            decoded_token_bytes_[index] =
                decode_byte_level_token(decoded_token_bytes_[index], static_cast<int>(index));
        }
    }
    bpe_merge_rules_        = load_bpe_merge_rules(model, tokenizer_label, vocab_token_to_id_);
    byte_token_ids_         = load_byte_token_ids(vocab_token_to_id_);
    if (family_ == TokenizerFamily::Gemma) {
        // Gemma's alphabet is characters, so every byte needs its `<0xXX>` fallback and the
        // U+2581 marker must exist: without them the encoder would differ from the checkpoint.
        byte_fallback_ids_     = load_byte_fallback_ids(vocab_token_to_id_);
        const auto marker      = vocab_token_to_id_.find(std::string(kGemmaSpaceMarker));
        const bool complete =
            marker != vocab_token_to_id_.end() &&
            std::all_of(byte_fallback_ids_.begin(), byte_fallback_ids_.end(),
                        [](int id) { return id >= 0; });
        if (!complete) {
            throw std::invalid_argument(
                "tokenizer.json lacks the Gemma alphabet: U+2581 marker and <0xXX> fallbacks");
        }
        gemma_marker_token_ = marker->second;
    }
    default_stop_token_ids_ = load_default_stop_token_ids(resources.generation_config_json);
}

std::vector<int> Tokenizer::encode(std::string_view text, EncodeOptions options) const {
    BoundaryEncodedText encoded = encode_with_boundaries(text, {}, options);
    return std::move(encoded.input_ids);
}

BoundaryEncodedText Tokenizer::encode_with_boundaries(
    std::string_view text, std::span<const std::size_t> byte_boundaries, EncodeOptions options,
    std::span<const text::ByteSpan> literal_spans) const {
    std::size_t previous_end = 0;
    for (const auto span : literal_spans) {
        if (span.begin < previous_end || span.begin >= span.end || span.end > text.size())
            throw std::invalid_argument(
                "Tokenizer literal spans must be ordered disjoint input ranges");
        previous_end = span.end;
    }
    BoundaryEncodedText encoded;
    encoded.boundaries.resize(byte_boundaries.size());
    std::vector<IndexedByteBoundary> boundaries;
    boundaries.reserve(byte_boundaries.size());
    for (std::size_t index = 0; index < byte_boundaries.size(); ++index) {
        if (byte_boundaries[index] > text.size()) {
            throw std::out_of_range("Tokenizer byte boundary exceeds input text");
        }
        boundaries.push_back(IndexedByteBoundary{.offset = byte_boundaries[index], .index = index});
    }
    std::ranges::stable_sort(boundaries, [](IndexedByteBoundary lhs, IndexedByteBoundary rhs) {
        return lhs.offset < rhs.offset;
    });
    if (options.max_tokens == 0) { return encoded; }

    std::size_t boundary_cursor     = 0;
    const auto boundary_end_through = [&](std::size_t offset) {
        std::size_t end = boundary_cursor;
        while (end < boundaries.size() && boundaries[end].offset <= offset) { ++end; }
        return end;
    };
    const SymbolContext symbols{.family         = family_,
                                .byte_token_ids = &byte_token_ids_,
                                .vocab          = &vocab_token_to_id_,
                                .marker_token   = gemma_marker_token_,
                                .byte_fallback  = &byte_fallback_ids_};
    const auto append_ordinary = [&](std::size_t begin, std::size_t end) {
        const std::size_t request_end = boundary_end_through(end);
        const bool complete          = append_ordinary_text(encoded, text.substr(begin, end - begin), begin,
                                 std::span<const IndexedByteBoundary>(boundaries)
                                     .subspan(boundary_cursor, request_end - boundary_cursor),
                                 bpe_merge_rules_, symbols, options.max_tokens);
        boundary_cursor = request_end;
        return complete;
    };

    if (!options.parse_added_tokens) {
        (void)append_ordinary(0, text.size());
        return encoded;
    }

    std::size_t ordinary_begin = 0;
    std::size_t pos            = 0;
    while (pos < text.size()) {
        const AddedToken* match_token = nullptr;
        const auto& candidates = added_token_candidates_[static_cast<unsigned char>(text[pos])];
        for (const std::size_t index : candidates) {
            const AddedToken& token = added_tokens_[index];
            if (token.content.size() <= text.size() - pos &&
                text.compare(pos, token.content.size(), token.content) == 0 &&
                !text::overlaps(literal_spans, pos, pos + token.content.size())) {
                match_token = &token;
                break;
            }
        }

        if (match_token == nullptr) {
            ++pos;
            continue;
        }

        if (!append_ordinary(ordinary_begin, pos)) { return encoded; }

        const std::size_t token_end       = pos + match_token->content.size();
        const std::size_t stable_frontier = encoded.input_ids.size();
        while (boundary_cursor < boundaries.size() &&
               boundaries[boundary_cursor].offset < token_end) {
            encoded.boundaries[boundaries[boundary_cursor].index] = TokenBoundaryResult{
                .exact_frontier = std::nullopt, .stable_frontier = stable_frontier};
            ++boundary_cursor;
        }
        encoded.input_ids.push_back(match_token->id);
        if (encoded.input_ids.size() == options.max_tokens) { return encoded; }
        while (boundary_cursor < boundaries.size() &&
               boundaries[boundary_cursor].offset == token_end) {
            encoded.boundaries[boundaries[boundary_cursor].index] =
                TokenBoundaryResult{.exact_frontier  = encoded.input_ids.size(),
                                    .stable_frontier = encoded.input_ids.size()};
            ++boundary_cursor;
        }
        pos            = token_end;
        ordinary_begin = pos;
    }
    (void)append_ordinary(ordinary_begin, text.size());
    return encoded;
}

std::string Tokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(id, options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
    return text;
}

DecodedTokenView Tokenizer::decoded_token(int id) const {
    if (id < 0) {
        throw std::invalid_argument("Tokenizer::decode received negative token id " +
                                    std::to_string(id));
    }
    const auto index = static_cast<std::size_t>(id);
    if (index >= decoded_token_bytes_.size() || index >= valid_token_ids_.size() ||
        !valid_token_ids_.at(index)) {
        throw std::out_of_range("Tokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }
    return DecodedTokenView{.bytes   = decoded_token_bytes_[index],
                            .special = special_token_ids_[index]};
}

std::string_view Tokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    const DecodedTokenView token = decoded_token(id);
    return skip_special_tokens && token.special ? std::string_view{} : token.bytes;
}

bool Tokenizer::is_special_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_token_ids_.size() &&
           special_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_valid_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::has_exact_token_domain(std::size_t size) const noexcept {
    return valid_token_ids_.size() == size &&
           std::ranges::find(valid_token_ids_, false) == valid_token_ids_.end();
}

} // namespace ninfer::models::qwen3_5::frontend
