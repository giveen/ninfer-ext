// The Nemotron-H frontend's load path: the artifact's own text resources -- the tokenizer, its
// config, the generation config and the chat template, as the loader resolves them -- build a
// frontend that prepares this checkpoint's prompt in both thinking modes.
//
// NINFER_NEMOTRON_ARTIFACT unset skips (77): the artifact is 45 GB and a routine run must not pay
// for it. `ninfer_nemotron_h_frontend_test` carries the protocol cases; this is the wiring they
// both rest on, kept separate so a resource or constructor failure is readable on its own.
#include "models/nemotron_h/frontend.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "artifact_resources.h"

#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

namespace nemotron = ninfer::models::nemotron_h;
namespace tf       = ninfer::models::qwen3_5::frontend;

constexpr std::uint32_t kMaxContext = 65536;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

PromptInput weather_prompt(std::string question, bool thinking) {
    PromptInput input;
    input.messages.push_back(
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = std::move(question)}}});
    input.options.enable_thinking = thinking;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object",)"
        R"("properties":{"city":{"type":"string"}},"required":["city"]}}})");
    return input;
}

} // namespace

int main() {
    const char* artifact = ninfer::test::nemotron_artifact_path();
    if (artifact == nullptr) {
        std::cout << "SKIP: NINFER_NEMOTRON_ARTIFACT is not set\n";
        return 77;
    }

    try {
        const ninfer::test::NemotronResources resources(artifact);
        const nemotron::FrontendResources& text = resources.frontend();
        expect(!text.tokenizer_json.empty() && !text.tokenizer_config_json.empty() &&
                   !text.generation_config_json.empty() && !text.chat_template_jinja.empty(),
               "the loader resolved an empty text resource");
        expect(text.tokenizer_json.front() == '{' && text.chat_template_jinja.front() != '\0',
               "the resolved resources are not the artifact's own text");
        std::cout << "resources: tokenizer.json " << text.tokenizer_json.size()
                  << " B, tokenizer_config.json " << text.tokenizer_config_json.size()
                  << " B, generation_config.json " << text.generation_config_json.size()
                  << " B, chat_template.jinja " << text.chat_template_jinja.size() << " B\n";

        const nemotron::Frontend frontend(text, kMaxContext);
        // The family's frontend is wired to the one shared tokenizer, so the ids it prepares are
        // the ids that tokenizer produces for the text the template rendered.
        const tf::Tokenizer tokenizer(
            tf::TokenizerResources{text.tokenizer_json, text.tokenizer_config_json,
                                   text.generation_config_json, tf::TokenizerFamily::Nemotron});
        for (const bool thinking : {false, true}) {
            const std::string what                = thinking ? "thinking" : "non-thinking";
            const PromptInput input               = weather_prompt("Weather in Paris?", thinking);
            const std::string rendered            = frontend.render(input);
            const nemotron::PreparedPrompt prompt = frontend.prepare(input);
            const std::vector<int> encoded        = tokenizer.encode(rendered);
            expect(!prompt.ids.empty(), what + ": the prepared prompt is empty");
            expect(prompt.summary().prompt_tokens == prompt.ids.size(),
                   what + ": prompt_tokens does not describe the prepared ids");
            expect(std::vector<TokenId>(encoded.begin(), encoded.end()) == prompt.ids,
                   what + ": the prepared ids are not the tokenizer's own encoding of the render");
            expect(prompt.summary().starts_in_reasoning == thinking,
                   what + ": starts_in_reasoning does not follow the request");
            expect(!prompt.summary().has_media, what + ": a text prompt reports media");
            expect(prompt.tool_names != nullptr && prompt.tool_names->size() == 1 &&
                       prompt.tool_names->front() == "get_weather",
                   what + ": the prompt did not carry its tool declaration");
            expect(prompt.tool_call_contract != nullptr, what + ": no tool contract was compiled");
            expect(frontend.count_tokens(input) == prompt.summary().prompt_tokens,
                   what + ": count_tokens disagrees with prepare");
            expect(prompt.preparation.tokenize_seconds <= prompt.preparation.seconds + 1e-9,
                   what + ": tokenize time exceeds the preparation it belongs to");
            std::cout << what << ": " << prompt.ids.size() << " prompt tokens, "
                      << (prompt.summary().starts_in_reasoning ? "starts" : "does not start")
                      << " in reasoning\n";

            // The caller owns a token sequence it hands over, and gets exactly it back.
            const nemotron::PreparedPrompt given =
                frontend.prepare_tokens(std::vector<TokenId>(prompt.ids));
            expect(given.ids == prompt.ids && given.summary().prompt_tokens == prompt.ids.size(),
                   what + ": prepare_tokens is not transparent");
            const std::vector<TokenId> plain = frontend.tokenize_text("Weather in Paris?");
            expect(!plain.empty(), what + ": tokenize_text produced no tokens");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "the load path failed\n";
        return 1;
    }
    std::cout << "the Nemotron-H frontend loads from the artifact's own resources\n";
    return 0;
}
