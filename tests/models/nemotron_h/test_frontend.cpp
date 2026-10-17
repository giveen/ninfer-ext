// The Nemotron-H frontend: the load path, the chat template against the artifact's own rendering,
// the sampling and stop defaults the checkpoint declares, every refusal by name, and the output
// decoder's reasoning and tool-call semantics.
//
// Everything here needs the 45 GB artifact, so NINFER_NEMOTRON_ARTIFACT unset skips (77). The chat
// cases come from `tools/verify/nemotron_h_frontend_fixture.py`, which renders them with the
// artifact's own template through transformers' `apply_chat_template`.
#include "models/nemotron_h/frontend.h"

#include "artifact/schema.h"
#include "artifact_resources.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

using Json         = nlohmann::ordered_json;
namespace nemotron = ninfer::models::nemotron_h;
using nemotron::Frontend;
using nemotron::OutputSession;
using nemotron::PreparedPrompt;
using nemotron::PublishedOutput;

constexpr std::uint32_t kMaxContext = 65536;

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

ChatRole role_of(const std::string& name) {
    if (name == "system") return ChatRole::System;
    if (name == "developer") return ChatRole::Developer;
    if (name == "user") return ChatRole::User;
    if (name == "assistant") return ChatRole::Assistant;
    if (name == "tool") return ChatRole::Tool;
    throw std::invalid_argument("unknown role " + name);
}

PromptInput prompt_of(const Json& test) {
    PromptInput input;
    for (const Json& message : test.at("messages")) {
        ChatMessage out;
        out.role = role_of(message.at("role"));
        out.parts.push_back(MessagePart{.text = message.value("content", std::string{})});
        out.reasoning_content = message.value("reasoning_content", std::string{});
        out.tool_call_id      = message.value("tool_call_id", std::string{});
        if (message.contains("tool_calls")) {
            for (const Json& call : message.at("tool_calls")) {
                out.tool_calls.push_back(ToolCall{
                    .id             = call.at("id"),
                    .name           = call.at("function").at("name"),
                    .arguments_json = call.at("function").at("arguments").dump(),
                });
            }
        }
        input.messages.push_back(std::move(out));
    }
    if (test.contains("tools")) {
        for (const Json& tool : test.at("tools")) input.options.tool_jsons.push_back(tool.dump());
    }
    input.options.enable_thinking = test.at("thinking").get<bool>();
    return input;
}

PromptInput user_prompt(std::string text, bool thinking = false) {
    PromptInput input;
    input.messages.push_back(
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = std::move(text)}}});
    input.options.enable_thinking = thinking;
    return input;
}

constexpr std::string_view kWeatherTool =
    R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object",)"
    R"("properties":{"city":{"type":"string"}},"required":["city"]}}})";

// What one round published, split by channel.
struct Round {
    ninfer::runtime::OutputDecision decision;
    std::string reasoning;
    std::string content;
};

void collect(OutputSession& session, Round& round) {
    PublishedOutput published = session.commit_preview();
    for (const OutputDelta& delta : published) {
        (delta.channel == OutputChannel::Reasoning ? round.reasoning : round.content) += delta.text;
    }
}

// One generated round. A budget equal to the round's length is what makes the round terminal, which
// is when a withheld tool-call region is decided.
Round run(OutputSession& session, std::span<const TokenId> ids, std::uint32_t budget,
          FinishReason limit = FinishReason::OutputLimit) {
    Round round;
    round.decision = session.preview_model(ids, budget, limit);
    collect(session, round);
    return round;
}

Round terminate(OutputSession& session, FinishReason reason) {
    Round round;
    round.decision = session.preview_terminal(reason);
    collect(session, round);
    return round;
}

// The refusal one request produces, or an empty string when it is accepted.
template <typename Call>
std::string refusal(Call&& call) {
    try {
        call();
        return {};
    } catch (const RequestError& error) { return error.what(); }
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
        const Frontend frontend(resources.frontend(), kMaxContext);

        // ---- The chat template, against the artifact's own rendering -------------------------
        const Json fixture =
            ninfer::artifact::parse_json(read_file(std::string(NINFER_SOURCE_DIR) +
                                                   "/tests/fixtures/nemotron_h/chat_cases.json"),
                                         "nemotron_h chat fixture");
        std::size_t cases = 0;
        for (const Json& test : fixture.at("cases")) {
            const std::string name     = test.at("name").get<std::string>();
            const PromptInput input    = prompt_of(test);
            const std::string rendered = frontend.render(input);
            if (rendered != test.at("text").get<std::string>()) {
                expect(false, "render of " + name +
                                  " does not match the template's own rendering: " +
                                  Json(rendered).dump());
                continue;
            }
            const PreparedPrompt prompt = frontend.prepare(input);
            std::vector<int> want;
            for (const Json& id : test.at("ids")) want.push_back(id.get<int>());
            const bool same_ids = want.size() == prompt.tokens().size() &&
                                  std::equal(want.begin(), want.end(), prompt.tokens().begin());
            expect(same_ids, "ids of " + name + " do not match the template's own rendering");
            expect(prompt.summary().prompt_tokens == prompt.tokens().size(),
                   "prompt_tokens does not describe the prompt");
            expect(prompt.summary().starts_in_reasoning == test.at("thinking").get<bool>(),
                   "starts_in_reasoning does not follow the request's thinking mode");
            expect(!prompt.summary().has_media, "a text prompt reports media");
            ++cases;
        }
        std::cout << "chat template: " << cases << " cases match the artifact's own rendering\n";

        // ---- Sampling and stops, from the artifact's own generation config -------------------
        const Json generation   = Json::parse(resources.frontend().generation_config_json);
        const float temperature = generation.at("temperature").get<float>();
        const float top_p       = generation.at("top_p").get<float>();
        std::vector<int> eos;
        for (const Json& id : generation.at("eos_token_id")) eos.push_back(id.get<int>());
        expect(temperature == 1.0F && top_p == 0.95F,
               "the checkpoint's own sampling recommendation changed: " + generation.dump());
        expect(eos == std::vector<int>({2, 11}),
               "the checkpoint's own end-of-sequence ids changed: " + generation.dump());
        for (const SamplingPreset& preset :
             {frontend.sampling_defaults().thinking, frontend.sampling_defaults().non_thinking}) {
            expect(preset.temperature == temperature,
                   "the registered temperature is not the checkpoint's");
            expect(preset.top_p == top_p, "the registered top_p is not the checkpoint's");
            expect(preset.top_k == 20,
                   "the registered top_k is not what the sampler can draw from");
        }
        std::cout << "sampling: temperature " << temperature << ", top_p " << top_p
                  << ", top_k 20 for both thinking modes\n";

        const PreparedPrompt prompt = frontend.prepare(user_prompt("Say hello."));
        for (const int id : eos) {
            OutputSession session = frontend.make_output_session(prompt, StopPolicy{});
            const Round round     = run(session, std::span<const TokenId>(&id, 1), 1);
            expect(round.decision.finish_reason == FinishReason::StopToken,
                   "end-of-sequence token " + std::to_string(id) + " does not stop the request");
        }
        std::cout << "stops: " << eos.size() << " end-of-sequence ids stop the request\n";

        // ---- The refusals, by name ----------------------------------------------------------
        const std::vector<std::string> messages{
            refusal([&] {
                OutputSession session = frontend.make_output_session(
                    prompt, StopPolicy{}, OutputOptions{},
                    ThinkingControlOptions{.budget = std::uint32_t{16}});
                (void)session;
            }),
            refusal([&] {
                OutputSession session = frontend.make_output_session(
                    prompt, StopPolicy{}, OutputOptions{}, ThinkingControlOptions{},
                    OutputConstraint::json_object());
                (void)session;
            }),
            refusal([&] {
                PromptInput with_tools = user_prompt("Weather?");
                with_tools.options.tool_jsons.push_back(std::string(kWeatherTool));
                const PreparedPrompt tool_prompt = frontend.prepare(with_tools);
                OutputSession session            = frontend.make_output_session(
                    tool_prompt, StopPolicy{}, OutputOptions{}, ThinkingControlOptions{}, {},
                    ToolChoice{.mode = ToolChoiceMode::Required});
                (void)session;
            }),
            refusal([&] {
                PromptInput with_tools = user_prompt("Weather?");
                with_tools.options.tool_jsons.push_back(std::string(kWeatherTool));
                const PreparedPrompt tool_prompt = frontend.prepare(with_tools);
                OutputSession session            = frontend.make_output_session(
                    tool_prompt, StopPolicy{}, OutputOptions{}, ThinkingControlOptions{}, {},
                    ToolChoice{.mode          = ToolChoiceMode::Auto,
                               .allowed_names = std::vector<std::string>{"get_weather"}});
                (void)session;
            }),
            refusal([&] {
                PromptInput with_image;
                ChatMessage message{.role = ChatRole::User};
                message.parts.push_back(MessagePart{.text = "What is this?"});
                MessagePart media;
                media.kind        = MessagePartKind::Media;
                media.media.kind  = MediaKind::Image;
                media.media.bytes = {1, 2, 3, 4};
                message.parts.push_back(std::move(media));
                with_image.messages.push_back(std::move(message));
                (void)frontend.prepare(with_image);
            }),
        };
        for (const std::string& message : messages) {
            expect(!message.empty(), "a refusal was expected");
            std::cout << "refused: " << message << '\n';
        }

        // ---- The output decoder --------------------------------------------------------------
        // A thinking prompt opens the block, and the close marker is in neither channel, so the
        // reasoning a caller replays through the template reproduces the model's own text.
        const PreparedPrompt thinking_prompt = frontend.prepare(user_prompt("Capital?", true));
        expect(thinking_prompt.summary().starts_in_reasoning,
               "a thinking prompt does not report that it starts in reasoning");
        const std::vector<TokenId> script =
            frontend.tokenize_text("Weighing it up</think>The capital of France is Paris.");
        OutputSession thinking = frontend.make_output_session(thinking_prompt, StopPolicy{});
        const Round split      = run(thinking, script, 4096);
        expect(split.reasoning == "Weighing it up",
               "reasoning channel is " + Json(split.reasoning).dump());
        expect(split.content == "The capital of France is Paris.",
               "content channel is " + Json(split.content).dump());
        expect(thinking.reasoning_tokens() != 0, "the reasoning tokens are not counted");
        std::cout << "reasoning: " << Json(split.reasoning).dump() << " then "
                  << Json(split.content).dump() << " over " << thinking.reasoning_tokens()
                  << " reasoning tokens\n";

        // ultra_v3's second rule: the model echoes the start marker, and the template's own
        // <think>\n is not repeated in the reasoning channel.
        OutputSession echoing = frontend.make_output_session(thinking_prompt, StopPolicy{});
        const std::vector<TokenId> echo = frontend.tokenize_text("<think>\nechoed</think>answer");
        const Round echoed              = run(echoing, echo, 4096);
        expect(echoed.reasoning == "echoed",
               "an echoed start marker is not dropped: " + Json(echoed.reasoning).dump());
        expect(echoed.content == "answer",
               "content after the echo is " + Json(echoed.content).dump());

        // Thinking off is no reasoning channel at all, including a stray close marker.
        OutputSession off = frontend.make_output_session(
            frontend.prepare(user_prompt("Capital?", false)), StopPolicy{});
        const std::vector<TokenId> stray = frontend.tokenize_text("stray</think>answer");
        const Round strayed              = run(off, stray, 4096);
        expect(strayed.reasoning.empty(),
               "a non-thinking request produced reasoning: " + Json(strayed.reasoning).dump());
        expect(strayed.content == "stray</think>answer",
               "a non-thinking request altered its content: " + Json(strayed.content).dump());

        // ultra_v3's third rule: the block never closes, and the request asked for content.
        PromptInput forced_input                       = user_prompt("Capital?", true);
        forced_input.options.chat_template_kwargs_json = R"({"force_nonempty_content": true})";
        const PreparedPrompt forced_prompt             = frontend.prepare(forced_input);
        expect(forced_prompt.force_nonempty_content,
               "the forced-content kwarg did not reach the decoder");
        OutputSession forced = frontend.make_output_session(forced_prompt, StopPolicy{});
        const std::vector<TokenId> only_reasoning =
            frontend.tokenize_text("Still thinking about it");
        (void)run(forced, only_reasoning, 4096);
        const Round limit = terminate(forced, FinishReason::OutputLimit);
        expect(limit.reasoning.empty() && limit.content == "Still thinking about it",
               "an unclosed block did not become content: " + Json(limit.reasoning).dump() + " / " +
                   Json(limit.content).dump());
        std::cout << "unclosed thinking block became content: " << Json(limit.content).dump()
                  << '\n';

        // A stop string, in the channel that declared it.
        StopPolicy stop_strings;
        stop_strings.strings.push_back(
            StopString{.text = "\n\n", .channel = OutputChannel::Content});
        OutputSession stopped = frontend.make_output_session(prompt, stop_strings);
        const std::vector<TokenId> paragraphs =
            frontend.tokenize_text("First paragraph.\n\nSecond paragraph.");
        const Round cut = run(stopped, paragraphs, 4096);
        expect(cut.decision.finish_reason == FinishReason::StopString,
               "a stop string did not stop the request");
        expect(cut.content == "First paragraph.",
               "stop-string content is " + Json(cut.content).dump());
        expect(stopped.matched_stop_string() == std::optional<std::string>("\n\n"),
               "the matched stop string is not reported");

        // A tool call, in the wire format the artifact's template declares.
        PromptInput weather = user_prompt("Weather in Paris?");
        weather.options.tool_jsons.push_back(std::string(kWeatherTool));
        const PreparedPrompt weather_prompt = frontend.prepare(weather);
        expect(weather_prompt.tool_names != nullptr && weather_prompt.tool_names->size() == 1 &&
                   weather_prompt.tool_names->front() == "get_weather",
               "the prompt did not carry its tool declaration");
        OutputSession calling = frontend.make_output_session(weather_prompt, StopPolicy{});
        const std::vector<TokenId> call_text = frontend.tokenize_text(
            "Checking the forecast.\n<tool_call>\n<function=get_weather>\n<parameter=city>\n"
            "Paris\n</parameter>\n</function>\n</tool_call>");
        // A call region is withheld until the round ends, so a caller streaming tokens sees the
        // prose and not half a call; this model ends its turn with <|im_end|>, which is a stop
        // token.
        const Round streamed = run(calling, call_text, 4096);
        expect(streamed.content == "Checking the forecast.",
               "the prose before a call is " + Json(streamed.content).dump());
        expect(calling.take_tool_calls().empty(), "a call was published before its region closed");
        const TokenId end_of_turn = eos.back();
        const Round answered      = run(calling, std::span<const TokenId>(&end_of_turn, 1), 1);
        expect(answered.decision.finish_reason == FinishReason::StopToken,
               "the end-of-turn token did not stop the request");
        const std::vector<GeneratedToolCall> calls = calling.take_tool_calls();
        expect(calls.size() == 1 && calls.front().name == "get_weather",
               "the tool call was not parsed: " + Json(answered.content).dump());
        if (calls.size() == 1) {
            expect(Json::parse(calls.front().arguments_json) == Json{{"city", "Paris"}},
                   "the tool arguments are " + calls.front().arguments_json);
        }
        expect(answered.content.empty(),
               "a parsed call left content behind: " + Json(answered.content).dump());
        std::cout << "tool call: " << calls.size() << " call(s) after "
                  << Json(streamed.content).dump() << '\n';

        // A malformed region is the model's own bytes, verbatim: nothing ever constrained it.
        OutputSession malformed = frontend.make_output_session(weather_prompt, StopPolicy{});
        const std::vector<TokenId> broken =
            frontend.tokenize_text("<tool_call>\n<function=get_weather>\nno parameter");
        (void)run(malformed, broken, 4096);
        const Round returned = run(malformed, std::span<const TokenId>(&end_of_turn, 1), 1);
        expect(malformed.take_tool_calls().empty(), "a malformed region produced a call");
        expect(returned.content == "<tool_call>\n<function=get_weather>\nno parameter",
               "a malformed region was not returned as content: " + Json(returned.content).dump());

        // Raw output is the model's bytes, with no protocol interpretation at all.
        OutputSession raw_session =
            frontend.make_output_session(thinking_prompt, StopPolicy{}, OutputOptions{.raw = true});
        const Round raw = run(raw_session, script, 4096);
        expect(raw.reasoning.empty() &&
                   raw.content == "Weighing it up</think>The capital of France is Paris.",
               "raw output was interpreted: " + Json(raw.content).dump());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "the frontend cases failed\n";
        return 1;
    }
    std::cout << "the Nemotron-H frontend matches the artifact's template and protocol\n";
    return 0;
}
