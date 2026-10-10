// Gemma 4 frontend: the chat template against HuggingFace's own rendering, and the output decoder's
// channel, tool-call and stop semantics.
//
// The template cases come from `tools/verify/gemma4_chat_fixture.py` (transformers' apply_chat_template
// over the checkpoint's own template); the rendered text and its token ids must match exactly. Point
// NINFER_GEMMA_CHECKPOINT at a Gemma 4 checkpoint directory to run.
#include "models/gemma4/frontend.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "artifact/schema.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma = ninfer::models::gemma4;
namespace tf    = ninfer::models::qwen3_5::frontend;
using Json      = nlohmann::json;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
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

struct Decoded {
    std::string reasoning;
    std::string content;
    FinishReason finish = FinishReason::None;
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
};

// Drives a session the way the Engine does: one token per round, each preview committed.
Decoded decode(gemma::OutputSession session, std::span<const int> tokens, std::uint32_t budget) {
    Decoded out;
    for (std::size_t index = 0; index < tokens.size() && out.finish == FinishReason::None; ++index) {
        const TokenId token = tokens[index];
        const auto decision = session.preview_model(std::span<const TokenId>(&token, 1),
                                                    budget - static_cast<std::uint32_t>(index),
                                                    FinishReason::OutputLimit);
        for (const OutputDelta& delta : session.commit_preview()) {
            (delta.channel == OutputChannel::Reasoning ? out.reasoning : out.content) += delta.text;
        }
        out.finish = decision.finish_reason;
    }
    out.calls       = session.take_tool_calls();
    out.diagnostics = session.tool_call_parse_diagnostics();
    return out;
}

void check_tool_call_syntax() {
    const auto call = gemma::parse_tool_call(
        "call:lookup{query:<|\"|>a, b: {c}<|\"|>,limit:-2.5,exact:true,missing:null,"
        "filters:{tags:[<|\"|>x<|\"|>,<|\"|>y<|\"|>],<|\"|>quoted key<|\"|>:[]}}");
    expect(call.has_value(), "a nested call parses");
    if (call) {
        expect(call->name == "lookup", "the call's name is its function");
        const Json arguments = Json::parse(call->arguments_json);
        expect(arguments.at("query") == "a, b: {c}", "a string keeps the syntax characters inside it");
        expect(arguments.at("limit") == -2.5, "a negative float parses");
        expect(arguments.at("exact") == true && arguments.at("missing").is_null(),
               "booleans and null parse");
        expect(arguments.at("filters").at("tags") == Json::array({"x", "y"}), "arrays parse");
        expect(arguments.at("filters").at("quoted key") == Json::array(), "a quoted key parses");
    }
    expect(gemma::parse_tool_call("call:f{}").has_value(), "an empty argument object parses");
    expect(!gemma::parse_tool_call("call:f{a:<|\"|>open}").has_value(),
           "an unterminated string is refused");
    expect(!gemma::parse_tool_call("call:f{a:1} trailing").has_value(), "trailing text is refused");
    expect(!gemma::parse_tool_call("call:bad name{}").has_value(), "a name with a space is refused");
    expect(!gemma::parse_tool_call("declaration:f{}").has_value(), "only a call is a call");
}

} // namespace

int main() {
    const char* checkpoint = std::getenv("NINFER_GEMMA_CHECKPOINT");
    check_tool_call_syntax();
    if (checkpoint == nullptr || *checkpoint == '\0') {
        std::cout << "skip: NINFER_GEMMA_CHECKPOINT is not set\n";
        return failures == 0 ? 77 : 1;
    }
    const std::string dir(checkpoint);
    const gemma::FrontendResources resources{
        .tokenizer_json         = read_file(dir + "/tokenizer.json"),
        .tokenizer_config_json  = read_file(dir + "/tokenizer_config.json"),
        .generation_config_json = read_file(dir + "/generation_config.json"),
        .chat_template_jinja    = read_file(dir + "/chat_template.jinja"),
    };
    const gemma::Frontend frontend(resources, 1u << 18);
    const tf::Tokenizer tokenizer(tf::TokenizerResources{
        resources.tokenizer_json, resources.tokenizer_config_json, resources.generation_config_json,
        tf::TokenizerFamily::Gemma});

    // The template, against HF.
    const Json fixture = ninfer::artifact::parse_json(
        read_file("tests/fixtures/gemma4/chat_template.json"), "gemma4 chat fixture");
    for (const Json& test : fixture.at("cases")) {
        const std::string name     = test.at("name");
        const std::string expected = test.at("text");
        const std::string rendered = frontend.render(prompt_of(test));
        expect(rendered == expected, name + ": the rendered prompt differs from HF's\n  ours: " +
                                         Json(rendered).dump() + "\n  HF:   " + Json(expected).dump());
        const gemma::PreparedPrompt prepared = frontend.prepare(prompt_of(test));
        const std::vector<TokenId> ids       = test.at("ids").get<std::vector<TokenId>>();
        expect(prepared.ids == ids, name + ": the prompt's token ids differ from HF's");
    }

    const ModelSamplingDefaults defaults = frontend.sampling_defaults();
    expect(defaults.thinking.temperature == 1.0F && defaults.thinking.top_k == 20 &&
               defaults.thinking.top_p == 0.95F,
           "sampling defaults come from generation_config.json");

    // A thinking answer: the channel becomes reasoning, the rest content, and <turn|> stops.
    PromptInput plain;
    plain.messages.push_back(ChatMessage{.parts = {MessagePart{.text = "2+2?"}}});
    const gemma::PreparedPrompt plain_prompt = frontend.prepare(plain);
    {
        const std::vector<int> tokens =
            tokenizer.encode("<|channel>thought\nAdd them.<channel|>The answer is 4.<turn|>ignored");
        const Decoded out = decode(frontend.make_output_session(plain_prompt, {}), tokens, 64);
        expect(out.reasoning == "Add them.", "the thought channel is reasoning: '" + out.reasoning + "'");
        expect(out.content == "The answer is 4.", "the answer is content: '" + out.content + "'");
        expect(out.finish == FinishReason::StopToken, "<turn|> stops the turn");
    }
    // Raw output keeps the protocol text and splits nothing.
    {
        const std::vector<int> tokens = tokenizer.encode("<|channel>thought\nA<channel|>B<turn|>");
        const Decoded out = decode(
            frontend.make_output_session(plain_prompt, {}, OutputOptions{.raw = true}), tokens, 64);
        expect(out.reasoning.empty() && out.content == "<|channel>thought\nA<channel|>B",
               "raw output is the model's text: '" + out.content + "'");
    }
    // A stop string ends the content where it starts.
    {
        StopPolicy stop;
        stop.strings.push_back(StopString{.text = "STOP"});
        const std::vector<int> tokens = tokenizer.encode("Hello STOP world");
        const Decoded out = decode(frontend.make_output_session(plain_prompt, stop), tokens, 64);
        expect(out.content == "Hello ", "a stop string truncates: '" + out.content + "'");
        expect(out.finish == FinishReason::StopString, "a stop string finishes the request");
    }
    // The budget ends a request without a stop.
    {
        const std::vector<int> tokens = tokenizer.encode("one two three four");
        const Decoded out = decode(frontend.make_output_session(plain_prompt, {}), tokens, 2);
        expect(out.finish == FinishReason::OutputLimit, "the budget finishes the request");
    }

    // A thinking budget: the channel's tokens count against it, and at the budget the session asks
    // for <channel|> to be forced. After the forced close the model's tokens are the answer.
    {
        gemma::OutputSession session = frontend.make_output_session(
            plain_prompt, {}, {}, ThinkingControlOptions{.budget = 4});
        const std::vector<int> thought = tokenizer.encode("<|channel>thought\nlong long long long");
        std::string reasoning, content;
        const auto collect = [&](const gemma::PublishedOutput& output) {
            for (const OutputDelta& delta : output) {
                (delta.channel == OutputChannel::Reasoning ? reasoning : content) += delta.text;
            }
        };
        runtime::OutputDecision decision{};
        std::size_t fed = 0;
        for (; fed < thought.size(); ++fed) {
            expect(session.model_token_budget_remaining(100) > 0, "the budget admits each token");
            const TokenId token = thought[fed];
            decision = session.preview_model(std::span<const TokenId>(&token, 1), 100,
                                             FinishReason::OutputLimit);
            collect(session.commit_preview());
            if (decision.continuation == runtime::ContinuationAction::ApplyTargetControl) break;
        }
        expect(fed == 3, "the fourth thinking token reaches the budget, after " +
                             std::to_string(fed + 1) + " tokens");
        expect(session.model_token_budget_remaining(100) == 0, "no model token passes a pending close");
        const auto control = session.pending_control_tokens();
        expect(control.size() == 1, "the forced span is one token");
        if (control.size() == 1) {
            const std::vector<TokenId> forced(control.begin(), control.end());
            const auto applied = session.preview_control(forced, 100);
            expect(applied.accepted_tokens == 1, "the close is accepted");
            collect(session.commit_preview());
        }
        const std::vector<int> answer = tokenizer.encode("Four.<turn|>");
        for (const int token : answer) {
            const TokenId id = token;
            decision = session.preview_model(std::span<const TokenId>(&id, 1), 100,
                                             FinishReason::OutputLimit);
            collect(session.commit_preview());
            if (decision.finish_reason != FinishReason::None) break;
        }
        // The four budgeted tokens are <|channel>, "thought", "\n" and "long".
        expect(reasoning == "long", "the reasoning is cut at the budget: '" + reasoning + "'");
        expect(content == "Four.", "the answer follows the forced close: '" + content + "'");
        const ThinkingBudgetStats stats = session.thinking_stats();
        expect(stats.applied && stats.model_thinking_tokens == 4 && stats.injected_tokens == 1,
               "the stats record the forced close");
    }

    // Tool calls.
    PromptInput with_tools = plain;
    with_tools.options.tool_jsons.push_back(
        Json{{"type", "function"},
             {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}}
            .dump());
    const gemma::PreparedPrompt tool_prompt = frontend.prepare(with_tools);
    {
        const std::vector<int> tokens = tokenizer.encode(
            "Checking.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>"
            "<|tool_response>");
        const Decoded out = decode(frontend.make_output_session(tool_prompt, {}), tokens, 64);
        expect(out.finish == FinishReason::StopToken, "<|tool_response> hands control to the tools");
        expect(out.content == "Checking.", "text before a call stays content: '" + out.content + "'");
        expect(out.calls.size() == 1, "one call is parsed");
        if (out.calls.size() == 1) {
            expect(out.calls[0].name == "get_weather", "the call names its function");
            expect(Json::parse(out.calls[0].arguments_json) == Json{{"city", "Paris"}, {"days", 2}},
                   "the arguments are JSON: " + out.calls[0].arguments_json);
        }
    }
    {
        const std::vector<int> tokens = tokenizer.encode(
            "<|tool_call>call:get_weather{city:<|\"|>Paris}<tool_call|><turn|>");
        const Decoded out = decode(frontend.make_output_session(tool_prompt, {}), tokens, 64);
        expect(out.calls.empty(), "a malformed call is not a call");
        expect(out.content == "<|tool_call>call:get_weather{city:<|\"|>Paris}<tool_call|>",
               "a malformed call returns to content: '" + out.content + "'");
        expect(out.diagnostics.fallback_reason == ToolCallParseFallbackReason::MalformedStructure,
               "the fallback says why");
    }
    {
        const std::vector<int> tokens =
            tokenizer.encode("<|tool_call>call:unknown{}<tool_call|><|tool_response>");
        const Decoded out = decode(frontend.make_output_session(tool_prompt, {}), tokens, 64);
        expect(out.calls.empty() &&
                   out.diagnostics.fallback_reason == ToolCallParseFallbackReason::UndeclaredTool,
               "a call to an undeclared tool is not a call");
    }
    {
        bool refused = false;
        try {
            (void)frontend.make_output_session(tool_prompt, {}, {}, {}, {},
                                               ToolChoice{.mode = ToolChoiceMode::Required});
        } catch (const RequestError& error) {
            refused = error.kind() == RequestErrorKind::InvalidToolConstraint;
        }
        expect(refused, "a required tool choice is refused rather than ignored");
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gemma4 frontend: " << fixture.at("cases").size()
              << " template cases match HF exactly; decoder checks pass\n";
    return 0;
}
