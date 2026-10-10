#pragma once

// Gemma 4's frontend: the chat template, the tokenizer, and the decoder that turns generated tokens
// back into the Engine's output channels.
//
// The protocol is Gemma's own, so nothing here is shared with the Qwen frontend except the format-
// neutral parts it is built from: the Jinja engine and the tokenizer implementation.
//
//   turns        <|turn>role\n ... <turn|>\n, the assistant's role being `model`
//   thinking     <|think|> at the top of the system turn switches it on; the model then opens its
//                answer with <|channel>thought\n ... <channel|>. With thinking off the template
//                closes an empty channel itself, so the answer is content from its first token.
//   tool calls   <|tool_call>call:NAME{key:value,...}<tool_call|>, strings delimited by <|"|>, and
//                then <|tool_response>, which is a default stop token: that is the model handing
//                control to the tool runner.

#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {
class Tokenizer;
} // namespace ninfer::models::qwen3_5::frontend

namespace ninfer::models::gemma4 {

// The text resources a frontend is built from, owned: an artifact's bytes live only as long as the
// load's Reader.
struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
    std::string chat_template_jinja;
};

class PreparedPrompt {
public:
    std::vector<TokenId> ids;
    PromptSummary descriptor;
    PromptPreparationStats preparation;
    // Tool declarations the prompt carried, by name, so the decoder can tell a declared call from
    // prose that merely looks like one.
    std::shared_ptr<const std::vector<std::string>> tool_names;

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return ids; }
    [[nodiscard]] const PromptSummary& summary() const noexcept { return descriptor; }
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept {
        return preparation;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return !ids.empty(); }
};

// One preview's published deltas, in order, adjacent deltas of one channel merged.
class PublishedOutput {
public:
    [[nodiscard]] bool empty() const noexcept { return deltas_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return deltas_.size(); }
    [[nodiscard]] auto begin(this auto& self) noexcept { return self.deltas_.begin(); }
    [[nodiscard]] auto end(this auto& self) noexcept { return self.deltas_.end(); }
    void append(OutputChannel channel, std::string text);
    void clear() noexcept { deltas_.clear(); }

private:
    std::vector<OutputDelta> deltas_;
};

class Frontend;

// The per-request output decoder the Engine drives: it previews each generated round, decides where
// the request stops, and publishes the committed text. The contract is the Engine's; its semantics
// are Gemma's protocol above.
class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;
    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    void discard_preview();

    // Gemma has no thinking budget and no forced control suffix: the model budget is the request's,
    // and no control is ever pending.
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
        return total_budget_remaining;
    }
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t control_suffix_tokens() const noexcept { return 0; }
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t) const noexcept {}

    // No output constraint reaches this session: the Program consumes no token masks, so the Engine
    // refuses a constrained request before it gets here.
    [[nodiscard]] bool constrained() const noexcept { return false; }
    void observe_constraint(bool, double) noexcept {}
    void constraint_uploaded(std::size_t) noexcept {}
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const {
        return {};
    }
    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>);

    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept { return {}; }
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const FrontendResources& resources, std::uint32_t max_context);
    ~Frontend();
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    Frontend(const Frontend&)            = delete;
    Frontend& operator=(const Frontend&) = delete;

    // Renders a conversation with the artifact's chat template and tokenizes it.
    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    // Exactly the ids given: the caller owns the sequence, beginning-of-sequence token included.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    // Text as the model would see it, which for this tokenizer starts with <bos>.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const { return {}; }
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output                       = {},
                        const ThinkingControlOptions& thinking            = {},
                        const std::optional<OutputConstraint>& constraint = {},
                        const ToolChoice& tool_choice                     = {}) const;

    // The rendered prompt text, without tokenizing it: what the template produced for `input`.
    [[nodiscard]] std::string render(const PromptInput& input,
                                     const PreparationControl& control = {}) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Parses the body of one Gemma tool call, `call:NAME{...}` without its delimiters, into a name and
// JSON arguments. Returns nothing when the text is not a well-formed call.
[[nodiscard]] std::optional<GeneratedToolCall> parse_tool_call(std::string_view body);

} // namespace ninfer::models::gemma4
