#pragma once

// Nemotron-H's frontend: the checkpoint's tokenizer, its chat template, the output decoder that
// turns generated tokens into the Engine's output channels, and the tool-call contract.
//
// The protocol is this checkpoint's own, so nothing here is shared with the Qwen or Gemma frontends
// except the format-neutral parts it is built from: the tokenizer implementation, the Jinja engine,
// and the tool-call parser that already owns this wire format.
//
//   turns        <|im_start|>role\n ... <|im_end|>\n
//   thinking     the assistant turn opens with <think>\n when thinking is on, and the model closes
//   it
//                with </think>; with thinking off the template closes an empty block itself, so the
//                answer is content from its first token and a stray </think> stays content.
//   tool calls   <tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n</parameter>\n</function>\n
//                </tool_call>, which is the wire format `parser/tool_call_parser` already parses.
//   images       this family has no vision component: a media part is refused, not rendered.
//
// This family's reasoning is not a controllable channel. Qwen's canned control sentence is not this
// checkpoint's, and nothing in the template can force the thought block closed, so a thinking
// budget is refused by name rather than approximated.

#include "models/qwen3_5/frontend/tool_contract.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::nemotron_h {

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
    // Tool declarations the prompt carried, by name, so a controller can tell whether a round had
    // tools at all.
    std::shared_ptr<const std::vector<std::string>> tool_names;
    // The compiled parameter contract the output decoder normalizes calls against. Null when the
    // prompt declared none.
    std::shared_ptr<const qwen3_5::frontend::ToolCallOutputContract> tool_call_contract;
    // `chat_template_kwargs.force_nonempty_content`: when the model never closes its thought block,
    // the reasoning it did produce is published as content instead. See OutputSession.
    bool force_nonempty_content = false;

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
// are this checkpoint's protocol above.
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

    // This family has no target-control token: nothing can close the thought block from the Engine,
    // so a budget never reaches a session and these answer their neutral values.
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] std::uint32_t control_suffix_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;

    // No output constraint reaches this session: the Program consumes no token masks, so the Engine
    // refuses a constrained request before it gets here.
    [[nodiscard]] bool constrained() const noexcept { return false; }

    void observe_constraint(bool, double) noexcept {}

    void constraint_uploaded(std::size_t) noexcept {}

    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const { return {}; }

    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>);

    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
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
    // Exactly the ids given: the caller owns the sequence.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    // Text as this tokenizer reads it. The chat template owns the model's framing and writes no
    // beginning-of-sequence token, so none is added here either.
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

} // namespace ninfer::models::nemotron_h
