#include "models/nemotron_h/frontend.h"

// The frontend is built from the format-neutral pieces only: the tokenizer implementation and the
// Jinja engine. The tool-call wire format the checkpoint's template emits is the one the runtime's
// own parser already owns, so the contract and the incremental decoder come from there.

#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/jinja.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::nemotron_h {
namespace {

namespace tf = qwen3_5::frontend;
using Json   = nlohmann::ordered_json;
using Clock  = std::chrono::steady_clock;

constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";
constexpr std::string_view kThinkOpen       = "<think>";
constexpr std::string_view kThinkClose      = "</think>";
constexpr std::string_view kTurnClose       = "<|im_end|>\n";

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

std::string_view media_kind_name(MediaKind kind) {
    return kind == MediaKind::Image ? "image" : "video";
}

Json parse_json(std::string_view text, std::string_view what) {
    try {
        return Json::parse(text);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string(what) + " is not valid JSON: " + error.what());
    }
}

void check_control(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "prompt preparation was cancelled");
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "prompt preparation exceeded its deadline");
    }
}

// Returns the bytes of `pending` that form complete UTF-8, replacing malformed sequences, and keeps
// an incomplete trailing sequence for the next token.
std::string consume_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }
        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U)
                    minimum = 0xa0U;
                else if (lead == 0xedU)
                    maximum = 0x9fU;
                else if (lead == 0xf0U)
                    minimum = 0x90U;
                else if (lead == 0xf4U)
                    maximum = 0x8fU;
            }
            if (byte < minimum || byte > maximum) {
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) continue;
        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() == 0 ? 0 : marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) return size;
    }
    return 0;
}

// The checkpoint's tokenizer_config.json, for the parts this family's behaviour depends on.
//
// There is deliberately no pad-token rule here. This checkpoint names `<|im_end|>` -- an added
// token its own vocabulary does contain -- and no path in the runtime pads this family's tokens, so
// Qwen's
// `<|endoftext|>` rule would restate another family's contract rather than this one's. What is
// checked is that the config does not ask the tokenizer to encode differently from the pipeline the
// artifact declares: a prefix space or a beginning-of-sequence token would change every prompt's
// ids, and the chat template writes the model's own `<|im_start|>` framing itself.
void validate_tokenizer_config(const Json& config) {
    if (config.value("add_prefix_space", false)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match this tokenizer's prefix semantics: "
            "add_prefix_space must be false");
    }
    if (config.value("add_bos_token", false)) {
        throw std::invalid_argument(
            "tokenizer_config.json adds a beginning-of-sequence token, which this family's chat "
            "template does not: the prompt is the template's own framing");
    }
}

} // namespace

void PublishedOutput::append(OutputChannel channel, std::string text) {
    if (text.empty()) return;
    if (!deltas_.empty() && deltas_.back().channel == channel) {
        deltas_.back().text += text;
    } else {
        deltas_.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

// ---- Frontend --------------------------------------------------------------------------------

struct Frontend::Impl {
    std::shared_ptr<const tf::Tokenizer> tokenizer;
    text::JinjaTemplate chat_template;
    std::uint32_t max_context = 0;
    StopPolicy defaults;
    ModelSamplingDefaults sampling;

    Impl(const FrontendResources& resources, std::uint32_t context)
        : tokenizer(std::make_shared<const tf::Tokenizer>(tf::TokenizerResources{
              .tokenizer_json         = resources.tokenizer_json,
              .tokenizer_config_json  = resources.tokenizer_config_json,
              .generation_config_json = resources.generation_config_json,
              .family                 = tf::TokenizerFamily::Nemotron,
          })),
          chat_template(resources.chat_template_jinja, "artifact:chat_template.jinja"),
          max_context(context) {
        if (resources.chat_template_jinja.empty()) {
            throw std::invalid_argument(
                "nemotron_h frontend: the artifact carries no chat template");
        }
        validate_tokenizer_config(
            parse_json(resources.tokenizer_config_json, "tokenizer_config.json"));
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        // The checkpoint's generation_config.json is the model's recommendation, and this family
        // does not distinguish a thinking preset from a non-thinking one: the template's <think>
        // block is the only switch, and it does not change the distribution's shape.
        const Json generation =
            parse_json(resources.generation_config_json, "generation_config.json");
        SamplingPreset preset;
        preset.temperature = generation.value("temperature", 1.0F);
        // The sampling Op keeps at most 20 candidates, so a larger recommendation is narrowed to
        // what the runtime can draw from.
        constexpr std::int32_t kSamplerTopK = 20;
        preset.top_k =
            std::min<std::int32_t>(generation.value("top_k", kSamplerTopK), kSamplerTopK);
        preset.top_p          = generation.value("top_p", 1.0F);
        sampling.thinking     = preset;
        sampling.non_thinking = preset;
    }

    struct Rendered {
        std::string text;
        bool thinking               = false;
        bool force_nonempty_content = false;
        std::shared_ptr<const std::vector<std::string>> tool_names;
        std::shared_ptr<const tf::ToolCallOutputContract> tool_contract;
    };

    Rendered render(const PromptInput& input, const PreparationControl& control) const {
        check_control(control);
        const PromptOptions& options = input.options;
        Json context                 = Json::object();
        if (!options.chat_template_kwargs_json.empty()) {
            Json kwargs = parse_json(options.chat_template_kwargs_json, "chat_template_kwargs");
            if (!kwargs.is_object()) {
                throw std::invalid_argument("chat_template_kwargs must be a JSON object");
            }
            for (auto& [key, value] : kwargs.items()) context[key] = value;
        }
        // The template's boolean parameters arrive straight from the request, so a mistyped keyword
        // is this frontend's request error rather than a truthy value inside the renderer.
        for (const char* name : {"enable_thinking", "medium_effort", "truncate_history_thinking",
                                 "force_nonempty_content"}) {
            if (context.contains(name) && !context[name].is_boolean()) {
                throw std::invalid_argument(std::string(name) + " must be a boolean");
            }
        }

        // The template's own default is thinking on, which is what its `enable_thinking is defined`
        // test falls back to; an explicit request field overrides a chat-template kwarg.
        bool thinking = context.value("enable_thinking", true);
        if (options.enable_thinking) {
            thinking = *options.enable_thinking;
        } else if (options.reasoning_effort) {
            thinking = *options.reasoning_effort != ReasoningEffort::None;
        }
        context["enable_thinking"] = thinking;
        // The family's knob for keeping an earlier turn's reasoning in the prompt is
        // `truncate_history_thinking`, and the request field means its inverse.
        if (options.preserve_thinking) {
            context["truncate_history_thinking"] = !*options.preserve_thinking;
        }

        const bool continuation =
            options.continuation == PromptContinuationMode::ContinueFinalAssistant;
        if (continuation &&
            (input.messages.empty() || input.messages.back().role != ChatRole::Assistant)) {
            throw std::invalid_argument("continuation requires a final assistant message");
        }
        context["add_generation_prompt"] = !continuation;

        Json messages = Json::array();
        for (const ChatMessage& message : input.messages) {
            Json value{{"role", role_name(message.role)}};
            const bool has_media = std::ranges::any_of(message.parts, [](const MessagePart& part) {
                return part.kind != MessagePartKind::Text;
            });
            if (has_media) {
                // Refused by name: this family's artifact carries no vision component, so there is
                // no image or video token for a part to become.
                const auto media = std::ranges::find_if(message.parts, [](const MessagePart& part) {
                    return part.kind != MessagePartKind::Text;
                });
                throw RequestError(
                    RequestErrorKind::InvalidMedia,
                    std::string("this Nemotron-H model has no vision component; a ") +
                        std::string(media_kind_name(media->media.kind)) + " part is not accepted");
            }
            std::string content;
            for (const MessagePart& part : message.parts) content += part.text;
            value["content"] = std::move(content);
            if (!message.reasoning_content.empty()) {
                value["reasoning_content"] = message.reasoning_content;
            }
            if (!message.tool_calls.empty()) {
                Json calls = Json::array();
                for (const ToolCall& call : message.tool_calls) {
                    Json arguments = call.arguments_json.empty()
                                         ? Json::object()
                                         : parse_json(call.arguments_json, "tool call arguments");
                    calls.push_back(
                        {{"id", call.id},
                         {"type", "function"},
                         {"function", {{"name", call.name}, {"arguments", arguments}}}});
                }
                value["tool_calls"] = std::move(calls);
            }
            if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
            messages.push_back(std::move(value));
        }
        context["messages"] = std::move(messages);

        if (!options.tool_jsons.empty()) {
            Json tools = Json::array();
            for (const std::string& tool : options.tool_jsons) {
                tools.push_back(parse_json(tool, "tool declaration"));
            }
            context["tools"] = std::move(tools);
        }

        text::TemplateRenderOptions execution{
            .checkpoint = [&] { check_control(control); },
        };
        std::string text = chat_template.render(context, execution).text;
        if (continuation) {
            // The final assistant turn is the one being continued, so its close is not part of it.
            if (!text.ends_with(kTurnClose)) {
                throw std::invalid_argument(
                    "chat template did not close the final assistant turn it is asked to continue");
            }
            text.resize(text.size() - kTurnClose.size());
        }
        auto names = std::make_shared<std::vector<std::string>>();
        for (const std::string& declaration : options.tool_jsons) {
            Json parsed = parse_json(declaration, "tool declaration");
            if (parsed.contains("function") && parsed["function"].contains("name") &&
                parsed["function"]["name"].is_string()) {
                names->push_back(parsed["function"]["name"].get<std::string>());
            }
        }
        return Rendered{
            .text = std::move(text),
            // A continued assistant turn is already past its thought block, so the decoder never
            // splits one there.
            .thinking               = thinking && !continuation,
            .force_nonempty_content = context.value("force_nonempty_content", false),
            .tool_names             = std::move(names),
            .tool_contract = tf::build_tool_call_output_contract(options.tool_jsons, true,
                                                                 options.tolerant_tool_calls),
        };
    }
};

Frontend::Frontend(const FrontendResources& resources, std::uint32_t max_context)
    : impl_(std::make_unique<Impl>(resources, max_context)) {
    if (max_context == 0) {
        throw std::invalid_argument("nemotron_h frontend: max_context must be nonzero");
    }
}

Frontend::~Frontend()                              = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;

std::string Frontend::render(const PromptInput& input, const PreparationControl& control) const {
    return impl_->render(input, control).text;
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start        = Clock::now();
    Impl::Rendered rendered = impl_->render(input, control);
    PreparedPrompt prompt;
    const auto tokenize_started    = Clock::now();
    const std::vector<int> encoded = impl_->tokenizer->encode(rendered.text);
    check_control(control);
    if (encoded.size() > impl_->max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prepared prompt has " + std::to_string(encoded.size()) +
                               " tokens, exceeding Engine max_context " +
                               std::to_string(impl_->max_context));
    }
    prompt.ids.assign(encoded.begin(), encoded.end());
    prompt.descriptor.has_media           = false;
    prompt.descriptor.prompt_tokens       = static_cast<std::uint32_t>(prompt.ids.size());
    prompt.descriptor.starts_in_reasoning = rendered.thinking;
    prompt.tool_names                     = std::move(rendered.tool_names);
    prompt.tool_call_contract             = std::move(rendered.tool_contract);
    prompt.force_nonempty_content         = rendered.force_nonempty_content;
    const auto done                       = Clock::now();
    prompt.preparation.seconds            = std::chrono::duration<double>(done - start).count();
    prompt.preparation.tokenize_seconds =
        std::chrono::duration<double>(done - tokenize_started).count();
    return prompt;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).descriptor.prompt_tokens;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    // Deliberately transparent: the token count a caller handed over is the count it gets back,
    // which is what the Engine's scoring contract checks against its own window.
    PreparedPrompt prompt;
    prompt.descriptor.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    prompt.ids                      = std::move(token_ids);
    return prompt;
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const std::vector<int> encoded = impl_->tokenizer->encode(text);
    return std::vector<TokenId>(encoded.begin(), encoded.end());
}

ModelSamplingDefaults Frontend::sampling_defaults() const { return impl_->sampling; }

// ---- OutputSession ---------------------------------------------------------------------------

namespace {

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

struct MarkerScan {
    std::optional<std::size_t> close;
    // A trailing prefix of the marker, held until the next bytes can complete it.
    std::size_t hold = 0;
};

// The vendor's parser splits on the close marker alone: the marker is in neither channel.
MarkerScan scan_close_marker(std::string_view text) {
    const std::size_t at = text.find(kThinkClose);
    if (at != std::string_view::npos) { return MarkerScan{.close = at, .hold = 0}; }
    return MarkerScan{.close = std::nullopt, .hold = longest_suffix_prefix(text, kThinkClose)};
}

struct DecoderState {
    // The thought block is open until its close marker arrives; the prompt may have opened it.
    bool in_reasoning = false;
    // The block's unpublished bytes. Normally only its head is held, until the marker question
    // below is decidable; with force_nonempty_content the whole block waits for its close.
    std::string reasoning;
    bool reasoning_head_resolved = false;
    // Bytes held because they could still complete the close marker.
    std::string close_pending;
    std::string utf8_pending;
    std::array<std::string, 2> stop_pending;
    std::optional<tf::ToolCallOutputDecoder> tool_output;
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
    std::uint32_t reasoning_tokens = 0;
    std::uint64_t decoded_bytes    = 0;
    std::optional<std::uint32_t> matched_stop;
    bool content_published = false;
    bool terminal          = false;
};

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

} // namespace

class OutputSession::Impl {
public:
    std::shared_ptr<const tf::Tokenizer> tokenizer;
    StopPolicy policy;
    bool preserve_special            = false;
    bool raw                         = false;
    bool split_reasoning             = false;
    bool force_nonempty_content      = false;
    bool parse_calls                 = true;
    std::size_t max_tool_name_length = 128;

    DecoderState state;
    DecoderState preview;
    PublishedOutput preview_output;
    bool preview_ready = false;
    std::vector<GeneratedToolCall> published_calls;

    // Streams `text` into one channel, holding back what could still become a stop string, and
    // records the earliest stop string the text completes.
    void feed_channel(DecoderState& s, OutputChannel channel, std::string_view text,
                      PublishedOutput& emitted, std::uint32_t committed, StopMatch* best) const {
        if (text.empty()) return;
        std::string combined          = s.stop_pending[channel_index(channel)];
        const std::size_t old_pending = combined.size();
        combined.append(text);
        const std::uint64_t combined_start = s.decoded_bytes - old_pending;
        if (best != nullptr) {
            for (std::size_t order = 0; order < policy.strings.size(); ++order) {
                const StopString& stop = policy.strings[order];
                if (stop.channel != channel || stop.text.empty()) continue;
                const std::size_t found = combined.find(stop.text);
                if (found == std::string::npos) continue;
                const std::uint64_t cut = combined_start + found;
                const bool earlier = !best->found || committed < best->committed_tokens ||
                                     (committed == best->committed_tokens &&
                                      (cut < best->byte_cut ||
                                       (cut == best->byte_cut && order < best->declaration_order)));
                if (!earlier) continue;
                PublishedOutput candidate = emitted;
                candidate.append(channel, combined.substr(0, found));
                if (stop.include_in_output) candidate.append(channel, stop.text);
                *best = StopMatch{.found             = true,
                                  .committed_tokens  = committed,
                                  .byte_cut          = cut,
                                  .declaration_order = static_cast<std::uint32_t>(order),
                                  .output            = std::move(candidate)};
            }
        }
        std::size_t hold = 0;
        for (const StopString& stop : policy.strings) {
            if (stop.channel == channel)
                hold = std::max(hold, longest_suffix_prefix(combined, stop.text));
        }
        if (channel == OutputChannel::Content) {
            if (combined.size() > hold) s.content_published = true;
        }
        emitted.append(channel, combined.substr(0, combined.size() - hold));
        s.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
        s.decoded_bytes += text.size();
    }

    void close_channel(DecoderState& s, OutputChannel channel, PublishedOutput& emitted) const {
        std::string& pending = s.stop_pending[channel_index(channel)];
        if (channel == OutputChannel::Content) {
            if (!pending.empty()) s.content_published = true;
        }
        emitted.append(channel, std::move(pending));
        pending.clear();
    }

    // The block's bytes that may be published as reasoning. The vendor reaches this by two rules in
    // sequence: the inherited parser splits on the *end* token, so the start marker is not part of
    // the reasoning text, and then this family drops exactly one leading newline when the start
    // token is present in the output at all -- "the chat template owns the newline in `<think>\n`
    // and re-adds it when it replays the turn". Its own comment names the case: "Live vLLM output
    // never contains the start token (the template prefilled it), so this only fires when the
    // parser is handed a full assistant span". So a span that carries the marker loses it and one
    // newline; a live span (no marker) is byte-identical either way.
    std::string take_reasoning_block(DecoderState& s) const {
        if (!s.reasoning_head_resolved) {
            s.reasoning_head_resolved = true;
            if (s.reasoning.starts_with(kThinkOpen)) {
                s.reasoning.erase(0, kThinkOpen.size());
                if (s.reasoning.starts_with('\n')) s.reasoning.erase(0, 1);
            }
        }
        return std::move(s.reasoning);
    }

    // Whether the held head is long enough to decide the marker question at all. The decision is
    // not just "is the marker here": it is "is the marker here, and is the newline the template
    // owns the next byte", because both are stripped together. Resolving at exactly the marker's
    // length would resolve on a head whose next byte has not arrived -- which happens whenever a
    // close-marker prefix holds the marker's last byte back -- and the newline would then be
    // published as a piece of its own.
    static bool reasoning_head_decidable(std::string_view held) noexcept {
        const std::size_t known = std::min(held.size(), kThinkOpen.size());
        return held.size() > kThinkOpen.size() ||
               held.substr(0, known) != kThinkOpen.substr(0, known);
    }

    void feed_reasoning(DecoderState& s, std::string_view bytes, PublishedOutput& emitted,
                        std::uint32_t committed, StopMatch* best) const {
        if (bytes.empty()) return;
        s.reasoning.append(bytes);
        // A request that asked for nonempty content holds the whole block: it may have to publish
        // it as content when the model never closes the block.
        if (force_nonempty_content || !reasoning_head_decidable(s.reasoning)) return;
        // `take_reasoning_block` owns the resolution: it is the one place that strips an echoed
        // start marker and the newline the template owns, so the flag must not be set before it
        // runs.
        feed_channel(s, OutputChannel::Reasoning, take_reasoning_block(s), emitted, committed,
                     best);
    }

    void close_reasoning(DecoderState& s, PublishedOutput& emitted, std::uint32_t committed,
                         StopMatch* best) const {
        feed_channel(s, OutputChannel::Reasoning, take_reasoning_block(s), emitted, committed,
                     best);
        close_channel(s, OutputChannel::Reasoning, emitted);
    }

    // Content-phase bytes: the thought block is closed, so what the model wrote is either a tool
    // call or ordinary text. Everything the decoder withholds is a possible call region.
    void feed_content(DecoderState& s, std::string_view bytes, PublishedOutput& emitted,
                      std::uint32_t committed, StopMatch* best) const {
        if (bytes.empty()) return;
        if (!s.tool_output) {
            feed_channel(s, OutputChannel::Content, bytes, emitted, committed, best);
            return;
        }
        feed_channel(s, OutputChannel::Content, s.tool_output->feed(bytes), emitted, committed,
                     best);
    }

    void route_bytes(DecoderState& s, std::string_view bytes, PublishedOutput& emitted,
                     std::uint32_t committed, StopMatch* best) const {
        if (bytes.empty()) return;
        if (!s.in_reasoning) {
            feed_content(s, bytes, emitted, committed, best);
            return;
        }
        std::string combined = s.close_pending;
        s.close_pending.clear();
        combined.append(bytes);
        const MarkerScan scan = scan_close_marker(combined);
        if (!scan.close) {
            s.close_pending = combined.substr(combined.size() - scan.hold);
            feed_reasoning(s, combined.substr(0, combined.size() - scan.hold), emitted, committed,
                           best);
            return;
        }
        feed_reasoning(s, combined.substr(0, *scan.close), emitted, committed, best);
        close_reasoning(s, emitted, committed, best);
        s.in_reasoning = false;
        feed_content(s, std::string_view(combined).substr(*scan.close + kThinkClose.size()),
                     emitted, committed, best);
    }

    // Publishes whatever a token's bytes hold, keeping an incomplete code point for the next one.
    void feed_bytes(DecoderState& s, std::string_view bytes, PublishedOutput& emitted,
                    std::uint32_t committed, StopMatch* best) const {
        s.utf8_pending.append(bytes);
        route_bytes(s, consume_utf8(s.utf8_pending), emitted, committed, best);
    }

    // The round is over: everything still held is published, and a block the model never closed is
    // published as content when the request asked for nonempty content.
    void terminalize(DecoderState& s, PublishedOutput& emitted, std::uint32_t committed,
                     FinishReason reason) const {
        std::string pending = consume_utf8(s.utf8_pending);
        if (!s.utf8_pending.empty()) {
            // An incomplete code point at the end is published as the replacement character,
            // exactly as a decoder with one more round to finish it would.
            s.utf8_pending.clear();
            pending.append(kUtf8Replacement);
        }
        feed_bytes(s, pending, emitted, committed, nullptr);
        if (s.in_reasoning) {
            if (force_nonempty_content && !s.content_published) {
                // ultra_v3's third rule: the model never closed its thought block, so what it did
                // produce is the answer rather than a reasoning channel.
                feed_channel(s, OutputChannel::Content, take_reasoning_block(s), emitted, committed,
                             nullptr);
                close_channel(s, OutputChannel::Reasoning, emitted);
            } else {
                close_reasoning(s, emitted, committed, nullptr);
            }
            s.in_reasoning = false;
        }
        if (s.tool_output) {
            tf::ToolCallOutputDecoder::Terminal finished = s.tool_output->finish(reason);
            s.calls                                      = std::move(finished.tool_calls);
            s.diagnostics                                = finished.diagnostics;
            // Through the content channel rather than straight into the output: bytes it withheld
            // may follow bytes this channel is still holding for a stop string.
            feed_channel(s, OutputChannel::Content, finished.content, emitted, committed, nullptr);
        }
        close_channel(s, OutputChannel::Content, emitted);
        s.terminal = true;
    }
};

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (impl_ == nullptr) throw std::logic_error("output session is empty");
    if (impl_->state.terminal) throw std::logic_error("output session is already terminal");
    if (impl_->preview_ready) throw std::logic_error("output session already has a preview");
    if (tokens.empty())
        throw std::invalid_argument("cannot preview an empty generated-token round");
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }
    Impl& impl   = *impl_;
    impl.preview = impl.state;
    impl.preview_output.clear();
    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        impl.preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens = count, .finish_reason = reason, .continuation = continuation};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count = static_cast<std::uint32_t>(index + 1);
        const TokenId token       = tokens[index];
        const bool stop_token =
            std::ranges::find(impl.policy.token_ids, token) != impl.policy.token_ids.end();
        if (stop_token) {
            if (impl.policy.publish_stop_token) {
                const tf::DecodedTokenView decoded = impl.tokenizer->decoded_token(token);
                impl.feed_bytes(impl.preview, decoded.bytes, impl.preview_output, count, nullptr);
            }
            impl.terminalize(impl.preview, impl.preview_output, count, FinishReason::StopToken);
            return complete(count, FinishReason::StopToken);
        }
        const tf::DecodedTokenView decoded = impl.tokenizer->decoded_token(token);
        // Counted before the special-token drop, so a protocol token inside the block still counts
        // as one of the round's reasoning tokens.
        if (impl.preview.in_reasoning) ++impl.preview.reasoning_tokens;
        if (impl.preserve_special || !decoded.special) {
            StopMatch match;
            impl.feed_bytes(impl.preview, decoded.bytes, impl.preview_output, count, &match);
            if (match.found) {
                impl.preview.utf8_pending.clear();
                impl.preview.stop_pending = {};
                impl.preview.terminal     = true;
                impl.preview.matched_stop = match.declaration_order;
                impl.preview_output       = std::move(match.output);
                return complete(match.committed_tokens, FinishReason::StopString);
            }
        }
    }
    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        impl.terminalize(impl.preview, impl.preview_output, count, limit_reason);
        return complete(count, limit_reason);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    // No budget ever reaches a session: see make_output_session's refusal.
    return total_budget_remaining;
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept { return {}; }

std::uint32_t OutputSession::control_suffix_tokens() const noexcept { return 0; }

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId>, std::uint32_t) {
    throw std::logic_error("this Nemotron-H output session has no target control to apply");
}

void OutputSession::validate_generation_capacity(std::uint32_t) const {}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) throw std::logic_error("output session is empty");
    if (impl_->state.terminal) throw std::logic_error("output session is already terminal");
    if (impl_->preview_ready) throw std::logic_error("output session already has a preview");
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview = impl_->state;
    impl_->preview_output.clear();
    impl_->terminalize(impl_->preview, impl_->preview_output, 0, reason);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

std::uint32_t OutputSession::grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
    throw std::logic_error("mask requested for unconstrained output");
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) std::terminate();
    using std::swap;
    swap(impl_->state, impl_->preview);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
    if (impl_->state.terminal) impl_->published_calls = std::move(impl_->state.calls);
    return output;
}

void OutputSession::discard_preview() {
    if (impl_ == nullptr) return;
    impl_->preview_ready = false;
    impl_->preview_output.clear();
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->published_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->state.diagnostics : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept { return {}; }

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop) return std::nullopt;
    return impl_->policy.strings.at(*impl_->state.matched_stop).text;
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    if (thinking.budget) {
        // Refused by name: Qwen's canned control sentence is not this family's, and the template
        // has no token that closes the thought block, so a budget could not be honoured as asked.
        throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                           "this Nemotron-H model has no thinking budget: its thought block is "
                           "closed by the model, "
                           "not by an Engine-injected control token");
    }
    if (constraint) {
        throw RequestError(RequestErrorKind::InvalidGrammar,
                           "this Nemotron-H model does not support output constraints: its Program "
                           "consumes no token masks");
    }
    if (tool_choice.mode == ToolChoiceMode::Required || tool_choice.allowed_names) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "this Nemotron-H model cannot force or restrict tool calls; only "
                           "tool_choice auto or none is supported",
                           {}, RequestErrorSource::Tools);
    }
    auto impl                       = std::make_unique<OutputSession::Impl>();
    impl->tokenizer                 = impl_->tokenizer;
    impl->raw                       = output.raw;
    impl->preserve_special          = output.raw || output.preserve_special_tokens;
    impl->parse_calls               = !output.raw && tool_choice.mode != ToolChoiceMode::None &&
                                      prompt.tool_call_contract != nullptr;
    impl->split_reasoning           = prompt.descriptor.starts_in_reasoning && !output.raw;
    impl->force_nonempty_content    = prompt.force_nonempty_content;
    impl->max_tool_name_length      = output.tool_name_max_length;
    impl->policy.strings            = caller_stop.strings;
    impl->policy.publish_stop_token = caller_stop.publish_stop_token;
    impl->policy.token_ids          = caller_stop.token_ids;
    if (caller_stop.include_model_defaults) {
        for (const TokenId token : impl_->defaults.token_ids) {
            if (std::ranges::find(impl->policy.token_ids, token) == impl->policy.token_ids.end()) {
                impl->policy.token_ids.push_back(token);
            }
        }
    }
    for (const TokenId token : impl->policy.token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::invalid_argument("stop token is outside the vocabulary");
        }
    }
    impl->state.in_reasoning = impl->split_reasoning;
    if (impl->parse_calls) {
        // A tool choice cannot constrain this model's output -- no token mask ever constrains it --
        // so the decoder must treat a malformed region as free output: the model's own bytes,
        // verbatim.
        auto selected = tf::select_tool_call_contract(prompt.tool_call_contract, tool_choice);
        if (selected) {
            auto free_output         = std::make_shared<tf::ToolCallOutputContract>(*selected);
            free_output->constrained = false;
            free_output->required    = false;
            selected                 = std::move(free_output);
        }
        impl->state.tool_output.emplace(std::move(selected), impl->max_tool_name_length);
    }
    return OutputSession(std::move(impl));
}

} // namespace ninfer::models::nemotron_h
