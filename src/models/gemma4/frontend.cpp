#include "models/gemma4/frontend.h"

#include "media/decode/decode.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/jinja.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::gemma4 {
namespace {

namespace tf = qwen3_5::frontend;
using Json   = nlohmann::ordered_json;
using Clock  = std::chrono::steady_clock;

constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";
constexpr std::string_view kBos             = "<bos>";
constexpr std::string_view kTurnClose       = "<turn|>\n";
constexpr std::string_view kModelTurn       = "<|turn>model\n";

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
        throw RequestError(RequestErrorKind::QueueTimeout, "prompt preparation exceeded its deadline");
    }
}

// The tokenizer's id for a marker the protocol depends on; it must be one added token.
TokenId marker_id(const tf::Tokenizer& tokenizer, std::string_view marker) {
    const std::vector<int> ids = tokenizer.encode(marker);
    if (ids.size() != 1) {
        throw std::invalid_argument("gemma4 frontend: the tokenizer has no single token for " +
                                    std::string(marker));
    }
    return ids.front();
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
                if (lead == 0xe0U) minimum = 0xa0U;
                else if (lead == 0xedU) maximum = 0x9fU;
                else if (lead == 0xf0U) minimum = 0x90U;
                else if (lead == 0xf4U) maximum = 0x8fU;
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

// ---- Tool-call argument syntax ----------------------------------------------------------------
//
// value  := string | object | array | number | true | false | null
// string := <|"|> any bytes up to the next <|"|> <|"|>
// object := { [key : value (, key : value)*] }      key is bare or a string
// array  := [ [value (, value)*] ]

constexpr std::string_view kQuote = "<|\"|>";

class ArgumentParser {
public:
    explicit ArgumentParser(std::string_view text) : text_(text) {}

    std::optional<Json> object_at_end() {
        auto value = object();
        skip_space();
        if (!value || offset_ != text_.size()) return std::nullopt;
        return value;
    }

private:
    void skip_space() {
        while (offset_ < text_.size() &&
               (text_[offset_] == ' ' || text_[offset_] == '\n' || text_[offset_] == '\t' ||
                text_[offset_] == '\r')) {
            ++offset_;
        }
    }

    bool take(std::string_view literal) {
        if (text_.substr(offset_).starts_with(literal)) {
            offset_ += literal.size();
            return true;
        }
        return false;
    }

    std::optional<std::string> quoted() {
        if (!take(kQuote)) return std::nullopt;
        const std::size_t close = text_.find(kQuote, offset_);
        if (close == std::string_view::npos) return std::nullopt;
        std::string out(text_.substr(offset_, close - offset_));
        offset_ = close + kQuote.size();
        return out;
    }

    std::optional<std::string> key() {
        skip_space();
        if (text_.substr(offset_).starts_with(kQuote)) return quoted();
        const std::size_t begin = offset_;
        while (offset_ < text_.size() && text_[offset_] != ':' && text_[offset_] != '}' &&
               text_[offset_] != ',') {
            ++offset_;
        }
        std::string name(text_.substr(begin, offset_ - begin));
        while (!name.empty() && (name.back() == ' ' || name.back() == '\n')) name.pop_back();
        if (name.empty()) return std::nullopt;
        return name;
    }

    std::optional<Json> object() {
        skip_space();
        if (!take("{")) return std::nullopt;
        Json out = Json::object();
        skip_space();
        if (take("}")) return out;
        while (true) {
            auto name = key();
            if (!name) return std::nullopt;
            skip_space();
            if (!take(":")) return std::nullopt;
            auto item = value();
            if (!item) return std::nullopt;
            out[*name] = std::move(*item);
            skip_space();
            if (take("}")) return out;
            if (!take(",")) return std::nullopt;
        }
    }

    std::optional<Json> array() {
        if (!take("[")) return std::nullopt;
        Json out = Json::array();
        skip_space();
        if (take("]")) return out;
        while (true) {
            auto item = value();
            if (!item) return std::nullopt;
            out.push_back(std::move(*item));
            skip_space();
            if (take("]")) return out;
            if (!take(",")) return std::nullopt;
        }
    }

    std::optional<Json> scalar() {
        const std::size_t begin = offset_;
        while (offset_ < text_.size() && text_[offset_] != ',' && text_[offset_] != '}' &&
               text_[offset_] != ']' && text_[offset_] != ' ' && text_[offset_] != '\n') {
            ++offset_;
        }
        const std::string_view token = text_.substr(begin, offset_ - begin);
        if (token == "true") return Json(true);
        if (token == "false") return Json(false);
        if (token == "null") return Json(nullptr);
        if (token.empty()) return std::nullopt;
        try {
            Json number = Json::parse(token);
            if (number.is_number()) return number;
        } catch (const Json::exception&) {}
        return std::nullopt;
    }

    std::optional<Json> value() {
        skip_space();
        if (text_.substr(offset_).starts_with(kQuote)) {
            auto text = quoted();
            if (!text) return std::nullopt;
            return Json(std::move(*text));
        }
        if (offset_ < text_.size() && text_[offset_] == '{') return object();
        if (offset_ < text_.size() && text_[offset_] == '[') return array();
        return scalar();
    }

    std::string_view text_;
    std::size_t offset_ = 0;
};

} // namespace

std::optional<GeneratedToolCall> parse_tool_call(std::string_view body) {
    constexpr std::string_view kCall = "call:";
    std::size_t begin                = 0;
    while (begin < body.size() && (body[begin] == ' ' || body[begin] == '\n')) ++begin;
    if (!body.substr(begin).starts_with(kCall)) return std::nullopt;
    begin += kCall.size();
    const std::size_t brace = body.find('{', begin);
    if (brace == std::string_view::npos || brace == begin) return std::nullopt;
    const std::string_view name = body.substr(begin, brace - begin);
    for (const char byte : name) {
        const bool allowed = std::isalnum(static_cast<unsigned char>(byte)) != 0 || byte == '_' ||
                             byte == '-' || byte == '.';
        if (!allowed) return std::nullopt;
    }
    ArgumentParser parser(body.substr(brace));
    auto arguments = parser.object_at_end();
    if (!arguments) return std::nullopt;
    return GeneratedToolCall{.name = std::string(name), .arguments_json = arguments->dump()};
}

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
    TokenId bos           = -1;
    TokenId channel_open  = -1;
    TokenId channel_close = -1;
    TokenId call_open     = -1;
    TokenId call_close    = -1;
    TokenId image_token   = -1;
    TokenId image_open    = -1;
    TokenId image_close   = -1;

    Impl(const FrontendResources& resources, std::uint32_t context)
        : tokenizer(std::make_shared<const tf::Tokenizer>(tf::TokenizerResources{
              .tokenizer_json         = resources.tokenizer_json,
              .tokenizer_config_json  = resources.tokenizer_config_json,
              .generation_config_json = resources.generation_config_json,
              .family                 = tf::TokenizerFamily::Gemma,
          })),
          chat_template(resources.chat_template_jinja, "artifact:chat_template.jinja"),
          max_context(context) {
        if (resources.chat_template_jinja.empty()) {
            throw std::invalid_argument("gemma4 frontend: the artifact carries no chat template");
        }
        bos           = marker_id(*tokenizer, kBos);
        channel_open  = marker_id(*tokenizer, "<|channel>");
        channel_close = marker_id(*tokenizer, "<channel|>");
        call_open     = marker_id(*tokenizer, "<|tool_call>");
        call_close    = marker_id(*tokenizer, "<tool_call|>");
        image_token   = marker_id(*tokenizer, "<|image|>");
        image_open    = marker_id(*tokenizer, "<|image>");
        image_close   = marker_id(*tokenizer, "<image|>");
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        // The checkpoint's generation_config.json is the model's recommendation, and Gemma does not
        // distinguish a thinking preset from a non-thinking one.
        const Json generation = parse_json(resources.generation_config_json, "generation_config.json");
        SamplingPreset preset;
        preset.temperature = generation.value("temperature", 1.0F);
        // The sampling Op keeps at most 20 candidates. The checkpoint recommends 64, so its default is
        // narrowed to what the runtime can draw from; with top_p 0.95 at temperature 1 the nucleus
        // usually fits inside 20, but not always, and that difference is real.
        constexpr std::int32_t kSamplerTopK = 20;
        preset.top_k = std::min<std::int32_t>(generation.value("top_k", kSamplerTopK), kSamplerTopK);
        preset.top_p       = generation.value("top_p", 1.0F);
        sampling.thinking     = preset;
        sampling.non_thinking = preset;
    }

    struct Rendered {
        std::string text;
        bool thinking = false;
        std::shared_ptr<const std::vector<std::string>> tool_names;
        // The image parts, in the order the template writes their placeholders.
        std::vector<const OwnedMedia*> images;
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

        bool thinking = context.value("enable_thinking", false);
        if (options.enable_thinking) {
            thinking = *options.enable_thinking;
        } else if (options.reasoning_effort) {
            thinking = *options.reasoning_effort != ReasoningEffort::None;
        }
        context["enable_thinking"] = thinking;
        if (options.preserve_thinking) context["preserve_thinking"] = *options.preserve_thinking;

        const bool continuation =
            options.continuation == PromptContinuationMode::ContinueFinalAssistant;
        if (continuation &&
            (input.messages.empty() || input.messages.back().role != ChatRole::Assistant)) {
            throw std::invalid_argument("continuation requires a final assistant message");
        }
        context["add_generation_prompt"] = !continuation;
        context["bos_token"]             = std::string(kBos);

        Json messages = Json::array();
        std::vector<const OwnedMedia*> images;
        for (const ChatMessage& message : input.messages) {
            Json value{{"role", role_name(message.role)}};
            const bool has_media = std::ranges::any_of(
                message.parts, [](const MessagePart& part) { return part.kind != MessagePartKind::Text; });
            if (!has_media) {
                std::string content;
                for (const MessagePart& part : message.parts) content += part.text;
                value["content"] = std::move(content);
            } else {
                // A message with media is a parts list, as transformers' processor passes it; the
                // template trims each text part and writes <|image|> for each image.
                if (message.role == ChatRole::Tool) {
                    throw RequestError(RequestErrorKind::InvalidMedia,
                                       "Gemma does not accept images in tool results");
                }
                Json parts = Json::array();
                for (const MessagePart& part : message.parts) {
                    if (part.kind == MessagePartKind::Text) {
                        parts.push_back({{"type", "text"}, {"text", part.text}});
                    } else if (part.media.kind != MediaKind::Image) {
                        throw RequestError(RequestErrorKind::InvalidMedia,
                                           "Gemma accepts images only, not video");
                    } else {
                        parts.push_back({{"type", "image"}});
                        images.push_back(&part.media);
                    }
                }
                value["content"] = std::move(parts);
            }
            if (!message.reasoning_content.empty()) {
                value["reasoning_content"] = message.reasoning_content;
            }
            if (!message.tool_calls.empty()) {
                Json calls = Json::array();
                for (const ToolCall& call : message.tool_calls) {
                    Json arguments = call.arguments_json.empty()
                                         ? Json::object()
                                         : parse_json(call.arguments_json, "tool call arguments");
                    calls.push_back({{"id", call.id},
                                     {"type", "function"},
                                     {"function", {{"name", call.name}, {"arguments", arguments}}}});
                }
                value["tool_calls"] = std::move(calls);
            }
            if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
            messages.push_back(std::move(value));
        }
        context["messages"] = std::move(messages);

        auto names = std::make_shared<std::vector<std::string>>();
        if (!options.tool_jsons.empty()) {
            Json tools = Json::array();
            for (const std::string& tool : options.tool_jsons) {
                Json parsed = parse_json(tool, "tool declaration");
                if (parsed.contains("function") && parsed["function"].contains("name")) {
                    names->push_back(parsed["function"]["name"].get<std::string>());
                }
                tools.push_back(std::move(parsed));
            }
            context["tools"] = std::move(tools);
        }

        const std::array<std::string, 1> control_variables{"bos_token"};
        text::TemplateRenderOptions execution{
            .checkpoint        = [&] { check_control(control); },
            .control_variables = control_variables,
        };
        std::string text = chat_template.render(context, execution).text;
        if (continuation) {
            // The final assistant turn is the one being continued, so its close is not part of it.
            if (!text.ends_with(kTurnClose)) {
                throw std::invalid_argument(
                    "chat template did not close the final assistant turn it is asked to continue");
            }
            text.resize(text.size() - kTurnClose.size());
        } else if (text.ends_with(kTurnClose)) {
            // After an assistant turn that both answered and called a tool, the template closes the
            // turn without opening the next one; the model needs its own turn to answer in.
            text += kModelTurn;
        }
        return Rendered{.text = std::move(text), .thinking = thinking && !continuation,
                        .tool_names = std::move(names), .images = std::move(images)};
    }
};

Frontend::Frontend(const FrontendResources& resources, std::uint32_t max_context)
    : impl_(std::make_unique<Impl>(resources, max_context)) {
    if (max_context == 0) throw std::invalid_argument("gemma4 frontend: max_context must be nonzero");
}

Frontend::~Frontend()                              = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;

std::string Frontend::render(const PromptInput& input, const PreparationControl& control) const {
    return impl_->render(input, control).text;
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start            = Clock::now();
    Impl::Rendered rendered     = impl_->render(input, control);
    PreparedPrompt prompt;
    std::vector<PreparedImage> images;
    const auto media_started = Clock::now();
    for (const OwnedMedia* media : rendered.images) {
        try {
            // A client that asked for an error instead of a downscale is answered before the decode.
            if (media->image_resize_policy == ImageResizePolicy::RejectOversized) {
                const media::decode::ImageInfo info =
                    media::decode::inspect_image(media->bytes, media::decode::Policy{});
                if (static_cast<std::uint64_t>(info.width) * static_cast<std::uint64_t>(info.height) >
                    kGemmaImageBudgetPixels) {
                    throw RequestError(RequestErrorKind::InvalidMedia,
                                       "image exceeds the Vision budget and oversized_image is 'error'");
                }
            }
            images.push_back(prepare_gemma_image(media->bytes, media::decode::Policy{},
                                                 [&] { check_control(control); }));
        } catch (const media::decode::Error& error) {
            throw RequestError(error.kind() == media::decode::ErrorKind::BudgetExceeded
                                   ? RequestErrorKind::MediaBudgetExceeded
                                   : RequestErrorKind::InvalidMedia,
                               std::string("image: ") + error.what());
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::InvalidMedia, std::string("image: ") + error.what());
        }
        prompt.preparation.media_bytes += media->bytes.size();
        prompt.preparation.raw_patches +=
            static_cast<std::uint64_t>(images.back().grid_width) * images.back().grid_height;
        prompt.preparation.vision_tokens += static_cast<std::uint64_t>(images.back().soft_tokens());
        prompt.preparation.patch_bytes += images.back().pixels->size() * sizeof(std::uint16_t);
    }
    prompt.preparation.media_items = images.size();
    prompt.preparation.media_preprocess_seconds =
        std::chrono::duration<double>(Clock::now() - media_started).count();
    prompt.preparation.media_preprocess_work_seconds = prompt.preparation.media_preprocess_seconds;

    const auto tokenize_started = Clock::now();
    const std::vector<int> encoded = impl_->tokenizer->encode(rendered.text);
    check_control(control);
    // Every image token the text holds is a placeholder the template wrote for an image part: one
    // the user typed would take an image's place, so the counts must agree.
    std::size_t placeholders = 0;
    for (const int id : encoded) placeholders += id == impl_->image_token;
    if (placeholders != images.size()) {
        throw RequestError(RequestErrorKind::InvalidMedia,
                           "the prompt text contains the image placeholder <|image|>");
    }
    prompt.ids.reserve(encoded.size());
    std::size_t next = 0;
    for (const int id : encoded) {
        if (id != impl_->image_token) {
            prompt.ids.push_back(id);
            continue;
        }
        PreparedImage& image = images[next++];
        prompt.ids.push_back(impl_->image_open);
        prompt.images.push_back({static_cast<std::uint32_t>(prompt.ids.size()), image});
        prompt.ids.insert(prompt.ids.end(), static_cast<std::size_t>(image.soft_tokens()),
                          impl_->image_token);
        prompt.ids.push_back(impl_->image_close);
    }
    if (prompt.ids.size() > impl_->max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prepared prompt has " + std::to_string(prompt.ids.size()) +
                               " tokens, exceeding Engine max_context " +
                               std::to_string(impl_->max_context));
    }
    prompt.descriptor.has_media = !prompt.images.empty();
    prompt.descriptor.prompt_tokens       = static_cast<std::uint32_t>(prompt.ids.size());
    prompt.descriptor.starts_in_reasoning = rendered.thinking;
    prompt.tool_names                     = std::move(rendered.tool_names);
    const auto done                       = Clock::now();
    prompt.preparation.seconds = std::chrono::duration<double>(done - start).count();
    prompt.preparation.tokenize_seconds =
        std::chrono::duration<double>(done - tokenize_started).count();
    return prompt;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).descriptor.prompt_tokens;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    // Deliberately transparent: the token count a caller handed over is the count it gets back, which
    // is what the Engine's scoring contract checks against its own window.
    PreparedPrompt prompt;
    prompt.descriptor.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    prompt.ids                      = std::move(token_ids);
    return prompt;
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const std::vector<int> encoded = impl_->tokenizer->encode(text);
    std::vector<TokenId> ids(encoded.begin(), encoded.end());
    // The model was trained on sequences that begin with <bos>; without it every probability in the
    // sequence moves. Prepending here and not in prepare_tokens keeps the count a caller planned its
    // scoring windows over.
    if (ids.empty() || ids.front() != impl_->bos) ids.insert(ids.begin(), impl_->bos);
    return ids;
}

ModelSamplingDefaults Frontend::sampling_defaults() const { return impl_->sampling; }

// ---- OutputSession ---------------------------------------------------------------------------

namespace {

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

enum class Region : std::uint8_t {
    Content,
    // After <|channel>, until the channel's name ends at its newline.
    ChannelHeader,
    Reasoning,
    ToolCall,
};

struct DecoderState {
    Region region = Region::Content;
    std::string utf8_pending;
    std::array<std::string, 2> stop_pending;
    std::string header;
    std::string call; // the bytes of the call being read, specials included
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
    std::uint32_t reasoning_tokens = 0;
    std::uint64_t decoded_bytes    = 0;
    std::optional<std::uint32_t> matched_stop;
    bool terminal = false;
    // Thinking budget: model-origin tokens inside the thought channel, and the forced close.
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool control_pending                = false;
    bool control_applied                = false;

    [[nodiscard]] bool thinking() const noexcept {
        return region == Region::Reasoning || region == Region::ChannelHeader;
    }
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
    bool preserve_special = false;
    bool raw              = false;
    bool parse_calls      = true;
    std::shared_ptr<const std::vector<std::string>> tool_names;
    TokenId channel_open  = -1;
    TokenId channel_close = -1;
    TokenId call_open     = -1;
    TokenId call_close    = -1;
    std::optional<std::uint32_t> thinking_budget;
    std::array<TokenId, 1> thinking_control{};

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
            if (stop.channel == channel) hold = std::max(hold, longest_suffix_prefix(combined, stop.text));
        }
        emitted.append(channel, combined.substr(0, combined.size() - hold));
        s.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
        s.decoded_bytes += text.size();
    }

    void close_channel(DecoderState& s, OutputChannel channel, PublishedOutput& emitted) const {
        std::string& pending = s.stop_pending[channel_index(channel)];
        emitted.append(channel, std::move(pending));
        pending.clear();
    }

    // Flushes bytes still waiting to complete a code point into the current channel.
    void flush_utf8(DecoderState& s, PublishedOutput& emitted, std::uint32_t committed,
                    StopMatch* best, bool terminal) const {
        std::string text = consume_utf8(s.utf8_pending);
        if (terminal && !s.utf8_pending.empty()) {
            s.utf8_pending.clear();
            text.append(kUtf8Replacement);
        }
        if (text.empty()) return;
        const OutputChannel channel =
            s.region == Region::Reasoning ? OutputChannel::Reasoning : OutputChannel::Content;
        feed_channel(s, channel, text, emitted, committed, best);
    }

    // A call region that does not parse is returned to the content it interrupted, markers and all, so
    // nothing the model wrote disappears.
    void finish_call(DecoderState& s, PublishedOutput& emitted, std::uint32_t committed,
                     StopMatch* best, bool closed) const {
        s.diagnostics.marker_seen = true;
        std::optional<GeneratedToolCall> call = closed ? parse_tool_call(s.call) : std::nullopt;
        if (call && tool_names && !tool_names->empty() &&
            std::ranges::find(*tool_names, call->name) == tool_names->end()) {
            s.diagnostics.fallback_reason = ToolCallParseFallbackReason::UndeclaredTool;
            call.reset();
        }
        if (call) {
            s.calls.push_back(std::move(*call));
            ++s.diagnostics.structured_call_count;
        } else {
            if (s.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
                s.diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            }
            std::string text = "<|tool_call>" + s.call;
            if (closed) text += "<tool_call|>";
            feed_channel(s, OutputChannel::Content, text, emitted, committed, best);
        }
        s.call.clear();
        s.region = Region::Content;
    }

    void terminalize(DecoderState& s, PublishedOutput& emitted, std::uint32_t committed) const {
        if (s.region == Region::ToolCall) {
            s.call += consume_utf8(s.utf8_pending);
            s.utf8_pending.clear();
            finish_call(s, emitted, committed, nullptr, false);
        }
        flush_utf8(s, emitted, committed, nullptr, true);
        if (s.region == Region::ChannelHeader) s.header.clear();
        close_channel(s, OutputChannel::Reasoning, emitted);
        close_channel(s, OutputChannel::Content, emitted);
        s.terminal = true;
    }

    // Feeds one generated token. Returns true when it completed a stop string.
    void feed_token(DecoderState& s, TokenId token, PublishedOutput& emitted,
                    std::uint32_t committed, StopMatch* best) const {
        const tf::DecodedTokenView decoded = tokenizer->decoded_token(token);
        if (!raw) {
            if (token == channel_open && s.region != Region::ToolCall) {
                flush_utf8(s, emitted, committed, best, true);
                close_channel(s, OutputChannel::Content, emitted);
                s.region = Region::ChannelHeader;
                s.header.clear();
                return;
            }
            if (token == channel_close &&
                (s.region == Region::Reasoning || s.region == Region::ChannelHeader)) {
                flush_utf8(s, emitted, committed, best, true);
                close_channel(s, OutputChannel::Reasoning, emitted);
                s.region = Region::Content;
                return;
            }
            if (token == channel_close) return; // a stray close outside a channel carries no text
            if (parse_calls && token == call_open && s.region == Region::Content) {
                flush_utf8(s, emitted, committed, best, true);
                s.region = Region::ToolCall;
                s.call.clear();
                return;
            }
            if (s.region == Region::ToolCall) {
                if (token == call_close) {
                    s.call += consume_utf8(s.utf8_pending);
                    s.utf8_pending.clear();
                    finish_call(s, emitted, committed, best, true);
                } else {
                    // Inside a call the string delimiter <|"|> is syntax, so specials are kept.
                    s.utf8_pending.append(decoded.bytes);
                    s.call += consume_utf8(s.utf8_pending);
                }
                return;
            }
        }
        if (s.region == Region::Reasoning || s.region == Region::ChannelHeader) ++s.reasoning_tokens;
        if (!preserve_special && decoded.special) return;
        if (s.region == Region::ChannelHeader) {
            s.header.append(decoded.bytes);
            const std::size_t newline = s.header.find('\n');
            if (newline == std::string::npos) return;
            std::string rest = s.header.substr(newline + 1);
            s.header.clear();
            s.region = Region::Reasoning;
            s.utf8_pending.append(rest);
            flush_utf8(s, emitted, committed, best, false);
            return;
        }
        s.utf8_pending.append(decoded.bytes);
        flush_utf8(s, emitted, committed, best, false);
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
    if (tokens.empty()) throw std::invalid_argument("cannot preview an empty generated-token round");
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit && limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }
    Impl& impl = *impl_;
    if (impl.state.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    impl.preview = impl.state;
    impl.preview_output.clear();
    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) impl.preview.control_pending = false;
        impl.preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens = count, .finish_reason = reason, .continuation = continuation};
    };
    const bool budgeted = impl.thinking_budget && !impl.state.control_applied;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count = static_cast<std::uint32_t>(index + 1);
        const TokenId token       = tokens[index];
        const bool stop_token =
            std::ranges::find(impl.policy.token_ids, token) != impl.policy.token_ids.end();
        if (stop_token) {
            if (impl.policy.publish_stop_token) {
                impl.feed_token(impl.preview, token, impl.preview_output, count, nullptr);
            }
            impl.terminalize(impl.preview, impl.preview_output, count);
            return complete(count, FinishReason::StopToken);
        }
        StopMatch match;
        impl.feed_token(impl.preview, token, impl.preview_output, count, &match);
        if (budgeted && impl.preview.thinking() &&
            ++impl.preview.model_thinking_tokens > *impl.thinking_budget) {
            throw std::logic_error("model output exceeded the licensed thinking budget");
        }
        if (match.found) {
            impl.preview.utf8_pending.clear();
            impl.preview.stop_pending = {};
            impl.preview.terminal     = true;
            impl.preview.matched_stop = match.declaration_order;
            impl.preview_output       = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }
    }
    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        impl.terminalize(impl.preview, impl.preview_output, count);
        return complete(count, limit_reason);
    }
    if (budgeted && impl.preview.thinking() &&
        impl.preview.model_thinking_tokens == *impl.thinking_budget) {
        impl.preview.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->thinking_budget || impl_->state.control_applied) {
        return total_budget_remaining;
    }
    const DecoderState& s = impl_->state;
    if (s.control_pending) return 0;
    // Outside the channel one token may open it, and that token counts against the budget.
    if (s.model_thinking_tokens >= *impl_->thinking_budget) return s.thinking() ? 0 : total_budget_remaining;
    return std::min(total_budget_remaining, *impl_->thinking_budget - s.model_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->state.control_pending) return {};
    return impl_->thinking_control;
}

std::uint32_t OutputSession::control_suffix_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->thinking_budget || impl_->state.control_applied) return 0;
    return static_cast<std::uint32_t>(impl_->thinking_control.size());
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) throw std::logic_error("output session is empty");
    if (!impl_->thinking_budget || effective_output_tokens <= *impl_->thinking_budget) return;
    if (effective_output_tokens - *impl_->thinking_budget < impl_->thinking_control.size() + 1U) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the channel close and "
            "one post-close model token");
    }
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) return {};
    return ThinkingBudgetStats{
        .configured_budget     = impl_->thinking_budget,
        .model_thinking_tokens = impl_->state.model_thinking_tokens,
        .injected_tokens       = impl_->state.injected_tokens,
        .applied               = impl_->state.control_applied,
    };
}

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
    impl_->terminalize(impl_->preview, impl_->preview_output, 0);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) throw std::logic_error("output session is empty");
    if (impl_->state.terminal) throw std::logic_error("output session is already terminal");
    if (impl_->preview_ready) throw std::logic_error("output session already has a preview");
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || !std::ranges::equal(tokens, expected)) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }
    Impl& impl   = *impl_;
    impl.preview = impl.state;
    impl.preview_output.clear();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        impl.feed_token(impl.preview, tokens[index], impl.preview_output,
                        static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (impl.preview.thinking()) {
        throw std::logic_error("the forced channel close did not end the thought channel");
    }
    impl.preview.control_pending = false;
    impl.preview.control_applied = true;
    impl.preview.injected_tokens = static_cast<std::uint32_t>(tokens.size());
    impl.preview_ready           = true;
    return runtime::OutputDecision{.accepted_tokens = static_cast<std::uint32_t>(tokens.size())};
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
    if (constraint) {
        throw RequestError(RequestErrorKind::InvalidGrammar,
                           "this Gemma model does not support output constraints");
    }
    const bool has_tools = prompt.tool_names && !prompt.tool_names->empty();
    if (has_tools && (tool_choice.mode == ToolChoiceMode::Required || tool_choice.allowed_names)) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "this Gemma model cannot force or restrict tool calls; only "
                           "tool_choice auto or none is supported",
                           {}, RequestErrorSource::Tools);
    }
    auto impl              = std::make_unique<OutputSession::Impl>();
    impl->tokenizer        = impl_->tokenizer;
    impl->raw              = output.raw;
    impl->preserve_special = output.raw || output.preserve_special_tokens;
    impl->parse_calls      = !output.raw && has_tools && tool_choice.mode != ToolChoiceMode::None;
    impl->tool_names       = prompt.tool_names;
    impl->channel_open     = impl_->channel_open;
    impl->channel_close    = impl_->channel_close;
    impl->call_open        = impl_->call_open;
    impl->call_close       = impl_->call_close;
    impl->thinking_budget  = thinking.budget;
    impl->thinking_control = {impl_->channel_close};
    impl->policy.strings   = caller_stop.strings;
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
    return OutputSession(std::move(impl));
}

} // namespace ninfer::models::gemma4
