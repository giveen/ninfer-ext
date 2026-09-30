#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

[[nodiscard]] inline LookupDraftMode parse_lookup_draft_mode(std::string_view value) {
    if (value == "off") { return LookupDraftMode::Off; }
    if (value == "auto") { return LookupDraftMode::Auto; }
    if (value == "always") { return LookupDraftMode::Always; }
    throw std::invalid_argument("invalid lookup-drafts mode: " + std::string(value));
}

[[nodiscard]] inline const char* lookup_draft_mode_name(LookupDraftMode mode) noexcept {
    switch (mode) {
    case LookupDraftMode::Off:
        return "off";
    case LookupDraftMode::Auto:
        return "auto";
    case LookupDraftMode::Always:
        return "always";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (options.lookup_drafts != LookupDraftMode::Off) {
        if (options.backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument("--lookup-drafts requires --spec mtp");
        }
        if (options.lookup_min_match < 3 || options.lookup_min_match > 32) {
            throw std::invalid_argument("--lookup-min-match must be in [3,32]");
        }
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full ||
            options.fixed_draft) {
            throw std::invalid_argument(
                "--draft-tokens, --lm-head-draft and --fixed-draft require --spec "
                "mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,7]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        if (options.fixed_draft) {
            throw std::invalid_argument("--fixed-draft applies only to --spec mtp");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        if (options.fixed_draft) {
            throw std::invalid_argument("--fixed-draft applies only to --spec mtp");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

// Applies the parsed-flag defaults, then validates: `--spec mtp` without `--draft-tokens` selects the
// adaptive policy with the longest draft, seven tokens. `--fixed-draft` still needs an explicit length.
inline void resolve_speculative_cli_options(SpeculativeOptions& options) {
    if (options.backend == SpeculativeBackend::Mtp && options.draft_tokens == 0 &&
        !options.fixed_draft) {
        options.draft_tokens = 7;
    }
    validate_speculative_cli_options(options);
}

} // namespace ninfer::product
