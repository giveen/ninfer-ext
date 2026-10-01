#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options cached =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hi", "--expert-cache", "20480"});
    failures += check(cached.expert_cache.mode == ninfer::ExpertCacheMode::Explicit &&
                          cached.expert_cache.explicit_bytes == 20480ULL * 1024ULL * 1024ULL,
                      "--expert-cache MiB did not select an explicit cache");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hi"}).expert_cache.mode ==
                          ninfer::ExpertCacheMode::Automatic,
                      "omitted --expert-cache is not automatic");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hi",
                                       "--expert-cache", "0"});
                      }),
                      "--expert-cache accepted zero");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hi", "--ngram-residency",
                             "mapped"})
                              .ngram_residency == ninfer::NgramResidency::Mapped,
                      "--ngram-residency mapped did not parse");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hi",
                                       "--ngram-residency", "pinned"});
                      }),
                      "--ngram-residency accepted an unknown mode");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").contains("--thinking-budget"),
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options adaptive_mtp =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
               "--draft-tokens", "7"});
    failures += check(adaptive_mtp.speculative.draft_tokens == 7 &&
                          !adaptive_mtp.speculative.fixed_draft,
                      "MTP did not default to adaptive drafting");
    const ninfer::cli::Options fixed_mtp =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
               "--draft-tokens", "3", "--fixed-draft"});
    failures += check(fixed_mtp.speculative.draft_tokens == 3 && fixed_mtp.speculative.fixed_draft,
                      "--fixed-draft was not preserved");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--draft-tokens", "8"});
                      }),
                      "CLI accepted an MTP draft window outside [1,7]");
    const ninfer::cli::Options default_mtp =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp"});
    failures += check(default_mtp.speculative.draft_tokens == 7 &&
                          !default_mtp.speculative.fixed_draft,
                      "--spec mtp alone did not select adaptive drafting up to 7");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--fixed-draft"});
                      }),
                      "--fixed-draft was accepted without a draft length");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--fixed-draft"});
                      }),
                      "--fixed-draft was accepted without a speculative backend");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "dflash", "--draft-tokens", "7", "--fixed-draft"});
                      }),
                      "--fixed-draft was accepted for DFlash");
    failures += check(ninfer::cli::usage_text("ninfer-cli").contains("--fixed-draft"),
                      "CLI help omits --fixed-draft");
    const ninfer::cli::Options lookup =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
               "--draft-tokens", "3", "--lookup-drafts", "always", "--lookup-min-match", "6"});
    failures += check(lookup.speculative.lookup_drafts == ninfer::LookupDraftMode::Always &&
                          lookup.speculative.lookup_min_match == 6,
                      "CLI did not preserve the lookup-drafts settings");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--lookup-drafts", "auto"});
                      }),
                      "lookup drafts were accepted without a speculative backend");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "dflash", "--draft-tokens", "7", "--lookup-drafts", "auto"});
                      }),
                      "lookup drafts were accepted for DFlash");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--draft-tokens", "3", "--lookup-drafts", "auto",
                                       "--lookup-min-match", "2"});
                      }),
                      "CLI accepted a lookup min match below 3");
    failures += check(ninfer::cli::usage_text("ninfer-cli").contains("--lookup-drafts"),
                      "CLI help omits --lookup-drafts");
    const ninfer::cli::Options wide_lookup =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
               "--draft-tokens", "15", "--fixed-draft", "--lookup-drafts", "always"});
    failures += check(wide_lookup.speculative.draft_tokens == 15 &&
                          wide_lookup.speculative.fixed_draft,
                      "CLI did not accept a lookup-width draft window");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--draft-tokens", "15", "--fixed-draft"});
                      }),
                      "a draft window above 7 was accepted without lookup drafting");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--draft-tokens", "15", "--lookup-drafts", "always"});
                      }),
                      "a wide lookup window was accepted without --fixed-draft");
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.contains("nvfp4") && help.contains("k8v4"),
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.contains("--log-level"),
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
