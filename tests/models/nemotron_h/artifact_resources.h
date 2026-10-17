#pragma once

// Shared gate and resource reader for the opt-in Nemotron-H frontend tests.
//
// Every one of them needs the artifact's text resources -- the tokenizer, its config, the
// generation config and the chat template -- and only the loader can hand those out: they borrow
// bytes the plan read out of the artifact, valid for as long as the plan lives. The reader below
// therefore owns the Reader and the LoadPlan together and copies what the frontend owns.

#include "artifact/reader.h"
#include "models/nemotron_h/frontend.h"
#include "models/nemotron_h/load.h"

#include <cstdlib>
#include <memory>
#include <string>

namespace ninfer::test {

// The artifact the frontend tests read. The routed banks make it 45 GB, so a routine run must not
// pay for it: an unset variable skips (77) instead.
[[nodiscard]] inline const char* nemotron_artifact_path() noexcept {
    const char* path = std::getenv("NINFER_NEMOTRON_ARTIFACT");
    return path != nullptr && *path != '\0' ? path : nullptr;
}

class NemotronResources {
public:
    explicit NemotronResources(const char* path)
        : reader_(std::make_unique<ninfer::artifact::Reader>(path)),
          plan_(std::make_unique<ninfer::models::nemotron_h::LoadPlan>(
              ninfer::models::nemotron_h::plan_load(*reader_))) {
        const ninfer::models::nemotron_h::TextResources& resources = plan_->resources();
        frontend_ = ninfer::models::nemotron_h::FrontendResources{
            .tokenizer_json         = std::string(resources.tokenizer_json),
            .tokenizer_config_json  = std::string(resources.tokenizer_config_json),
            .generation_config_json = std::string(resources.generation_config_json),
            .chat_template_jinja    = std::string(resources.chat_template_jinja),
        };
    }

    [[nodiscard]] const ninfer::models::nemotron_h::FrontendResources& frontend() const noexcept {
        return frontend_;
    }

private:
    std::unique_ptr<ninfer::artifact::Reader> reader_;
    std::unique_ptr<ninfer::models::nemotron_h::LoadPlan> plan_;
    ninfer::models::nemotron_h::FrontendResources frontend_;
};

} // namespace ninfer::test
