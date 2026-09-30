#pragma once
// Shared gate for the opt-in real-model Engine tests.
//
// Each of these tests loads a model from NINFER_TEST_ARTIFACT, which CTest sets
// from the matching NINFER_ARTIFACT_* cache variable. The tests share one opt-in
// surface but not one artifact: a DFlash2 artifact cannot serve the DFlash test,
// and an artifact without a proposal head cannot serve the MoE test. An artifact
// that simply does not carry the component or proposal head the test requests is
// therefore "not applicable" and skips, exactly like a missing artifact. Any
// other failure stays a failure (<c>ArtifactError</c> for a corrupt artifact
// included), so a broken artifact is never mistaken for a skip.
#include <exception>
#include <iostream>
#include <string_view>

namespace ninfer::test {

// True when the artifact does not carry what the test asked it to load. The
// messages are the component/proposal diagnostics from the artifact schema and
// the Qwen3.5 loader.
[[nodiscard]] inline bool artifact_component_mismatch(const std::exception& error) noexcept {
    const std::string_view message = error.what();
    return message.find("missing component") != std::string_view::npos ||
           message.find("absent from artifact") != std::string_view::npos ||
           message.find("proposal domain is smaller") != std::string_view::npos;
}

// Terminal result for a real-test failure: a component mismatch is a skip (77),
// anything else is a failure (1). Prints the reason either way.
[[nodiscard]] inline int real_test_error(const std::exception& error) {
    if (artifact_component_mismatch(error)) {
        std::cout << "skip: artifact does not carry the required component: " << error.what()
                  << '\n';
        return 77;
    }
    std::cerr << error.what() << '\n';
    return 1;
}

} // namespace ninfer::test
