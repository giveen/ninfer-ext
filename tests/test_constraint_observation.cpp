// ninfer::product - constraint observation wire schema checks.
#include "product/constraint_observation.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <string>

namespace {

using Json = nlohmann::ordered_json;
using ninfer::ConstraintCacheAccess;
using ninfer::ConstraintObservation;
using ninfer::ConstraintOutputBranch;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Json observe(const ConstraintObservation& observation) {
    return ninfer::product::constraint_observation_json(observation);
}

int test_absent_is_null() {
    return check(ninfer::product::constraint_observation_json(std::nullopt).is_null(),
                 "an absent observation must serialize as null");
}

int test_branch_and_cache_names() {
    int failures = 0;
    for (const auto& [branch, name] : {std::pair{ConstraintOutputBranch::Content, "content"},
                                       std::pair{ConstraintOutputBranch::Tools, "tools"},
                                       std::pair{ConstraintOutputBranch::Undecided, "undecided"}}) {
        ConstraintObservation observation;
        observation.branch = branch;
        failures += check(observe(observation).at("branch") == name,
                          std::string("branch ") + name + " must serialize as its name");
    }
    for (const auto& [cache, name] : {std::pair{ConstraintCacheAccess::Hit, "hit"},
                                      std::pair{ConstraintCacheAccess::Built, "built"},
                                      std::pair{ConstraintCacheAccess::Waited, "waited"}}) {
        ConstraintObservation observation;
        observation.cache = cache;
        failures += check(observe(observation).at("cache") == name,
                          std::string("cache ") + name + " must serialize as its name");
    }
    return failures;
}

int test_flags_and_mask_counts() {
    ConstraintObservation observation;
    observation.branch            = ConstraintOutputBranch::Tools;
    observation.complete          = true;
    observation.terminated        = true;
    observation.mask_positions    = 37;
    observation.mask_upload_bytes = 4096;
    const Json payload            = observe(observation);
    int failures                  = 0;
    failures += check(payload.at("complete") == true, "complete must round-trip");
    failures += check(payload.at("terminated") == true, "terminated must round-trip");
    failures += check(payload.at("mask_positions") == 37, "mask_positions must round-trip");
    failures += check(payload.at("mask_upload_bytes") == 4096, "mask_upload_bytes must round-trip");
    failures += check(payload.size() == 6, "an observation without timings carries six keys");
    failures += check(!payload.contains("timings_seconds"),
                      "an observation without timings omits timings_seconds");
    return failures;
}

int test_timings_are_conditional() {
    int failures = 0;
    ConstraintObservation collected;
    collected.timings_collected = true;
    collected.prepare_seconds   = 0.5;
    collected.mask_seconds      = 0.25;
    collected.matcher_seconds   = 1.5;
    const Json payload          = observe(collected);
    failures += check(payload.contains("timings_seconds"),
                      "a collected timing must serialize timings_seconds");
    if (payload.contains("timings_seconds")) {
        const Json& timings = payload.at("timings_seconds");
        failures += check(timings.at("prepare") == 0.5, "prepare seconds must round-trip");
        failures += check(timings.at("mask") == 0.25, "mask seconds must round-trip");
        failures += check(timings.at("matcher") == 1.5, "matcher seconds must round-trip");
        failures += check(timings.size() == 3, "timings_seconds carries three keys");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_absent_is_null();
    failures += test_branch_and_cache_names();
    failures += test_flags_and_mask_counts();
    failures += test_timings_are_conditional();
    if (failures != 0) {
        std::cerr << failures << " constraint observation checks failed\n";
        return 1;
    }
    std::cout << "Constraint observation checks passed\n";
    return 0;
}
