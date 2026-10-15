#pragma once

// Sensitivity-probe targets: every EXL3 projection the model prepared, keyed by its logical name.
// The M5 sensitivity measurement perturbs one weight at a time and needs to name it, but a prepared
// `Weight` carries only its trellis plane -- the name is known here, at preparation.

#include "core/weight.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

struct ProbeTarget {
    std::string name;
    const void* qdata = nullptr;
    std::int32_t n    = 0;
    std::int32_t k    = 0;
};

inline std::vector<ProbeTarget>& probe_targets() {
    static std::vector<ProbeTarget> targets;
    return targets;
}

// Drop every recorded target. Preparation is the only writer, so a new preparation starts from an
// empty registry: otherwise a stale address can read storage a destroyed model freed, or a target
// that happens to reuse one is silently skipped by register_probe_target's address check.
inline void clear_probe_targets() { probe_targets().clear();
}

// Records one prepared EXL3 projection. A no-op for every other format, and for a repeat of a plane
// already recorded (preparation visits some parameters more than once).
inline void register_probe_target(const std::string& name, const Weight& weight) {
    if (weight.qtype != QType::EXL3_MUL1 || weight.qdata == nullptr) { return; }
    for (const auto& target : probe_targets()) {
        if (target.qdata == weight.qdata) { return; }
    }
    probe_targets().push_back(ProbeTarget{name, weight.qdata, weight.n, weight.k});
}

} // namespace ninfer::models::qwen3_5::execution
