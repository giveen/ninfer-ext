// Gemma 4 text config: the artifact's own config parses into the fields the Program consumes, and
// every invariant the mathematics depends on is refused when it does not hold.
#include "models/gemma4/config.h"

#include "artifact/schema.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using ninfer::artifact::Json;
using ninfer::models::gemma4::MixerKind;
using ninfer::models::gemma4::parse_text_config;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// The value must be refused: this build cannot execute it, so it must not be read as something else.
void expect_rejected(const Json& config, const std::string& what) {
    try {
        (void)parse_text_config(config);
        expect(false, what + " was accepted");
    } catch (const ninfer::artifact::ArtifactError&) {}
}

} // namespace

int main() {
    const std::string fixture =
        read_file(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/gemma4/text_config.json");
    const Json record         = Json::parse(fixture);
    const Json config         = record.at("config");
    const auto text           = parse_text_config(config);

    expect(text.architecture == ninfer::models::Architecture::Gemma4, "architecture");
    expect(text.hidden_size == 5376 && text.intermediate_size == 21504, "widths");
    expect(text.vocab_size == 262144, "vocabulary");
    expect(text.num_hidden_layers == 60 && text.layer_types.size() == 60, "layers");
    expect(text.global_layers == 10, "global layer count");
    expect(text.layer_types.at(5) == MixerKind::FullAttention &&
           text.layer_types.at(59) == MixerKind::FullAttention &&
           text.layer_types.at(0) == MixerKind::SlidingAttention, "layer pattern");
    expect(text.sliding.num_key_value_heads == 16 && text.sliding.head_dim == 256, "sliding geometry");
    expect(text.global.shared.num_key_value_heads == 4 && text.global.shared.head_dim == 512,
           "global geometry");
    expect(text.sliding.rotary_dim == 256 && text.global.shared.rotary_dim == 128, "rotary widths");
    expect(text.global.rope_angles == 64 && text.global.denominator_head_dim == 512,
           "proportional rope");
    expect(text.sliding_window == 1024, "window");
    expect(text.attention_scale == 1.0F, "attention scale");
    expect(std::fabs(text.final_logit_softcapping - 30.0F) < 1e-6F, "soft cap");
    expect(std::fabs(text.embedding_scale - std::sqrt(5376.0F)) < 1e-3F, "embedding scale");
    expect(text.tie_word_embeddings && text.hidden_act == "gelu_pytorch_tanh", "head and activation");
    // The geometry accessor follows the layer's own kind.
    expect(&text.geometry(0) == &text.sliding, "layer 0 is sliding");
    expect(&text.geometry(5) == &text.global.shared, "layer 5 is global");

    // Refusals: an unknown member, and each invariant the implementation relies on.
    Json extra = config;
    extra["attention_dropout"] = 0.0;
    expect_rejected(extra, "an unknown member");
    Json offset = config;
    offset["norm_unit_offset"] = true;
    expect_rejected(offset, "a norm unit offset");
    Json scaled = config;
    scaled["attention_scale"] = 1.0 / std::sqrt(256.0);
    expect_rejected(scaled, "an attention scale other than 1");
    Json pattern = config;
    pattern["layer_types"][5] = "sliding_attention";
    expect_rejected(pattern, "a layer pattern without global attention every sixth");
    Json angles = config;
    angles["rope_parameters"]["full_attention"]["rope_angles"] = 64 * 2;
    expect_rejected(angles, "proportional rope angles that disagree with the rotary width");
    Json tied = config;
    tied["tie_word_embeddings"] = false;
    expect_rejected(tied, "an untied head");

    if (failures == 0) {
        std::cout << "OK: Gemma 4 text config parses and its invariants are enforced\n";
    }
    return failures == 0 ? 0 : 1;
}
