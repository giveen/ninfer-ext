// The Nemotron-H text config: a hybrid block list whose FFN shape varies per layer.
//
// Two things are worth a test here and they are different in kind. The first is arithmetic: the
// widths derived from the checkpoint's own numbers -- 128 heads of 64 = 8192 inner, the
// convolution's 10240 channels, the 18560-wide input projection -- are what the rest of the port
// sizes kernels and workspaces from, and this checkpoint's numbers are unusual enough (a 1024-wide
// latent, banks of 240 to 496 experts) that eyeballing them is not a check.
//
// The second is the rule that keeps a per-layer bank from being assumed: a layer's shape may not be
// silently mixed. The checkpoint's own config class refuses a field that some blocks set and others
// leave global, and so does this parser, because the alternative is a model that loads and computes
// the wrong thing.

#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "models/nemotron_h/config.h"

#include <nlohmann/json.hpp>

#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;
using models::nemotron_h::BlockKind;
using models::nemotron_h::parse_config;
using Json = nlohmann::json;

// The real checkpoint's constants, with a four-entry block list in place of its 88 so the test
// stays small while every per-layer field still varies.
Json base_config() {
    return Json{
        {"hidden_size", 4096},
        {"vocab_size", 131072},
        {"max_position_embeddings", 262144},
        {"tie_word_embeddings", false},
        {"layer_norm_epsilon", 1e-5},
        {"moe_latent_size", 1024},
        {"moe_shared_expert_intermediate_size", 5376},
        {"mamba", Json{{"mamba_num_heads", 128},
                       {"mamba_head_dim", 64},
                       {"ssm_state_size", 128},
                       {"n_groups", 8},
                       {"conv_kernel", 4},
                       {"chunk_size", 128},
                       {"time_step_min", 0.001},
                       {"use_conv_bias", true}}},
        {"attention",
         Json{{"num_attention_heads", 32}, {"num_key_value_heads", 2}, {"head_dim", 128}}},
        {"router", Json{{"routed_scaling_factor", 5.0},
                        {"norm_topk_prob", true},
                        {"e_score_correction_bias", true},
                        {"n_group", 1},
                        {"topk_group", 1}}},
        {"block_configs", Json::array({
                              Json{{"block_type", "mamba"}},
                              Json{{"block_type", "moe"},
                                   {"n_routed_experts", 240},
                                   {"num_experts_per_tok", 9},
                                   {"moe_intermediate_size", 2048}},
                              Json{{"block_type", "attention"}},
                              Json{{"block_type", "moe"},
                                   {"n_routed_experts", 496},
                                   {"num_experts_per_tok", 9},
                                   {"moe_intermediate_size", 2560}},
                          })},
    };
}

models::nemotron_h::Config parse(const Json& text_config) {
    Fixture fixture;
    fixture.root["components"]["text"]["config"] = text_config;
    fixture.write();
    Reader reader(fixture.entry);
    return parse_config(reader.directory());
}

void derived_widths_match_the_checkpoint() {
    const auto config = parse(base_config());
    require(config.text.num_hidden_layers() == 4, "block list length lost");
    require(config.text.moe_layer_count() == 2, "MoE layer count wrong");

    require(config.text.block_kind(0) == BlockKind::Mamba, "layer 0 is not mamba");
    require(config.text.block_kind(1) == BlockKind::Moe, "layer 1 is not moe");
    require(config.text.block_kind(2) == BlockKind::Attention, "layer 2 is not attention");
    require(config.text.block_kind(3) == BlockKind::Moe, "layer 3 is not moe");

    require(config.text.mamba.has_value(), "mamba config missing although mamba blocks exist");
    const auto& mamba = *config.text.mamba;
    // 128 heads x 64 head_dim, then the convolution sees x, B and C: 8192 + 2*8*128, and in_proj
    // emits z, x/B/C and dt: 2*8192 + 10240 + 128.
    require(mamba.inner_size() == 8192, "mamba inner size is not 8192");
    require(mamba.conv_channels() == 10240, "mamba conv channels are not 10240");
    require(mamba.in_projection_size() == 18560, "mamba in_projection size is not 18560");
    require(mamba.conv_bias, "the depthwise convolution's bias was dropped");

    require(config.text.attention.has_value(), "attention config missing");
    require(config.text.attention->query_width() == 4096, "query width is not 4096");
    require(config.text.attention->key_width() == 256, "key width is not 256");
    require(config.text.attention->rope_theta == 10000.0F, "unset rope_theta did not default");

    require(config.text.router.scaling_factor == 5.0F, "routed_scaling_factor lost");
    require(config.text.router.normalize_top_k, "norm_topk_prob lost");
    require(config.text.router.correction_bias, "the correction bias buffer was dropped");
    require(config.text.router.groups == 1 && config.text.router.groups_used == 1,
            "the degenerate group setting changed");
}

void banks_differ_per_layer() {
    const auto config = parse(base_config());
    const auto& first = config.text.moe_bank(1);
    const auto& last  = config.text.moe_bank(3);

    require(first.num_experts == 240 && last.num_experts == 496,
            "per-layer expert counts were flattened");
    require(first.intermediate_size == 2048 && last.intermediate_size == 2560,
            "per-layer expert widths were flattened");
    require(first.latent_size == 1024 && first.shared_intermediate_size == 5376,
            "the model-wide latent and shared widths were lost");
    require(first.latent(), "a 1024-wide latent bank does not report a latent");
    // Two matrices per expert, in the latent space: 2 * 1024 * 2048.
    require(first.parameters_per_expert(4096) == 4194304, "expert parameter count is wrong");
    require(first.expert_input_width(4096) == 1024, "expert input is not the latent width");
}

void a_non_moe_layer_has_no_bank() {
    const auto config = parse(base_config());
    rejects([&] { (void)config.text.moe_bank(0); }, "a mamba layer produced an expert bank");
    rejects([&] { (void)config.text.moe_bank(4); }, "an out-of-range layer produced a bank");
    rejects([&] { (void)config.text.block_kind(4); }, "an out-of-range layer produced a kind");
}

void per_layer_fields_may_not_be_mixed() {
    // A MoE block that omits its shape: the parser must refuse rather than fall back to a global.
    auto missing                = base_config();
    missing["block_configs"][3] = Json{{"block_type", "moe"}};
    rejects([&] { (void)parse(missing); }, "a moe block without its shape was accepted");

    // A mamba block that carries one: the shape would be read from a layer that cannot use it.
    auto extra                                    = base_config();
    extra["block_configs"][0]["n_routed_experts"] = 240;
    rejects([&] { (void)parse(extra); }, "a non-moe block with expert fields was accepted");

    auto unknown                              = base_config();
    unknown["block_configs"][2]["block_type"] = "hybrid";
    rejects([&] { (void)parse(unknown); }, "an unknown block type was accepted");
}

void inconsistent_dimensions_are_refused() {
    auto heads                   = base_config();
    heads["num_attention_heads"] = 33;
    rejects([&] { (void)parse(heads); }, "32 query heads with 2 KV heads accepted 33");

    auto groups        = base_config();
    groups["n_groups"] = 7; // 8192 is not divisible by 7
    rejects([&] { (void)parse(groups); }, "a non-integral norm group size was accepted");

    auto empty             = base_config();
    empty["block_configs"] = Json::array();
    rejects([&] { (void)parse(empty); }, "an empty block list was accepted");

    auto zero_time_step             = base_config();
    zero_time_step["time_step_min"] = 0.0;
    rejects([&] { (void)parse(zero_time_step); }, "a zero time-step floor was accepted");
}

} // namespace

int main() {
    try {
        derived_widths_match_the_checkpoint();
        banks_differ_per_layer();
        a_non_moe_layer_has_no_bank();
        per_layer_fields_may_not_be_mixed();
        inconsistent_dimensions_are_refused();
    } catch (const std::exception& error) {
        std::cerr << "nemotron_h config: " << error.what() << "\n";
        return 1;
    }
    std::cout << "nemotron_h config: ok\n";
    return 0;
}
