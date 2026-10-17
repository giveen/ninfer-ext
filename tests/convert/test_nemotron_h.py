"""Nemotron-H adapter: the config the artifact carries, the bank layout, and the recipe.

What this adapter emits is the contract `src/models/nemotron_h/config.cpp` enforces -- its parser
validates every member it is given, so a field from another level is a refusal rather than a field it
ignores -- and the fixtures under `tests/fixtures/nemotron_h` are the released BF16 checkpoint's own
`llm_config` beside the config the adapter derives from it.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools.convert.methods import fp8_row_maxabs, nvfp4_mse
from tools.convert.model import Model, Parameter
from tools.convert.nemotron_h import (
    _tap_major,
    is_lilicorr_draft,
    is_nemotron_h,
    lilicorr_config,
    text_config,
)
from tools.convert.official_recipes import RECIPES, nemotron_h_74b_nvfp4
from tools.convert.recipe import Recipe
from tools.convert.resources import (
    NEMOTRON_SPLIT_PATTERN,
    normalize_nemotron_tokenizer,
    validate_nemotron_tokenizer,
)
from tools.convert.sources.logical import array_source

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "nemotron_h"
# The exact member sets src/models/nemotron_h/config.cpp accepts, level by level.
_TEXT_KEYS = {
    "hidden_size",
    "vocab_size",
    "max_position_embeddings",
    "tie_word_embeddings",
    "layer_norm_epsilon",
    "block_configs",
    "mamba",
    "attention",
    "router",
    "moe_latent_size",
    "moe_shared_expert_intermediate_size",
}
_MAMBA_KEYS = {
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "n_groups",
    "conv_kernel",
    "chunk_size",
    "time_step_min",
    "use_conv_bias",
}
_ATTENTION_KEYS = {"num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta"}
_ROUTER_KEYS = {"routed_scaling_factor", "norm_topk_prob", "n_group", "topk_group",
                "e_score_correction_bias"}


def _llm() -> dict:
    return json.loads((_FIXTURES / "llm_config.json").read_text())


def _draft() -> dict:
    return json.loads((_FIXTURES / "lilicorr_config.json").read_text())


def test_the_lilicorr_draft_is_recognized_and_normalized_against_the_tower() -> None:
    target = text_config({"llm_config": _llm()})
    draft = lilicorr_config(_draft(), target)
    assert is_lilicorr_draft(_draft())
    assert not is_lilicorr_draft({"architectures": ["DFlashDraftModel"]})
    # The release's own structure: five sliding-attention blocks over six of this tower's taps.
    assert draft["num_hidden_layers"] == 5
    assert draft["layer_types"] == ["sliding_attention"] * 5
    assert draft["block_size"] == 8
    assert draft["sliding_window"] == 1024
    assert draft["rope_parameters"]["rope_theta"] == 10000.0
    assert draft["max_position_embeddings"] == 262144
    assert draft["hidden_size"] == target["hidden_size"]
    assert draft["dflash_config"]["target_layer_ids"] == [1, 18, 35, 51, 68, 85]
    assert draft["dflash_config"]["mask_token_id"] == 990
    assert draft["dflash_config"]["attention_sink_bias"] is True
    assert draft["dflash_config"]["conv_kernel_size"] == 2
    assert draft["dflash_config"]["conv_group_size"] == 16
    assert draft["dflash_config"]["lilicorr_candidate_topk"] == 8
    assert draft["dflash_config"]["lilicorr_hidden_size"] == 1024


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 2048),
        ("num_hidden_layers", 4),
        ("rope_theta", 1e7),
        ("sliding_window", 4096),
        ("attention_sink_bias", False),
    ],
)
def test_a_draft_that_disagrees_with_the_family_is_refused(field, value) -> None:
    target = text_config({"llm_config": _llm()})
    raw = _draft()
    raw[field] = value
    with pytest.raises(ValueError):
        lilicorr_config(raw, target)


def test_a_draft_over_other_taps_is_refused() -> None:
    target = text_config({"llm_config": _llm()})
    raw = _draft()
    raw["dflash_config"]["target_layer_ids"] = [1, 18, 35]
    with pytest.raises(ValueError):
        lilicorr_config(raw, target)
    # A tap past the tower's blocks is refused as well, not silently accepted.
    raw = _draft()
    raw["dflash_config"]["target_layer_ids"] = [1, 18, 35, 51, 68, 88]
    with pytest.raises(ValueError):
        lilicorr_config(raw, target)


def test_emitted_config_matches_the_committed_fixture() -> None:
    assert text_config({"llm_config": _llm()}) == json.loads(
        (_FIXTURES / "text_config.json").read_text()
    )


def test_the_emitted_members_are_exactly_the_parsers() -> None:
    config = text_config({"llm_config": _llm()})
    assert set(config) == _TEXT_KEYS
    assert set(config["mamba"]) == _MAMBA_KEYS
    # `rope_theta` is optional and this checkpoint declares none: the reference has no positional
    # encoding at all.
    assert set(config["attention"]) == _ATTENTION_KEYS - {"rope_theta"}
    assert set(config["router"]) == _ROUTER_KEYS
    for block in config["block_configs"]:
        expected = {"block_type"}
        if block["block_type"] == "moe":
            expected |= {"n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"}
        assert set(block) == expected


def test_the_block_list_is_what_decides_each_layers_mixer() -> None:
    config = text_config({"llm_config": _llm()})
    kinds = [block["block_type"] for block in config["block_configs"]]
    assert len(kinds) == 88
    assert (kinds.count("mamba"), kinds.count("moe"), kinds.count("attention")) == (40, 40, 8)
    assert kinds[:8] == ["mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention"]


def test_every_bank_the_family_has_is_accepted_and_no_other() -> None:
    config = text_config({"llm_config": _llm()})
    pairs = {
        (block["n_routed_experts"], block["moe_intermediate_size"])
        for block in config["block_configs"]
        if block["block_type"] == "moe"
    }
    assert len(pairs) == 12, "the release has twelve distinct (experts, width) pairs"

    source = _llm()
    for index, block in enumerate(source["block_configs"]):
        if block["block_type"] != "moe":
            continue
        source["block_configs"][index]["n_routed_experts"] = 512  # the config's bound, not a bank
        break
    with pytest.raises(ValueError, match="unsupported bank width"):
        text_config({"llm_config": source})


def test_a_non_moe_block_may_not_declare_a_bank() -> None:
    source = _llm()
    source["block_configs"][0]["n_routed_experts"] = 240
    with pytest.raises(ValueError, match="declares"):
        text_config({"llm_config": source})


def test_mtp_layers_are_refused_rather_than_ignored() -> None:
    source = _llm()
    source["mtp_block_configs"] = [{"block_type": "moe", "n_routed_experts": 240,
                                    "num_experts_per_tok": 9, "moe_intermediate_size": 2048}]
    with pytest.raises(ValueError, match="MTP"):
        text_config({"llm_config": source})


def test_the_layer_type_list_must_agree_with_the_blocks() -> None:
    source = _llm()
    source["layers_block_type"] = list(reversed(source["layers_block_type"]))
    with pytest.raises(ValueError, match="layers_block_type"):
        text_config({"llm_config": source})


def test_a_source_that_disagrees_with_the_family_is_refused() -> None:
    source = _llm()
    source["chunk_size"] = 64
    with pytest.raises(ValueError, match="chunk_size"):
        text_config({"llm_config": source})


def test_only_the_bolt_architectures_are_recognized() -> None:
    assert is_nemotron_h({"architectures": ["NemotronH_Omni_Reasoning_V3"]})
    assert is_nemotron_h({"architectures": ["NemotronHPuzzleForCausalLM"]})
    assert is_nemotron_h({"llm_config": {"architectures": ["NemotronHPuzzleForCausalLM"]}})
    assert not is_nemotron_h({"architectures": ["Gemma4ForCausalLM"]})
    assert not is_nemotron_h({"architectures": ["Qwen4ExpForCausalLM"]})


def test_the_convolution_weight_is_stored_tap_major() -> None:
    # The Op indexes `weight[j * C + c]`; the checkpoint's `[C, 1, K]` is `c * K + j`.
    channels, kernel = 5, 4
    source = torch.arange(channels * kernel, dtype=torch.float32).reshape(channels, 1, kernel)
    flat = _tap_major(array_source(source, "conv"), (channels, kernel)).values()
    for channel in range(channels):
        for tap in range(kernel):
            assert flat[tap * channels + channel].item() == source[channel, 0, tap].item()


def _layer_model() -> Model:
    """One Mamba, one attention and one MoE layer in the vocabulary the artifact binds."""

    config = text_config({"llm_config": _llm()})
    model = Model({"text": {"config": config}})

    def add(name: str, shape: tuple[int, ...], inputs: tuple[str, ...] = ("x",)) -> None:
        model.add(
            Parameter(name, shape, array_source(torch.ones(shape, dtype=torch.bfloat16), name),
                      inputs=inputs)
        )

    add("text/token_embedding", (8, 4), ())
    add("text/output_head", (8, 4), ("text/final_hidden",))
    add("text/final_norm", (4,), ())
    add("text/layers/0/block_norm", (4,), ())
    add("text/layers/0/mamba/in_projection", (8, 4), ("text/layers/0/mixer_input",))
    add("text/layers/0/mamba/out_projection", (4, 4), ("text/layers/0/mamba/gated_norm_output",))
    add("text/layers/0/mamba/convolution", (4, 4), ())
    add("text/layers/0/mamba/a_log", (4,), ())
    add("text/layers/7/attention/query", (4, 4), ("text/layers/7/mixer_input",))
    add("text/layers/7/attention/output", (4, 4), ("text/layers/7/attention/attended",))
    add("text/layers/1/block_norm", (4,), ())
    add("text/layers/1/moe/router", (8, 4), ("text/layers/1/ffn_input",))
    add("text/layers/1/moe/correction_bias", (8,), ())
    add("text/layers/1/moe/latent_in", (4, 4), ("text/layers/1/ffn_input",))
    add("text/layers/1/moe/latent_out", (4, 4), ("text/layers/1/moe/expert_product",))
    add("text/layers/1/moe/shared/up", (4, 4), ("text/layers/1/ffn_input",))
    add("text/layers/1/moe/shared/down", (4, 4), ("text/layers/1/moe/shared/product",))
    for expert in range(2):
        add(f"text/layers/1/moe/experts/{expert}/up", (4, 4), ("text/layers/1/moe/expert_input",))
        add(f"text/layers/1/moe/experts/{expert}/down", (4, 4),
            (f"text/layers/1/moe/experts/{expert}/product",))
    model.packing_groups.append(
        ("text/layers/1/moe/experts/0/up", "text/layers/1/moe/experts/1/up")
    )
    model.packing_groups.append(
        ("text/layers/1/moe/experts/0/down", "text/layers/1/moe/experts/1/down")
    )
    return model


def _selection(model: Model, name: str, function=nemotron_h_74b_nvfp4):
    recipe = Recipe(model)
    function(model, recipe, {})
    selections = recipe.selections[name]
    assert len(selections) == 1, f"{name} is split across {len(selections)} selections"
    return recipe, selections[0]


def test_the_recipe_is_registered() -> None:
    assert RECIPES["nemotron_h_74b_nvfp4"] is nemotron_h_74b_nvfp4


def test_nvfp4_lands_on_the_experts_and_fp8_on_the_mixer_projections_and_the_head() -> None:
    model = _layer_model()
    for name in ("text/layers/1/moe/experts/0/up", "text/layers/1/moe/experts/1/down",
                 "text/layers/1/moe/shared/up", "text/layers/1/moe/shared/down"):
        selection = _selection(model, name)[1]
        assert selection.format == "nvfp4"
        assert selection.method is nvfp4_mse
    for name in ("text/layers/0/mamba/in_projection", "text/layers/0/mamba/out_projection",
                 "text/layers/7/attention/query", "text/layers/7/attention/output",
                 "text/output_head"):
        selection = _selection(model, name)[1]
        assert selection.format == "fp8_e4m3fn_row_bf16"
        assert selection.method is fp8_row_maxabs
    # The latent projections, the router, the embedding, the norms and the Mamba per-head vectors
    # stay at their direct default, which is BF16 for a value source.
    for name in ("text/layers/1/moe/latent_in", "text/layers/1/moe/latent_out",
                 "text/layers/1/moe/router", "text/token_embedding", "text/final_norm",
                 "text/layers/0/block_norm", "text/layers/0/mamba/a_log"):
        assert _selection(model, name)[1].format == "bf16"


def test_the_recipe_makes_each_bank_one_parent() -> None:
    model = _layer_model()
    recipe, _ = _selection(model, "text/layers/1/moe/experts/0/up")
    groups = [names for names, _ in recipe.explicit_groups]
    assert (
        "text/layers/1/moe/experts/0/up",
        "text/layers/1/moe/experts/1/up",
    ) in groups
    assert (
        "text/layers/1/moe/experts/0/down",
        "text/layers/1/moe/experts/1/down",
    ) in groups


def test_a_w4a16_bank_grants_no_a4_use() -> None:
    # A full-precision source carries no calibrated activation divisor, and the engine refuses an A4
    # use without one, so the banks are A16Only -- the released W4A16 assignment.
    model = _layer_model()
    recipe, _ = _selection(model, "text/layers/1/moe/experts/0/up")
    assert recipe.policies[("text/layers/1/moe/experts/0/up",
                            "text/layers/1/moe/expert_input")] == "A16Only"


def _tokenizer() -> dict:
    """The released tokenizer's pipeline description, without its 131,072-entry vocabulary."""
    return {
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": NEMOTRON_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "model": {"type": "BPE", "ignore_merges": True, "byte_fallback": False},
    }


def test_the_ignore_merges_fast_path_is_cleared_and_nothing_else_is_touched() -> None:
    tokenizer = _tokenizer()
    assert normalize_nemotron_tokenizer(tokenizer) is True
    assert tokenizer["model"]["ignore_merges"] is False
    # Idempotent, so a source that already declares plain BPE is left alone.
    assert normalize_nemotron_tokenizer(tokenizer) is False
    validate_nemotron_tokenizer(tokenizer)


def test_a_pipeline_that_is_not_the_released_one_is_refused() -> None:
    tokenizer = _tokenizer()
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "qwen"
    with pytest.raises(ValueError, match="Nanotron-v2"):
        validate_nemotron_tokenizer(tokenizer)
    # A normalizer would rewrite the text before the split, which this family never does.
    tokenizer = _tokenizer()
    tokenizer["normalizer"] = {"type": "NFC"}
    with pytest.raises(ValueError, match="normalizer"):
        validate_nemotron_tokenizer(tokenizer)
