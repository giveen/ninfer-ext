"""Gemma 4 text config: the profile split, and the MoE block the 26B-A4B adds.

The artifact's own config is what the loader parses, so what this adapter emits is the contract that
src/models/gemma4/config.cpp enforces (its tests/models/gemma4/test_config.cpp read the same
fixtures). Both profiles are pinned against those fixtures, which were generated from the real
checkpoints: `google/gemma-4-31B-it` and the 26B-A4B source on this workstation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.convert.gemma4 import text_config

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gemma4"

# The two checkpoint families the adapter accepts, by layer count. Everything else about them is the
# same and is checked against the source below.
_PROFILES = {
    60: {
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
    },
    30: {
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "enable_moe_block": True,
        "num_experts": 128,
        "top_k_experts": 8,
        "moe_intermediate_size": 704,
    },
}


def _config(layers: int = 60) -> dict:
    """A source config in the checkpoint's own shape for the profile with `layers` layers."""
    rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    }
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            **_PROFILES[layers],
            "num_hidden_layers": layers,
            "vocab_size": 262_144,
            "max_position_embeddings": 262_144,
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": 1024,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "layer_types": [
                "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
                for index in range(layers)
            ],
            "rope_parameters": rope,
        },
    }


def _committed(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())["config"]


@pytest.mark.parametrize(
    ("layers", "fixture"),
    [(60, "text_config.json"), (30, "text_config_26b.json")],
)
def test_emitted_config_matches_the_committed_fixture(layers, fixture):
    """The loader's authority is the emitted config, so both profiles are pinned to it."""
    assert text_config(_config(layers)) == _committed(fixture)


def test_only_the_moe_profile_carries_the_moe_block():
    dense = text_config(_config(60))
    moe = text_config(_config(30))
    assert "enable_moe_block" not in dense and "num_experts" not in dense
    assert moe["enable_moe_block"] is True
    assert moe["num_experts"] == 128 and moe["top_k_experts"] == 8
    assert moe["moe_intermediate_size"] == 704
    # The dense MLP is kept in parallel with the routed experts, not replaced by them.
    assert moe["intermediate_size"] == 2112


def test_the_profile_is_selected_by_the_layer_count():
    # A 30-layer source wearing the 31B widths is refused by the profile it lands in.
    wrong = _config(30)
    wrong["text_config"]["hidden_size"] = 5376
    with pytest.raises(ValueError, match="Gemma 4 26B-A4B expects text.hidden_size"):
        text_config(wrong)
    with pytest.raises(ValueError, match="unsupported Gemma 4 text layer count"):
        text_config({**_config(60), "text_config": {**_config(60)["text_config"],
                                                    "num_hidden_layers": 48}})


def test_a_misdeclared_moe_block_is_refused():
    # The flag is a property of the family: a 26B source must declare it and a 31B source must not.
    for layers, value in ((30, False), (60, True)):
        wrong = _config(layers)
        wrong["text_config"]["enable_moe_block"] = value
        with pytest.raises(ValueError, match="enable_moe_block"):
            text_config(wrong)
    for key, value in (("num_experts", 64), ("top_k_experts", 4), ("moe_intermediate_size", 1408)):
        wrong = _config(30)
        wrong["text_config"][key] = value
        with pytest.raises(ValueError, match=f"expects text.{key}"):
            text_config(wrong)


def test_fixed_mathematics_is_checked_on_both_profiles():
    for layers in (60, 30):
        for key, invalid in (
            ("attention_bias", True),
            ("attention_k_eq_v", False),
            ("head_dim", 512),
            ("sliding_window", 2048),
            ("final_logit_softcapping", 0.0),
        ):
            changed = deepcopy(_config(layers))
            changed["text_config"][key] = invalid
            with pytest.raises(ValueError, match=key):
                text_config(changed)
        changed = deepcopy(_config(layers))
        changed["text_config"]["layer_types"][5] = "sliding_attention"
        with pytest.raises(ValueError, match="every sixth layer"):
            text_config(changed)
