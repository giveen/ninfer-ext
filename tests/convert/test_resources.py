from __future__ import annotations

import pathlib
import re

import pytest

from tools.convert.resources import (
    QWEN_SPLIT_PATTERN,
    normalize_tokenizer,
    normalize_tokenizer_config,
    normalize_vision_config,
    validate_tokenizer_resources,
    validate_vision_resources,
)


def _pipeline(split_regex: str, *, add_prefix_space: bool = True, use_regex: bool = True) -> dict:
    return {
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": split_regex},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": add_prefix_space,
                    "use_regex": use_regex,
                },
            ],
        },
    }


def _added_token(index: int, content: str, *, special: bool = False) -> dict:
    return {
        "id": index,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": special,
    }


def _valid_resources() -> tuple[dict, dict, dict]:
    tokenizer = {
        "model": {"vocab": {"a": 0, "b": 1}, "merges": []},
        "added_tokens": [_added_token(2, "<|endoftext|>", special=True)],
    }
    tokenizer.update(_pipeline(QWEN_SPLIT_PATTERN, add_prefix_space=False, use_regex=False))
    config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "pad_token": "<|endoftext|>",
        "added_tokens_decoder": {"2": {
            "content": "<|endoftext|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }},
    }
    generation = {"eos_token_id": 2}
    return tokenizer, config, generation


def test_normalize_rewrites_an_unsupported_split() -> None:
    # swift-next ships a newer Transformers split regex without the combining-mark class; the
    # runtime validates the description exactly, so it is rewritten.
    tokenizer = _pipeline(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+")
    assert normalize_tokenizer(tokenizer) is True
    parts = tokenizer["pre_tokenizer"]["pretokenizers"]
    assert parts[0]["pattern"]["Regex"] == QWEN_SPLIT_PATTERN
    assert parts[0]["behavior"] == "Isolated"
    assert parts[0]["invert"] is False
    assert parts[1]["add_prefix_space"] is False
    assert parts[1]["use_regex"] is False


def test_normalize_leaves_a_supported_pipeline_alone() -> None:
    tokenizer = _pipeline(QWEN_SPLIT_PATTERN, add_prefix_space=False, use_regex=False)
    assert normalize_tokenizer(tokenizer) is False


def test_normalize_ignores_a_pipeline_it_does_not_understand() -> None:
    assert normalize_tokenizer({}) is False
    assert normalize_tokenizer({"pre_tokenizer": {"type": "Metaspace"}}) is False


def test_normalize_config_synthesizes_the_decoder_map_and_prefix_semantics() -> None:
    # Newer Transformers writes added tokens only in tokenizer.json and omits add_bos_token; the
    # runtime requires the decoder map, the prefix flags, and the official pad token.
    tokenizer = {"added_tokens": [_added_token(2, "<|endoftext|>", special=True)]}
    config: dict = {}
    assert normalize_tokenizer_config(tokenizer, config) is True
    entry = config["added_tokens_decoder"]["2"]
    assert entry["content"] == "<|endoftext|>"
    assert entry["special"] is True
    assert entry["single_word"] is False
    assert config["add_bos_token"] is False
    assert config["add_prefix_space"] is False
    assert config["pad_token"] == "<|endoftext|>"
    # A config that already matches is untouched.
    assert normalize_tokenizer_config(tokenizer, dict(config)) is False


def test_validate_accepts_a_supported_resource_set() -> None:
    validate_tokenizer_resources(*_valid_resources())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda t, c, g: t["model"].pop("vocab"), id="missing-vocab"),
        pytest.param(lambda t, c, g: t["model"].update({"ignore_merges": True}), id="ignore-merges"),
        pytest.param(lambda t, c, g: t.update(_pipeline(r"\p{L}+")), id="split-pattern"),
        pytest.param(lambda t, c, g: t.update({"normalizer": {"type": "NFD"}}), id="normalizer"),
        pytest.param(lambda t, c, g: c.pop("added_tokens_decoder"), id="missing-decoder"),
        pytest.param(lambda t, c, g: c.update({"add_bos_token": True}), id="add-bos-token"),
        pytest.param(lambda t, c, g: c.pop("add_bos_token"), id="missing-add-bos-token"),
        pytest.param(lambda t, c, g: c.update({"pad_token": "<|pad|>"}), id="wrong-pad-token"),
        pytest.param(
            lambda t, c, g: c["added_tokens_decoder"]["2"].update({"lstrip": True}),
            id="unsupported-flag",
        ),
        pytest.param(lambda t, c, g: g.pop("eos_token_id"), id="missing-eos"),
    ],
)
def test_validate_rejects_an_unsupported_resource_set(mutate) -> None:
    tokenizer, config, generation = _valid_resources()
    mutate(tokenizer, config, generation)
    with pytest.raises(ValueError):
        validate_tokenizer_resources(tokenizer, config, generation)


def _valid_vision() -> tuple[dict, dict]:
    config = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    }
    return dict(config), dict(config)


def test_normalize_vision_drops_null_members() -> None:
    # A present JSON null is invalid for the runtime, which reads an absent member as its default.
    config = {"rescale_factor": None, "patch_size": 16}
    assert normalize_vision_config(config) is True
    assert config == {"patch_size": 16}
    assert normalize_vision_config(config) is False


def test_validate_vision_accepts_a_supported_resource_set() -> None:
    validate_vision_resources(*_valid_vision())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda i, v: i.update({"do_resize": False}), id="disabled-pipeline"),
        pytest.param(lambda i, v: i.update({"resample": 2}), id="non-bicubic-resample"),
        pytest.param(lambda i, v: i.update({"image_mean": [0.5, 0.5]}), id="bad-mean"),
        pytest.param(lambda i, v: i.update({"rescale_factor": 1.0}), id="bad-rescale"),
        pytest.param(lambda i, v: i.pop("image_std"), id="missing-std"),
        pytest.param(lambda i, v: i.pop("size"), id="missing-size"),
        pytest.param(lambda i, v: i.update({"size": {"shortest_edge": 0}}), id="bad-edge"),
        pytest.param(lambda i, v: v.update({"fps": 1.0}), id="bad-fps"),
        pytest.param(lambda i, v: v.update({"min_frames": 1}), id="bad-min-frames"),
        pytest.param(lambda i, v: v.update({"max_frames": 1}), id="bad-max-frames"),
    ],
)
def test_validate_vision_rejects_an_unsupported_resource_set(mutate) -> None:
    preprocessor, video = _valid_vision()
    mutate(preprocessor, video)
    with pytest.raises(ValueError):
        validate_vision_resources(preprocessor, video)


def test_converter_pattern_matches_the_runtime() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src/models/qwen3_5/frontend/tokenizer.cpp"
    ).read_text()
    match = re.search(r'kQwenSplitPattern\s*=\s*\n\s*R"qwen\((.*?)\)qwen"', source, re.S)
    assert match is not None, "the runtime split pattern was not found"
    assert match.group(1) == QWEN_SPLIT_PATTERN
