from __future__ import annotations

import pathlib
import re

from tools.convert.resources import QWEN_SPLIT_PATTERN, normalize_tokenizer


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


def test_converter_pattern_matches_the_runtime() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src/models/qwen3_5/frontend/tokenizer.cpp"
    ).read_text()
    match = re.search(r'kQwenSplitPattern\s*=\s*\n\s*R"qwen\((.*?)\)qwen"', source, re.S)
    assert match is not None, "the runtime split pattern was not found"
    assert match.group(1) == QWEN_SPLIT_PATTERN
