"""Select final resource bytes and derive the tokenizer's public token domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")

# The tokenizer transformation the runtime implements. A checkpoint's tokenizer.json may describe
# an equivalent pipeline with a different description (for example a newer Transformers split regex
# without the combining-mark class); the runtime validates the description exactly, so the resource
# is normalized to this form at conversion time.
QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Values the compiled Vision pixel pipeline in
# src/models/qwen3_5/frontend/frontend.cpp expects from the preprocessor resources. The compiled
# 16/2/2 patch geometry is cross-checked against the model's Vision config by load_resources, since a
# synthetic model may use a smaller geometry for mapping tests.
VISION_RESCALE_FACTOR = 1.0 / 255.0
VISION_VIDEO_FPS = 2.0
VISION_VIDEO_MIN_FRAMES = 4
VISION_VIDEO_MAX_FRAMES = 768


def normalize_tokenizer(tokenizer: dict) -> bool:
    """Rewrite the tokenizer pipeline to the form the runtime implements. Returns True when the
    value changed. Vocabulary and special tokens are untouched."""
    pre = tokenizer.get("pre_tokenizer")
    if not isinstance(pre, dict) or pre.get("type") != "Sequence":
        return False
    parts = pre.get("pretokenizers")
    if not isinstance(parts, list) or len(parts) != 2:
        return False
    split, bytes_level = parts
    if not isinstance(split, dict) or split.get("type") != "Split":
        return False
    if not isinstance(bytes_level, dict) or bytes_level.get("type") != "ByteLevel":
        return False
    changed = False
    if split.setdefault("pattern", {}).get("Regex") != QWEN_SPLIT_PATTERN:
        split["pattern"] = {"Regex": QWEN_SPLIT_PATTERN}
        changed = True
    if split.get("behavior") != "Isolated":
        split["behavior"] = "Isolated"
        changed = True
    if split.get("invert", False) is not False:
        split["invert"] = False
        changed = True
    if bytes_level.get("add_prefix_space", True) is not False:
        bytes_level["add_prefix_space"] = False
        changed = True
    if bytes_level.get("use_regex", True) is not False:
        bytes_level["use_regex"] = False
        changed = True
    return changed


def normalize_tokenizer_config(tokenizer: dict, config: dict, *, family: str = "qwen") -> bool:
    """Rewrite tokenizer_config.json to the contract the runtime enforces: the added-token decoder
    map it cross-checks, the Qwen prefix semantics, and the official pad token. Newer Transformers
    writes added tokens only in tokenizer.json and omits add_bos_token; the runtime requires both.
    Returns True when the value changed."""
    changed = False
    if config.get("add_bos_token", True) is not False:
        config["add_bos_token"] = False
        changed = True
    if config.get("add_prefix_space", True) is not False:
        config["add_prefix_space"] = False
        changed = True
    if family != "gemma" and config.get("pad_token") != "<|endoftext|>":
        # The Qwen pad token is the runtime's; Gemma keeps the pad its checkpoint names, because
        # it has no <|endoftext|> in its vocabulary.
        config["pad_token"] = "<|endoftext|>"
        changed = True
    existing = config.get("added_tokens_decoder")
    if isinstance(existing, dict) and existing:
        return changed
    added = tokenizer.get("added_tokens")
    if not isinstance(added, list) or not added:
        return changed
    decoder: dict[str, dict] = {}
    for item in added:
        if not isinstance(item, dict) or "id" not in item or "content" not in item:
            continue
        decoder[str(item["id"])] = {
            "content": item["content"],
            "single_word": item.get("single_word", False),
            "lstrip": item.get("lstrip", False),
            "rstrip": item.get("rstrip", False),
            "normalized": item.get("normalized", False),
            "special": item.get("special", False),
        }
    if not decoder:
        return changed
    config["added_tokens_decoder"] = decoder
    return True


GEMMA_SPACE_MARKER = "\u2581"


def validate_gemma_tokenizer(tokenizer: dict) -> None:
    """Refuse a Gemma resource set the runtime would refuse.

    Mirrors the Gemma half of validate_pipeline in src/models/qwen3_5/frontend/tokenizer.cpp. The
    pipeline is not rewritten: its semantics are part of the checkpoint, and a rewrite would make
    the artifact tokenize differently from the model that produced it."""

    def fail(message: str) -> None:
        raise ValueError(f"tokenizer resources the runtime cannot load: {message}")

    normalizer = tokenizer.get("normalizer")
    if normalizer is not None:
        pattern = normalizer.get("pattern") if isinstance(normalizer, dict) else None
        if (
            not isinstance(normalizer, dict)
            or normalizer.get("type") != "Replace"
            or not isinstance(pattern, dict)
            or pattern.get("String") != " "
            or normalizer.get("content") != GEMMA_SPACE_MARKER
        ):
            fail("Gemma tokenizer.json normalizer must replace a space with U+2581")
    pre = tokenizer.get("pre_tokenizer")
    if pre is not None:
        pattern = pre.get("pattern") if isinstance(pre, dict) else None
        if (
            not isinstance(pre, dict)
            or pre.get("type") != "Split"
            or not isinstance(pattern, dict)
            or pattern.get("String") != " "
            or pre.get("behavior") != "MergedWithPrevious"
            or pre.get("invert", False) is not False
        ):
            fail("Gemma tokenizer.json pre_tokenizer must be one space Split, MergedWithPrevious")
    decoder = tokenizer.get("decoder")
    if decoder is not None:
        steps = decoder.get("decoders") if isinstance(decoder, dict) else None
        first = steps[0] if isinstance(steps, list) and steps else None
        if (
            not isinstance(decoder, dict)
            or decoder.get("type") != "Sequence"
            or not isinstance(steps, list)
            or len(steps) != 3
            or not isinstance(first, dict)
            or first.get("type") != "Replace"
            or first.get("content") != " "
            or not isinstance(first.get("pattern"), dict)
            or first["pattern"].get("String") != GEMMA_SPACE_MARKER
            or steps[1].get("type") != "ByteFallback"
            or steps[2].get("type") != "Fuse"
        ):
            fail("Gemma tokenizer.json decoder must be Replace, ByteFallback, Fuse")
    post = tokenizer.get("post_processor")
    if isinstance(post, dict) and post.get("type") != "TemplateProcessing":
        fail("Gemma tokenizer.json post_processor must add no tokens")
    if isinstance(post, dict) and post.get("special_tokens"):
        fail("Gemma tokenizer.json post_processor must add no tokens")
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("byte_fallback") is not True:
        fail("Gemma tokenizer.json model.byte_fallback must be true")


def validate_tokenizer_resources(tokenizer: dict, config: dict, generation: dict, *,
                                  family: str = "qwen") -> None:
    """Reject a resource set the runtime tokenizer loader would refuse, before writing the artifact.

    Mirrors the checks in src/models/qwen3_5/frontend/tokenizer.cpp: the pipeline description, the
    model fields, the added-token entries of both resources, and the generation eos token."""

    def fail(message: str) -> None:
        raise ValueError(f"tokenizer resources the runtime cannot load: {message}")

    model = tokenizer.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        fail("tokenizer.json model.vocab is required")
    if not isinstance(model.get("merges"), list):
        fail("tokenizer.json model.merges must be an array")
    if model.get("ignore_merges", False) is not False:
        fail("tokenizer.json model.ignore_merges must be false")
    if model.get("dropout") not in (None, 0):
        fail("tokenizer.json model.dropout must be null or zero")
    for field in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(field) not in (None, ""):
            fail(f"tokenizer.json model.{field} must be empty")
    if family == "gemma":
        pre = None  # checked by validate_gemma_tokenizer, which the pipeline is not rewritten for
    elif (normalizer := tokenizer.get("normalizer")) is not None and normalizer.get("type") != "NFC":
        fail("tokenizer.json normalizer must be NFC")
    pre = tokenizer.get("pre_tokenizer") if family != "gemma" else None
    if pre is not None:
        parts = pre.get("pretokenizers")
        if pre.get("type") != "Sequence" or not isinstance(parts, list) or len(parts) != 2:
            fail("tokenizer.json pre_tokenizer must be a two-step Sequence")
        split, bytes_level = parts
        if (
            split.get("type") != "Split"
            or not isinstance(split.get("pattern"), dict)
            or split["pattern"].get("Regex") != QWEN_SPLIT_PATTERN
            or split.get("behavior") != "Isolated"
            or split.get("invert", False) is not False
        ):
            fail("tokenizer.json pre_tokenizer.Split does not match the runtime pattern")
        if (
            bytes_level.get("type") != "ByteLevel"
            or bytes_level.get("add_prefix_space", True) is not False
            or bytes_level.get("use_regex", True) is not False
        ):
            fail("tokenizer.json pre_tokenizer.ByteLevel flags are unsupported")
    if family != "gemma":
        decoder = tokenizer.get("decoder")
        if decoder is not None and decoder.get("type") != "ByteLevel":
            fail("tokenizer.json decoder must be ByteLevel")
        post = tokenizer.get("post_processor")
        if post is not None and post.get("type") != "ByteLevel":
            fail("tokenizer.json post_processor must be ByteLevel or null")
    added = tokenizer.get("added_tokens")
    if not isinstance(added, list):
        fail("tokenizer.json added_tokens must be an array")
    required = ("id", "content", "single_word", "lstrip", "rstrip", "normalized", "special")
    for item in added:
        if not isinstance(item, dict) or any(field not in item for field in required):
            fail("tokenizer.json added_tokens items need id/content and their boolean flags")
        if not isinstance(item["content"], str) or not item["content"]:
            fail("tokenizer.json added_tokens items need non-empty content")
        if item["single_word"] or item["lstrip"] or item["rstrip"] or item["normalized"]:
            fail("tokenizer.json added_tokens only support single_word/lstrip/rstrip/normalized false")
    decoder_map = config.get("added_tokens_decoder")
    if not isinstance(decoder_map, dict):
        fail("tokenizer_config.json added_tokens_decoder is required")
    for key, item in decoder_map.items():
        if not key.isdigit():
            fail("tokenizer_config.json added_tokens_decoder keys must be token ids")
        if not isinstance(item, dict) or any(field not in item for field in required[1:]):
            fail("tokenizer_config.json added_tokens_decoder items need content and their flags")
        if not isinstance(item["content"], str) or not item["content"]:
            fail("tokenizer_config.json added_tokens_decoder items need non-empty content")
        if (
            item["single_word"]
            or item["lstrip"]
            or item["rstrip"]
            or item["normalized"]
        ):
            fail("tokenizer_config.json added_tokens_decoder only supports false token flags")
    if config.get("add_bos_token", True) is not False:
        fail("tokenizer_config.json add_bos_token must be false")
    if config.get("add_prefix_space", True) is not False:
        fail("tokenizer_config.json add_prefix_space must be false")
    if family != "gemma" and config.get("pad_token") != "<|endoftext|>":
        fail("tokenizer_config.json pad_token must be <|endoftext|>")
    eos = generation.get("eos_token_id")
    if type(eos) is not int and not (isinstance(eos, list) and eos and all(type(v) is int for v in eos)):
        fail("generation_config.json eos_token_id must be an integer or a non-empty array")


def normalize_vision_config(config: dict) -> bool:
    """Drop JSON null members. The runtime reads an absent member as its default, but a present null
    is invalid, so a checkpoint that writes null for "unset" is rewritten to the default form."""
    nulls = [key for key, value in config.items() if value is None]
    for key in nulls:
        del config[key]
    return bool(nulls)


def validate_vision_resources(preprocessor: dict, video: dict) -> None:
    """Reject a vision resource set the runtime's pixel preprocessor would refuse.

    Mirrors validate_pixel_pipeline and processor_options in
    src/models/qwen3_5/frontend/frontend.cpp."""

    def fail(message: str) -> None:
        raise ValueError(f"vision resources the runtime cannot load: {message}")

    for name, config in (
        ("preprocessor_config.json", preprocessor),
        ("video_preprocessor_config.json", video),
    ):
        for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if flag in config and config[flag] is not True:
                fail(f"{name}.{flag} must be true")
        if "resample" in config and config["resample"] != 3:
            fail(f"{name}.resample must be 3 (bicubic)")
        for field in ("image_mean", "image_std"):
            values = config.get(field)
            if not isinstance(values, list) or len(values) != 3 or any(v != 0.5 for v in values):
                fail(f"{name}.{field} must be three 0.5 values")
        if "rescale_factor" in config and config["rescale_factor"] != VISION_RESCALE_FACTOR:
            fail(f"{name}.rescale_factor must be {VISION_RESCALE_FACTOR}")
        size = config.get("size")
        if not isinstance(size, dict):
            fail(f"{name}.size must be an object")
        for edge in ("shortest_edge", "longest_edge"):
            value = size.get(edge)
            if type(value) is not int or value <= 0:
                fail(f"{name}.size.{edge} must be a positive integer")
    if video.get("fps", VISION_VIDEO_FPS) != VISION_VIDEO_FPS:
        fail("video_preprocessor_config.json.fps must be 2")
    if video.get("min_frames", VISION_VIDEO_MIN_FRAMES) != VISION_VIDEO_MIN_FRAMES:
        fail("video_preprocessor_config.json.min_frames must be 4")
    if video.get("max_frames", VISION_VIDEO_MAX_FRAMES) != VISION_VIDEO_MAX_FRAMES:
        fail("video_preprocessor_config.json.max_frames must be 768")


def token_domain(
    tokenizer: dict, config: dict, vocab_size: int
) -> tuple[int, tuple[int, ...]]:
    ids: dict[int, str] = {}
    special: dict[int, bool] = {}

    def add(index: object, content: object) -> None:
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"invalid tokenizer entry id={index!r}, content={content!r}"
            )
        if index in ids and ids[index] != content:
            raise ValueError(f"tokenizer resources disagree about token {index}")
        ids[index] = content

    def add_special(index: int, token: dict) -> None:
        flag = token.get("special", False)
        if type(flag) is not bool:
            raise ValueError(f"token {index}: special must be boolean")
        if index in special and special[index] != flag:
            raise ValueError(
                f"tokenizer resources disagree about token {index} special flag"
            )
        special[index] = flag

    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer model must provide a vocabulary mapping")
    for token, index in vocabulary.items():
        add(index, token)
    for token in tokenizer.get("added_tokens", []):
        add(token["id"], token["content"])
        add_special(token["id"], token)
    for raw_id, token in config.get("added_tokens_decoder", {}).items():
        index = int(raw_id)
        add(index, token["content"])
        add_special(index, token)
    if not ids or set(ids) != set(range(max(ids) + 1)):
        raise ValueError("this tokenizer requires a contiguous public token ID domain")
    return len(ids), tuple(sorted(index for index, flag in special.items() if flag))


def load_resources(
    model_dir: Path,
    *,
    vocab_size: int,
    vision_config: Mapping[str, int] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
    family: str = "qwen",
) -> tuple[dict[str, dict[str, str]], dict[str, bytes], int, tuple[int, ...]]:
    overrides = {} if overrides is None else dict(overrides)
    roles = {"text": TEXT_RESOURCES}
    if vision_config is not None:
        roles["vision"] = VISION_RESOURCES
    allowed = {role for names in roles.values() for role in names}
    if overrides.keys() - allowed:
        raise ValueError(
            f"resource overrides have no selected consumer: {sorted(overrides.keys()-allowed)}"
        )
    references: dict[str, dict[str, str]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, dict] = {}
    for component, names in roles.items():
        references[component] = {}
        for role in names:
            path = Path(overrides[role]) if role in overrides else model_dir / role
            data = path.read_bytes()
            if not data:
                raise ValueError(f"{path}: resource is empty")
            text = data.decode("utf-8")
            if role.endswith(".json"):
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: resource must contain a JSON object")
                parsed[role] = value
                if component == "vision":
                    for field, config_field in (
                        ("patch_size", "patch_size"),
                        ("temporal_patch_size", "temporal_patch_size"),
                        ("merge_size", "spatial_merge_size"),
                    ):
                        if (
                            type(value.get(field)) is not int
                            or value[field] != vision_config[config_field]
                        ):
                            raise ValueError(
                                f"{path}: {field} differs from Vision config"
                            )
            object_id = f"resource/{component}/{role}"
            references[component][role] = object_id
            payloads[object_id] = data
    if family == "gemma":
        validate_gemma_tokenizer(parsed["tokenizer.json"])
    elif normalize_tokenizer(parsed["tokenizer.json"]):
        payloads["resource/text/tokenizer.json"] = json.dumps(
            parsed["tokenizer.json"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    if normalize_tokenizer_config(
        parsed["tokenizer.json"], parsed["tokenizer_config.json"], family=family
    ):
        payloads["resource/text/tokenizer_config.json"] = json.dumps(
            parsed["tokenizer_config.json"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    if "vision" in roles:
        for role in VISION_RESOURCES:
            if normalize_vision_config(parsed[role]):
                payloads[f"resource/vision/{role}"] = json.dumps(
                    parsed[role], ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
        validate_vision_resources(
            parsed["preprocessor_config.json"], parsed["video_preprocessor_config.json"]
        )
    validate_tokenizer_resources(
        parsed["tokenizer.json"],
        parsed["tokenizer_config.json"],
        parsed.get("generation_config.json", {}),
        family=family,
    )
    count, special = token_domain(
        parsed["tokenizer.json"], parsed["tokenizer_config.json"], vocab_size
    )
    return references, payloads, count, special
