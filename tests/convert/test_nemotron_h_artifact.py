"""Nemotron-H 74B A9B artifact structure: the parameter set against the checkpoint's own tensors.

`tools/verify/check_nemotron_artifact.py` proves numerical agreement over a sample of the artifact's
objects. This file proves the part a sample cannot: that the artifact's parameter set is *exactly*
the checkpoint's language tensors -- every tensor mapped once, no parameter that no tensor explains
-- each declaring the shape the adapter declares for it, each in the format class the recipe assigns
it, and each MoE layer's expert count read back from the checkpoint's own `experts.<e>.up.weight`
tensors rather than from `config.json`, so a config that lies about a bank is caught.

It reads metadata only: the checkpoint's `model.safetensors.index.json`, the 63 shard headers, and
the artifact's directory. No payload byte is decoded, so it costs about a second and can run
whenever the artifact is present. It is therefore silent about *values*: numerical agreement --
including the depthwise convolution's tap-major payload order and the expert row ranges the banks
pack -- is `check_nemotron_artifact.py`'s job, and this file only pins the declaration the payload
is read through. It also pins this one release's structure (88 blocks, 13,824 routed experts), so a
different Nemotron-H checkpoint needs its expectations updated here.

    NINFER_NEMOTRON_ARTIFACT=out/nemotron_h_74b_nvfp4.ninfer \\
    NINFER_NEMOTRON_SOURCE=/mnt/storage/models/nemo/74B \\
    .venv/bin/python -m pytest tests/convert/test_nemotron_h_artifact.py
"""

from __future__ import annotations

import collections
import dataclasses
from math import prod
import os
import re
from pathlib import Path

import pytest

from tools.artifact.layouts import block_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, binding_parts
from tools.convert.sources.safetensors import SafetensorsSource

ARTIFACT_ENV = "NINFER_NEMOTRON_ARTIFACT"
SOURCE_ENV = "NINFER_NEMOTRON_SOURCE"

# The four format classes the recipe assigns, one per role the recipe names.
NVFP4 = "nvfp4"
FP8 = "fp8_e4m3fn_row_bf16"
BF16 = "bf16"
FP32 = "fp32"

# The adapter's naming rule, written down as data so the artifact can be checked without running the
# converter (`tools/convert/nemotron_h.py`'s `_NemotronBuilder` emits exactly these names). Each
# suffix after `language_model.backbone.layers.<i>.` maps to the artifact's suffix, with the
# transform the adapter declares for the checkpoint's stored shape.
_LAYER_SUFFIX = {
    "norm.weight": ("block_norm", None),
    "mixer.in_proj.weight": ("mamba/in_projection", None),
    "mixer.conv1d.weight": ("mamba/convolution", "tap_major"),
    "mixer.conv1d.bias": ("mamba/convolution_bias", None),
    "mixer.A_log": ("mamba/a_log", None),
    "mixer.D": ("mamba/d_skip", None),
    "mixer.dt_bias": ("mamba/dt_bias", None),
    "mixer.norm.weight": ("mamba/norm", None),
    "mixer.out_proj.weight": ("mamba/out_projection", None),
    "mixer.q_proj.weight": ("attention/query", None),
    "mixer.k_proj.weight": ("attention/key", None),
    "mixer.v_proj.weight": ("attention/value", None),
    "mixer.o_proj.weight": ("attention/output", None),
    "mixer.gate.weight": ("moe/router", None),
    "mixer.gate.e_score_correction_bias": ("moe/correction_bias", None),
    "mixer.fc1_latent_proj.weight": ("moe/latent_in", None),
    "mixer.fc2_latent_proj.weight": ("moe/latent_out", None),
    "mixer.shared_experts.up_proj.weight": ("moe/shared/up", None),
    "mixer.shared_experts.down_proj.weight": ("moe/shared/down", None),
}
_EXPERT_SUFFIX = re.compile(r"^mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")
_LAYER_PREFIX = re.compile(r"^language_model\.backbone\.layers\.(\d+)\.(.+)$")
_TOP_LEVEL = {
    "language_model.backbone.embeddings.weight": ("text/token_embedding", None),
    "language_model.lm_head.weight": ("text/output_head", None),
    "language_model.backbone.norm_f.weight": ("text/final_norm", None),
}
# The tensor names that carry a MoE layer's structure, read from the checkpoint itself.
_EXPERT_KEY = re.compile(
    r"^language_model\.backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.weight$"
)

# The release's shape, asserted rather than derived, so a checkpoint that grew or shrank a block is
# a failure here instead of a silently different set of comparisons.
_LAYERS = 88
_MOE_LAYERS = 40
_EXPERT_TOTAL = 13824
_EXPERT_DISTRIBUTION = {240: 12, 304: 6, 368: 9, 432: 10, 496: 3}
_UNPLACED = {"vision_model": 391, "mlp1": 3}
_PACKING_GROUPS = 80
# The parameter classes, by the roles the recipe names: routed and shared experts are NVFP4, the
# Mamba and attention projections and the head are the engine's FP8 route, the four per-head and
# per-expert vectors the scan and the router read are FP32, everything else stays BF16.
_FORMAT_CLASSES = (
    (NVFP4, ("/moe/experts/", "/moe/shared/")),
    (FP8, ("/attention/", "/mamba/in_projection", "/mamba/out_projection")),
    (FP32, ("/mamba/a_log", "/mamba/d_skip", "/mamba/dt_bias", "/moe/correction_bias")),
    (
        BF16,
        (
            "/block_norm",
            "/mamba/norm",
            "/mamba/convolution",
            "/mamba/convolution_bias",
            "/moe/router",
            "/moe/latent_in",
            "/moe/latent_out",
        ),
    ),
)
# Parameters whose class is not implied by a path marker.
_FORMAT_EXACT = {"text/output_head": FP8, "text/token_embedding": BF16, "text/final_norm": BF16}
_FORMAT_CLASS_COUNTS = {NVFP4: 27728, FP8: 113, BF16: 330, FP32: 160}


def checkpoint_target(key: str) -> tuple[str, str | None] | None:
    """The artifact parameter `key` becomes, with the transform declared for its stored shape."""
    if key in _TOP_LEVEL:
        return _TOP_LEVEL[key]
    match = _LAYER_PREFIX.match(key)
    if match is None:
        return None
    layer, suffix = int(match.group(1)), match.group(2)
    if suffix in _LAYER_SUFFIX:
        name, transform = _LAYER_SUFFIX[suffix]
        return f"text/layers/{layer}/{name}", transform
    expert = _EXPERT_SUFFIX.match(suffix)
    if expert is not None:
        return f"text/layers/{layer}/moe/experts/{expert.group(1)}/{expert.group(2)}", None
    return None


def declared_shape(shape: tuple[int, ...], transform: str | None) -> tuple[int, ...]:
    """The shape the artifact must declare for a checkpoint tensor stored as `shape`."""
    if transform is None:
        return tuple(shape)
    if transform == "tap_major":
        # `[channels, 1, kernel]` with the tap on the outer axis: the singleton drops and the stored
        # payload becomes `j * channels + c`, which is what the convolution Op indexes.
        assert len(shape) == 3 and shape[1] == 1, shape
        return (shape[0], shape[2])
    raise AssertionError(f"unknown shape transform {transform!r}")


def artifact_shape(artifact: Artifact, name: str) -> tuple[int, ...]:
    """The logical shape of one bound parameter, read from its object or its row ranges."""
    parts = binding_parts(artifact.directory.bindings[name], artifact.by_id, name)
    only = parts[0] if len(parts) == 1 else None
    if only is not None:
        obj = artifact.object(only[0])
        if only[1] == 0 and only[2] == prod(obj.shape):
            return tuple(obj.shape)
    shapes = {artifact.object(object_id).shape for object_id, _, _ in parts}
    assert len(shapes) == 1, f"{name}: parts span object shapes {sorted(shapes)}"
    shape = shapes.pop()
    if len(shape) == 1:
        return (sum(end - begin for _, begin, end in parts),)
    assert len(shape) == 2, f"{name}: unexpected object rank {shape}"
    rows = 0
    for _, begin, end in parts:
        assert begin % shape[1] == 0 and end % shape[1] == 0, f"{name}: unaligned row range"
        rows += (end - begin) // shape[1]
    return (rows, shape[1])


def parameter_format(artifact: Artifact, name: str) -> str:
    """The format class one bound parameter is stored in, which every part of it must share."""
    formats = {
        artifact.object(object_id).format
        for object_id, _, _ in binding_parts(artifact.directory.bindings[name], artifact.by_id, name)
    }
    assert len(formats) == 1, f"{name}: spans formats {sorted(formats)}"
    return formats.pop()


def expected_format(name: str) -> str:
    """The class the recipe assigns `name`, with the class it expected on failure."""
    if name in _FORMAT_EXACT:
        return _FORMAT_EXACT[name]
    for kind, markers in _FORMAT_CLASSES:
        if any(marker in name for marker in markers):
            return kind
    raise AssertionError(f"no format class covers {name!r}")


def expert_index(weight_map: dict[str, Path]) -> dict[int, dict[str, set[int]]]:
    """Each MoE layer's expert indices per role, from the checkpoint's own tensor names."""
    found: dict[int, dict[str, set[int]]] = collections.defaultdict(
        lambda: {"up": set(), "down": set()}
    )
    for key in weight_map:
        match = _EXPERT_KEY.match(key)
        if match is not None:
            found[int(match.group(1))][match.group(3)].add(int(match.group(2)))
    return dict(found)


def layer_kinds(weight_map: dict[str, Path]) -> dict[int, str]:
    """Each language layer's mixer family, from the checkpoint's own tensor names."""
    families = (("mixer.experts", "moe"), ("mixer.in_proj", "mamba"), ("mixer.q_proj", "attention"))
    kinds: dict[int, str] = {}
    for key in weight_map:
        match = _LAYER_PREFIX.match(key)
        if match is None:
            continue
        layer, suffix = int(match.group(1)), match.group(2)
        for marker, kind in families:
            if suffix.startswith(marker):
                kinds[layer] = kind
    return kinds


def bank_members(artifact: Artifact) -> dict[str, list[tuple[int, int, str]]]:
    """Every tensor object one or more parameters read, as `object -> [(begin, end, parameter)]`."""
    members: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for name in artifact.directory.bindings:
        for object_id, begin, end in binding_parts(
            artifact.directory.bindings[name], artifact.by_id, name
        ):
            members[object_id].append((begin, end, name))
    return dict(members)


@dataclasses.dataclass(frozen=True, slots=True)
class Cross:
    """One pass over the checkpoint's tensor list and the artifact's directory."""

    artifact: Artifact
    source: SafetensorsSource
    weight_map: dict[str, Path]
    pairs: list[tuple[str, str, str | None]]
    unmappable: list[str]
    missing: list[tuple[str, str]]
    duplicates: list[tuple[str, str, str]]
    unexplained: list[str]
    unplaced: collections.Counter
    members: dict[str, list[tuple[int, int, str]]]

    def bank(self, layer: int, role: str) -> tuple[str, list[tuple[int, int, str]]]:
        """The one object holding MoE layer `layer`'s `role` bank, members sorted by row range."""
        name = f"text/layers/{layer}/moe/experts/0/{role}"
        object_id = self.artifact.directory.bindings[name]["parts"][0]["object"]
        return object_id, sorted(self.members[object_id])


def cross_check(artifact: Artifact, source: SafetensorsSource) -> Cross:
    bindings = set(artifact.directory.bindings)
    pairs, unmappable, missing, duplicates = [], [], [], []
    seen: dict[str, str] = {}
    unplaced: collections.Counter = collections.Counter()
    for key in sorted(source.weight_map):
        if not key.startswith("language_model."):
            unplaced[key.split(".")[0]] += 1
            continue
        target = checkpoint_target(key)
        if target is None:
            unmappable.append(key)
            continue
        name, transform = target
        if name not in bindings:
            missing.append((key, name))
            continue
        if name in seen:
            duplicates.append((name, seen[name], key))
            continue
        seen[name] = key
        pairs.append((name, key, transform))
    return Cross(
        artifact=artifact,
        source=source,
        weight_map=source.weight_map,
        pairs=pairs,
        unmappable=unmappable,
        missing=missing,
        duplicates=duplicates,
        unexplained=sorted(bindings - set(seen)),
        unplaced=unplaced,
        members=bank_members(artifact),
    )


@pytest.fixture(scope="module")
def cross() -> Cross:
    """The converted artifact and the checkpoint it came from, or a clean skip when unset."""
    artifact_path = os.environ.get(ARTIFACT_ENV)
    source_path = os.environ.get(SOURCE_ENV)
    if not artifact_path or not source_path:
        pytest.skip(f"set {ARTIFACT_ENV} and {SOURCE_ENV} to check a converted artifact")
    artifact = Artifact.open(artifact_path)
    source = SafetensorsSource(source_path)
    try:
        yield cross_check(artifact, source)
    finally:
        source.close()
        artifact.close()


def examples(rows: list, limit: int = 5) -> str:
    return "\n".join(f"  {row}" for row in rows[:limit])


def test_parameter_set_is_exactly_the_language_tensors(cross: Cross) -> None:
    """Every language tensor is one parameter, and every parameter is one language tensor."""
    assert not cross.unmappable, (
        f"{len(cross.unmappable)} language tensors have no artifact name:\n"
        f"{examples(cross.unmappable)}"
    )
    assert not cross.missing, (
        f"{len(cross.missing)} tensors map to an absent parameter:\n{examples(cross.missing)}"
    )
    assert not cross.duplicates, (
        f"{len(cross.duplicates)} parameters are claimed twice:\n{examples(cross.duplicates)}"
    )
    assert not cross.unexplained, (
        f"{len(cross.unexplained)} parameters no checkpoint tensor explains:\n"
        f"{examples(cross.unexplained)}"
    )
    assert len(cross.pairs) == len(cross.weight_map) - sum(cross.unplaced.values()), (
        f"{len(cross.pairs)} mapped of {len(cross.weight_map)} source tensors"
    )


def test_unplaced_tensors_are_the_vision_tower_and_projector(cross: Cross) -> None:
    """The tensors this component does not map are the other two components, and nothing else."""
    assert dict(cross.unplaced) == _UNPLACED, (
        f"unplaced components {dict(cross.unplaced)} are not {_UNPLACED}"
    )
    assert sum(cross.unplaced.values()) == 394, f"unplaced total {sum(cross.unplaced.values())}"


def test_declared_shapes_match_the_checkpoint(cross: Cross) -> None:
    """Every parameter declares the checkpoint tensor's shape under the adapter's transform."""
    mismatches = []
    for name, key, transform in cross.pairs:
        stored = cross.source.describe(key).shape
        want = declared_shape(stored, transform)
        got = artifact_shape(cross.artifact, name)
        if want != got:
            mismatches.append(f"{name} from {key}: checkpoint {stored} -> declared {got} != {want}")
    assert not mismatches, f"{len(mismatches)} shape mismatches:\n{examples(mismatches)}"


def test_parameter_formats_are_the_recipe_classes(cross: Cross) -> None:
    """Each parameter sits in the format class its role is assigned, named on failure."""
    counts: collections.Counter = collections.Counter()
    mismatches = []
    for name in sorted(cross.artifact.directory.bindings):
        want = expected_format(name)
        got = parameter_format(cross.artifact, name)
        counts[want] += 1
        if want != got:
            mismatches.append(f"{name}: expected {want}, stored as {got}")
    assert not mismatches, f"{len(mismatches)} format mismatches:\n{examples(mismatches)}"
    assert dict(counts) == _FORMAT_CLASS_COUNTS, f"class counts {dict(counts)}"


def test_expert_counts_come_from_the_checkpoint_per_layer(cross: Cross) -> None:
    """Each MoE layer's bank holds exactly the experts the checkpoint has for that layer."""
    artifact = cross.artifact
    found = expert_index(cross.weight_map)
    assert sorted(found) == sorted(
        layer for layer, kind in layer_kinds(cross.weight_map).items() if kind == "moe"
    )
    assert len(found) == _MOE_LAYERS, f"{len(found)} MoE layers, expected {_MOE_LAYERS}"

    latent, hidden = cross.source.describe(
        f"language_model.backbone.layers.{min(found)}.mixer.fc1_latent_proj.weight"
    ).shape
    problems, counts = [], []
    for layer in sorted(found):
        up, down = found[layer]["up"], found[layer]["down"]
        experts = len(up)
        counts.append(experts)
        source_up = cross.source.describe(
            f"language_model.backbone.layers.{layer}.mixer.experts.0.up_proj.weight"
        ).shape
        width, columns = source_up
        up_id, up_members = cross.bank(layer, "up")
        down_id, down_members = cross.bank(layer, "down")
        observed = {
            "experts": experts,
            "indices": up == down == set(range(experts)),
            "router": artifact_shape(artifact, f"text/layers/{layer}/moe/router"),
            "correction_bias": artifact_shape(
                artifact, f"text/layers/{layer}/moe/correction_bias"
            ),
            "up_bank": artifact.object(up_id).shape,
            "down_bank": artifact.object(down_id).shape,
            "up_members": len(up_members),
            "down_members": len(down_members),
        }
        if not (
            observed["indices"]
            and columns == latent
            and observed["router"] == (experts, hidden)
            and observed["correction_bias"] == (experts,)
            and observed["up_bank"] == (experts * width, latent)
            and observed["down_bank"] == (experts * latent, width)
            and observed["up_members"] == experts
            and observed["down_members"] == experts
        ):
            problems.append(f"layer {layer}: {observed}")
    assert not problems, f"{len(problems)} MoE layers disagree with the checkpoint:\n" + examples(
        problems
    )
    assert collections.Counter(counts) == _EXPERT_DISTRIBUTION, (
        f"per-layer expert counts {dict(collections.Counter(counts))}"
    )
    assert sum(counts) == _EXPERT_TOTAL, f"{sum(counts)} experts of {_EXPERT_TOTAL}"


def test_bank_groups_and_divisors_follow_the_counts(cross: Cross) -> None:
    """One up and one down bank per MoE layer, each a group of experts over one divisor."""
    artifact = cross.artifact
    found = expert_index(cross.weight_map)
    grouped = {
        object_id: sorted(members)
        for object_id, members in cross.members.items()
        if len(members) > 1
    }
    assert len(grouped) == _PACKING_GROUPS, (
        f"{len(grouped)} packing groups, expected {_PACKING_GROUPS}"
    )

    problems, seen = [], set()
    for object_id, members in sorted(grouped.items()):
        obj = artifact.object(object_id)
        experts = len(members)
        cursor = 0
        for begin, end, name in members:
            if begin != cursor or obj.format != NVFP4:
                problems.append(f"{object_id}: {name} starts at {begin}, not {cursor} ({obj.format})")
            cursor = end
        if cursor != prod(obj.shape) or obj.divisors != 1:
            problems.append(f"{object_id}: covers {cursor} of {prod(obj.shape)} over {obj.divisors}")
        geometry = block_scale_geometry(obj.format, obj.shape, obj.divisors)
        if geometry.payload_bytes != obj.bytes:
            problems.append(f"{object_id}: {obj.bytes} bytes, geometry says {geometry.payload_bytes}")
        layer, role = members[0][2].split("/")[2], members[0][2].rsplit("/", 1)[1]
        expected = [f"text/layers/{layer}/moe/experts/{expert}/{role}" for expert in range(experts)]
        if [member[2] for member in members] != expected:
            problems.append(f"{object_id}: members are not experts 0..{experts - 1} in order")
        if found.get(int(layer), {}).get(role) != set(range(experts)):
            problems.append(f"{object_id}: layer {layer} {role} disagrees with the checkpoint")
        seen.add((int(layer), role))
    assert not problems, f"{len(problems)} bank layout problems:\n{examples(problems)}"
    assert seen == {(layer, role) for layer in found for role in ("up", "down")}, (
        f"{len(seen)} (layer, role) banks, expected {2 * len(found)}"
    )

    shared = [
        members
        for members in cross.members.values()
        if len(members) == 1 and "/moe/shared/" in members[0][2]
    ]
    assert len(shared) == 2 * len(found), (
        f"{len(shared)} shared expert objects, expected {2 * len(found)}"
    )


def test_stored_config_agrees_with_the_checkpoint_tensors(cross: Cross) -> None:
    """The artifact's own config repeats what the checkpoint's tensors say, so neither can lie."""
    config = cross.artifact.directory.components["text"]["config"]
    blocks = config["block_configs"]
    found = expert_index(cross.weight_map)
    kinds = layer_kinds(cross.weight_map)
    assert len(blocks) == _LAYERS == len(kinds), f"{len(blocks)} blocks, {len(kinds)} layers"

    problems = []
    for layer, block in enumerate(blocks):
        if block["block_type"] != kinds[layer]:
            problems.append(f"layer {layer}: config {block['block_type']}, tensors {kinds[layer]}")
            continue
        if block["block_type"] != "moe":
            continue
        experts = len(found[layer]["up"])
        width = cross.source.describe(
            f"language_model.backbone.layers.{layer}.mixer.experts.0.up_proj.weight"
        ).shape[0]
        if block["n_routed_experts"] != experts or block["moe_intermediate_size"] != width:
            problems.append(
                f"layer {layer}: config ({block['n_routed_experts']}, "
                f"{block['moe_intermediate_size']}) vs tensors ({experts}, {width})"
            )
    assert not problems, f"{len(problems)} config entries disagree with the checkpoint:\n" + examples(
        problems
    )


def test_resources_are_not_parameters(cross: Cross) -> None:
    """The artifact's resources are the four files, and every tensor object is bound."""
    artifact = cross.artifact
    resources = {obj.id for obj in artifact.objects if isinstance(obj, ResourceObject)}
    tensors = {obj.id for obj in artifact.objects if not isinstance(obj, ResourceObject)}
    referenced = set(cross.members)
    assert len(resources) == 4, f"{len(resources)} resources: {sorted(resources)}"
    assert not resources & referenced, f"bound resources: {sorted(resources & referenced)}"
    assert referenced == tensors, f"{len(tensors - referenced)} tensor objects are unbound"
    assert set(artifact.directory.components["text"]["resources"].values()) == resources
    assert len(artifact.directory.bindings) == 28331, len(artifact.directory.bindings)
