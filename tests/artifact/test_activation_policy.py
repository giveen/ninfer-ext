from __future__ import annotations

from pathlib import Path
import struct

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, ResourceSpec, TensorSpec
from tools.artifact.set_activation_policy import set_activation_policy
from tools.artifact.writer import ArtifactWriter


def _two_use_artifact(path: Path) -> None:
    specs = [
        TensorSpec("w", (2, 2), "bf16", "contiguous_le_v1"),
        ResourceSpec("template", 5),
    ]
    components = {
        "text": {"config": {}, "resources": {"chat_template.jinja": "template"}}
    }
    bindings = {
        "text/layers/0/mlp/gate": {"object": "w"},
        "text/layers/0/attention/query": {"object": "w"},
    }
    uses = [
        {
            "parameter": "text/layers/0/mlp/gate",
            "input": "text/layers/0/ffn_input",
            "activation_policy": "A16Only",
        },
        {
            "parameter": "text/layers/0/attention/query",
            "input": "text/layers/0/attention_input",
            "activation_policy": "A16Only",
        },
    ]
    with ArtifactWriter(
        path, specs, components=components, bindings=bindings, uses=uses
    ) as writer:
        writer.write_object("w", struct.pack("<4H", 0, 0x8000, 0x3F80, 0x7FC1))
        writer.write_object("template", b"hello")


def _policies(path: Path) -> dict[str, str | None]:
    with Artifact(path) as artifact:
        return {use["parameter"]: use.get("activation_policy") for use in artifact.directory.uses}


def test_sets_matching_use_and_keeps_payload(tmp_path):
    source = tmp_path / "in.ninfer"
    _two_use_artifact(source)
    with Artifact(source) as artifact:
        source_id = artifact.artifact_id
        source_w = artifact.read_object("w")
        source_template = artifact.read_object("template")

    target = tmp_path / "out.ninfer"
    changed = set_activation_policy(source, target, ["*/mlp/gate"], "AllowA8")
    assert changed == 1

    assert _policies(source)["text/layers/0/mlp/gate"] == "A16Only"
    with Artifact(target) as artifact:
        assert artifact.artifact_id != source_id
        assert artifact.read_object("w") == source_w
        assert artifact.read_object("template") == source_template
    policies = _policies(target)
    assert policies["text/layers/0/mlp/gate"] == "AllowA8"
    assert policies["text/layers/0/attention/query"] == "A16Only"


def test_rejects_unmatched_pattern(tmp_path):
    source = tmp_path / "in.ninfer"
    _two_use_artifact(source)
    with pytest.raises(ArtifactError, match="no Use matched"):
        set_activation_policy(source, tmp_path / "out.ninfer", ["*/mlp/down"], "AllowA8")
    assert not (tmp_path / "out.ninfer").exists()


def test_rejects_already_set_policy(tmp_path):
    source = tmp_path / "in.ninfer"
    _two_use_artifact(source)
    with pytest.raises(ArtifactError, match="already use"):
        set_activation_policy(source, tmp_path / "out.ninfer", ["*/mlp/gate"], "A16Only")


def test_rejects_existing_or_same_output(tmp_path):
    source = tmp_path / "in.ninfer"
    _two_use_artifact(source)
    existing = tmp_path / "existing.ninfer"
    existing.write_bytes(b"old")
    with pytest.raises(ArtifactError, match="new path"):
        set_activation_policy(source, existing, ["*/mlp/gate"], "AllowA8")
    assert existing.read_bytes() == b"old"
    with pytest.raises(ArtifactError, match="new path"):
        set_activation_policy(source, source, ["*/mlp/gate"], "AllowA8")


def test_rejects_unsupported_policy(tmp_path):
    source = tmp_path / "in.ninfer"
    _two_use_artifact(source)
    with pytest.raises(ArtifactError, match="unsupported activation policy"):
        set_activation_policy(source, tmp_path / "out.ninfer", ["*/mlp/gate"], "AllowA3")
