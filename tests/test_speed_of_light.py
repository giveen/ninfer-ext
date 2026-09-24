"""Accounting checks for the offline inference roof estimate."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.bench.speed_of_light import (
    _model_work,
    _projection_geometry,
    estimate,
)


def fake_artifact(
    config: dict, byte_multipliers: dict[str, int] | None = None
):
    byte_multipliers = byte_multipliers or {}
    projections, expert_groups = _projection_geometry(config)
    shapes = dict(projections)
    shapes["text/output_head"] = config["hidden_size"] * config["vocab_size"]
    for experts in expert_groups:
        for expert in experts:
            for name in expert:
                shapes[name] = shapes[
                    name.replace("/experts/1/", "/experts/0/")
                ]
    objects = {}
    for name, count in shapes.items():
        objects[name] = TensorObject(
            name,
            (count,),
            "bf16",
            "row_major",
            0,
            count * byte_multipliers.get(name, 2),
        )
    return SimpleNamespace(
        directory=SimpleNamespace(
            components={"text": {"config": config}},
            bindings={name: {"object": name} for name in shapes},
        ),
        by_id=objects,
        artifact_id=b"\x01" * 16,
    )


class SpeedOfLightTests(unittest.TestCase):
    def test_real_artifact_directory_with_split_binding(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 2,
            "vocab_size": 4,
            "layer_types": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("head", (8,), "bf16", "contiguous_le_v1")],
                components={"text": {"config": config}},
                bindings={
                    "text/output_head": {
                        "parts": [
                            {"object": "head", "range": [0, 4]},
                            {"object": "head", "range": [4, 8]},
                        ]
                    }
                },
            ) as writer:
                writer.write_object("head", bytes(16))
            with Artifact(path) as artifact:
                body, head, weight_bytes = _model_work(artifact)
        assert (body, head, weight_bytes) == (0, 16, 16)

    def test_prefill_head_runs_once_and_decode_head_runs_per_token(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 6,
        }
        artifact = fake_artifact(config)
        report = {
            "artifact_type": "ninfer_bench_report",
            "schema_version": 15,
            "load": {"architecture": "Qwen3_5ForCausalLM", "name": "test"},
            "config": {"speculative_backend": "none"},
            "environment": {"gpu_name": "test gpu"},
            "memory": {"kv_capacity": 8, "kv_payload_bytes": 64},
            "tests": [
                {
                    "label": "pp2",
                    "kind": "pp",
                    "n_prompt": 2,
                    "n_gen": 0,
                    "prefill_seconds_mean": 0.5,
                    "decode_seconds_mean": 0,
                },
                {
                    "label": "tg3",
                    "kind": "tg",
                    "n_prompt": 0,
                    "n_gen": 3,
                    "prefill_seconds_mean": 0.1,
                    "decode_seconds_mean": 1,
                },
            ],
        }
        result = estimate(report, artifact, 1, 1)
        assert result["body_projection_flops_per_token"] == 272
        assert result["output_head_flops_per_call"] == 64
        assert (
            result["tests"][0]["phases"]["prefill"]["projection_flops"] == 608
        )
        assert (
            result["tests"][1]["phases"]["decode"]["projection_flops"] == 1008
        )
        assert (
            result["tests"][1]["phases"]["decode"][
                "estimated_weight_bytes_per_token"
            ]
            == 336
        )
        assert (
            result["tests"][1]["phases"]["decode"]["estimated_kv_read_bytes"]
            == 72
        )
        assert "decode" not in result["tests"][0]["phases"]
        assert "prefill" not in result["tests"][1]["phases"]

    def test_moe_traffic_uses_smallest_selectable_experts(self):
        config = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 2,
        }
        expensive = {
            f"text/layers/0/moe/experts/1/{role}": 8
            for role in ("gate", "up", "down")
        }
        artifact = fake_artifact(config, expensive)
        _, _, selected_bytes = _model_work(artifact)
        assert selected_bytes == 304
        assert selected_bytes == _model_work(fake_artifact(config))[2]

    def test_speculative_report_is_rejected(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": [],
        }
        artifact = fake_artifact(config)
        report = {
            "artifact_type": "ninfer_bench_report",
            "schema_version": 15,
            "load": {"architecture": "Qwen3_5ForCausalLM"},
            "config": {"speculative_backend": "mtp"},
        }
        with self.assertRaisesRegex(ValueError, "--spec none"):  # noqa: PT027
            estimate(report, artifact, 1, None)


if __name__ == "__main__":
    unittest.main()
