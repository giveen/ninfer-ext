from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file
import pytest
import torch

from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource, tensor_source


_SHAPES = ((13, 128), (5, 256), (32, 64), (7, 192), (64, 128), (3, 1024))


def _write_source(tmp_path):
    save_file(
        {
            f"w{index}": torch.randn(*shape, dtype=torch.bfloat16)
            for index, shape in enumerate(_SHAPES)
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")


def _convert(tmp_path, name, *, workers):
    model = Model({"text": {"config": {}}})
    with SafetensorsSource(tmp_path) as reader:
        for index, shape in enumerate(_SHAPES):
            model.add(
                Parameter(
                    f"p{index}",
                    shape,
                    tensor_source(reader, f"w{index}", shape),
                )
            )
        recipe = Recipe(model)
        for index in range(len(_SHAPES)):
            recipe.assign(f"p{index}", format="q6_g64_fp16", method="grouped_absmax")
        path = tmp_path / f"{name}.ninfer"
        report = convert(
            model,
            recipe,
            path,
            device="cpu",
            rows_per_chunk=4,
            workers=workers,
        )
    with Artifact(path) as artifact:
        objects = {obj.id: artifact.read_object(obj.id) for obj in artifact.directory.objects}
    return path, report, objects


def test_parallel_cpu_conversion_is_byte_identical(tmp_path) -> None:
    _write_source(tmp_path)
    _, serial_report, serial = _convert(tmp_path, "serial", workers=1)
    _, parallel_report, parallel = _convert(tmp_path, "parallel", workers=2)

    assert serial_report["workers"] == 1
    assert parallel_report["workers"] == 2
    assert serial.keys() == parallel.keys()
    for object_id, payload in serial.items():
        assert parallel[object_id] == payload, object_id


def test_parallel_conversion_rejects_invalid_workers(tmp_path) -> None:
    _write_source(tmp_path)
    model = Model({"text": {"config": {}}})
    with SafetensorsSource(tmp_path) as reader:
        model.add(
            Parameter(
                "p0",
                _SHAPES[0],
                tensor_source(reader, "w0", _SHAPES[0]),
            )
        )
        recipe = Recipe(model)
        recipe.assign("p0", format="q6_g64_fp16", method="grouped_absmax")
        with pytest.raises(ValueError, match="positive integer"):
            convert(model, recipe, tmp_path / "zero.ninfer", device="cpu", workers=0)
        with pytest.raises(ValueError, match="requires --device cpu"):
            convert(model, recipe, tmp_path / "cuda.ninfer", device="cuda", workers=2)
    assert not (tmp_path / "zero.ninfer").exists()
    assert not (tmp_path / "cuda.ninfer").exists()


def test_parallel_conversion_records_workers_in_report(tmp_path) -> None:
    _write_source(tmp_path)
    path, report, _ = _convert(tmp_path, "report", workers=3)
    stored = json.loads(Path(str(path) + ".conversion.json").read_text(encoding="utf-8"))
    assert stored["workers"] == 3
    assert len(report["methods"]) == len(_SHAPES)
