"""End-to-end CPU check: a ninfer-quantize source store becomes an exl3_mul1 artifact.

It exercises the recipe, `import_encoded`, the EXL3 branch of the tensor writer, and the stored
plane layout, then decodes the artifact's own bytes independently in FP64.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from tools.artifact.codecs import exl3
from tools.artifact.layouts import exl3_geometry
from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import qwen3_8_27b_exl3
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.sources.safetensors import SafetensorsSource


def _store(tmp_path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    n = k = 128
    half_bits = 3
    generator = torch.Generator().manual_seed(9)
    trellis = torch.randint(
        0, 256, (n // 16, k // 16, 16 * half_bits), dtype=torch.uint8, generator=generator
    )
    su = (torch.rand(k, generator=generator) + 0.5).float()
    sv = (torch.rand(n, generator=generator) + 0.5).float()
    directory = tmp_path / "source"
    directory.mkdir()
    save_file(
        {
            "text/layers/0/mlp/gate.trellis": trellis,
            "text/layers/0/mlp/gate.su": su,
            "text/layers/0/mlp/gate.sv": sv,
        },
        str(directory / "exl3.safetensors"),
    )
    return trellis, su, sv, directory / "exl3.safetensors"


def test_quantizer_store_flows_into_an_exl3_artifact(tmp_path) -> None:
    n = k = 128
    half_bits = 3
    trellis, su, sv, store_path = _store(tmp_path)

    model = Model({"text": {"config": {}}})
    model.add(
        Parameter(
            "text/layers/0/mlp/gate",
            (n, k),
            array_source(torch.ones((n, k), dtype=torch.bfloat16), "gate"),
            inputs=("input",),
        )
    )
    model.add(
        Parameter(
            "text/layers/0/mlp/down",
            (n, k),
            array_source(torch.ones((n, k), dtype=torch.bfloat16), "down"),
            inputs=("input",),
        )
    )

    recipe = Recipe(model)
    with SafetensorsSource(store_path) as store:
        qwen3_8_27b_exl3(model, recipe, {"quantized": store})
    output = tmp_path / "model.ninfer"
    convert(model, recipe, output, device="cpu")

    with Artifact(output) as artifact:
        exl3_objects = [
            obj for obj in artifact.objects if getattr(obj, "format", None) == "exl3_mul1"
        ]
        assert len(exl3_objects) == 1
        obj = exl3_objects[0]
        assert obj.shape == (n, k) and obj.bitrate_half_bits == half_bits
        payload = artifact.read_object(obj.id)

    geometry = exl3_geometry("exl3_mul1", (n, k), half_bits)
    stored_trellis = torch.frombuffer(
        bytearray(payload[0 : geometry.trellis_bytes]), dtype=torch.uint8
    ).reshape(n // 16, k // 16, 16 * half_bits)
    stored_su = torch.frombuffer(
        bytearray(
            payload[geometry.input_scale_offset : geometry.input_scale_offset + geometry.input_scale_bytes]
        ),
        dtype=torch.float32,
    )
    stored_sv = torch.frombuffer(
        bytearray(
            payload[
                geometry.output_scale_offset : geometry.output_scale_offset
                + geometry.output_scale_bytes
            ]
        ),
        dtype=torch.float32,
    )
    assert torch.equal(stored_trellis, trellis)
    assert torch.equal(stored_su, su)
    assert torch.equal(stored_sv, sv)
    assert torch.allclose(
        exl3.decode(stored_trellis, stored_su, stored_sv, half_bits),
        exl3.decode(trellis, su, sv, half_bits),
        rtol=0,
        atol=0,
    )
