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


def test_grouped_parent_store_reassembles_one_shared_matrix(tmp_path) -> None:
    """The model executes gate/up as one parent, so the store holds one [256, 128] matrix keyed by
    gate and the recipe binds each member to its row range. The artifact must hold that parent
    exactly, not a row-reordered or per-member copy."""
    parent_rows = 256
    k = 128
    half_bits = 3
    generator = torch.Generator().manual_seed(11)
    trellis = torch.randint(
        0,
        256,
        (parent_rows // 16, k // 16, 16 * half_bits),
        dtype=torch.uint8,
        generator=generator,
    )
    su = (torch.rand(k, generator=generator) + 0.5).float()
    sv = (torch.rand(parent_rows, generator=generator) + 0.5).float()
    directory = tmp_path / "grouped"
    directory.mkdir()
    save_file(
        {
            "text/layers/0/mlp/gate.trellis": trellis,
            "text/layers/0/mlp/gate.su": su,
            "text/layers/0/mlp/gate.sv": sv,
        },
        str(directory / "exl3.safetensors"),
    )

    model = Model({"text": {"config": {}}})
    for name in ("text/layers/0/mlp/gate", "text/layers/0/mlp/up"):
        model.add(
            Parameter(
                name,
                (k, k),
                array_source(torch.ones((k, k), dtype=torch.bfloat16), name),
                inputs=("input",),
            )
        )
    model.packing_groups.append(("text/layers/0/mlp/gate", "text/layers/0/mlp/up"))

    recipe = Recipe(model)
    with SafetensorsSource(directory / "exl3.safetensors") as store:
        qwen3_8_27b_exl3(model, recipe, {"quantized": store})
    output = tmp_path / "grouped.ninfer"
    convert(model, recipe, output, device="cpu")

    with Artifact(output) as artifact:
        exl3_objects = [
            obj for obj in artifact.objects if getattr(obj, "format", None) == "exl3_mul1"
        ]
        assert len(exl3_objects) == 1
        obj = exl3_objects[0]
        assert obj.shape == (parent_rows, k) and obj.bitrate_half_bits == half_bits
        payload = artifact.read_object(obj.id)

    geometry = exl3_geometry("exl3_mul1", (parent_rows, k), half_bits)
    stored_trellis = torch.frombuffer(
        bytearray(payload[0 : geometry.trellis_bytes]), dtype=torch.uint8
    ).reshape(parent_rows // 16, k // 16, 16 * half_bits)
    stored_su = torch.frombuffer(
        bytearray(
            payload[
                geometry.input_scale_offset : geometry.input_scale_offset
                + geometry.input_scale_bytes
            ]
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
    assert torch.equal(
        exl3.decode(stored_trellis, stored_su, stored_sv, half_bits),
        exl3.decode(trellis, su, sv, half_bits),
    )


def test_flash_next_experts_become_banks_with_one_scale_set_per_expert(tmp_path) -> None:
    """Separately quantised experts keep their own input scales inside one bank object."""

    from tools.convert.official_recipes import qwen3_8_flash_next_exl3

    experts, rows, k, half_bits = 3, 128, 128, 3
    generator = torch.Generator().manual_seed(21)
    tensors, expected = {}, {}
    names = []
    for expert in range(experts):
        for role in ("gate", "up", "down"):
            name = f"text/layers/0/moe/experts/{expert}/{role}"
            trellis = torch.randint(
                0, 256, (rows // 16, k // 16, 16 * half_bits), dtype=torch.uint8, generator=generator
            )
            su = (torch.rand(k, generator=generator) + 0.5).float()
            sv = (torch.rand(rows, generator=generator) + 0.5).float()
            tensors |= {name + ".trellis": trellis, name + ".su": su, name + ".sv": sv}
            expected[name] = exl3.decode(trellis, su, sv, half_bits)
            names.append(name)
    directory = tmp_path / "store"
    directory.mkdir()
    save_file(tensors, str(directory / "exl3.safetensors"))

    model = Model({"text": {"config": {"model_type": "qwen4_exp_text"}}})

    def add(name, shape, inputs):
        model.add(Parameter(name, shape, array_source(torch.ones(shape, dtype=torch.bfloat16), name), inputs=inputs))

    add("text/token_embedding", (128, 128), ())
    add("text/output_head", (128, 128), ())
    for name in names:
        add(name, (rows, k), ("x",))
    model.packing_groups.append(
        tuple(n for n in names if not n.endswith("/down"))  # gate0, up0, gate1, up1, ...
    )
    model.packing_groups.append(tuple(n for n in names if n.endswith("/down")))

    recipe = Recipe(model)
    with SafetensorsSource(directory / "exl3.safetensors") as store:
        qwen3_8_flash_next_exl3(model, recipe, {"quantized": store})
    output = tmp_path / "bank.ninfer"
    convert(model, recipe, output, device="cpu")

    with Artifact(output) as artifact:
        banks = [o for o in artifact.objects if getattr(o, "format", None) == "exl3_mul1"]
        assert sorted((o.shape, o.divisors) for o in banks) == [
            ((experts * rows, k), experts),
            ((2 * experts * rows, k), 2 * experts),
        ]
        for bank in banks:
            sets = bank.divisors
            g = exl3_geometry("exl3_mul1", bank.shape, half_bits, sets)
            payload = artifact.read_object(bank.id)
            assert len(payload) == g.payload_bytes
            trellis = torch.frombuffer(bytearray(payload[: g.trellis_bytes]), dtype=torch.uint8).reshape(
                bank.shape[0] // 16, k // 16, 16 * half_bits
            )
            su = torch.frombuffer(
                bytearray(payload[g.input_scale_offset : g.input_scale_offset + g.input_scale_bytes]),
                dtype=torch.float32,
            ).reshape(sets, k)
            sv = torch.frombuffer(
                bytearray(payload[g.output_scale_offset : g.output_scale_offset + g.output_scale_bytes]),
                dtype=torch.float32,
            )
            decoded = exl3.decode(trellis, su, sv, half_bits)
            members = [n for n in names if (n.endswith("/down")) == (bank.divisors == experts)]
            for index, name in enumerate(members):
                assert torch.equal(decoded[index * rows : (index + 1) * rows], expected[name]), name
