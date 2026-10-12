from __future__ import annotations

import torch

from tools.artifact.codecs.row_group import decode_row_group_words, encode_row_group
from tools.convert.quantization.row_group import quantize_rows, quantize_rows_on_grid

FORMAT = "q4_g32_fp16_rows"


def test_quantization_zero_groups_ties_and_reconstruction_bound() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(64, 160, generator=generator)
    source[3] = 0.0  # an all-zero row has zero scales and zero codes
    source[5, 32:64] = 0.0  # one zero group inside a live row
    quantized = quantize_rows(source, FORMAT, device="cpu")

    assert quantized.codes.dtype == torch.int8 and quantized.scales.dtype == torch.float16
    assert quantized.codes.abs().max().item() <= 7
    assert quantized.scales[3].tolist() == [0.0] * 5
    assert quantized.codes[3].abs().sum().item() == 0
    assert quantized.codes[5, 32:64].abs().sum().item() == 0

    # The largest value of every live group lands on a full-scale code.
    grouped = source.reshape(64, 5, 32)
    top = grouped.abs().argmax(dim=2)
    live = quantized.scales.float() > 0
    peak = torch.gather(quantized.codes.reshape(64, 5, 32), 2, top.unsqueeze(2)).squeeze(2).abs()
    assert bool((peak[live] == 7).all())

    # Reconstruction error is at most half a step of the group's own scale.
    payload = encode_row_group(quantized.codes, quantized.scales, FORMAT, source.shape)
    codes, scales = decode_row_group_words(payload, FORMAT, source.shape)
    assert torch.equal(codes, quantized.codes)
    restored = (codes.float().reshape(64, 5, 32) * scales.float().unsqueeze(2)).reshape(64, 160)
    bound = (scales.float() * 0.5 + 1e-6).unsqueeze(2).expand(64, 5, 32).reshape(64, 160)
    assert bool(((restored - source).abs() <= bound * 1.001).all())


def test_streamed_row_group_conversion_round_trips_through_the_artifact(tmp_path) -> None:
    import json

    from safetensors.torch import save_file

    from tools.artifact.codecs.row_group import decode_row_group_words
    from tools.artifact.reader import Artifact
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource, tensor_source

    name = "model.language_model.table.weight"
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(70, 160, generator=generator).to(torch.bfloat16)
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text("{}")
    model = Model({"text": {"config": {}}})
    with SafetensorsSource(tmp_path) as reader:
        model.add(Parameter("table", tuple(source.shape), tensor_source(reader, name, tuple(source.shape))))
        recipe = Recipe(model)
        recipe.assign("table", format=FORMAT, method="q4_rows_maxabs")
        path = tmp_path / "table.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=8)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["table"]["object"]
        payload = artifact.read_object(object_id)
    codes, scales = decode_row_group_words(payload, FORMAT, source.shape)
    expected = quantize_rows(source.float(), FORMAT, device="cpu")
    assert torch.equal(codes, expected.codes)
    assert torch.equal(scales.view(torch.int16), expected.scales.view(torch.int16))


def test_grid_recovery_reproduces_a_learned_int4_grid_exactly() -> None:
    # A QAT source: per group a learned step d and integer codes in [-8, 7] whose extremes need not
    # reach -8 or 7, stored as BF16(d * k) -- the case max|x| / 7 gets wrong.
    generator = torch.Generator().manual_seed(11)
    n, k = 32, 256
    steps = torch.exp(torch.empty(n, k // 32, 1).uniform_(-7.0, -4.0, generator=generator))
    reach = torch.randint(3, 9, (n, k // 32, 1), generator=generator)
    codes = torch.randint(-8, 8, (n, k // 32, 32), generator=generator)
    codes = torch.maximum(torch.minimum(codes, reach - 1), -reach)
    source = (codes.float() * steps).reshape(n, k).to(torch.bfloat16)
    source[2] = 0.0
    # One group on no 4-bit grid at all: the fallback still bounds its error.
    source[4, :32] = torch.linspace(-1.0, 1.0, 32).to(torch.bfloat16) * 1e-2

    quantized = quantize_rows_on_grid(source, FORMAT, device="cpu")
    assert quantized.codes.dtype == torch.int8 and quantized.scales.dtype == torch.float16
    assert int(quantized.codes.min()) >= -8 and int(quantized.codes.max()) <= 7
    assert quantized.scales[2].tolist() == [0.0] * (k // 32)
    restored = (
        quantized.codes.float().reshape(n, k // 32, 32) * quantized.scales.float().unsqueeze(2)
    ).reshape(n, k)
    on_grid = torch.ones(n, k, dtype=torch.bool)
    on_grid[4, :32] = False
    # Every on-grid value comes back within one BF16 ulp: the step is exact, and only its binary16
    # rounding can move the product across a BF16 rounding boundary.
    words = lambda t: t.to(torch.bfloat16).view(torch.int16).int()  # noqa: E731
    ulps = (words(restored) - words(source.float())).abs()[on_grid]
    assert int(ulps.max()) <= 1

    # The maxabs method, by contrast, lands far off the grid.
    plain = quantize_rows(source.float(), FORMAT, device="cpu")
    plain_restored = (
        plain.codes.float().reshape(n, k // 32, 32) * plain.scales.float().unsqueeze(2)
    ).reshape(n, k)
    assert int((words(plain_restored) - words(source.float())).abs()[on_grid].max()) > 1
    # The group on no grid takes the least-squares step, no worse than max|x| / 7.
    target = source[4, :32].float()
    assert float(((restored[4, :32] - target) ** 2).sum()) <= float(
        ((plain_restored[4, :32] - target) ** 2).sum()
    )
