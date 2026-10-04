from __future__ import annotations

import torch

from tools.artifact.codecs.row_group import decode_row_group_words, encode_row_group
from tools.convert.quantization.row_group import quantize_rows

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
