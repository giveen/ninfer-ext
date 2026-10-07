from __future__ import annotations

import struct

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs import exl3
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.logical import select_rows


def test_nvfp4_source_preserves_words_and_decodes_independently(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight_packed": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
            "proj.input_global_scale": torch.tensor([1.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        expected = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        expected = torch.stack((expected / 2, expected))
        assert torch.equal(source.values().reshape(2, 16), expected)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        assert source.values(16, 16).numel() == 0


def test_sharded_checkpoint_without_index_maps_every_shard(tmp_path):
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.arange(4, dtype=torch.float32)
    save_file({"a.weight": first}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": second}, str(tmp_path / "model-00002-of-00002.safetensors"))
    with SafetensorsSource(tmp_path) as store:
        assert store.has("a.weight") and store.has("b.weight")
        assert torch.equal(store.read_flat("a.weight").reshape(2, 3), first)
        assert torch.equal(store.read_flat("b.weight", 1, 3), second[1:3])


def test_sharded_checkpoint_without_index_rejects_a_duplicated_tensor(tmp_path):
    save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"a.weight": torch.ones(2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    with pytest.raises(ValueError, match="appears in"):
        SafetensorsSource(tmp_path)


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))


def test_exl3_source_preserves_tiles_and_decodes_independently(tmp_path):
    n = k = 128
    half_bits = 3
    generator = torch.Generator().manual_seed(11)
    trellis = torch.randint(
        0, 256, (n // 16, k // 16, 16 * half_bits), dtype=torch.uint8, generator=generator
    )
    su = (torch.rand(k, generator=generator) + 0.5).float()
    sv = (torch.rand(n, generator=generator) + 0.5).float()
    save_file(
        {"proj.trellis": trellis, "proj.su": su, "proj.sv": sv},
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        # A missing explicit format still detects the native quantizer's tiles.
        source = matrix_source(store, "proj.weight", (n, k))
        assert source.shape == (n, k)
        assert source.bitrate_half_bits() == half_bits
        words = source.read_encoded(0, n)
        assert words.format == "exl3_mul1"
        assert torch.equal(words.codes, trellis)
        assert torch.equal(words.scales, sv)
        assert torch.equal(words.input_scales, su)
        # Rows move in whole 16-row tiles.
        assert torch.equal(source.read_encoded(16, 48).codes, trellis[1:3])
        with pytest.raises(ValueError, match="whole 16-row tiles"):
            source.read_encoded(0, 8)
        # Logical values are the exact FP64 decode of the stored planes.
        expected = exl3.decode(trellis, su, sv, half_bits).float()
        assert torch.allclose(source.values().reshape(n, k), expected, rtol=0, atol=0)
        explicit = matrix_source(store, "proj.weight", (n, k), "exl3_mul1")
        assert explicit.bitrate_half_bits() == half_bits


def test_modelopt_nvfp4_source_is_detected(tmp_path):
    # ModelOpt writes NVFP4 as `weight`/`weight_scale`/`weight_scale_2` instead of the
    # compressed-tensors `weight_packed`/`weight_global_scale` spelling.
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": torch.tensor([[0x38]], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
            "proj.input_scale": torch.tensor(1.5, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (1, 16))
        words = source.read_encoded(0, 1)
        assert torch.equal(words.codes, codes)
        assert words.weight_divisor == struct.pack("<f", 0.5)  # 1 / weight_scale_2
        assert torch.equal(
            source.values().reshape(1, 16),
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                  0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0]]
            ),
        )
        assert source.input_divisor() == struct.pack("<f", 1.0 / 1.5)


def test_modelopt_per_tensor_fp8_source_is_detected(tmp_path):
    # ModelOpt per-tensor FP8 carries one scalar F32 scale, which has no exact encoded form.
    weight = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
    save_file(
        {
            "proj.weight": weight,
            "proj.weight_scale": torch.tensor(0.25, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 2))
        assert torch.equal(
            source.values().reshape(2, 2),
            torch.tensor([[0.25, -0.5], [0.125, 1.0]]),
        )
        with pytest.raises(ValueError, match="does not provide encoded"):
            source.read_encoded(0, 1)
