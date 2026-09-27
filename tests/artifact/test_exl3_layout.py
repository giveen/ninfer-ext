from __future__ import annotations

import pytest

from tools.artifact.formats import EXL3_MUL1, get_format
from tools.artifact.layouts import exl3_geometry, encoded_size, get_layout
from tools.artifact.schema import TensorSpec, plan_objects, validate_encoding


def test_exl3_format_and_layout_are_registered() -> None:
    assert get_format("exl3_mul1") is EXL3_MUL1
    layout = get_layout("trellis_t16_v1")
    assert layout.alignment == 256
    assert layout.formats == frozenset(("exl3_mul1",))


@pytest.mark.parametrize("bitrate_half_bits", range(2, 17))
def test_exl3_tile_size_and_plane_geometry(bitrate_half_bits: int) -> None:
    geometry = exl3_geometry("exl3_mul1", (128, 128), bitrate_half_bits)
    assert geometry.tiles_n == geometry.tiles_k == 8
    assert geometry.tile_bytes == 16 * bitrate_half_bits
    assert geometry.trellis_bytes == 64 * geometry.tile_bytes
    assert geometry.input_scale_offset == geometry.trellis_bytes
    assert geometry.input_scale_bytes == 512
    assert geometry.output_scale_offset == geometry.input_scale_offset + 512
    assert geometry.output_scale_bytes == 512
    assert (
        encoded_size(
            "trellis_t16_v1",
            "exl3_mul1",
            (128, 128),
            bitrate_half_bits=bitrate_half_bits,
        )
        == geometry.payload_bytes
    )


def test_exl3_qwen_mlp_geometry_and_artifact_metadata() -> None:
    geometry = exl3_geometry("exl3_mul1", (17408, 5120), 8)
    assert (geometry.tiles_n, geometry.tiles_k) == (1088, 320)
    assert geometry.trellis_bytes == 44_564_480
    assert geometry.payload_bytes == 44_654_592

    (obj,) = plan_objects(
        [
            TensorSpec(
                "weight/exl3",
                (17408, 5120),
                "exl3_mul1",
                "trellis_t16_v1",
                bitrate_half_bits=8,
            )
        ]
    )
    validate_encoding(obj)
    encoded = obj.to_json()
    assert encoded["bitrate_half_bits"] == 8
    assert encoded["bytes"] == geometry.payload_bytes


@pytest.mark.parametrize("bitrate_half_bits", [None, 0, 1, 17, True, 8.0, "8"])
def test_exl3_requires_an_integer_half_bit_rate(bitrate_half_bits: object) -> None:
    with pytest.raises(ValueError, match="bitrate_half_bits"):
        encoded_size(
            "trellis_t16_v1",
            "exl3_mul1",
            (128, 128),
            bitrate_half_bits=bitrate_half_bits,  # type: ignore[arg-type]
        )


def test_exl3_rejects_unsupported_layout_shape_and_metadata() -> None:
    with pytest.raises(ValueError, match="does not accept format"):
        encoded_size("row_split_k128_v1", "exl3_mul1", (128, 128), bitrate_half_bits=8)
    with pytest.raises(ValueError, match="requires both matrix dimensions"):
        encoded_size("trellis_t16_v1", "exl3_mul1", (128, 129), bitrate_half_bits=8)
    with pytest.raises(ValueError, match="bitrate_half_bits is only valid"):
        encoded_size("contiguous_le_v1", "bf16", (128, 128), bitrate_half_bits=8)
    with pytest.raises(ValueError, match="stores one divisor"):
        encoded_size("trellis_t16_v1", "exl3_mul1", (128, 128), 2, 8)
