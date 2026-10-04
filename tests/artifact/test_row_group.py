from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.row_group import (
    decode_row_group_words,
    dequantize_row_group,
    encode_row_group,
)
from tools.artifact.layouts import encoded_size, row_group_geometry

FORMAT = "q4_g32_fp16_rows"


def _half_words(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float16)


def test_row_group_layout_known_words_padding_and_reconstruction():
    # Rows are 160 wide like the n-gram table: five groups, 80 code bytes and 10 scale bytes each.
    shape = (2, 160)
    geometry = row_group_geometry(FORMAT, shape)
    assert (
        geometry.code_row_bytes,
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_row_bytes,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (80, 160, 256, 10, 20, 276)
    assert encoded_size("row_group_v1", FORMAT, shape) == 276

    codes = torch.zeros(2, 160, dtype=torch.int8)
    codes[0, :4] = torch.tensor([-8, -1, 0, 7], dtype=torch.int8)
    codes[1, 158:] = torch.tensor([3, -3], dtype=torch.int8)
    scales = torch.stack((_half_words(0.5, 1.0, 2.0, 0.25, 4.0), _half_words(1.0, 1.0, 1.0, 1.0, 0.125)))
    payload = encode_row_group(codes, scales, FORMAT, shape)

    # The stored nibble is code + 8; element 2i is the low nibble.
    assert payload[0] == (-8 + 8) | ((-1 + 8) << 4)
    assert payload[1] == (0 + 8) | ((7 + 8) << 4)
    assert payload[79 + 80] == (3 + 8) | ((-3 + 8) << 4)
    assert payload[160:256] == bytes(96)
    assert payload[256:266] == struct.pack("<5e", 0.5, 1.0, 2.0, 0.25, 4.0)
    assert payload[266:] == struct.pack("<5e", 1.0, 1.0, 1.0, 1.0, 0.125)

    decoded_codes, decoded_scales = decode_row_group_words(payload, FORMAT, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))
    values = dequantize_row_group(payload, FORMAT, shape)
    assert values[0, :4].tolist() == [-4.0, -0.5, 0.0, 3.5]
    assert values[1, 158:].tolist() == [0.375, -0.375]


def test_row_group_rejects_bad_shapes_codes_and_scales():
    with pytest.raises(ValueError, match="multiple of 32"):
        row_group_geometry(FORMAT, (1, 48))
    codes = torch.zeros(1, 32, dtype=torch.int8)
    scale = _half_words(1.0).reshape(1, 1)
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        encode_row_group(torch.full((1, 32), 8, dtype=torch.int8), scale, FORMAT, (1, 32))
    with pytest.raises(ValueError, match="nonnegative finite"):
        encode_row_group(codes, _half_words(-1.0).reshape(1, 1), FORMAT, (1, 32))
    with pytest.raises(ValueError, match="zero group scale"):
        encode_row_group(torch.ones(1, 32, dtype=torch.int8), _half_words(0.0).reshape(1, 1), FORMAT, (1, 32))
    with pytest.raises(ValueError, match="expected"):
        decode_row_group_words(bytes(10), FORMAT, (1, 32))
