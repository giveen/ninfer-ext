"""Signed 4-bit group codes and binary16 group scales in row_group_v1 layout.

Row r stores its K codes as K/2 bytes (element 2i in the low nibble, 2i+1 in the high nibble, each
code offset by 8 so the stored nibble is `code + 8`), and its K/G scales as binary16 words in a
separate plane, so a row is addressable without any padding.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import RowGroupFormat, get_format
from ..layouts import row_group_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor


def _format(format: str | RowGroupFormat) -> RowGroupFormat:
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, RowGroupFormat):
        raise TypeError("row_group_v1 requires a row-grouped format")
    return spec


def validate_row_group_words(
    codes: torch.Tensor, scales: torch.Tensor, spec: RowGroupFormat
) -> None:
    if bool(((codes < spec.qmin) | (codes > spec.qmax)).any()):
        raise ValueError(f"row-grouped codes must lie in [{spec.qmin}, {spec.qmax}]")
    words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    if bool((((words & 0x8000) != 0) | ((words & 0x7C00) == 0x7C00)).any()):
        raise ValueError("row-grouped scales must be nonnegative finite binary16 words")
    grouped = codes.reshape(codes.shape[0], scales.shape[1], spec.group_size)
    if bool(((words == 0).unsqueeze(2) & (grouped != 0)).any()):
        raise ValueError("a zero group scale requires all-zero codes")


def encode_row_group(
    codes: torch.Tensor,
    scales: torch.Tensor,
    format: str | RowGroupFormat,
    shape: Sequence[int],
) -> bytes:
    """Encode int8 codes [N,K] and binary16 scales [N,K/G] as one row_group_v1 payload."""

    spec = _format(format)
    g = row_group_geometry(spec, shape)
    if codes.dtype != torch.int8 or tuple(codes.shape) != (g.n, g.k):
        raise TypeError(f"codes must be int8 with shape ({g.n}, {g.k})")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (g.n, g.groups_per_row):
        raise TypeError(f"scales must be float16 with shape ({g.n}, {g.groups_per_row})")
    codes = codes.detach().contiguous().cpu()
    scales = scales.detach().contiguous().cpu()
    validate_row_group_words(codes, scales, spec)
    nibbles = (codes.to(torch.int16) + 8).to(torch.uint8).reshape(g.n, g.k // 2, 2)
    packed = nibbles[:, :, 0] | (nibbles[:, :, 1] << 4)
    payload = bytearray(g.payload_bytes)
    payload[: g.code_plane_bytes] = packed.numpy().tobytes()
    payload[g.scale_plane_offset : g.scale_plane_offset + g.scale_plane_bytes] = (
        scales.view(torch.int16).numpy().astype("<i2").tobytes()
    )
    return bytes(payload)


def decode_row_group_words(
    payload: Payload,
    format: str | RowGroupFormat,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode int8 codes [N,K] and binary16 scales [N,K/G] from a row_group_v1 payload."""

    spec = _format(format)
    g = row_group_geometry(spec, shape)
    if _payload_length(payload) != g.payload_bytes:
        raise ValueError(
            f"row-grouped payload has {_payload_length(payload)} bytes, expected {g.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    packed = raw[: g.code_plane_bytes].clone().reshape(g.n, g.k // 2)
    low = (packed & 0xF).to(torch.int16) - 8
    high = (packed >> 4).to(torch.int16) - 8
    codes = torch.stack((low, high), dim=2).reshape(g.n, g.k).to(torch.int8)
    scale_bytes = raw[g.scale_plane_offset : g.scale_plane_offset + g.scale_plane_bytes]
    scales = scale_bytes.clone().view(torch.int16).reshape(g.n, g.groups_per_row).view(torch.float16)
    validate_row_group_words(codes, scales, spec)
    return codes, scales


def dequantize_row_group(
    payload: Payload,
    format: str | RowGroupFormat,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a row-grouped matrix: code * its group's scale."""

    spec = _format(format)
    codes, scales = decode_row_group_words(payload, spec, shape)
    n, k = codes.shape
    grouped = codes.float().reshape(n, k // spec.group_size, spec.group_size)
    return (grouped * scales.float().unsqueeze(2)).reshape(n, k).to(dtype)
