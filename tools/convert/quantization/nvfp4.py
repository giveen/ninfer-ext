"""Weight-only NVFP4 quantization: E2M1 codes, E4M3FN per-16 scales and one FP32 divisor.

For values ``W`` and a positive divisor ``d`` (normally ``6 * 448 / max|W|``), each 16-column block
stores ``s = e4m3(max|W_block| * d / 6)`` and codes ``c = e2m1(W * d / s)`` rounded to the nearest
representable value (ties toward the smaller magnitude). Decoding gives
``W' = e2m1(c) * s / d`` exactly as the ``nvfp4`` format defines it.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MIDPOINTS = (_MAGNITUDES[1:] + _MAGNITUDES[:-1]) / 2
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0


@dataclass(frozen=True, slots=True)
class Nvfp4Rows:
    codes: torch.Tensor  # uint8 [rows, K/2], low nibble first
    scales: torch.Tensor  # uint8 [rows, K/16] E4M3FN words
    divisor: bytes  # FP32 little-endian word


def divisor_for(amax: float) -> bytes:
    """The FP32 divisor that maps ``amax`` to the largest block scale times the largest code."""
    if not amax > 0 or amax != amax or amax == float("inf"):
        amax = 1.0
    return struct.pack("<f", _E2M1_MAX * _E4M3_MAX / amax)


def quantize_rows(values: torch.Tensor, divisor: bytes, device: str = "cpu") -> Nvfp4Rows:
    """Quantize ``values [rows, K]`` against the given divisor."""
    rows, k = values.shape
    if k % 16:
        raise ValueError("NVFP4 rows require K divisible by 16")
    d = struct.unpack("<f", divisor)[0]
    x = values.to(device=device, dtype=torch.float32).reshape(rows, k // 16, 16) * d
    amax = x.abs().amax(dim=-1)
    scale = (amax / _E2M1_MAX).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    decoded = scale.float()
    safe = torch.where(decoded > 0, decoded, torch.ones_like(decoded))
    scaled = (x / safe[..., None]).clamp(-_E2M1_MAX, _E2M1_MAX)
    scaled = torch.where(decoded[..., None] > 0, scaled, torch.zeros_like(scaled))
    magnitude = torch.bucketize(scaled.abs(), _MIDPOINTS.to(scaled.device), right=False)
    negative = (scaled < 0) & (magnitude > 0)
    words = (magnitude | (negative.to(torch.int64) << 3)).to(torch.uint8).reshape(rows, k)
    codes = words[:, 0::2] | (words[:, 1::2] << 4)
    return Nvfp4Rows(codes.cpu(), scale.view(torch.uint8).cpu(), divisor)
