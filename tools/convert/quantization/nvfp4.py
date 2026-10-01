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


def e4m3_scale_candidates(device: str | torch.device = "cpu") -> torch.Tensor:
    """The 126 positive finite E4M3FN values, the per-block scale candidates ModelOpt sweeps."""
    words = torch.arange(0, 128, dtype=torch.uint8, device=device)
    values = words.view(torch.float8_e4m3fn).float()
    return values[torch.isfinite(values) & (values > 0)]


def _mse_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-block E4M3 scale minimising ``sum((x - decode(x))^2)`` over the candidate set.

    ``x`` is ``[rows, blocks, 16]`` already scaled by the divisor. The search mirrors ModelOpt's
    NVFP4 weight-MSE FP8 sweep: every block evaluates all 126 representable E4M3 scales and keeps
    the one with the smallest reconstruction error, rather than the max-magnitude scale.
    """
    candidates = e4m3_scale_candidates(x.device)
    magnitudes = _MAGNITUDES.to(x.device)
    midpoints = _MIDPOINTS.to(x.device)
    best_error = None
    best_scale = None
    for candidate in candidates:
        scale = torch.full(x.shape[:-1], float(candidate), device=x.device, dtype=torch.float32)
        scaled = (x / scale[..., None]).clamp(-_E2M1_MAX, _E2M1_MAX)
        magnitude = torch.bucketize(scaled.abs(), midpoints, right=False)
        decoded = torch.where(scaled < 0, -1.0, 1.0) * magnitudes[magnitude] * scale[..., None]
        error = ((x - decoded) ** 2).sum(dim=-1)
        if best_error is None:
            best_error, best_scale = error, scale
        else:
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, scale, best_scale)
    # An all-zero block stores a zero scale, matching the max-magnitude convention, rather than
    # the smallest positive candidate the sweep would otherwise keep.
    empty = x.abs().amax(dim=-1) == 0
    return torch.where(empty, torch.zeros_like(best_scale), best_scale)


def quantize_rows(values: torch.Tensor, divisor: bytes, device: str = "cpu",
                  scale_search: str = "absmax") -> Nvfp4Rows:
    """Quantize ``values [rows, K]`` against the given divisor.

    ``scale_search`` selects each 16-column block's E4M3 scale: ``"absmax"`` uses the
    max-magnitude scale, ``"mse"`` the reconstruction-error-minimising scale.
    """
    rows, k = values.shape
    if k % 16:
        raise ValueError("NVFP4 rows require K divisible by 16")
    d = struct.unpack("<f", divisor)[0]
    x = values.to(device=device, dtype=torch.float32).reshape(rows, k // 16, 16) * d
    if scale_search == "absmax":
        amax = x.abs().amax(dim=-1)
        scale = (amax / _E2M1_MAX).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    elif scale_search == "mse":
        scale = _mse_scale(x).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unknown NVFP4 scale search: {scale_search!r}")
    decoded = scale.float()
    safe = torch.where(decoded > 0, decoded, torch.ones_like(decoded))
    scaled = (x / safe[..., None]).clamp(-_E2M1_MAX, _E2M1_MAX)
    scaled = torch.where(decoded[..., None] > 0, scaled, torch.zeros_like(scaled))
    magnitude = torch.bucketize(scaled.abs(), _MIDPOINTS.to(scaled.device), right=False)
    negative = (scaled < 0) & (magnitude > 0)
    words = (magnitude | (negative.to(torch.int64) << 3)).to(torch.uint8).reshape(rows, k)
    codes = words[:, 0::2] | (words[:, 1::2] << 4)
    return Nvfp4Rows(codes.cpu(), scale.view(torch.uint8).cpu(), divisor)
