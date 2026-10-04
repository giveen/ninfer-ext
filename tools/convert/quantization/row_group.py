"""Row-wise grouped symmetric 4-bit quantization for the row_group_v1 layout.

Same transform as the grouped formats: one binary16 scale per group of G values, scale = group max
magnitude / qmax rounded to binary16 before the codes are chosen, codes = round(value / scale)
clamped to [-qmax, qmax]. Only the layout differs: rows are not padded to a K multiple.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.artifact.formats import RowGroupFormat, get_format

from .groupwise import _canonical_scale_words, pick_device


@dataclass(frozen=True, slots=True)
class RowGroupWords:
    codes: torch.Tensor  # int8 [N, K]
    scales: torch.Tensor  # float16 [N, K / G]


def quantize_rows(
    values: torch.Tensor,
    format: str | RowGroupFormat,
    *,
    device: str | torch.device | None = None,
) -> RowGroupWords:
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, RowGroupFormat):
        raise ValueError("row-grouped quantization requires a row-grouped format")
    if values.dim() != 2 or not values.dtype.is_floating_point:
        raise ValueError("row-grouped quantization needs floating-point values [N, K]")
    n, k = values.shape
    if k % spec.group_size:
        raise ValueError(f"K={k} is not a multiple of the group size {spec.group_size}")
    qmax = spec.qmax
    where = pick_device(device or "cuda")
    grouped = values.to(where, dtype=torch.float32).reshape(n, k // spec.group_size, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    scale, reciprocal = _canonical_scale_words(max_abs, qmax)
    reciprocal = reciprocal.to(where)
    codes = torch.round(grouped * reciprocal.unsqueeze(2)).clamp_(-qmax, qmax)
    return RowGroupWords(codes=codes.reshape(n, k).to(torch.int8).cpu(), scales=scale)
