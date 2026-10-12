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


# The relative residual a value may keep and still count as lying on a candidate grid: two BF16
# half-ulps, since a QAT checkpoint stores every grid value d*k rounded to BF16.
_ON_GRID_RELATIVE = 2.0**-8
# The fallback for groups on no grid: steps max|x| / q over this range, by squared error.
_FALLBACK_DIVISORS = torch.arange(4.0, 8.0 + 1e-6, 1.0 / 16.0)


def quantize_rows_on_grid(
    values: torch.Tensor,
    format: str | RowGroupFormat,
    *,
    device: str | torch.device | None = None,
) -> RowGroupWords:
    """Recover the integer grid a quantization-aware-trained checkpoint was trained on.

    A QAT checkpoint stores each group of G values as `d * k` with `k` an integer in
    [qmin, qmax] and a learned step `d`, rounded to BF16; its largest magnitude need not be
    `qmax * d`. Per group, this takes the coarsest step `max|x| / q` for q = 1 .. (qmax - qmin + 1)
    under which every value is such a multiple to BF16 precision, so the group is represented
    exactly up to the binary16 rounding of `d`. A group on no such grid takes the step in
    `max|x| / [4, 8]` with the least squared error. Codes are chosen against the binary16 scale.
    """
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, RowGroupFormat):
        raise ValueError("row-grouped quantization requires a row-grouped format")
    if values.dim() != 2 or not values.dtype.is_floating_point:
        raise ValueError("row-grouped quantization needs floating-point values [N, K]")
    n, k = values.shape
    if k % spec.group_size:
        raise ValueError(f"K={k} is not a multiple of the group size {spec.group_size}")
    where = pick_device(device or "cuda")
    grouped = values.to(where, dtype=torch.float32).reshape(n, k // spec.group_size, spec.group_size)
    if not torch.isfinite(grouped).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    tolerance = grouped.abs() * _ON_GRID_RELATIVE

    def codes_for(step: torch.Tensor) -> torch.Tensor:
        safe = torch.where(step > 0, step, torch.ones_like(step))
        return torch.round(grouped / safe).clamp_(spec.qmin, spec.qmax)

    step = torch.zeros_like(max_abs)
    found = max_abs == 0
    for q in range(1, spec.qmax - spec.qmin + 2):
        candidate = max_abs / q
        on_grid = ((codes_for(candidate) * candidate - grouped).abs() <= tolerance).all(
            dim=2, keepdim=True
        )
        take = on_grid & ~found
        step = torch.where(take, candidate, step)
        found |= on_grid

    if not bool(found.all()):
        # Only the off-grid groups pay for the search.
        missing = ~found.squeeze(2)
        values = grouped[missing]
        peak = max_abs[missing]
        best_error = torch.full_like(peak, float("inf"))
        best_step = torch.zeros_like(peak)
        for divisor in _FALLBACK_DIVISORS.tolist():
            candidate = (peak / divisor).half().float()
            safe = torch.where(candidate > 0, candidate, torch.ones_like(candidate))
            codes = torch.round(values / safe).clamp_(spec.qmin, spec.qmax)
            error = ((codes * candidate - values) ** 2).sum(dim=1, keepdim=True)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_step = torch.where(better, candidate, best_step)
        step[missing] = best_step

    scale = step.half()
    if bool(((scale == 0) & (max_abs > 0)).any()):
        raise ValueError("a recovered step underflows binary16")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("a recovered step overflows binary16")
    codes = codes_for(scale.float())
    return RowGroupWords(
        codes=codes.reshape(n, k).to(torch.int8).cpu(), scales=scale.squeeze(2).cpu()
    )
