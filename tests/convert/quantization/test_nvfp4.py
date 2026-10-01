from __future__ import annotations

import struct

import pytest
import torch

from tools.convert.quantization.nvfp4 import (
    Nvfp4Rows,
    divisor_for,
    e4m3_scale_candidates,
    quantize_rows,
)

_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _decode(rows: Nvfp4Rows, shape: tuple[int, int]) -> torch.Tensor:
    """Independent decode of the stored representation: ``e2m1(code) * e4m3(scale) / divisor``."""
    n, k = shape
    codes = rows.codes.view(torch.uint8).reshape(n, k // 2)
    words = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(n, k)
    sign = torch.where(words >= 8, -1.0, 1.0)
    magnitude = _MAGNITUDES[(words & 7).long()]
    scales = rows.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    divisor = struct.unpack("<f", rows.divisor)[0]
    return sign * magnitude * scales / divisor


def test_e4m3_candidates_are_the_126_positive_finite_values() -> None:
    candidates = e4m3_scale_candidates("cpu")
    assert candidates.numel() == 126
    assert candidates.min() > 0
    assert candidates.max() == 448.0
    assert torch.isfinite(candidates).all()


def test_absmax_round_trip_uses_the_stored_scale_and_divisor() -> None:
    # One 16-column block whose largest magnitude maps to the largest E4M3 scale and code.
    values = torch.tensor(
        [[6.0, -3.0, 0.0, 1.5, 0.5, -6.0, 2.0, 4.0] + [0.0] * 8], dtype=torch.float32
    )
    divisor = divisor_for(float(values.abs().max()))
    rows = quantize_rows(values, divisor, device="cpu", scale_search="absmax")
    assert rows.scales.view(torch.float8_e4m3fn).float().item() == 448.0
    assert torch.allclose(_decode(rows, tuple(values.shape)), values, atol=1e-6)


def test_mse_scale_search_never_exceeds_absmax_reconstruction_error() -> None:
    torch.manual_seed(0)
    values = torch.randn(128, 512, dtype=torch.float32) * 0.02
    values[0, 0] = 1.0  # a heavy tail makes the max-magnitude scale a poor fit for other blocks
    divisor = divisor_for(float(values.abs().max()))
    absmax = quantize_rows(values, divisor, device="cpu", scale_search="absmax")
    mse = quantize_rows(values, divisor, device="cpu", scale_search="mse")
    error_absmax = ((values - _decode(absmax, tuple(values.shape))) ** 2).sum()
    error_mse = ((values - _decode(mse, tuple(values.shape))) ** 2).sum()
    assert error_mse <= error_absmax
    # The sweep is expected to be a strict improvement on non-degenerate data.
    assert error_mse < error_absmax * 0.95


def test_mse_scales_are_representable_e4m3_words() -> None:
    torch.manual_seed(1)
    values = torch.randn(32, 256, dtype=torch.float32)
    divisor = divisor_for(float(values.abs().max()))
    rows = quantize_rows(values, divisor, device="cpu", scale_search="mse")
    scales = rows.scales.view(torch.float8_e4m3fn).float()
    candidates = set(e4m3_scale_candidates("cpu").tolist()) | {0.0}
    assert all(float(scale) in candidates for scale in scales.flatten())


def test_zero_blocks_store_a_zero_scale() -> None:
    values = torch.zeros(2, 32, dtype=torch.float32)
    divisor = divisor_for(1.0)
    rows = quantize_rows(values, divisor, device="cpu", scale_search="mse")
    assert torch.count_nonzero(rows.scales.view(torch.float8_e4m3fn).float()) == 0
    assert torch.count_nonzero(rows.codes) == 0


def test_unknown_scale_search_is_rejected() -> None:
    with pytest.raises(ValueError):
        quantize_rows(torch.ones(1, 16), divisor_for(1.0), device="cpu", scale_search="bogus")
