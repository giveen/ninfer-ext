"""EXL3 tensors produced by NInfer's native quantizer (ninfer-quantize).

For a matrix ``prefix.weight`` of logical shape [N, K] the source file holds

- ``prefix.trellis``: U8 [N/16, K/16, 16 * bitrate_half_bits], trellis_t16_v1 tiles in
  output-major order (the bitrate follows from the last dimension);
- ``prefix.su``: F32 [K] input scales; ``prefix.sv``: F32 [N] output scales.

Encoded rows are imported unchanged; logical values are the exact FP64 decode, rounded to FP32.
"""

from __future__ import annotations

from functools import cache

import torch

from tools.artifact.codecs import exl3
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource


def has_exl3(store: SafetensorsSource, prefix: str) -> bool:
    return store.has(prefix + ".trellis")


def exl3_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int]
) -> LogicalSource:
    n, k = shape
    info = store.describe(prefix + ".trellis")
    if (
        info.dtype != "U8"
        or len(info.shape) != 3
        or tuple(info.shape[:2]) != (n // 16, k // 16)
        or info.shape[2] % 16
        or n % 128
        or k % 128
    ):
        raise ValueError(f"{prefix}.trellis: expected U8[{n // 16}, {k // 16}, 16 * rate] tiles")
    half_bits = info.shape[2] // 16
    if not 2 <= half_bits <= 16:
        raise ValueError(f"{prefix}.trellis: tile size implies an unsupported rate")
    tile_row_bytes = (k // 16) * info.shape[2]
    for name, length in ((".su", k), (".sv", n)):
        scale = store.describe(prefix + name)
        if scale.dtype != "F32" or tuple(scale.shape) != (length,):
            raise ValueError(f"{prefix}{name}: expected F32[{length}]")

    @cache
    def input_scales() -> torch.Tensor:
        return store.read_flat(prefix + ".su")

    def encoded(begin: int, end: int) -> EncodedRows:
        if begin % 16 or end % 16 or not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: EXL3 rows must be whole 16-row tiles")
        tiles = store.read_flat(
            prefix + ".trellis", begin // 16 * tile_row_bytes, end // 16 * tile_row_bytes
        ).reshape((end - begin) // 16, k // 16, info.shape[2])
        return EncodedRows(
            "exl3_mul1",
            tiles,
            store.read_flat(prefix + ".sv", begin, end),
            input_scales=input_scales(),
        )

    @cache
    def decoded() -> torch.Tensor:
        rows = encoded(0, n)
        return exl3.decode(rows.codes, rows.input_scales, rows.scales, half_bits).float()

    return LogicalSource(
        (n, k),
        f"{store.path}:{prefix}",
        lambda begin, end: decoded().reshape(-1)[begin:end],
        encoded,
        bitrate_half_bits=lambda: half_bits,
    )
