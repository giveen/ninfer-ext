"""Interpret NVIDIA ModelOpt NVFP4, block-FP8 and per-tensor FP8 checkpoint fields.

ModelOpt stores multipliers where the v3 formats store divisors or row multipliers:

- NVFP4 ``weight`` (U8 packed E2M1, low nibble first), ``weight_scale`` (E4M3FN per 16 columns),
  ``weight_scale_2`` (FP32 multiplier) and ``input_scale`` (FP32 activation multiplier). The v3
  weight divisor is ``1 / weight_scale_2``, rounded once to FP32; codes and block scales are
  preserved bit-exactly.
- Block FP8 ``weight`` (E4M3FN) with ``weight_scale_inv`` (BF16 multiplier per 128x128 tile).
  It has no v3 encoded counterpart and is exposed as logical values only.
- Per-tensor FP8 ``weight`` (E4M3FN) with one BF16 ``weight_scale`` multiplier. It maps exactly to
  the row-scaled FP8 format with that multiplier on every row.
"""

from __future__ import annotations

from functools import cache
from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _signature(store: SafetensorsSource, name: str, shape: tuple[int, ...], dtype: str) -> None:
    info = store.describe(name)
    if tuple(info.shape) != tuple(shape) or info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}{shape}, got {info.dtype}{info.shape}")


def _scalar_f32(store: SafetensorsSource, name: str) -> float:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected one FP32 value")
    return float(store.read_flat(name)[0])


def reciprocal_word(value: float, label: str) -> bytes:
    """FP32 word of ``1 / value``; the only rounding a ModelOpt multiplier import performs."""
    if not value > 0 or value != value or value == float("inf"):
        raise ValueError(f"{label}: multiplier must be finite and positive")
    word = struct.pack("<f", 1.0 / value)
    if not valid_positive_fp32_word(struct.unpack("<I", word)[0]):
        raise ValueError(f"{label}: reciprocal is not a positive finite FP32 value")
    return word


def decode_nvfp4(codes: torch.Tensor, scales: torch.Tensor, divisor: bytes) -> torch.Tensor:
    """Exact represented values ``e2m1 * e4m3 / d_w`` of packed rows."""
    rows = codes.shape[0]
    words = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, -1)
    values = _E2M1[words.long()]
    block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * block / struct.unpack("<f", divisor)[0]


def nvfp4_source(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> LogicalSource:
    """One ModelOpt NVFP4 matrix ``prefix.weight`` with logical shape ``[N,K]``."""
    n, k = shape
    if k % 16:
        raise ValueError(f"{prefix}: NVFP4 K must be divisible by 16")

    # Each chunk of the matrix names its divisor; read the scalar once, not once per chunk.
    @cache
    def divisor() -> bytes:
        return reciprocal_word(_scalar_f32(store, prefix + ".weight_scale_2"), prefix)

    @cache
    def input_divisor() -> bytes:
        return reciprocal_word(_scalar_f32(store, prefix + ".input_scale"), prefix)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        _signature(store, prefix + ".weight", (n, k // 2), "U8")
        _signature(store, prefix + ".weight_scale", (n, k // 16), "F8_E4M3")
        codes = store.read_flat(prefix + ".weight", begin * (k // 2), end * (k // 2))
        scales = store.read_flat(prefix + ".weight_scale", begin * (k // 16), end * (k // 16))
        scales = scales.view(torch.uint8).reshape(end - begin, k // 16)
        if bool((scales > 0x7E).any()):
            raise ValueError(f"{prefix}.weight_scale: expected nonnegative finite E4M3FN")
        return EncodedRows("nvfp4", codes.reshape(end - begin, k // 2), scales, divisor())

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = decode_nvfp4(words.codes, words.scales, words.weight_divisor)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape, f"{store.path}:{prefix} (modelopt nvfp4)", read, encoded, divisor, input_divisor
    )


def fp8_block_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], block: int = 128
) -> LogicalSource:
    """ModelOpt/DeepSeek block FP8 ``weight * weight_scale_inv[tile]`` as logical values."""
    n, k = shape
    tiles = ((n + block - 1) // block, (k + block - 1) // block)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        _signature(store, prefix + ".weight", shape, "F8_E4M3")
        _signature(store, prefix + ".weight_scale_inv", tiles, "BF16")
        first, last = begin // k, (end + k - 1) // k
        codes = store.read_flat(prefix + ".weight", first * k, last * k).reshape(last - first, k)
        scale = store.read_flat(prefix + ".weight_scale_inv").reshape(tiles).float()
        rows = torch.arange(first, last) // block
        tile = scale[rows].repeat_interleave(block, dim=1)[:, :k]
        values = codes.view(torch.float8_e4m3fn).float() * tile
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} (block fp8)", read)


def fp8_tensor_rows_source(
    store: SafetensorsSource,
    shards: list[str],
    scale: str,
    shape: tuple[int, int],
) -> LogicalSource:
    """Row-concatenated per-tensor FP8 shards sharing one BF16 multiplier.

    Encoded rows carry the shared multiplier as every row scale, which is the exact
    ``fp8_e4m3fn_row_bf16`` representation of the same values.
    """
    n, k = shape
    counts = []
    for name in shards:
        info = store.describe(name)
        if info.dtype != "F8_E4M3" or len(info.shape) != 2 or info.shape[1] != k:
            raise ValueError(f"{name}: expected F8_E4M3 rows of width {k}")
        counts.append(info.shape[0])
    if sum(counts) != n:
        raise ValueError(f"{shards[0]}: shard rows {sum(counts)} differ from {n}")
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)

    def multiplier() -> torch.Tensor:
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != 1:
            raise ValueError(f"{scale}: expected one BF16 multiplier")
        return store.read_flat(scale).reshape(1)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{shards[0]}: invalid encoded rows [{begin},{end})")
        pieces = []
        for index, name in enumerate(shards):
            low, high = max(begin, starts[index]), min(end, starts[index + 1])
            if low < high:
                local = low - starts[index], high - starts[index]
                pieces.append(store.read_flat(name, local[0] * k, local[1] * k))
        codes = torch.cat(pieces).view(torch.uint8).reshape(end - begin, k)
        scales = multiplier().expand(end - begin).contiguous()
        validate_fp8_row_words(codes, scales)
        return EncodedRows("fp8_e4m3fn_row_bf16", codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = words.codes.view(torch.float8_e4m3fn).float() * words.scales.float()[:, None]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{shards[0]}+{len(shards)-1} (fp8)", read, encoded)
