"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    DirectFormat,
    Exl3Format,
    Fp8RowFormat,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    weight_divisor_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Exl3Geometry:
    n: int
    k: int
    bitrate_half_bits: int
    tiles_n: int
    tiles_k: int
    tile_bytes: int
    trellis_bytes: int
    input_scale_offset: int
    input_scale_bytes: int
    output_scale_offset: int
    output_scale_bytes: int
    payload_bytes: int
    scale_sets: int = 1


CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)
TRELLIS_T16_V1 = Layout("trellis_t16_v1", 256, frozenset(("exl3_mul1",)))

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            TRELLIS_T16_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int], divisors: int = 1
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    if divisors < 1 or n % divisors:
        raise ValueError("NVFP4 divisor count must divide the plane's rows")
    # Each divisor covers one source matrix, and a 128-row scale tile may not span two of them:
    # the swizzled scale plane is addressed in whole tiles and the engine takes the divisor from
    # the row.
    if divisors > 1 and (n // divisors) % 128:
        raise ValueError(
            "each NVFP4 divisor must cover a whole number of 128-row scale tiles"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        weight_divisor_count=divisors,
        payload_bytes=weight_divisor_offset + 4 * divisors,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def exl3_geometry(
    format: str | Exl3Format,
    shape: Sequence[int],
    bitrate_half_bits: int | None,
    scale_sets: int = 1,
) -> Exl3Geometry:
    """Geometry of `scale_sets` stacked [N / scale_sets, K] matrices, each with its own suh[K].

    The trellis plane and svh[N] are those of the whole [N, K] object; only suh repeats, once per
    stacked matrix, so one set covers N / scale_sets consecutive rows. One set is the plain matrix.
    """
    spec = _format(format)
    if not isinstance(spec, Exl3Format):
        raise ValueError("trellis_t16_v1 requires exl3_mul1")
    if (
        isinstance(bitrate_half_bits, bool)
        or not isinstance(bitrate_half_bits, int)
        or not 2 <= bitrate_half_bits <= 16
    ):
        raise ValueError("EXL3 bitrate_half_bits must be an integer in [2, 16]")
    n, k = _shape(shape, rank=2)
    if n % 128 or k % 128:
        raise ValueError("trellis_t16_v1 requires both matrix dimensions divisible by 128")
    if isinstance(scale_sets, bool) or not isinstance(scale_sets, int) or scale_sets < 1:
        raise ValueError("EXL3 scale set count must be a positive integer")
    # Each stacked matrix is transformed on its own: a 128-row Hadamard block may not span two.
    if n % scale_sets or (n // scale_sets) % 128:
        raise ValueError(
            "each EXL3 scale set must cover a whole number of 128-row blocks of the object"
        )

    tiles_n = n // 16
    tiles_k = k // 16
    tile_bytes = 16 * bitrate_half_bits
    trellis_bytes = tiles_n * tiles_k * tile_bytes
    input_scale_offset = align_up(trellis_bytes, PLANE_ALIGNMENT)
    input_scale_bytes = k * 4 * scale_sets
    output_scale_offset = align_up(
        input_scale_offset + input_scale_bytes, PLANE_ALIGNMENT
    )
    output_scale_bytes = n * 4
    return Exl3Geometry(
        n=n,
        k=k,
        bitrate_half_bits=bitrate_half_bits,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        tile_bytes=tile_bytes,
        trellis_bytes=trellis_bytes,
        input_scale_offset=input_scale_offset,
        input_scale_bytes=input_scale_bytes,
        output_scale_offset=output_scale_offset,
        output_scale_bytes=output_scale_bytes,
        payload_bytes=output_scale_offset + output_scale_bytes,
        scale_sets=scale_sets,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    divisors: int = 1,
    bitrate_half_bits: int | None = None,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if divisors != 1 and layout_spec not in (BLOCK_SCALE_K16_M128X4_V1, TRELLIS_T16_V1):
        raise ValueError(
            f"layout {layout_spec.name!r} stores one divisor, not {divisors}"
        )
    if bitrate_half_bits is not None and not isinstance(numeric_spec, Exl3Format):
        raise ValueError("bitrate_half_bits is only valid for exl3_mul1")
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape, divisors).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is TRELLIS_T16_V1:
        if not isinstance(numeric_spec, Exl3Format):
            raise ValueError("trellis_t16_v1 requires exl3_mul1")
        return exl3_geometry(numeric_spec, shape, bitrate_half_bits, divisors).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
