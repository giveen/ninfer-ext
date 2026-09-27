"""EXL3 mul1 trellis tiles (trellis_t16_v1): exact state unpacking and FP64 decoding.

A tensor of shape [N, K] (output channels by input channels) is stored as N/16 x K/16 tiles in
output-major order, each tile a tail-biting bitstream of 16 * bitrate_half_bits bytes, plus FP32
input scales su[K] and output scales sv[N]. With Z[k][n] = mul1(state) the exact integers of the
tiles, the represented matrix is

    W[n][k] = (diag(su) · H128_k · Z · H128_n · diag(sv))[k][n]

where H128 is the normalized 128-point Sylvester Hadamard applied blockwise
(docs/maintainer/storage-layouts.md section 9).
"""

from __future__ import annotations

from functools import cache

import torch

MUL1 = 0x83DCD12D
TILE = 16
TILE_STATES = 256


@cache
def tile_order() -> tuple[torch.Tensor, torch.Tensor]:
    """(k, n) within the tile of each of the 256 states (mma.m16n8k16 B-fragment order)."""
    t = torch.arange(TILE_STATES)
    k = 2 * ((t >> 3) & 3) + (t & 1) + 8 * ((t >> 1) & 1)
    n = (t >> 5) + 8 * ((t >> 2) & 1)
    return k, n


def step_widths(bitrate_half_bits: int) -> torch.Tensor:
    if isinstance(bitrate_half_bits, bool) or not 2 <= bitrate_half_bits <= 16:
        raise ValueError("EXL3 bitrate_half_bits must be an integer in [2, 16]")
    t = torch.arange(TILE_STATES)
    return bitrate_half_bits // 2 + ((bitrate_half_bits & 1) & t)


@cache
def _window_bits(bitrate_half_bits: int) -> torch.Tensor:
    """Stream bit index of window bit b (low bit oldest) for each state: [256, 16]."""
    total = 128 * bitrate_half_bits
    end = torch.cumsum(step_widths(bitrate_half_bits), 0)
    begin = (end - 16) % total
    return (begin.unsqueeze(1) + torch.arange(16)) % total


def unpack_states(trellis: torch.Tensor, bitrate_half_bits: int) -> torch.Tensor:
    """States [..., 256] (int64) of tiles [..., 16 * bitrate_half_bits] (uint8)."""
    if trellis.dtype != torch.uint8 or trellis.shape[-1] != 16 * bitrate_half_bits:
        raise ValueError("EXL3 tiles must be uint8 with 16 * bitrate_half_bits bytes each")
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((trellis.unsqueeze(-1) >> shifts) & 1).flatten(-2).long()  # little-endian stream
    window = bits[..., _window_bits(bitrate_half_bits)]
    return (window << torch.arange(16)).sum(-1)


def pack_states(states: torch.Tensor, bitrate_half_bits: int) -> torch.Tensor:
    """Tiles [..., 16 * bitrate_half_bits] (uint8) of circular states [..., 256]."""
    widths = step_widths(bitrate_half_bits)
    previous = torch.roll(states, 1, dims=-1)
    expected = (previous >> widths) | ((states >> (16 - widths)) << (16 - widths))
    if not bool((states == expected).all()) or bool(((states < 0) | (states > 0xFFFF)).any()):
        raise ValueError("EXL3 states do not form circular trellis paths")
    # Each state contributes its top `width` bits to the stream, lowest of them first.
    total = 128 * bitrate_half_bits
    owner = torch.repeat_interleave(torch.arange(TILE_STATES), widths)
    first = torch.cumsum(widths, 0) - widths
    bit = torch.arange(total) - first[owner]
    bits = (states[..., owner] >> (16 - widths[owner] + bit)) & 1
    grouped = bits.view(*states.shape[:-1], total // 8, 8)
    return (grouped << torch.arange(8)).sum(-1).to(torch.uint8)


def mul1(states: torch.Tensor) -> torch.Tensor:
    """Exact codebook integers in [-510, 510]."""
    product = (states.long() * MUL1) & 0xFFFFFFFF
    total = sum((product >> shift) & 0xFF for shift in (0, 8, 16, 24))
    return total - 510


@cache
def hadamard128() -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / 128**0.5


def decode(
    trellis: torch.Tensor, su: torch.Tensor, sv: torch.Tensor, bitrate_half_bits: int
) -> torch.Tensor:
    """FP64 matrix [N, K] represented by tiles [N/16, K/16, tile_bytes] and scales su[K], sv[N]."""
    tiles_n, tiles_k = trellis.shape[:2]
    n, k = tiles_n * TILE, tiles_k * TILE
    if su.shape != (k,) or sv.shape != (n,) or n % 128 or k % 128:
        raise ValueError("EXL3 decode needs su[K], sv[N] and 128-aligned dimensions")
    z_tiles = mul1(unpack_states(trellis, bitrate_half_bits)).double()  # [tn, tk, 256]
    tk_idx, tn_idx = tile_order()
    z = torch.zeros((tiles_k, TILE, tiles_n, TILE), dtype=torch.float64)
    z[:, tk_idx, :, tn_idx] = z_tiles.permute(2, 1, 0)
    z = z.reshape(k, n)
    h = hadamard128()
    z = (h @ z.view(k // 128, 128, n)).view(k, n)  # H128 along k
    z = (z.view(k, n // 128, 128) @ h).view(k, n)  # H128 along n
    w_kn = su.double().unsqueeze(1) * z * sv.double().unsqueeze(0)
    return w_kn.T.contiguous()
