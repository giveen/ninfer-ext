from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs import exl3
from tools.artifact.layouts import exl3_geometry
from tools.artifact.schema import TensorObject
from tools.artifact.tensor_output import TensorOutput


def test_unpack_follows_the_little_endian_circular_window() -> None:
    # K = 2: stream bit 0 set. State t is the 16-bit window ending at bit 2(t+1), low bit oldest,
    # so bit 0 sits at window position 14 - 2t for t = 0..7 and in no other window.
    tile = torch.zeros(64, dtype=torch.uint8)
    tile[0] = 1
    states = exl3.unpack_states(tile, 4)
    expected = torch.zeros(256, dtype=torch.long)
    expected[:8] = 1 << (14 - 2 * torch.arange(8))
    assert torch.equal(states, expected)


def test_mul1_codebook_known_values() -> None:
    # 0x83DCD12D bytes sum to 605.
    assert exl3.mul1(torch.tensor([0, 1])).tolist() == [-510, 95]


@pytest.mark.parametrize("bitrate_half_bits", range(2, 17))
def test_pack_and_unpack_round_trip(bitrate_half_bits: int) -> None:
    generator = torch.Generator().manual_seed(bitrate_half_bits)
    tiles = torch.randint(0, 256, (3, 16 * bitrate_half_bits), dtype=torch.uint8, generator=generator)
    states = exl3.unpack_states(tiles, bitrate_half_bits)
    assert torch.equal(exl3.pack_states(states, bitrate_half_bits), tiles)
    broken = states.clone()
    broken[0, 5] ^= 1  # low bit belongs to the previous state's window
    with pytest.raises(ValueError, match="circular"):
        exl3.pack_states(broken, bitrate_half_bits)


def test_decode_matches_the_explicit_formula() -> None:
    # W[n][k] = su[k] sv[n] sum_ij H[k][i] Z[i][j] H[j][n], Z built independently tile by tile.
    generator = torch.Generator().manual_seed(7)
    n = k = 128
    tiles = torch.randint(0, 256, (n // 16, k // 16, 48), dtype=torch.uint8, generator=generator)
    su = torch.rand(k, generator=generator) + 0.5
    sv = torch.rand(n, generator=generator) + 0.5
    z = torch.zeros((k, n), dtype=torch.float64)
    states = exl3.unpack_states(tiles, 3)
    for nt in range(n // 16):
        for kt in range(k // 16):
            for t in range(256):
                kk = 2 * ((t >> 3) & 3) + (t & 1) + 8 * ((t >> 1) & 1)
                nn = (t >> 5) + 8 * ((t >> 2) & 1)
                z[kt * 16 + kk, nt * 16 + nn] = float(exl3.mul1(states[nt, kt, t]))
    h = torch.tensor(
        [[(-1.0) ** bin(a & b).count("1") for b in range(128)] for a in range(128)],
        dtype=torch.float64,
    ) / 128**0.5
    expected = (su.double().unsqueeze(1) * (h @ z @ h) * sv.double().unsqueeze(0)).T
    assert torch.allclose(exl3.decode(tiles, su, sv, 3), expected, rtol=0, atol=1e-9)


class _MemoryWriter:
    def __init__(self, obj: TensorObject) -> None:
        self.by_id = {obj.id: obj}
        self.data = bytearray(obj.bytes)
        self.written = bytearray(obj.bytes)

    def write_region(self, object_id: str, offset: int, data) -> None:
        self.data[offset : offset + len(data)] = bytes(data)
        self.written[offset : offset + len(data)] = b"\x01" * len(data)

    def write_zeros(self, object_id: str, offset: int, count: int) -> None:
        self.data[offset : offset + count] = bytes(count)
        self.written[offset : offset + count] = b"\x01" * count


def test_tensor_output_places_trellis_and_scale_planes() -> None:
    n, k, half_bits = 256, 128, 5
    g = exl3_geometry("exl3_mul1", (n, k), half_bits)
    obj = TensorObject("w", (n, k), "exl3_mul1", "trellis_t16_v1", 0, g.payload_bytes, 1, half_bits)
    writer = _MemoryWriter(obj)
    output = TensorOutput(writer, "w")
    generator = torch.Generator().manual_seed(3)
    tiles = torch.randint(0, 256, (n // 16, k // 16, 16 * half_bits), dtype=torch.uint8, generator=generator)
    su = torch.rand(k, generator=generator)
    sv = torch.rand(n, generator=generator)
    # Two row blocks, written out of order, as a stacked parent's sources would be.
    output.write_codes(128, tiles[8:], sv[128:], input_scales=su)
    output.write_codes(0, tiles[:8], sv[:128], input_scales=su)
    data = bytes(writer.data)
    assert data[: g.trellis_bytes] == tiles.numpy().tobytes()
    assert data[g.input_scale_offset : g.input_scale_offset + g.input_scale_bytes] == su.numpy().tobytes()
    assert data[g.output_scale_offset :] == sv.numpy().tobytes()
    assert all(writer.written)  # every byte, including plane padding, was written
    with pytest.raises(ValueError, match="share their input scales"):
        output.write_codes(0, tiles[:8], sv[:128], input_scales=su + 1)
    with pytest.raises(ValueError, match="16-row tiles"):
        output.write_codes(8, tiles[:1], sv[:16], input_scales=su)


def test_stacked_object_places_one_input_scale_set_per_matrix_and_decodes_each() -> None:
    sets, rows, k, half_bits = 3, 128, 128, 4
    n = sets * rows
    g = exl3_geometry("exl3_mul1", (n, k), half_bits, sets)
    obj = TensorObject("bank", (n, k), "exl3_mul1", "trellis_t16_v1", 0, g.payload_bytes, sets, half_bits)
    writer = _MemoryWriter(obj)
    output = TensorOutput(writer, "bank")
    generator = torch.Generator().manual_seed(11)
    tiles = torch.randint(0, 256, (n // 16, k // 16, 16 * half_bits), dtype=torch.uint8, generator=generator)
    su = torch.rand(sets, k, generator=generator) + 0.5
    sv = torch.rand(n, generator=generator) + 0.5
    # Matrices arrive in any order, each in two row blocks of its own set.
    for s in (2, 0, 1):
        for begin in (rows // 2, 0):
            lo = s * rows + begin
            output.write_codes(lo, tiles[lo // 16 : (lo + rows // 2) // 16], sv[lo : lo + rows // 2], input_scales=su[s])
    data = bytes(writer.data)
    assert data[: g.trellis_bytes] == tiles.numpy().tobytes()
    assert data[g.input_scale_offset : g.input_scale_offset + g.input_scale_bytes] == su.numpy().tobytes()
    assert data[g.output_scale_offset :] == sv.numpy().tobytes()
    assert all(writer.written)
    # Each set's scales are its own; a block under another set's scales is refused.
    with pytest.raises(ValueError, match="share their input scales"):
        output.write_codes(0, tiles[:8], sv[:rows], input_scales=su[1])
    # A block straddling the boundary between two stacked matrices has no single owner.
    with pytest.raises(ValueError, match="straddles two scale sets"):
        output.write_codes(rows - 16, tiles[(rows - 16) // 16 : (rows + 16) // 16], sv[rows - 16 : rows + 16], input_scales=su[0])
    # The logical matrix is the stack of the separately decoded ones.
    stacked = exl3.decode(tiles, su, sv, half_bits)
    separate = torch.cat(
        [
            exl3.decode(tiles[s * rows // 16 : (s + 1) * rows // 16], su[s], sv[s * rows : (s + 1) * rows], half_bits)
            for s in range(sets)
        ]
    )
    assert stacked.shape == (n, k) and torch.equal(stacked, separate)
    with pytest.raises(ValueError, match="stacked decode"):
        exl3.decode(tiles, su[:2], sv, half_bits)
