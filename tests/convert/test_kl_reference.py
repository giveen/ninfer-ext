"""The KL reference byte layout is a contract with the C++ reader in apps/perplexity/reference.cpp.

Both producer bugs so far were silent layout errors that still produced a finite KL, so pin the
header offsets, the payload size and the window protocol rather than trusting an end-to-end run.
"""

from __future__ import annotations

import numpy as np

from tools.perplexity import kl_reference


def test_payload_follows_the_96_byte_header(tmp_path):
    path = tmp_path / "reference.bin"
    logits = np.arange(12, dtype=np.float32).reshape(3, 4)
    digest = kl_reference.write_reference(path, context=4, stride=2, text="hello",
                                          positions=[1, 5, 9], logits=logits, vocab_size=4)
    raw = path.read_bytes()

    # 96-byte header, a u32 per position, then a BF16 word per logit: a 64-byte shift anywhere here
    # is exactly the bug that misaligned every row.
    assert len(raw) == 96 + 3 * 4 + 3 * 4 * 2
    assert raw[:8] == b"NINFKL1\0"
    version, vocab, rows, context, stride, reserved = np.frombuffer(raw[8:32], dtype="<u4")
    assert (version, vocab, rows, context, stride, reserved) == (1, 4, 3, 4, 2, 0)
    assert raw[32:96].decode("ascii") == digest
    assert list(np.frombuffer(raw[96:108], dtype="<u4")) == [1, 5, 9]

    # FP32 n -> BF16 bit pattern is the top half of the FP32 word.
    stored = np.frombuffer(raw[108:], dtype="<u2")
    assert list(stored) == [int(np.float32(value).view(np.uint32) >> 16) for value in range(12)]


def test_window_protocol_matches_the_cpp_plan(tmp_path):
    # Same case the C++ evaluation test pins: tokens=10, context=6, stride=2.
    assert kl_reference.plan_windows(10, 6, 2) == [
        (0, 6, 1, 6, 1),
        (2, 8, 6, 8, 4),
        (4, 10, 8, 10, 4),
    ]


def test_duplicate_positions_are_refused(tmp_path):
    path = tmp_path / "reference.bin"
    import pytest

    with pytest.raises(ValueError):
        kl_reference.write_reference(path, context=4, stride=2, text="hello",
                                     positions=[1, 1], logits=np.zeros((2, 4), dtype=np.float32),
                                     vocab_size=4)
