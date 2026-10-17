"""Rotations used to make a row Gaussian before a fixed-point quantizer.

The transform itself lives in `tools/artifact/rotation.py` because a rotated weight is a *stored
representation*: the engine applied the same transform to the activation with `ops::block_hadamard`
while that Op existed. The measurements in `docs/maintainer/llama-cpp-cuda-pr-survey.md` (§A3) are what
remains of it: the Op, the artifact's per-Use rotation field, the converter assignment and the Gemma 4
wiring were reverted after the engine measured the rotated artifact 9% worse in perplexity, so the
names below are the study's, not a live artifact field. This module re-exports them for the
verification harnesses and adds the activation quantizers those harnesses emulate.
"""
from __future__ import annotations

import torch

from tools.artifact.rotation import (DECLARABLE, ROTATIONS, block_size, regular_hadamard,  # noqa: F401
                                     rotate_k, rotate_values, rotation_matrix,
                                     sylvester_hadamard)

__all__ = ["DECLARABLE", "ROTATIONS", "block_size", "regular_hadamard", "rotate_k", "rotate_values",
           "rotation_matrix", "sylvester_hadamard", "int8_rows", "quantize_activation",
           "weight_rotation_name"]


def int8_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax symmetric INT8, as NInfer's `q4_a8_quantize_kernel` defines it.

    The runner sums in int32 with the token scale and the weight's group scale applied afterwards, so
    this returns the represented values, which is the error the route actually carries.
    """
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(x / safe).clamp(-127.0, 127.0) * safe


def quantize_activation(x: torch.Tensor, mode: str) -> torch.Tensor:
    """`i8` per-token INT8; `i8r<block>` the same after a K-block rotation; `r<block>` the rotation
    alone, which pairs with a rotated weight while leaving the activation unquantized."""
    if mode == "i8":
        return int8_rows(x)
    if mode.startswith("i8r"):
        return int8_rows(rotate_k(x, "regular" + mode[3:]))
    if mode.startswith("r"):
        return rotate_k(x, "regular" + mode[1:])
    raise ValueError(f"unknown activation mode {mode!r}")


def weight_rotation_name(mode: str) -> str | None:
    """The rotation a weight must be encoded with to pair with `mode`'s activation, by name."""
    if mode.startswith("i8r") or mode.startswith("r"):
        return "regular" + (mode[3:] if mode.startswith("i8r") else mode[1:])
    return None
