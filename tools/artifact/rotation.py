"""Rotations used by the A3 rotation study (`docs/maintainer/llama-cpp-cuda-pr-survey.md`).

`R` is block-diagonal along K: a rotated weight is stored as `W @ R^T` and the activation it multiplies
must be rotated with the same `R`, because `(x R) @ (W R)^T == x @ W^T` in real arithmetic while each
block of the activation is Gaussianized — which is what a fixed-point quantizer wants.

`sylvester256` is the normalized Sylvester transform the KV codecs and EXL3 already use
(`ops/kv_cache/hadamard_d256.cuh`), and it is the matrix the engine's Op implemented while it existed.
The `regular*` names exist for study against the ConvRot/QuaRot lineage. Nothing here is reachable from
an artifact any more: the per-Use rotation field was reverted with the feature.
"""

from __future__ import annotations

import math

import torch

# The names the study measured; the artifact's rotation field they corresponded to was reverted.
DECLARABLE = frozenset(("sylvester256",))


def sylvester_hadamard(size: int) -> torch.Tensor:
    """Natural-order Sylvester Hadamard, normalized: H[i][j] = (-1)^popcount(i & j) / sqrt(size)."""
    r = torch.ones(1, 1)
    while r.shape[0] < size:
        r = torch.cat([torch.cat([r, r], dim=1), torch.cat([r, -r], dim=1)], dim=0) / math.sqrt(2.0)
    if r.shape[0] != size:
        raise ValueError(f"Sylvester Hadamard needs a power of two, got {size}")
    return r


def regular_hadamard(size: int) -> torch.Tensor:
    """Kronecker power of a normalized 4x4 Hadamard with no all-ones column (ConvRot's transform)."""
    h4 = torch.tensor([[1.0, 1.0, 1.0, -1.0],
                       [1.0, 1.0, -1.0, 1.0],
                       [1.0, -1.0, 1.0, 1.0],
                       [-1.0, 1.0, 1.0, 1.0]]) / 2.0
    r = h4
    while r.shape[0] < size:
        r = torch.kron(r, h4)
    if r.shape[0] != size:
        raise ValueError(f"regular Hadamard needs a power of four, got {size}")
    return r


_BY_NAME = {
    "sylvester128": lambda: sylvester_hadamard(128),
    "sylvester256": lambda: sylvester_hadamard(256),
    # A regular (ConvRot-family) transform is a Kronecker power of H4, so only powers of four exist:
    # 64 and 256. ConvRot itself registers only the 256-point form.
    "regular64": lambda: regular_hadamard(64),
    "regular256": lambda: regular_hadamard(256),
}

# name -> matrix, for callers that iterate the candidates.
ROTATIONS = {name: build() for name, build in _BY_NAME.items()}


def rotation_matrix(name: str) -> torch.Tensor:
    try:
        return ROTATIONS[name]
    except KeyError:
        raise ValueError(f"unknown rotation {name!r}") from None


def block_size(name: str) -> int:
    return rotation_matrix(name).shape[0]


def rotate_k(x: torch.Tensor, name: str) -> torch.Tensor:
    """Rotate the last (K) dimension of `x` in blocks, in FP32.

    Weights are rotated here, at conversion, before their quantizer: quantizing first and rotating the
    codes afterwards would take a second lossy step and the stored codes would no longer be the ones
    the engine's activation rotation pairs with.
    """
    r = rotation_matrix(name)
    size = r.shape[0]
    if x.shape[-1] % size:
        raise ValueError(f"K={x.shape[-1]} is not a multiple of the {size}-point block")
    lead = x.shape[:-1]
    blocks = x.to(torch.float32).reshape(*lead, x.shape[-1] // size, size)
    return (blocks @ r.to(x.device).T).reshape(*lead, x.shape[-1])


def rotate_values(values: torch.Tensor, name: str, k: int) -> torch.Tensor:
    """Rotate a flat, row-aligned element sequence whose rows are `k` long."""
    if k % block_size(name):
        raise ValueError(f"K={k} is not a multiple of the {block_size(name)}-point block")
    if values.numel() % k:
        raise ValueError("a rotated read must cover whole rows")
    return rotate_k(values.reshape(-1, k), name).reshape(-1)
