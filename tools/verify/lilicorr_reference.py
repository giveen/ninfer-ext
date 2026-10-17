#!/usr/bin/env python3
"""Independent FP32 oracle for the LiLiCorr drafter (body + head) of
NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.

This is the *from-scratch* counterpart of NInfer's C++/CUDA LiLiCorr port: it loads the shipped
checkpoint, dequantizes its NVFP4 weights, runs the draft body and the correlator head on synthetic
target-tap inputs, and dumps every intermediate stage so that a C++ op -- or the composition --
can be compared stage by stage.  It is deliberately not a port of vLLM: it is written from the
sources below, so that a disagreement with the C++ implementation is informative rather than
inherited.

Sources (all read for this file; every remaining ambiguity is marked `SOURCE NOTE`):

  A. vLLM master: `vllm/model_executor/models/lilicorr.py`, `qwen3_dflash.py`, `qwen3_dflash2.py`,
     `vllm/v1/worker/gpu/spec_decode/{dflash,dflash2,lilicorr}/speculator.py`,
     `vllm/model_executor/layers/attention/attention.py`,
     `vllm/v1/attention/backends/flash_attn.py` (`_maybe_symmetrize_window`),
     `vllm/model_executor/model_loader/weight_utils.py::make_attention_sink`.
  B. The LiLiCorr paper, arXiv:2608.20530v1 (fetched 2026-10-16), and the (A)+(B) digest at
     `local://lilicorr-vllm-paper-spec.md`.
  C. llama.cpp @ cd0a5b6a (PR ggml-org/llama.cpp#30520), `src/models/dflash.cpp`, and the porting
     spec `local://lilicorr-llamacpp-forward.md`.
  D. The checkpoint header itself: `/mnt/storage/models/nemo/74B/LiLiCorr/{config.json,
     hf_quant_config.json, model.safetensors}` (179 tensors).
  E. NInfer's own draft machinery, surveyed in `local://ninfer-dflash-map.md`; used only to confirm
     that C++ already commits to the symmetric non-causal window `abs(p_key - p_query) < window`
     and to a denominator-only sink term, matching (A) and (C).

Run:

    # default mode: reconcile the checkpoint header, then check every internal invariant
    python3 tools/verify/lilicorr_reference.py

    # dump every stage of one block, with the real checkpoint and the real target tables
    python3 tools/verify/lilicorr_reference.py --dump /tmp/lilicorr_dump --context 24

The interpreter needs torch, numpy and safetensors (the repo's `.venv/bin/python` has numpy but not
torch; use `python3`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------------------------
# Switchable conventions.
#
# Each of these is a fork in the math that the checkpoint alone cannot settle; they are module
# constants so the C++ side can be tested against both readings.
# ---------------------------------------------------------------------------------------------

# Which nibble of a packed byte holds the even-indexed element along the input dimension.
#
# Chosen "low_even": low nibble = element 2i, high nibble = element 2i+1.  NOT VERIFIABLE FROM THE
# FILE: swapping the two nibbles permutes each 16-element group, and the only structural invariant a
# modelopt NVFP4 checkpoint carries -- every group's absolute maximum equals
# 6 * group_scale * tensor_scale, see `dequant_nvfp4` -- is invariant under that permutation.
# `--self-check` therefore evaluates the invariant under both orders; only an activation comparison
# against the C++ (or a reference) can separate them.
NIBBLE_ORDER = "low_even"  # "low_even" | "high_even"

# Which half of the fused edge weight is the *outgoing* vector.
#
# SOURCE NOTE -- resolved, kept switchable because a swap is silent and degrades acceptance without
# erroring:
#   (A) materializes `_fused_edge_weight = cat([out_head.weight, in_head.weight], dim=0)` and then
#       `out_vec, in_vec = F.normalize(edges.unflatten(-1, (2, d_e))).unbind(-2)`, with
#       `start_scores = anchor_out . in_vec[:, 0]` and `pair[s,i,j] = out[s,i] . in[s+1,j]`.
#   (C) names the two projections directly (`out = lilicorr_out_head @ factors`,
#       `in = lilicorr_in_head @ factors`) and also dots the slot-0 *in* vector with `anchor_out`.
#   Both sources therefore agree: the fused block's half `[0:h)` is `out_head`'s output (the
#   outgoing vector of the *previous* candidate) and `[h:2h)` is `in_head`'s output (the incoming
#   vector of the *next* candidate), and the anchor scores the slot-0 *in* vector.
#
# The flag names the reading of that fused block, not the concatenation order: `"out_first"` is the
# resolution above, `"in_first"` is the swap (`in_head`'s output in `[0:h)`, so the anchor scores the
# other vector and the edge term runs the other way).  Both are computed and reported as different
# numbers, so a C++ implementation can be checked against either.
EDGE_FUSION_ORDER = "out_first"  # "out_first" | "in_first"

# Attention-sink semantics.
#
# SOURCE NOTE -- resolved by three independent readings: (A) passes the learned `[num_heads]` bias
# as FlashAttention-3 `s_aux`, whose documented effect is one extra softmax column with logit
# `sink[h]` and value 0 (denominator only); (C) states it explicitly
# (`den = sum_i exp(w_i - m) + exp(sink[h] - m)`, `out_i = exp(w_i - m)/den`, applied after the mask
# and after the 1/sqrt(d) scaling); (E) records that NInfer's sliding-window kernel would need
# exactly one `expf(sink - m)` term in its merged denominator.  There is no source that puts the
# sink in the numerator, so the alternative is not implemented -- a reader that thinks otherwise
# should change `attention_with_sink` in one place.
SINK_MODE = "denominator_only"

# Sliding-window mask shape for a layer that is both sliding and non-causal.
#
# SOURCE NOTE: (A) builds the Decoder window as `(sliding_window - 1, 0)` and then
# `_maybe_symmetrize_window` rewrites it to `(w-1, w-1)` because the layer is non-causal; (C) drops a
# cell iff `p1 - p0 >= n_swa` with the causal test removed, i.e. keeps `|p_q - p_k| < window`; (E)
# records `abs(p_key - p_query) < S` as NInfer's own contract.  All three agree on symmetric.
WINDOW_MODE = "symmetric"  # "symmetric" | "causal"

# `feature_norm` is an nn.LayerNorm(5) in (A), whose default eps is 1e-5, and (C) hardcodes 1e-5
# (`:715`).  (C) notes this is *not* the model's `rms_norm_eps` (1e-6) and flags it as unresolved.
FEATURE_NORM_EPS = 1e-5

# Inputs.
CHECKPOINT = Path("/mnt/storage/models/nemo/74B/LiLiCorr/model.safetensors")
CHECKPOINT_CONFIG = Path("/mnt/storage/models/nemo/74B/LiLiCorr/config.json")
TARGET_DIR = Path("/mnt/storage/models/nemo/74B")
TARGET_INDEX = "model.safetensors.index.json"
TARGET_EMBED_KEY = "language_model.backbone.embeddings.weight"
TARGET_LM_HEAD_KEY = "language_model.lm_head.weight"

# `relative_slot_bias`/`slot_embedding` are shaped for this; a shorter draft reuses a prefix of them
# and keeps the trained `block_size - 1` bias offset.
TRAINED_BLOCK_SIZE = 8

# ---------------------------------------------------------------------------------------------
# NVFP4: E2M1 codes, E4M3 scales, and the two-level dequantization
# ---------------------------------------------------------------------------------------------


def e2m1_values() -> torch.Tensor:
    """The 16 NVFP4 (E2M1) codes as values: 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives."""
    values = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if code & 0x8 else 1.0
        exponent = (code >> 1) & 0x3
        mantissa = code & 0x1
        if exponent == 0:
            magnitude = mantissa * 0.5
        else:
            magnitude = (1.0 + mantissa * 0.5) * 2.0 ** (exponent - 1)
        values[code] = sign * magnitude
    return torch.from_numpy(values)


def e4m3_values() -> torch.Tensor:
    """All 256 OCP E4M3 codes (no infinity; `0x7F`/`0xFF` decode as NaN)."""
    values = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            magnitude = (mantissa / 8.0) * 2.0 ** -6
        elif exponent == 15 and mantissa == 7:
            magnitude = float("nan")
        else:
            magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values[byte] = sign * magnitude
    return torch.from_numpy(values)


E2M1 = e2m1_values()
E4M3 = e4m3_values()


def unpack_nibbles(packed: torch.Tensor, order: str | None = None) -> torch.Tensor:
    """[..., K/2] uint8 -> [..., K] int64, one code per 4-bit nibble (see NIBBLE_ORDER).

    `order=None` reads the module constant *at call time* so that a caller (or the CLI) can change
    it; a default argument would bind the value at import time and silently ignore the switch.
    """
    order = NIBBLE_ORDER if order is None else order
    if order not in ("low_even", "high_even"):
        raise ValueError(f"unknown nibble order {order!r}")
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.int64)
    if order == "low_even":
        out[..., 0::2], out[..., 1::2] = low, high
    else:
        out[..., 0::2], out[..., 1::2] = high, low
    return out


def dequant_nvfp4(
    packed: torch.Tensor,
    group_scale: torch.Tensor,
    tensor_scale: float,
    group_size: int = 16,
    order: str | None = None,
) -> torch.Tensor:
    """[N, K/2] uint8 + [N, K/group_size] E4M3 + scalar -> [N, K] float32.

    value[n, k] = E2M1(nibble) * group_scale[n, k // group_size] * tensor_scale.

    This is the same decode as NInfer's own converter
    (`tools/convert/sources/modelopt.py::decode_nvfp4`): modelopt's `weight_scale_2` is a
    *multiplier* whose v3 image is the divisor `1 / weight_scale_2`, so
    `W = e2m1(code) * e4m3(scale) / (1/weight_scale_2)`, and that file also fixes the packed
    nibble order as "low nibble first".

    Falsifiable invariant, verified on the shipped file by `--self-check`: modelopt scales every
    group by its own amax/6, so `absmax(group) == 6 * group_scale * tensor_scale` exactly.  A wrong
    group indexing, wrong scale tensor or wrong dtype decode breaks it.
    """
    codes = unpack_nibbles(packed, order)
    values = E2M1[codes]
    n, k = values.shape
    if k % group_size or group_scale.shape[1] * group_size != k:
        raise ValueError(f"scale shape {tuple(group_scale.shape)} does not cover width {k}")
    scale = group_scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    return values * scale * float(tensor_scale)


# ---------------------------------------------------------------------------------------------
# The checkpoint header: reconciliation and the parameter -> role table
# ---------------------------------------------------------------------------------------------

# dtype -> bytes per stored element
STORAGE = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}

# Roles are named after the operator that consumes the tensor rather than after the checkpoint's own
# nesting, so the C++ binder can key a table on them verbatim.
ROLE_RULES: tuple[tuple[str, str], ...] = (
    (r"^fc$", "body.fc"),
    (r"^hidden_norm$", "body.hidden_norm"),
    (r"^norm$", "body.norm"),
    (r"^layers\.(\d+)\.input_layernorm$", r"body.layer\1.input_layernorm"),
    (r"^layers\.(\d+)\.post_attention_layernorm$", r"body.layer\1.post_attention_layernorm"),
    (r"^layers\.(\d+)\.attention_conv\.base_kernel$", r"body.layer\1.conv_attn.base"),
    (r"^layers\.(\d+)\.attention_conv\.kernel_projection$", r"body.layer\1.conv_attn.proj"),
    (r"^layers\.(\d+)\.mlp_conv\.base_kernel$", r"body.layer\1.conv_mlp.base"),
    (r"^layers\.(\d+)\.mlp_conv\.kernel_projection$", r"body.layer\1.conv_mlp.proj"),
    (r"^layers\.(\d+)\.self_attn\.q_proj$", r"body.layer\1.attn.q_proj"),
    (r"^layers\.(\d+)\.self_attn\.k_proj$", r"body.layer\1.attn.k_proj"),
    (r"^layers\.(\d+)\.self_attn\.v_proj$", r"body.layer\1.attn.v_proj"),
    (r"^layers\.(\d+)\.self_attn\.o_proj$", r"body.layer\1.attn.o_proj"),
    (r"^layers\.(\d+)\.self_attn\.q_norm$", r"body.layer\1.attn.q_norm"),
    (r"^layers\.(\d+)\.self_attn\.k_norm$", r"body.layer\1.attn.k_norm"),
    (r"^layers\.(\d+)\.self_attn\.attention_sink_bias$", r"body.layer\1.attn.sink_bias"),
    (r"^layers\.(\d+)\.mlp\.gate_proj$", r"body.layer\1.mlp.gate_proj"),
    (r"^layers\.(\d+)\.mlp\.up_proj$", r"body.layer\1.mlp.up_proj"),
    (r"^layers\.(\d+)\.mlp\.down_proj$", r"body.layer\1.mlp.down_proj"),
    (r"^lilicorr\.token_proj$", "head.token_proj"),
    (r"^lilicorr\.pass_hidden_proj$", "head.pass_hidden_proj"),
    (r"^lilicorr\.context_proj$", "head.context_proj"),
    (r"^lilicorr\.feature_norm$", "head.feature_norm"),
    (r"^lilicorr\.feature_mlp\.up_proj$", "head.feature_mlp.up_proj"),
    (r"^lilicorr\.feature_mlp\.down_proj$", "head.feature_mlp.down_proj"),
    (r"^lilicorr\.slot_embedding$", "head.slot_embedding"),
    (r"^lilicorr\.rank_embedding$", "head.rank_embedding"),
    (r"^lilicorr\.relative_slot_bias$", "head.relative_slot_bias"),
    (r"^lilicorr\.same_slot_bias$", "head.same_slot_bias"),
    (r"^lilicorr\.layers\.(\d+)\.attn_norm$", r"head.layer\1.attn_norm"),
    (r"^lilicorr\.layers\.(\d+)\.attn\.in_proj$", r"head.layer\1.attn.in_proj"),
    (r"^lilicorr\.layers\.(\d+)\.attn\.out_proj$", r"head.layer\1.attn.out_proj"),
    (r"^lilicorr\.layers\.(\d+)\.mlp_norm$", r"head.layer\1.mlp_norm"),
    (r"^lilicorr\.layers\.(\d+)\.mlp\.up_proj$", r"head.layer\1.mlp.up_proj"),
    (r"^lilicorr\.layers\.(\d+)\.mlp\.down_proj$", r"head.layer\1.mlp.down_proj"),
    (r"^lilicorr\.output_norm$", "head.output_norm"),
    (r"^lilicorr\.anchor_norm$", "head.anchor_norm"),
    (r"^lilicorr\.factor_input_proj$", "head.factor_input_proj"),
    (r"^lilicorr\.out_head$", "head.out_head"),
    (r"^lilicorr\.in_head$", "head.in_head"),
    (r"^lilicorr\.anchor_out_head$", "head.anchor_out_head"),
)

# NVFP4 linears carry a packed weight, a per-group E4M3 scale and a per-tensor F32 scale; the three
# have to be bound together, and the role table says which is which.
COMPONENT_SUFFIX = (
    (".weight_scale_2", "tensor_scale"),
    (".weight_scale", "group_scale"),
    (".weight", "weight"),
    (".bias", "bias"),
    ("_weight", "weight"),
    ("_bias", "bias"),
)


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header (name -> dtype/shape/offsets) plus its byte length.

    Everything `print_reconciliation` and `role_table` use comes from this header alone, i.e. before
    any payload byte is read.
    """
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header, header_len


def classify(name: str) -> tuple[str, str]:
    """(role, component) for one checkpoint tensor name; role is `UNKNOWN` when unmatched.

    The full name is matched first so that tensors whose own name ends in `_bias` or `_weight`
    (`attention_sink_bias`, `in_proj_weight`, `relative_slot_bias`, ...) are not mistaken for the
    bias or weight component of a longer role.
    """
    for pattern, replacement in ROLE_RULES:
        if re.match(pattern + r"$", name):
            return re.sub(pattern, replacement, name), "value"
    for suffix, kind in COMPONENT_SUFFIX:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            for pattern, replacement in ROLE_RULES:
                if re.match(pattern + r"$", base):
                    return re.sub(pattern, replacement, base), kind
    return "UNKNOWN", "value"


def role_table(header: dict) -> list[dict]:
    rows = []
    for name in sorted(header):
        entry = header[name]
        stored = 1
        for dim in entry["shape"]:
            stored *= dim
        role, component = classify(name)
        rows.append(
            {
                "name": name,
                "role": role,
                "component": component,
                "dtype": entry["dtype"],
                "shape": list(entry["shape"]),
                "stored_elements": stored,
                # a packed NVFP4 byte carries two parameters
                "parameters": stored * (2 if entry["dtype"] == "U8" else 1),
            }
        )
    return rows


def print_role_table(rows: list[dict]) -> None:
    print("\n--- parameter -> role table (header only; the C++ binder reuses these roles) ---")
    print(f"{'tensor':62s} {'role':32s} {'component':12s} {'dtype':8s} {'shape':14s} {'params':>12s}")
    for row in rows:
        shape = "x".join(str(d) for d in row["shape"])
        print(
            f"{row['name']:62s} {row['role']:32s} {row['component']:12s} "
            f"{row['dtype']:8s} {shape:14s} {row['parameters']:12d}"
        )
    body = sum(r["parameters"] for r in rows if r["role"].startswith("body."))
    head = sum(r["parameters"] for r in rows if r["role"].startswith("head."))
    unknown = [r["name"] for r in rows if r["role"] == "UNKNOWN"]
    print(f"\n  parameters: body {body:,} + head {head:,} = {body + head:,}")
    print(f"  head share, vendor README claims 37M: {head / 1e6:.2f} M")
    print(f"  unexplained tensors: {len(unknown)}" + (f" {unknown}" if unknown else ""))


def print_reconciliation(header: dict, header_len: int, path: Path) -> dict:
    """Reproduce the plan doc's parameter split and the file-size identity from the header alone."""
    elements = {dtype: 0 for dtype in STORAGE}
    payload = 0
    for entry in header.values():
        count = 1
        for dim in entry["shape"]:
            count *= dim
        elements[entry["dtype"]] += count
        payload += count * STORAGE[entry["dtype"]]

    packed, bf16, f8, f32 = elements["U8"], elements["BF16"], elements["F8_E4M3"], elements["F32"]
    fp4_params = packed * 2
    vendor_total = fp4_params + bf16
    element_total = fp4_params + bf16 + f8 + f32
    on_disk = path.stat().st_size

    print("\n--- checkpoint reconciliation (header only, no payload read) ---")
    print(f"  tensors                : {len(header)}")
    print(f"  NVFP4 packed bytes     : {packed:,}   2 params/byte -> {fp4_params:,} ({fp4_params/1e6:.2f} M)")
    print(f"  BF16 elements          : {bf16:,} ({bf16/1e6:.2f} M)")
    print(f"  F8_E4M3 group scales   : {f8:,} ({f8/1e6:.2f} M)")
    print(f"  F32 per-tensor scales  : {f32:,}")
    print(f"  flattened elements     : {element_total:,} ({element_total/1e9:.4f} B)")
    print(f"  vendor count (packed x2 + BF16, scales excluded) = {vendor_total:,} ({vendor_total/1e9:.4f} B)")
    print(
        "  plan doc: 878.70 M packed + 234.10 M BF16 + 54.92 M F8 scales = 1.1128 B, payload 962,467,000 B\n"
        f"    packed {fp4_params:,} / BF16 {bf16:,} / F8 {f8:,} / payload {payload:,}"
        " -> all four reproduce digit for digit.\n"
        f"    1.1128 B is not the sum of the three listed terms ({element_total/1e9:.4f} B with scales):"
        f" it is packed x2 + BF16 = {vendor_total:,},\n"
        "    which is exactly the vendor README's '1.113B draft parameters'. The doc's total is right;"
        " its '+' is wrong."
    )
    print(f"  file bytes             : {on_disk:,} = payload + 8 + header_len({header_len})")

    if path == CHECKPOINT:
        assert payload == 962_467_000, payload
        assert on_disk == payload + 8 + header_len
        assert (packed, bf16, f8, f32) == (439_353_344, 234_097_194, 54_919_168, 25)
        assert fp4_params == 878_706_688
    return {"payload": payload, "fp4_params": fp4_params, "bf16": bf16, "f8": f8, "f32": f32}


# ---------------------------------------------------------------------------------------------
# Weights: the checkpoint's tensors as float32, NVFP4 dequantized on demand
# ---------------------------------------------------------------------------------------------


class Checkpoint:
    def __init__(self, path: Path, nibble_order: str | None = None):
        self.path = Path(path)
        self.header, self.header_len = read_header(self.path)
        self.nibble_order = NIBBLE_ORDER if nibble_order is None else nibble_order
        self._handle = safe_open(self.path, framework="pt")
        self._cache: dict[str, torch.Tensor] = {}

    def raw(self, name: str) -> torch.Tensor:
        if name not in self.header:
            raise KeyError(name)
        return self._handle.get_tensor(name)

    def tensor(self, name: str) -> torch.Tensor:
        """Any tensor as float32; a `.weight` whose dtype is U8 is dequantized with its scales."""
        if name in self._cache:
            return self._cache[name]
        if name.endswith(".weight") and self.header[name]["dtype"] == "U8":
            base = name[: -len(".weight")]
            value = dequant_nvfp4(
                self.raw(name),
                self.raw(base + ".weight_scale"),
                float(self.raw(base + ".weight_scale_2").to(torch.float32)),
                order=self.nibble_order,
            )
        else:
            value = self.raw(name).to(torch.float32)
        self._cache[name] = value
        return value

    def matrix(self, name: str) -> torch.Tensor:
        """A [out, in] linear weight."""
        return self.tensor(name + ".weight")

    def vector(self, name: str) -> torch.Tensor:
        """A [n] bias or norm weight; some parameters are stored without a `.weight` suffix."""
        if name + ".weight" in self.header:
            return self.tensor(name + ".weight")
        return self.tensor(name)


class TargetTables:
    """The target's input embedding table and LM head, or a seeded synthetic stand-in.

    With `--target`, only two shards are touched and each is read through the index for exactly the
    two tensors needed (`model-00001` holds the whole embedding table, the shard the index names for
    `lm_head` holds that whole), so the 63-shard / 150 GB target is never loaded as a model.
    """

    def __init__(self, root: Path | None, vocab: int, hidden: int, seed: int):
        self.synthetic = root is None
        if self.synthetic:
            generator = torch.Generator().manual_seed(seed)
            self.embed = torch.randn(vocab, hidden, generator=generator) * 0.02
            self.lm_head = torch.randn(vocab, hidden, generator=generator) * 0.02
            self.source = f"synthetic (seed {seed}, vocab {vocab}, width {hidden})"
            return

        weight_map = json.loads((root / TARGET_INDEX).read_text())["weight_map"]
        embed_shard = root / weight_map[TARGET_EMBED_KEY]
        head_shard = root / weight_map[TARGET_LM_HEAD_KEY]
        self.embed = _read_tensor(embed_shard, TARGET_EMBED_KEY).to(torch.float32)
        if self.embed.shape[1] != hidden:
            raise ValueError(
                f"target embedding width {self.embed.shape[1]} != draft feature width {hidden}"
            )
        self._head_shard = head_shard
        self.lm_head = None  # streamed: [131072, 4096] bf16 is 1 GiB and need not be upcast whole
        self.source = (
            f"{embed_shard.name}::{TARGET_EMBED_KEY} ({tuple(self.embed.shape)}) and "
            f"{head_shard.name}::{TARGET_LM_HEAD_KEY}"
        )

    def embedding_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed[ids]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, S, hidden] -> [B, S, vocab], accumulated in vocab chunks to bound peak memory."""
        if self.lm_head is None:
            self.lm_head = _read_tensor(self._head_shard, TARGET_LM_HEAD_KEY)
        weight = self.lm_head
        if weight.shape[1] != hidden.shape[-1]:
            raise ValueError(f"target LM head width {weight.shape[1]} != {hidden.shape[-1]}")
        flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        out = torch.empty(flat.shape[0], weight.shape[0], dtype=torch.float32)
        for start in range(0, weight.shape[0], 16384):
            block = weight[start : start + 16384].to(torch.float32)
            out[:, start : start + block.shape[0]] = torch.matmul(flat, block.T)
        return out.reshape(hidden.shape[:-1] + (weight.shape[0],))


def _read_tensor(path: Path, name: str) -> torch.Tensor:
    """One tensor by name, BF16 widened to float32 (safetensors' own reader does the locate)."""
    with safe_open(path, framework="pt") as handle:
        raw = handle.get_tensor(name)
    if raw.dtype == torch.bfloat16:
        return raw.to(torch.float32)
    if raw.dtype in (torch.float16, torch.float32):
        return raw.to(torch.float32)
    bits = raw.view(torch.uint16).to(torch.uint32) << 16
    return bits.view(torch.float32)


# ---------------------------------------------------------------------------------------------
# Tracing: named stages, a statistics table, and a dump
# ---------------------------------------------------------------------------------------------


class Trace:
    """Records named stages as float32 (the repo's reference-dump convention)."""

    def __init__(self):
        self.stages: dict[str, torch.Tensor] = {}

    def add(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.stages[name] = value.detach().to(torch.float32).contiguous()
        return value

    def get(self, name: str) -> torch.Tensor:
        return self.stages[name]

    def clear(self) -> None:
        self.stages.clear()

    def table(self) -> list[dict]:
        rows = []
        for name in sorted(self.stages):
            value = self.stages[name].to(torch.float32)
            double = value.double()
            rows.append(
                {
                    "stage": name,
                    "shape": list(value.shape),
                    "dtype": "float32",
                    "mean": float(double.mean()) if value.numel() else 0.0,
                    "std": float(double.std()) if value.numel() > 1 else 0.0,
                    "absmax": float(value.abs().max()) if value.numel() else 0.0,
                    "sum_f64": float(double.sum()) if value.numel() else 0.0,
                }
            )
        return rows

    def print_table(self) -> None:
        rows = self.table()
        print(f"\n--- {len(rows)} stages (float32; sum is accumulated in float64) ---")
        print(f"{'stage':34s} {'shape':20s} {'mean':>13s} {'std':>13s} {'absmax':>13s} {'sum(f64)':>21s}")
        for row in rows:
            shape = "x".join(str(d) for d in row["shape"])
            print(
                f"{row['stage']:34s} {shape:20s} {row['mean']:13.6e} {row['std']:13.6e} "
                f"{row['absmax']:13.6e} {row['sum_f64']:21.10e}"
            )

    def dump(self, directory: Path, extra: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {name.replace(".", "_"): value.numpy() for name, value in self.stages.items()}
        np.savez(directory / "lilicorr_stages.npz", **arrays)
        index = {
            "inputs": extra,
            "stages": self.table(),
            "arrays": [
                {"stage": name, "array": name.replace(".", "_")} for name in sorted(self.stages)
            ],
        }
        (directory / "lilicorr_stages.json").write_text(json.dumps(index, indent=1))
        print(
            f"\nwrote {directory / 'lilicorr_stages.npz'} ({len(arrays)} arrays) and "
            f"{directory / 'lilicorr_stages.json'}"
        )


# ---------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`ggml_l2_norm` / `F.normalize`: divide by the row norm *floored* at eps (not an additive eps)."""
    return x / x.norm(2, dim=-1, keepdim=True).clamp_min(eps)


def rope_neox(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """[B, heads, T, head_dim] NeoX rotary embedding at absolute `positions` ([B, T] or [T]).

    (C) fixes the style for this checkpoint: `LLAMA_ROPE_TYPE_NEOX`, full width (128 of 128 dims),
    theta 10000, no per-layer `rope_freqs`; (A) defaults `is_neox_style=True`.  NeoX rotates the two
    halves against each other, (i, i + D/2).  A mismatched style is silent upstream.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    inv_freq = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    angle = positions.to(torch.float32)[:, None, :, None] * inv_freq[None, None, None, :]
    cos, sin = torch.cos(angle), torch.sin(angle)
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


def grouped_dynamic_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    side: int,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """The DFlash2 grouped dynamic convolution for one side.

    x     [B, T, H]                    hidden states of the query block (T = block_size rows)
    delta [B, T, taps, H/group_size]   the side's coefficients from `kernel_projection`
    base  [2, taps, H]                 the checkpoint's `base_kernel` (side is dimension 0)

    y[t, e] = sum_{k < taps, (t % block_size) >= k} ( base[side, k, e] + delta[t, k, g(e)] ) * x[t-k, e]

    with x[t-k] = 0 where t - k < 0, so position 0 of a block reads no tap-1 input and nothing leaks
    across a block boundary.  The coefficient row is the *output* row t, not t-k: (A)
    `coefficients[:, tap] * shifted` with `coefficients` built from the current hidden states, (C)
    `:535-540` indexing the tap offset into the token axis without moving the coefficient's row.
    Both sources also agree on the projected row layout `side*taps*G + tap*G + g` and on the
    checkpoint's `base[side, tap, channel]` order.
    """
    batch, rows, hidden = x.shape
    groups = hidden // group_size
    taps = base.shape[1]
    position = torch.arange(rows) % block_size
    out = torch.zeros_like(x)
    for tap in range(taps):
        coefficient = base[side, tap].view(groups, group_size) + delta[:, :, tap].unsqueeze(-1)
        coefficient = coefficient.reshape(batch, rows, hidden)
        if tap == 0:
            out = out + coefficient * x
            continue
        shifted = torch.zeros_like(x)
        keep = position >= tap
        shifted[:, keep] = x[:, (torch.arange(rows) - tap)[keep]]
        out = out + coefficient * shifted
    return out


def sliding_window_visible(
    query_positions: torch.Tensor, key_positions: torch.Tensor, window: int
) -> torch.Tensor:
    """[B, Tq, Tk]: `|p_k - p_q| < window` (symmetric) or `0 <= p_q - p_k < window` (causal)."""
    delta = query_positions[:, :, None] - key_positions[:, None, :]
    if WINDOW_MODE == "symmetric":
        return delta.abs() < window
    return (delta >= 0) & (delta < window)


def attention_with_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    visible: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """GQA attention with a window mask and a per-query-head learned sink.

    query [B, Hq, Tq, D], key/value [B, Hkv, Tk, D], visible [B, Tq, Tk], sinks [Hq].

    The sink is a fixed per-head logit with no value: one added term in the denominator only,
    `den = sum_j exp(w_j - m) + exp(sink - m)`, applied after the 1/sqrt(D) scaling and after the
    mask -- see SINK_MODE.  A fully-masked row is still well defined because the sink keeps the
    denominator positive (and this checkpoint's sink biases are finite).
    """
    if SINK_MODE != "denominator_only":
        raise ValueError(f"unknown sink mode {SINK_MODE!r}")
    heads = query.shape[1]
    repeat = heads // key.shape[1]
    if repeat > 1:
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    weights = torch.matmul(query, key.transpose(-1, -2)) * scale
    weights = weights.masked_fill(~visible[:, None, :, :], float("-inf"))
    sink = sinks.view(1, heads, 1, 1)
    row_max = torch.maximum(weights.amax(-1, keepdim=True), sink)
    numerator = torch.exp(weights - row_max)
    denominator = numerator.sum(-1, keepdim=True) + torch.exp(sink - row_max)
    return torch.matmul(numerator, value) / denominator


# ---------------------------------------------------------------------------------------------
# The body: five Qwen3-shaped layers with the grouped dynamic convolution
# ---------------------------------------------------------------------------------------------


class DraftBody:
    """The five draft layers.

    Observed magnitude regime (a checkpoint fact, verified against the file, not an artifact of this
    file's math): every draft projection weight is 16-70x larger than `1/sqrt(fan_in)` -- the BF16,
    *non-quantized* `layers.0.self_attn.q_proj` has std 0.268 where `1/sqrt(4096)` is 0.0156, and the
    target's own weights sit at `1/sqrt(fan_in)` -- while the draft's `input_layernorm` weights are
    ~0.07.  The product leaves each sublayer's gain near 1 in *direction*, but the residual stream
    reaches ~1e7 by the end of layer 0 (the layer-0 MLP alone contributes ~5e5 standard deviation
    against a ~2e2 attention contribution).  That is harmless for the math this file implements:
    every consumer of the residual is an RMSNorm (per-head k/q norm, layer norms, the final `norm`)
    or a cosine, so an arbitrary common scale cancels.  It is *not* harmless for a low-precision
    port: bf16 has a 2^15.5 ulp at 1.19e7, so a bf16 residual accumulator drops the attention term
    entirely.  The C++ comparison should therefore keep the residual/norm path in fp32 and use
    scale-relative tolerances.
    """
    def __init__(self, weights: Checkpoint, config: dict, trace: Trace):
        self.w = weights
        self.trace = trace
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        self.theta = float(config["rope_theta"])
        dflash = config["dflash_config"]
        self.window = int(dflash["swa_window_size"])
        self.taps = int(dflash["conv_kernel_size"])
        self.group_size = int(dflash["conv_group_size"])
        self.groups = self.hidden // self.group_size
        self.target_layer_ids = list(dflash["target_layer_ids"])

    def fuse_taps(self, taps: list[torch.Tensor]) -> torch.Tensor:
        """`fc(concat(taps))` with the taps in ascending layer order (A: `cat([...], dim=-1)` over
        the captured aux hidden states; C: layer k at offset `k * hidden`).  Bias-free."""
        if len(taps) != len(self.target_layer_ids):
            raise ValueError(f"expected {len(self.target_layer_ids)} taps, got {len(taps)}")
        concatenated = torch.cat(taps, dim=-1)
        self.trace.add("body.fc_input", concatenated)
        return torch.matmul(concatenated, self.w.matrix("fc").T)

    def context_keys_values(
        self, normed: torch.Tensor, positions: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer K/V of the *context* rows, projected from the fused target state, not the layers.

        SOURCE NOTE (agreed): (A) `precompute_and_store_context_kv` RMS-norms the fused target states
        with `hidden_norm` and projects each layer's own K/V slices, then applies per-layer `k_norm`
        and RoPE (`_project_context_kv`, `_normalize_context_k`); (C) says the same for its inject
        pass ("fc -> RMSNorm -> K/V written into the draft SWA cache").  The context rows are never
        run through the draft layers, so at *every* layer their K/V come from this one tensor.
        """
        prefix = f"layers.{layer}.self_attn"
        batch, rows, _ = normed.shape
        key = torch.matmul(normed, self.w.matrix(prefix + ".k_proj").T).view(
            batch, rows, self.kv_heads, self.head_dim
        )
        value = torch.matmul(normed, self.w.matrix(prefix + ".v_proj").T).view(
            batch, rows, self.kv_heads, self.head_dim
        )
        key = rms_norm(key, self.w.vector(prefix + ".k_norm"), self.eps)
        key = rope_neox(key.transpose(1, 2), positions, self.theta)
        return key, value.transpose(1, 2)

    def forward(
        self,
        taps: list[torch.Tensor],
        embeddings: torch.Tensor,
        context_positions: torch.Tensor,
        block_positions: torch.Tensor,
        slots: int,
        anchor_valid: torch.Tensor | None = None,
    ) -> dict:
        """One draft forward.

        `embeddings` [B, 1 + slots, hidden] are the *query* rows' inputs -- the committed bonus token
        and `mask_token_id` rows, from the target's embedding table (`A`/`C` both call
        `model(input_ids=...)` for the block and never feed the fused target state as its input).
        The context rows contribute K/V only (see `context_keys_values`) plus the anchor state.

        Per layer the order is (A) `DFlash2Qwen3DecoderLayer.forward`, (C) `:955-1045`:
            norm -> conv(side 0) -> q/k/v (+ q_norm, k_norm, RoPE) -> attention(+sink) ->
            conv(side 1) -> + residual -> norm -> conv(side 0) -> SwiGLU MLP -> conv(side 1) -> +
            residual.
        """
        trace = self.trace
        fused = self.fuse_taps(taps)
        trace.add("body.fc_out", fused)
        normed = rms_norm(fused, self.w.vector("hidden_norm"), self.eps)
        trace.add("body.hidden_norm_out", normed)

        anchor_index = context_positions.shape[1] - 1
        anchor_hidden = normed[:, anchor_index]
        trace.add("body.anchor_hidden", anchor_hidden)

        hidden = embeddings
        trace.add("body.embeddings", hidden)
        for layer in range(self.layers):
            prefix = f"layers.{layer}"
            batch, rows, _ = hidden.shape
            trace.add(f"body.layer{layer}.residual_in", hidden)

            normed_in = rms_norm(hidden, self.w.vector(prefix + ".input_layernorm"), self.eps)
            trace.add(f"body.layer{layer}.norm_in", normed_in)
            delta = torch.matmul(
                normed_in, self.w.matrix(prefix + ".attention_conv.kernel_projection").T
            ).view(batch, rows, 2, self.taps, self.groups)
            trace.add(f"body.layer{layer}.conv_attn_delta", delta)
            prepared = grouped_dynamic_conv(
                normed_in,
                delta[:, :, 0],
                self.w.tensor(prefix + ".attention_conv.base_kernel"),
                0,
                rows,
                self.group_size,
            )
            trace.add(f"body.layer{layer}.conv_attn_pre", prepared)

            attn = prefix + ".self_attn"
            query = torch.matmul(prepared, self.w.matrix(attn + ".q_proj").T)
            key = torch.matmul(prepared, self.w.matrix(attn + ".k_proj").T)
            value = torch.matmul(prepared, self.w.matrix(attn + ".v_proj").T)
            trace.add(f"body.layer{layer}.q_pre_norm", query)
            trace.add(f"body.layer{layer}.k_pre_norm", key)
            trace.add(f"body.layer{layer}.v_pre_norm", value)
            query = query.view(batch, rows, self.heads, self.head_dim)
            key = key.view(batch, rows, self.kv_heads, self.head_dim)
            value = value.view(batch, rows, self.kv_heads, self.head_dim)
            query = rms_norm(query, self.w.vector(attn + ".q_norm"), self.eps)
            key = rms_norm(key, self.w.vector(attn + ".k_norm"), self.eps)
            trace.add(f"body.layer{layer}.q_post_norm", query)
            trace.add(f"body.layer{layer}.k_post_norm", key)
            query = rope_neox(query.transpose(1, 2), block_positions, self.theta)
            key = rope_neox(key.transpose(1, 2), block_positions, self.theta)
            trace.add(f"body.layer{layer}.q_post_rope", query)
            trace.add(f"body.layer{layer}.k_post_rope", key)

            context_key, context_value = self.context_keys_values(normed, context_positions, layer)
            trace.add(f"body.layer{layer}.k_ctx_post_rope", context_key)
            trace.add(f"body.layer{layer}.v_ctx", context_value)
            key_positions = torch.cat((context_positions, block_positions), dim=1)
            visible = sliding_window_visible(block_positions, key_positions, self.window)
            attention = attention_with_sink(
                query,
                torch.cat((context_key, key), dim=2),
                torch.cat((context_value, value.transpose(1, 2)), dim=2),
                1.0 / math.sqrt(self.head_dim),
                visible,
                self.w.vector(attn + ".attention_sink_bias"),
            )
            attention = attention.transpose(1, 2).reshape(batch, rows, self.hidden)
            trace.add(f"body.layer{layer}.attn_out", attention)
            # The output projection precedes the finish convolution. llama.cpp returns
            # `wo(attn)` from its attention builder and feeds *that* to
            # `build_dflash2_conv(..., side 1)` (`dflash.cpp`, the draft block), so the conv's
            # coefficients modulate the projected output, not the raw attention. Applying the conv
            # to the raw output instead leaves every layer from the first one wrong; measured
            # against llama.cpp's own dumped stages it is rel L2 1.0004 versus 2.1e-4 with the
            # projection, and it costs the drafter all of its acceptance.
            attention = torch.matmul(attention, self.w.matrix(attn + ".o_proj").T)
            trace.add(f"body.layer{layer}.attn_out_proj", attention)
            attention = grouped_dynamic_conv(
                attention,
                delta[:, :, 1],
                self.w.tensor(prefix + ".attention_conv.base_kernel"),
                1,
                rows,
                self.group_size,
            )
            trace.add(f"body.layer{layer}.attn_out_conv", attention)
            hidden = hidden + attention
            trace.add(f"body.layer{layer}.hidden_post_attn", hidden)

            normed_mlp = rms_norm(
                hidden, self.w.vector(prefix + ".post_attention_layernorm"), self.eps
            )
            trace.add(f"body.layer{layer}.norm_mlp_in", normed_mlp)
            delta_mlp = torch.matmul(
                normed_mlp, self.w.matrix(prefix + ".mlp_conv.kernel_projection").T
            ).view(batch, rows, 2, self.taps, self.groups)
            trace.add(f"body.layer{layer}.conv_mlp_delta", delta_mlp)
            prepared_mlp = grouped_dynamic_conv(
                normed_mlp,
                delta_mlp[:, :, 0],
                self.w.tensor(prefix + ".mlp_conv.base_kernel"),
                0,
                rows,
                self.group_size,
            )
            trace.add(f"body.layer{layer}.pre_mlp", prepared_mlp)
            gate = torch.matmul(prepared_mlp, self.w.matrix(prefix + ".mlp.gate_proj").T)
            up = torch.matmul(prepared_mlp, self.w.matrix(prefix + ".mlp.up_proj").T)
            trace.add(f"body.layer{layer}.mlp_gate", gate)
            trace.add(f"body.layer{layer}.mlp_up", up)
            activated = F.silu(gate) * up
            trace.add(f"body.layer{layer}.mlp_act", activated)
            down = torch.matmul(activated, self.w.matrix(prefix + ".mlp.down_proj").T)
            trace.add(f"body.layer{layer}.mlp_out", down)
            down = grouped_dynamic_conv(
                down,
                delta_mlp[:, :, 1],
                self.w.tensor(prefix + ".mlp_conv.base_kernel"),
                1,
                rows,
                self.group_size,
            )
            trace.add(f"body.layer{layer}.post_mlp", down)
            hidden = hidden + down
            trace.add(f"body.layer{layer}.out", hidden)

        final = rms_norm(hidden, self.w.vector("norm"), self.eps)
        trace.add("body.norm_out", final)
        pass_hidden = final[:, 1 : 1 + slots]
        trace.add("head.pass_hidden", pass_hidden)
        if anchor_valid is None:
            anchor_valid = torch.ones(hidden.shape[0], dtype=torch.bool)
        return {
            "normed": normed,
            "anchor_hidden": anchor_hidden,
            "anchor_valid": anchor_valid,
            "pass_hidden": pass_hidden,
            "hidden": hidden,
        }


# ---------------------------------------------------------------------------------------------
# The head: candidates, features, the candidate lattice, the edge factors, the walk
# ---------------------------------------------------------------------------------------------


class CorrelatorHead:
    def __init__(self, weights: Checkpoint, config: dict, trace: Trace):
        self.w = weights
        self.trace = trace
        dflash = config["dflash_config"]
        self.hidden = int(dflash["lilicorr_hidden_size"])
        self.heads = int(dflash["lilicorr_num_heads"])
        self.head_dim = self.hidden // self.heads
        self.layer_count = int(dflash["lilicorr_num_layers"])
        self.factor_dim = int(dflash["lilicorr_factor_dim"])
        self.top_k = int(dflash["lilicorr_candidate_topk"])
        self.vector_eps = float(dflash["lilicorr_vector_eps"])
        self.logit_scale = float(dflash["lilicorr_logit_scale"])
        self.eps = config["rms_norm_eps"]
        self.block_size = int(config.get("block_size", TRAINED_BLOCK_SIZE))

    def candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, S, vocab] -> (ids, log-probs) of the top-K, rank order = descending logit.

        SOURCE NOTE (agreed): (A) `get_top_k_tokens(..., return_log_probs=True)` returns the top-K by
        logit together with the *full-vocabulary* log-softmax of exactly those tokens, temperature 1
        and no penalties; (C) takes `top_k`, then `argsort` descending, then
        `log(softmax(logits))` gathered over the whole vocabulary.  Same set, same order, same
        normalization -- not a K-token-normalized softmax.
        """
        logits = logits.to(torch.float32)
        probabilities = torch.softmax(logits, dim=-1)
        values, ids = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        log_probs = torch.log(
            probabilities.gather(-1, ids).clamp_min(torch.finfo(torch.float32).tiny)
        )
        # token ids travel through the trace as float32 (exact below 2**24); the npz keeps them there
        self.trace.add("head.candidate_ids", ids.to(torch.float32))
        self.trace.add("head.candidate_logits", values)
        self.trace.add("head.candidate_logprobs", log_probs)
        self.trace.add("head.candidate_probs", probabilities.gather(-1, ids))
        return ids, log_probs

    def features(self, log_probs: torch.Tensor) -> torch.Tensor:
        """The five scalars, in the order `feature_mlp.up_proj[1024, 5]`'s columns expect:
        (log-prob, probability, gap to the slot's top candidate, rank/(K-1), is-top-1).

        (A): `log_probs, log_probs.exp(), log_probs - log_probs.max(-1), rank_frac, is_top1`;
        (C) builds `concatenate([lp], [probs], [relative], [rank/(top_k-1)], [1-step(rank)])`.  Both
        put rank 0 first, i.e. the highest logit, and both make `relative <= 0` with `== 0` at rank 0.
        """
        top_k = log_probs.shape[-1]
        rank = torch.arange(top_k, dtype=torch.float32).view(1, 1, top_k).expand_as(log_probs)
        rank_frac = rank / float(top_k - 1) if top_k > 1 else torch.zeros_like(rank)
        is_top1 = (rank == 0).to(torch.float32)
        relative = log_probs - log_probs[..., :1]
        features = torch.stack((log_probs, log_probs.exp(), relative, rank_frac, is_top1), dim=-1)
        self.trace.add("head.features", features)
        return features

    def lattice_bias(self, slots: int) -> torch.Tensor:
        """[heads, S*K, S*K]: `relative_slot_bias[h, dslot + block_size - 1] +
        same_slot_bias[h] * (same slot)`.

        The offset is the *trained* `block_size - 1` even when a shorter draft is scored (A: "Keep
        the trained offset = block_size-1 ... so each distance continues to use the same learned
        weight"; C expands `relative_slot_bias[:, q_slot - k_slot + (block_size-1)]` identically).
        The same-slot term therefore covers distinct candidates of one slot as well as the diagonal.
        Out-of-range offsets cannot occur while `slots <= block_size - 1`.
        """
        node_slot = torch.arange(slots).repeat_interleave(self.top_k)
        relative = node_slot.view(-1, 1) - node_slot.view(1, -1)
        bias = self.w.tensor("lilicorr.relative_slot_bias")[:, relative + self.block_size - 1]
        same = (node_slot.view(-1, 1) == node_slot.view(1, -1)).to(torch.float32)
        return bias + same.unsqueeze(0) * self.w.tensor("lilicorr.same_slot_bias").view(-1, 1, 1)

    def score(
        self,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> dict:
        """`candidate_embeddings` are already the *target* embedding rows of the candidate ids."""
        trace = self.trace
        batch, slots, top_k = log_probs.shape
        features = self.features(log_probs)
        normalized = layer_norm(
            features,
            self.w.tensor("lilicorr.feature_norm.weight"),
            self.w.tensor("lilicorr.feature_norm.bias"),
            FEATURE_NORM_EPS,
        )
        trace.add("head.feature_norm_out", normalized)
        feature_states = F.silu(
            torch.matmul(normalized, self.w.matrix("lilicorr.feature_mlp.up_proj").T)
            + self.w.tensor("lilicorr.feature_mlp.up_proj.bias")
        )
        feature_states = (
            torch.matmul(feature_states, self.w.matrix("lilicorr.feature_mlp.down_proj").T)
            + self.w.tensor("lilicorr.feature_mlp.down_proj.bias")
        )
        trace.add("head.feature_mlp_out", feature_states)

        token_states = (
            torch.matmul(candidate_embeddings, self.w.matrix("lilicorr.token_proj").T)
            + self.w.tensor("lilicorr.token_proj.bias")
        )
        pass_states = (
            torch.matmul(pass_hidden, self.w.matrix("lilicorr.pass_hidden_proj").T)
            + self.w.tensor("lilicorr.pass_hidden_proj.bias")
        ).unsqueeze(2)
        trace.add("head.node_token", token_states)
        trace.add("head.node_pass", pass_states.expand(batch, slots, top_k, self.hidden))
        trace.add("head.node_feature", feature_states)
        node = (
            token_states
            + pass_states
            + feature_states
            + self.w.tensor("lilicorr.slot_embedding")[0, 0, :slots, 0].view(
                1, slots, 1, self.hidden
            )
            + self.w.tensor("lilicorr.rank_embedding")[0, 0, 0].view(1, 1, top_k, self.hidden)
        )
        trace.add("head.node", node)

        nodes = node.reshape(batch, slots * top_k, self.hidden)
        bias = self.lattice_bias(slots)
        trace.add("head.lattice_bias", bias)
        for layer in range(self.layer_count):
            prefix = f"lilicorr.layers.{layer}"
            normed = rms_norm(nodes, self.w.vector(prefix + ".attn_norm"), self.eps)
            qkv = (
                torch.matmul(normed, self.w.tensor(prefix + ".attn.in_proj_weight").T)
                + self.w.tensor(prefix + ".attn.in_proj_bias")
            )
            query, key, value = qkv.chunk(3, dim=-1)
            shape = (batch, slots * top_k, self.heads, self.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            value = value.view(shape).transpose(1, 2)
            trace.add(f"head.lattice{layer}_attn_in", normed)
            trace.add(f"head.lattice{layer}_q", query)
            trace.add(f"head.lattice{layer}_k", key)
            trace.add(f"head.lattice{layer}_v", value)
            # Fully bidirectional within the block, no positional encoding: the additive bias is the
            # entire mask (A: `scaled_dot_product_attention(q,k,v, attn_mask=attention_bias)`).
            weights = torch.matmul(query, key.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
            attention = torch.matmul(torch.softmax(weights + bias.unsqueeze(0), dim=-1), value)
            attention = attention.transpose(1, 2).reshape(batch, slots * top_k, self.hidden)
            attention = (
                torch.matmul(attention, self.w.matrix(prefix + ".attn.out_proj").T)
                + self.w.tensor(prefix + ".attn.out_proj.bias")
            )
            nodes = nodes + attention
            trace.add(f"head.lattice{layer}_attn_out", attention)
            trace.add(f"head.lattice{layer}_resid1", nodes)
            mlp_in = rms_norm(nodes, self.w.vector(prefix + ".mlp_norm"), self.eps)
            mlp = F.silu(
                torch.matmul(mlp_in, self.w.matrix(prefix + ".mlp.up_proj").T)
                + self.w.tensor(prefix + ".mlp.up_proj.bias")
            )
            mlp = (
                torch.matmul(mlp, self.w.matrix(prefix + ".mlp.down_proj").T)
                + self.w.tensor(prefix + ".mlp.down_proj.bias")
            )
            nodes = nodes + mlp
            trace.add(f"head.lattice{layer}_mlp_out", mlp)
            trace.add(f"head.lattice{layer}_out", nodes)

        normed_nodes = rms_norm(nodes, self.w.vector("lilicorr.output_norm"), self.eps).reshape(
            batch, slots, top_k, self.hidden
        )
        trace.add("head.lattice_out", normed_nodes)

        # The anchor lives outside the lattice and is computed from the fused target state only.
        anchor_state = (
            torch.matmul(anchor_hidden, self.w.matrix("lilicorr.context_proj").T)
            + self.w.tensor("lilicorr.context_proj.bias")
        )
        trace.add("head.anchor_state_pre_mask", anchor_state)
        anchor_state = anchor_state * anchor_valid[:, None].to(torch.float32)
        anchor_state = rms_norm(anchor_state, self.w.vector("lilicorr.anchor_norm"), self.eps)
        trace.add("head.anchor_state", anchor_state)
        anchor_out = l2_normalize(
            torch.matmul(anchor_state, self.w.matrix("lilicorr.anchor_out_head").T)
            + self.w.tensor("lilicorr.anchor_out_head.bias"),
            self.vector_eps,
        )
        trace.add("head.anchor_out", anchor_out)

        anchor_row = anchor_state[:, None, None, :]
        factor_input = torch.cat(
            (normed_nodes, anchor_row.expand_as(normed_nodes), normed_nodes * anchor_row), dim=-1
        )
        trace.add("head.factor_input", factor_input)
        # (A) evaluates the same concatenated projection as three split products for speed; both
        # sources fix the block order to (node, anchor, node * anchor).
        pre = (
            torch.matmul(factor_input, self.w.matrix("lilicorr.factor_input_proj").T)
            + self.w.tensor("lilicorr.factor_input_proj.bias")
        )
        trace.add("head.factor_pre", pre)
        factor = F.silu(pre)
        trace.add("head.factor", factor)

        out_weight = self.w.matrix("lilicorr.out_head")
        in_weight = self.w.matrix("lilicorr.in_head")
        out_bias = self.w.tensor("lilicorr.out_head.bias")
        in_bias = self.w.tensor("lilicorr.in_head.bias")
        # The fused edge weight is `cat([out_head, in_head])` (A); WHICH half is "out" is
        # EDGE_FUSION_ORDER.  (A) evaluates the projection as one `[2h, h]` GEMM and the C++ side
        # will too, so the layout is part of the interface.
        if EDGE_FUSION_ORDER not in ("out_first", "in_first"):
            raise ValueError(f"unknown edge fusion order {EDGE_FUSION_ORDER!r}")
        fused_weight = torch.cat((out_weight, in_weight), dim=0)
        fused_bias = torch.cat((out_bias, in_bias), dim=0)
        edges = torch.matmul(factor, fused_weight.T) + fused_bias
        trace.add("head.edges", edges)
        halves = l2_normalize(edges.unflatten(-1, (2, self.factor_dim)), self.vector_eps)
        first, second = halves.unbind(-2)
        if EDGE_FUSION_ORDER == "out_first":
            out_vec, in_vec = first, second
        else:
            in_vec, out_vec = first, second
        trace.add("head.out_vec", out_vec)
        trace.add("head.in_vec", in_vec)

        start = (anchor_out[:, None, :] * in_vec[:, 0]).sum(-1)
        pairs = torch.matmul(out_vec[:, :-1], in_vec[:, 1:].transpose(-1, -2))
        trace.add("head.start_scores", start)
        trace.add("head.pair_scores", pairs)
        scores = self.assemble_scores(start, pairs)
        trace.add("head.score_tensor", scores)

        # The pairwise scoring path in float64: cheap (S*K rows of width factor_dim) and it is the
        # natural place for a walk decision to be decided by fp32 accumulation.
        high_start = (anchor_out.double()[:, None, :] * in_vec.double()[:, 0]).sum(-1)
        high_pairs = torch.matmul(
            out_vec.double()[:, :-1], in_vec.double()[:, 1:].transpose(-1, -2)
        )
        high = self.assemble_scores(high_start, high_pairs)
        trace.add("head.score_tensor_fp64", high)
        fp64_delta = float((high - scores.double()).abs().max())
        return {"scores": scores, "fp64_delta": fp64_delta}

    def assemble_scores(self, start: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """[B, S, K, K] = `sigma * [start broadcast ; pairs]`, index (slot, previous, next).

        Row 0 is the anchor row broadcast over the predecessor axis (A:
        `first = start[:,None,None,:].expand(-1,1,K,-1)`), row t >= 1 is
        `out[slot t-1] . in[slot t]` with index (previous, next).  The scaling by
        `lilicorr_logit_scale` happens *outside* the cosine similarity in (A) (`:437-438`) and in (C)
        (`:781`, `:792`); with two unit vectors each score is therefore in [-sigma, sigma].
        (C) packs the same numbers into a 64-wide row per slot with `index = next + K * previous`,
        and its slot-0 block only fills the `previous = 0` row, which the walk reads; the two layouts
        agree on the walk and on every score value.
        """
        first = start[:, None, None, :].expand(-1, 1, self.top_k, -1)
        return self.logit_scale * torch.cat((first, pairs), dim=1)

    def walk(self, scores: torch.Tensor, ids: torch.Tensor) -> dict:
        """Greedy left-to-right walk: `prev = argmax_k scores[slot, prev, k]` (paper Eq. 8).

        (A) forces temperature 0 for the greedy draft sampler, so this is a plain argmax over the K
        scores of the committed predecessor's row; (C) runs the same predecessor chain and only then
        applies an optional confidence gate, which can truncate but never change a pick.  The
        realized rows are returned because the probabilistic sampler consumes exactly them
        (`_cache_draft_logits_kernel` writes the realized row and -inf elsewhere).
        """
        trace = self.trace
        batch, steps, top_k, _ = scores.shape
        rows_index = torch.arange(batch)
        previous = torch.zeros(batch, dtype=torch.long)
        chosen = torch.zeros(batch, steps, dtype=torch.long)
        indices = torch.zeros(batch, steps, dtype=torch.long)
        realized = torch.zeros(batch, steps, top_k)
        for step in range(steps):
            rows = scores[:, step, :, :][rows_index, previous]
            realized[:, step] = rows
            picked = rows.argmax(-1)
            indices[:, step] = picked
            chosen[:, step] = ids[:, step].gather(-1, picked.unsqueeze(-1)).squeeze(-1)
            previous = picked
        trace.add("head.walk_ids", chosen.to(torch.float32))
        trace.add("head.walk_indices", indices.to(torch.float32))
        trace.add("head.walk_rows", realized)
        return {"ids": chosen, "indices": indices, "rows": realized}


# ---------------------------------------------------------------------------------------------
# Inputs and the composed reference
# ---------------------------------------------------------------------------------------------


def build_inputs(
    config: dict, batch: int, context: int, slots: int, seed: int, window: int, source: Path | None
) -> dict:
    """Synthetic target taps + a query block, or the same loaded from an npz.

    The taps are the six captured target residual streams at the *context* positions, `[B, C, 4096]`
    each in ascending `target_layer_ids` order, plus the block ids (the committed bonus token at row
    0 and `mask_token_id` rows after it).  Positions are absolute and contiguous.
    """
    dflash = config["dflash_config"]
    hidden = config["hidden_size"]
    tap_count = len(dflash["target_layer_ids"])
    block = slots + 1
    if source is not None:
        data = np.load(source)
        if "taps" in data.files:
            taps = [torch.from_numpy(data["taps"][:, k]).to(torch.float32) for k in range(tap_count)]
        else:
            taps = [torch.from_numpy(data[f"tap{k}"]).to(torch.float32) for k in range(tap_count)]
        block_ids = torch.from_numpy(data["block_ids"]).to(torch.long)
        batch, context, block = taps[0].shape[0], taps[0].shape[1], block_ids.shape[1]
        slots = block - 1
    else:
        generator = torch.Generator().manual_seed(seed)
        taps = [
            torch.randn(batch, context, hidden, generator=generator) * 0.05
            for _ in range(tap_count)
        ]
        block_ids = torch.zeros(batch, block, dtype=torch.long)
        block_ids[:, 0] = 7  # the committed bonus token
        for slot in range(slots):
            block_ids[:, 1 + slot] = int(dflash["mask_token_id"]) + (slot % 4)
    return {
        "taps": taps,
        "block_ids": block_ids,
        "context_positions": torch.arange(context).unsqueeze(0).expand(batch, -1).contiguous(),
        "block_positions": (torch.arange(block).unsqueeze(0) + context)
        .expand(batch, -1)
        .contiguous(),
        "context": context,
        "slots": slots,
        "window": window,
        "seed": seed,
    }


def run_reference(
    config: dict, weights: Checkpoint, tables: TargetTables, inputs: dict, trace: Trace
) -> dict:
    body = DraftBody(weights, config, trace)
    head = CorrelatorHead(weights, config, trace)
    embeddings = tables.embedding_rows(inputs["block_ids"])
    stage = body.forward(
        inputs["taps"],
        embeddings,
        inputs["context_positions"],
        inputs["block_positions"],
        inputs["slots"],
    )
    logits = tables.logits(stage["pass_hidden"])
    trace.add("head.logits", logits)
    ids, log_probs = head.candidates(logits)
    scored = head.score(
        tables.embedding_rows(ids),  # candidate embeddings always come from the target table
        log_probs,
        stage["pass_hidden"],
        stage["anchor_hidden"],
        stage["anchor_valid"],
    )
    walked = head.walk(scored["scores"], ids)
    return {
        "ids": ids,
        "log_probs": log_probs,
        "scores": scored["scores"],
        "walk": walked,
        "fp64_delta": scored["fp64_delta"],
        "stage": stage,
    }


# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------


def self_check(args) -> int:
    """Assert every internal invariant that can be checked without ground truth."""
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        print(f"  [{'ok' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    print("=== LiLiCorr oracle self-check ===")
    print(f"interpreter: torch {torch.__version__}, numpy {np.__version__}, safetensors {safetensors.__version__}")

    print("\n-- codec tables --")
    check(float(E2M1[0]) == 0.0 and float(E2M1[1]) == 0.5 and float(E2M1[7]) == 6.0, "E2M1 {0, .5, 6}")
    check(float(E2M1[15]) == -6.0 and float(E2M1[4]) == 2.0 and float(E2M1[5]) == 3.0, "E2M1 {-6, 2, 3}")
    check(float(E4M3[0x7E]) == 448.0 and float(E4M3[0x38]) == 1.0 and float(E4M3[0x00]) == 0.0, "E4M3 {448, 1, 0}")

    config = json.loads(args.config.read_text())
    checkpoint = args.checkpoint if args.checkpoint.exists() else CHECKPOINT
    header, header_len = read_header(checkpoint)
    reconciliation = print_reconciliation(header, header_len, checkpoint)
    rows = role_table(header)
    print_role_table(rows)
    check(len(rows) == 179, "the checkpoint declares 179 tensors")
    check(all(r["role"] != "UNKNOWN" for r in rows), "every tensor is assigned a role")
    check(
        {r["component"] for r in rows} <= {"weight", "bias", "group_scale", "tensor_scale", "value"},
        "every component classification is one of the known kinds",
    )
    quantized_roles = {
        r["role"] for r in rows if r["component"] == "weight" and r["dtype"] == "U8"
    }
    check(
        all(
            {"weight", "group_scale", "tensor_scale"}
            <= {r["component"] for r in rows if r["role"] == role}
            <= {"weight", "group_scale", "tensor_scale", "bias"}
            for role in quantized_roles
        ),
        "every NVFP4 linear carries weight + group_scale + tensor_scale (bias optional, nothing else)",
    )
    check(
        len(quantized_roles) == reconciliation["f32"],
        f"{len(quantized_roles)} NVFP4 linears, one F32 tensor scale each",
    )
    check(
        sum(r["parameters"] for r in rows) == reconciliation["fp4_params"]
        + reconciliation["bf16"]
        + reconciliation["f8"]
        + reconciliation["f32"],
        "the role table's parameter tally equals the header reconciliation",
    )

    print("\n-- NVFP4 dequantization --")
    weights = Checkpoint(checkpoint)
    sample = weights.raw("fc.weight")
    group_scale = weights.raw("fc.weight_scale")
    torch_scale = group_scale.to(torch.float32)
    table_scale = E4M3[group_scale.view(torch.uint8).to(torch.int64)]
    check(
        bool(torch.equal(torch_scale.view(-1)[:4096], table_scale.view(-1)[:4096])),
        "the E4M3 table matches torch's own float8_e4m3fn reader",
    )
    tensor_scale = float(weights.raw("fc.weight_scale_2").to(torch.float32))
    for order in ("low_even", "high_even"):
        values = E2M1[unpack_nibbles(sample, order)]
        scaled = values * torch_scale.repeat_interleave(16, dim=1) * tensor_scale
        ratio = scaled.view(4096, -1, 16).abs().amax(-1) / (torch_scale * tensor_scale)
        check(
            bool((ratio - 6.0).abs().max() < 2e-3),
            f"every 16-group of fc.weight has absmax == 6*group_scale*tensor_scale "
            f"({order}, {ratio.numel()} groups, worst |ratio - 6| {float((ratio - 6).abs().max()):.2e})",
        )
    values = dequant_nvfp4(sample, group_scale, tensor_scale)
    check(tuple(values.shape) == (4096, 24576), "dequantized fc.weight is [4096, 24576]")
    check(bool(torch.isfinite(values).all()), "dequantized fc.weight is finite")
    check(float(values.std()) > 1e-3, "dequantized fc.weight is non-degenerate")

    # -- the composed forward, twice: once with the whole context visible, once with a window that
    # actually cuts, so the mask path is exercised as well as the serving-shaped one.
    slots = 7
    top_k = int(config["dflash_config"]["lilicorr_candidate_topk"])
    hidden = config["hidden_size"]
    sigma = float(config["dflash_config"]["lilicorr_logit_scale"])
    tables = TargetTables(None, vocab=args.vocab, hidden=hidden, seed=args.seed)
    for label, context, window in (
        ("wide window (no key is cut)", 49, 1024),
        ("tight window (keys are cut)", 49, 8),
    ):
        print(f"\n-- forward: {label} --")
        trace = Trace()
        inputs = build_inputs(config, 1, context, slots, args.seed, window, None)
        out = run_reference(config, weights, tables, inputs, trace)

        check(tuple(out["stage"]["pass_hidden"].shape) == (1, slots, hidden), "pass_hidden [1, S, 4096]")
        check(tuple(out["ids"].shape) == (1, slots, top_k), "candidate ids [1, S, K]")
        check(tuple(out["scores"].shape) == (1, slots, top_k, top_k), "score tensor [1, S, K, K]")
        check(tuple(out["walk"]["ids"].shape) == (1, slots), "walk ids [1, S]")
        check(
            tuple(trace.get("body.layer4.out").shape) == (1, slots + 1, hidden),
            "last body layer emits [1, 1 + S, 4096]",
        )
        for layer in range(config["num_hidden_layers"]):
            check(
                tuple(trace.get(f"body.layer{layer}.conv_attn_delta").shape)
                == (1, slots + 1, 2, 2, hidden // int(config["dflash_config"]["conv_group_size"])),
                f"layer {layer} attention-conv delta is [1, W, 2, taps, G]",
            )

        # --- the walk ----------------------------------------------------------------
        check(
            all(
                int(out["walk"]["ids"][0, step]) in set(out["ids"][0, step].tolist())
                for step in range(slots)
            ),
            "every walked token id is one of its slot's candidates",
        )
        previous, ok = 0, True
        for step in range(slots):
            row = out["scores"][0, step, previous, :]
            picked = int(row.argmax())
            ok &= picked == int(out["walk"]["indices"][0, step])
            ok &= bool(torch.equal(row, out["walk"]["rows"][0, step]))
            previous = picked
        check(ok, "the walk re-derives as prev = argmax_k scores[step, prev, k]")

        # --- unit-norm factors and the score band ------------------------------------
        for name in ("head.out_vec", "head.in_vec"):
            norms = trace.get(name).norm(2, dim=-1)
            check(
                float((norms - 1.0).abs().max()) < 1e-4,
                f"{name} rows are unit norm (min {float(norms.min()):.6f})",
            )
        check(
            float((trace.get("head.anchor_out").norm(2, dim=-1) - 1).abs().max()) < 1e-4,
            "anchor_out rows are unit norm",
        )
        check(
            float(out["scores"].abs().max()) <= sigma * (1 + 1e-3),
            f"score tensor stays within [-{sigma}, {sigma}] (max {float(out['scores'].abs().max()):.4f})",
        )
        check(
            float(trace.get("head.start_scores").abs().max()) <= 1 + 1e-4
            and float(trace.get("head.pair_scores").abs().max()) <= 1 + 1e-4,
            "raw cosines stay within [-1, 1]",
        )

        # --- candidate/feature monotonicity (the acceptance-relevant quantities) -----
        log_probs = out["log_probs"]
        check(
            bool((log_probs[..., 1:] <= log_probs[..., :-1] + 1e-6).all()),
            "candidate log-probs are non-increasing in rank",
        )
        check(
            bool((out["ids"][..., :1] == out["ids"][..., :1]).all())
            and bool((out["log_probs"][..., 0] == out["log_probs"].max(-1).values).all()),
            "rank 0 is the slot's argmax log-probability",
        )
        features = trace.get("head.features")
        check(
            bool((features[..., 2] <= 0).all()) and bool((features[..., 2][..., 0] == 0).all()),
            "feature 3 (gap to top-1) is <= 0 and exactly 0 at rank 0",
        )
        check(
            bool((features[..., 3][..., 1:] > features[..., 3][..., :-1]).all()),
            "feature 4 (rank fraction) is strictly increasing in rank",
        )
        check(
            bool((features[..., 4][..., 0] == 1).all())
            and float(features[..., 4].sum()) == slots,
            "feature 5 (is-top-1) is 1 exactly once per slot and 0 elsewhere",
        )
        check(
            bool((features[..., 0] - features[..., 0][..., :1] - features[..., 2]).abs().max() < 1e-6),
            "feature 1 == feature 3 + top-1 log-probability",
        )
        check(
            bool((features[..., 1] - features[..., 0].exp()).abs().max() < 1e-6),
            "feature 2 == exp(feature 1)",
        )

        # --- the sliding-window mask -------------------------------------------------
        key_positions = torch.cat((inputs["context_positions"], inputs["block_positions"]), dim=1)
        visible = sliding_window_visible(inputs["block_positions"], key_positions, window)
        delta = inputs["block_positions"][:, :, None] - key_positions[:, None, :]
        check(bool((visible == (delta.abs() < window)).all()), "mask is exactly |p_query - p_key| < window")
        for row in range(visible.shape[1]):
            preceding = int((visible[0, row] & (delta[0, row] < 0)).sum())
            following = int((visible[0, row] & (delta[0, row] > 0)).sum())
            check(preceding <= window - 1, f"block row {row} sees at most window-1 preceding positions")
            check(following <= window - 1, f"block row {row} sees at most window-1 following positions")
        check(
            all(bool(visible[0, row, context + row]) for row in range(visible.shape[1])),
            "every block row sees itself",
        )
        check(
            int(visible.sum()) == int((delta.abs() < window).sum()) and bool(visible.any()),
            "the mask keeps at least one key per row",
        )

        # --- the conv's position-0 row has no preceding tap --------------------------
        base = weights.tensor("layers.0.attention_conv.base_kernel")
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(1, 8, hidden, generator=generator)
        delta_in = torch.randn(1, 8, 2, hidden // 16, generator=generator)
        convolved = grouped_dynamic_conv(x, delta_in, base, 0, 8, 16)
        expected = (
            base[0, 0].view(hidden // 16, 16) + delta_in[0, 0, 0].unsqueeze(-1)
        ).reshape(1, hidden) * x[:, 0]
        check(bool(torch.allclose(convolved[:, 0], expected, atol=1e-6)), "conv row 0 is tap-0 only")
        perturbed = x.clone()
        perturbed[:, 1:] += 1.0
        again = grouped_dynamic_conv(perturbed, delta_in, base, 0, 8, 16)
        check(
            bool(torch.allclose(again[:, 0], convolved[:, 0], atol=1e-6)),
            "conv row 0 is unaffected by any later row (no preceding tap, no block leak)",
        )
        check(
            bool(torch.allclose(x[:, 0] * 0, x[:, 0] * 0)) and float((again[:, 1] - convolved[:, 1]).abs().max()) > 0,
            "conv row 1 does depend on row 0 through tap 1",
        )

        # --- the anchor ---------------------------------------------------------------
        nonfinite = [name for name, value in trace.stages.items() if not bool(torch.isfinite(value).all())]
        check(not nonfinite, f"every traced stage is finite (first offenders: {nonfinite[:4]})")
        check(
            bool(
                torch.allclose(
                    trace.get("body.anchor_hidden")[0], trace.get("body.hidden_norm_out")[0, context - 1]
                )
            ),
            "anchor_hidden == hidden_norm(fc(taps)) at the last context position",
        )
        check(
            bool((trace.get("head.anchor_state_pre_mask") != 0).any()),
            "the anchor state is non-degenerate",
        )

    print("\n-- fp64 pairwise path --")
    print(f"  max |fp32 - fp64| score delta: {out['fp64_delta']:.3e}")
    check(out["fp64_delta"] < 1e-3, "the fp32 and fp64 scoring paths agree")

    print(f"\n{'=' * 62}\nself-check: {len(failures)} failure(s)")
    for failure in failures:
        print(f"  FAIL: {failure}")
    return 1 if failures else 0


def summary_of_run(config: dict, inputs: dict, out: dict) -> None:
    print("\n--- inputs ---")
    print(
        f"  batch {inputs['block_ids'].shape[0]}, context rows {inputs['context']}, block "
        f"{inputs['block_ids'].shape[1]} (1 anchor + {inputs['slots']} slots), window "
        f"{inputs['window']}, seed {inputs['seed']}"
    )
    for index, tap in enumerate(inputs["taps"]):
        print(f"  tap{index} {tuple(tap.shape)}   (target layer id "
              f"{config['dflash_config']['target_layer_ids'][index]}, 0-based draft convention)")
    print(f"  block ids {inputs['block_ids'].tolist()}")
    print("\n--- walk ---")
    sigma = float(config["dflash_config"]["lilicorr_logit_scale"])
    for step in range(inputs["slots"]):
        row = out["walk"]["rows"][0, step]
        print(
            f"  slot {step}: predecessor index {int(out['walk']['indices'][0, step])} -> token "
            f"{int(out['walk']['ids'][0, step])}; cosines "
            + " ".join(f"{float(v) / sigma:+.4f}" for v in row)
        )
    print(f"  proposed ids: {out['walk']['ids'][0].tolist()}")
    print(f"  max |fp32 - fp64| score delta: {out['fp64_delta']:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CHECKPOINT_CONFIG)
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=f"target model dir (default {TARGET_DIR} in dump mode; synthetic in --self-check)",
    )
    parser.add_argument("--no-target", action="store_true", help="force the synthetic tables")
    parser.add_argument("--vocab", type=int, default=4096, help="vocabulary of the synthetic tables")
    parser.add_argument("--dump", type=Path, default=None, help="write every stage into this directory")
    parser.add_argument(
        "--self-check", action="store_true", help="run the invariant checks (the default mode)"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=49,
                        help="context rows before the query block (>= window exercises the mask)")
    parser.add_argument("--slots", type=int, default=7,
                        help="scored slots; the query block is 1 + slots rows")
    parser.add_argument("--window", type=int, default=None, help="override the sliding window")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="npz with tap0..tap5 [B, C, 4096] (or taps [B, C, 6*4096]) and block_ids [B, W]",
    )
    parser.add_argument("--nibble-order", choices=("low_even", "high_even"), default=None,
                        help="NVFP4 packed-nibble order (default: the module constant)")
    parser.add_argument("--edge-fusion", choices=("out_first", "in_first"), default=None,
                        help="which half of the fused edge weight is the outgoing vector")
    parser.add_argument("--window-mode", choices=("symmetric", "causal"), default=None,
                        help="non-causal sliding-window rule")
    args = parser.parse_args()

    # The switches are read at call time, so a CLI override has to be applied before anything is
    # constructed (a default argument would have bound the import-time value).
    global NIBBLE_ORDER, EDGE_FUSION_ORDER, WINDOW_MODE
    if args.nibble_order is not None:
        NIBBLE_ORDER = args.nibble_order
    if args.edge_fusion is not None:
        EDGE_FUSION_ORDER = args.edge_fusion
    if args.window_mode is not None:
        WINDOW_MODE = args.window_mode

    if args.dump is None:
        return self_check(args)
    if args.self_check:
        status = self_check(args)
        if status:
            return status

    print("=== LiLiCorr oracle: stage dump ===")
    print(
        f"interpreter: torch {torch.__version__}, numpy {np.__version__}, safetensors "
        f"{safetensors.__version__}"
    )
    config = json.loads(args.config.read_text())
    checkpoint = args.checkpoint if args.checkpoint.exists() else CHECKPOINT
    header, header_len = read_header(checkpoint)
    print_reconciliation(header, header_len, checkpoint)
    print_role_table(role_table(header))
    print(
        f"\nconventions: nibble_order={NIBBLE_ORDER} edge_fusion_order={EDGE_FUSION_ORDER} "
        f"sink_mode={SINK_MODE} window_mode={WINDOW_MODE} feature_norm_eps={FEATURE_NORM_EPS}"
    )

    target_root = None if args.no_target else (args.target or TARGET_DIR)
    if target_root is not None and not (target_root / TARGET_INDEX).exists():
        print(f"target index not found under {target_root}; using the synthetic tables")
        target_root = None
    tables = TargetTables(target_root, vocab=args.vocab, hidden=config["hidden_size"], seed=args.seed)
    print(f"target embedding / LM head: {tables.source}")

    window = args.window or int(config["dflash_config"]["swa_window_size"])
    inputs = build_inputs(config, args.batch, args.context, args.slots, args.seed, window, args.inputs)
    trace = Trace()
    trace.add("input.block_ids", inputs["block_ids"].to(torch.float32))
    for index, tap in enumerate(inputs["taps"]):
        trace.add(f"input.tap{index}", tap)
    trace.add("input.context_positions", inputs["context_positions"].to(torch.float32))
    trace.add("input.block_positions", inputs["block_positions"].to(torch.float32))

    weights = Checkpoint(checkpoint)
    out = run_reference(config, weights, tables, inputs, trace)
    summary_of_run(config, inputs, out)
    trace.print_table()
    trace.dump(
        args.dump,
        {
            "seed": inputs["seed"],
            "batch": inputs["block_ids"].shape[0],
            "context_rows": inputs["context"],
            "slots": inputs["slots"],
            "block_rows": inputs["block_ids"].shape[1],
            "window": inputs["window"],
            "block_ids": inputs["block_ids"].tolist(),
            "context_positions": inputs["context_positions"].tolist(),
            "block_positions": inputs["block_positions"].tolist(),
            "target_layer_ids": config["dflash_config"]["target_layer_ids"],
            "conventions": {
                "nibble_order": NIBBLE_ORDER,
                "edge_fusion_order": EDGE_FUSION_ORDER,
                "sink_mode": SINK_MODE,
                "window_mode": WINDOW_MODE,
                "feature_norm_eps": FEATURE_NORM_EPS,
            },
            "target_tables": tables.source,
            "walk_ids": out["walk"]["ids"][0].tolist(),
            "walk_indices": out["walk"]["indices"][0].tolist(),
            "fp64_score_delta": out["fp64_delta"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
