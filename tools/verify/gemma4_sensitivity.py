"""Per-tensor quantization sensitivity of Gemma 4 31B against its own BF16 checkpoint.

For every layer and every projection group (attention q/k/v, attention output, MLP gate+up, MLP down)
this replaces that one group with a candidate format, quantized and decoded by the converter's own
encoders, runs the rest of the model in BF16, and measures the KL divergence of the final
distribution from the all-BF16 reference over a calibration text. The embedding and the head are
measured the same way. The result ranks where bits pay and is the input to a mixed-precision recipe.

The checkpoint (59 GB) does not fit a 32 GB device, so its layers stay in pinned host memory and are
streamed through the GPU one at a time. The reference pass keeps every layer's input on the host;
a candidate at layer L starts from that input, so only layers L..59 run for it, and all candidates of
one layer share each streamed layer.

What a candidate simulates: the weight's exact decoded values, rounded to BF16 for the matmul (the
engine multiplies the exact decoded value; the extra BF16 rounding is far below every format's own
error except Q8's, where it is comparable). `nvfp4_a4` also quantizes the projection inputs to NVFP4
with a per-call divisor, which approximates the W4A4 route, whose divisor is calibrated offline.
Attention runs with FP32 scores and softmax, so the reference is slightly more exact than HF's BF16.

Run with an interpreter that has torch with CUDA and `tokenizers`, from the repository root:

    /mnt/storage/ninfer/eval/.venv/bin/python -m tools.verify.gemma4_sensitivity --out <file.json>

`--validate <ids.i32> <hf_logprobs.f32>` instead scores one BOS-prefixed sequence with the BF16
reference and compares it with HF's per-target log-probabilities, which checks this reimplementation.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from tools.artifact.formats import get_format
from tools.convert.quantization import groupwise, nvfp4

CKPT = Path("/mnt/storage/models/gemma/full-31b")
CORPUS = Path("eval/corpora/perplexity-1m/data")
torch.set_grad_enabled(False)
DEV = torch.device("cuda")
# Activation dtype between Ops: BF16 as the checkpoint runs, or FP32 (`--fp32-activations`), which
# removes the BF16 rounding of the residual stream from both the reference and every candidate.
ACT = torch.bfloat16

CFG = json.loads((CKPT / "config.json").read_text())
TEXT = CFG.get("text_config", CFG)
H = int(TEXT["hidden_size"])
LAYERS = int(TEXT["num_hidden_layers"])
TYPES = TEXT["layer_types"]
EPS = float(TEXT["rms_norm_eps"])
HQ = int(TEXT["num_attention_heads"])
WINDOW = int(TEXT["sliding_window"])
CAP = float(TEXT["final_logit_softcapping"])
SLIDING = dict(d=int(TEXT["head_dim"]), hkv=int(TEXT["num_key_value_heads"]),
               theta=10000.0, pairs=int(TEXT["head_dim"]) // 2)
GD = int(TEXT["global_head_dim"])
GLOBAL = dict(d=GD, hkv=int(TEXT["num_global_key_value_heads"]), theta=1000000.0,
              pairs=int(round(GD * 0.25)) // 2)

GROUPS = {
    "attn_qkv": ("q_proj", "k_proj", "v_proj"),
    "attn_o": ("o_proj",),
    "mlp_gu": ("gate_proj", "up_proj"),
    "mlp_down": ("down_proj",),
}
WEIGHT_FORMATS = ("fp8", "q8", "q6", "q5", "q4", "nvfp4")
GROUP_FORMATS = {
    "attn_qkv": WEIGHT_FORMATS,
    "attn_o": WEIGHT_FORMATS,
    "mlp_gu": WEIGHT_FORMATS + ("nvfp4_a4",),
    "mlp_down": WEIGHT_FORMATS + ("nvfp4_a4",),
}
# Stored bits per weight, scales included, for the byte budget.
BITS = {"bf16": 16.0, "fp8": 8.0, "q8": 8.5, "q6": 6.25, "q5": 5.25, "q4": 4.25,
        "nvfp4": 4.5, "nvfp4_a4": 4.5}
GROUPWISE = {"q8": "q8_g32_fp16", "q6": "q6_g64_fp16", "q5": "q5_g64_fp16", "q4": "q4_g64_fp16"}


# ---------------------------------------------------------------- weights
class Checkpoint:
    """The BF16 checkpoint in pinned host memory, by tensor name."""

    def __init__(self):
        index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
        self.host: dict[str, torch.Tensor] = {}
        files: dict[str, list[str]] = {}
        for name, file in index.items():
            if name.startswith("model.language_model."):
                files.setdefault(file, []).append(name)
        start = time.time()
        for file, names in files.items():
            with safe_open(str(CKPT / file), framework="pt", device="cpu") as f:
                for name in names:
                    self.host[name[len("model.language_model."):]] = f.get_tensor(name).pin_memory()
        print(f"checkpoint pinned: {len(self.host)} tensors, {time.time() - start:.0f}s", flush=True)

    def layer(self, index: int) -> dict[str, torch.Tensor]:
        prefix = f"layers.{index}."
        return {k[len(prefix):]: v.to(DEV, non_blocking=True)
                for k, v in self.host.items() if k.startswith(prefix)}


def projection(weights: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    key = ("mlp." if name in ("gate_proj", "up_proj", "down_proj") else "self_attn.") + name + ".weight"
    return weights.get(key)


# ---------------------------------------------------------------- formats (decode back to BF16)
def fake_fp8_rows(w: torch.Tensor) -> torch.Tensor:
    x = w.float()
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = (amax.double() / 448.0).float().to(torch.bfloat16).float()
    scale = torch.where((scale == 0) & (amax > 0), torch.full_like(scale, 2.0 ** -133), scale)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (x / safe).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return codes * scale


def fake_groupwise(w: torch.Tensor, fmt: str) -> torch.Tensor:
    spec = get_format(GROUPWISE[fmt])
    q = groupwise.quantize_matrix(w, spec, device=DEV)
    n, k = w.shape
    values = q.codes.float() * q.scales.float().unsqueeze(-1)
    return values.reshape(n, -1)[:, :k]


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fake_nvfp4(w: torch.Tensor, divisor: bytes) -> torch.Tensor:
    d = struct.unpack("<f", divisor)[0]
    q = nvfp4.quantize_rows(w, divisor, device=str(DEV), scale_search="mse")
    codes = q.codes.to(DEV)
    words = torch.stack([codes & 0xF, codes >> 4], dim=-1).reshape(w.shape[0], -1).long()
    magnitude = _E2M1.to(DEV)[words & 7]
    values = torch.where((words & 8) != 0, -magnitude, magnitude)
    scales = q.scales.to(DEV).view(torch.float8_e4m3fn).float()
    values = values.reshape(w.shape[0], -1, 16) * scales.unsqueeze(-1) / d
    return values.reshape(w.shape)


def fake_weight(w: torch.Tensor, fmt: str) -> torch.Tensor:
    """Every format is row-local except NVFP4's one divisor per tensor, so rows go in blocks that
    bound the encoders' temporaries (the 1.4G-element embedding would not fit at once)."""
    if fmt == "nvfp4":
        divisor = nvfp4.divisor_for(float(w.abs().amax().float()))
        encode = lambda block: fake_nvfp4(block, divisor)
    elif fmt == "fp8":
        encode = fake_fp8_rows
    elif fmt in GROUPWISE:
        encode = lambda block: fake_groupwise(block, fmt)
    else:
        raise ValueError(fmt)
    rows = max(128, (1 << 25) // w.shape[1] // 128 * 128)
    return torch.cat([encode(w[r:r + rows]) for r in range(0, w.shape[0], rows)])


def fake_nvfp4_activation(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 activations: E4M3 per-16 block scales under one per-call FP32 divisor."""
    shape = x.shape
    v = x.float().reshape(-1, shape[-1] // 16, 16)
    amax = float(v.abs().amax())
    d = 6.0 * 448.0 / amax if amax > 0 else 1.0
    v = v * d
    scale = (v.abs().amax(dim=-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled = (v / safe.unsqueeze(-1)).clamp(-6.0, 6.0)
    mids = (_E2M1[1:] + _E2M1[:-1]).to(x.device) / 2
    index = torch.bucketize(scaled.abs(), mids, right=False)
    q = torch.where(scaled < 0, -1.0, 1.0) * _E2M1.to(x.device)[index] * scale.unsqueeze(-1) / d
    return q.reshape(shape).to(x.dtype)


# ---------------------------------------------------------------- model
def rmsnorm(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y.to(ACT)


_ROPE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def rope(x: torch.Tensor, geometry: dict) -> torch.Tensor:
    """x [S, heads, L, d]; rotates pairs (i, i + d/2) for i < pairs with denominator d."""
    length, d, pairs = x.shape[2], geometry["d"], geometry["pairs"]
    key = (length, d, pairs, geometry["theta"])
    if key not in _ROPE:
        i = torch.arange(pairs, dtype=torch.float64, device=x.device)
        angle = torch.arange(length, dtype=torch.float64, device=x.device)[:, None] * \
            torch.pow(torch.tensor(geometry["theta"], dtype=torch.float64), -2.0 * i / d)
        _ROPE[key] = (torch.cos(angle).float(), torch.sin(angle).float())
    cos, sin = _ROPE[key]
    y = x.float()
    lo, hi = y[..., :pairs].clone(), y[..., d // 2:d // 2 + pairs].clone()
    y[..., :pairs] = lo * cos - hi * sin
    y[..., d // 2:d // 2 + pairs] = lo * sin + hi * cos
    return y


_MASK: dict[tuple, torch.Tensor] = {}


def mask(length: int, sliding: bool) -> torch.Tensor:
    key = (length, sliding)
    if key not in _MASK:
        q = torch.arange(length, device=DEV)[:, None]
        k = torch.arange(length, device=DEV)[None, :]
        visible = (q >= k) & ((q - k < WINDOW) if sliding else True)
        _MASK[key] = torch.where(visible, 0.0, float("-inf"))
    return _MASK[key]


def linear(x: torch.Tensor, w: torch.Tensor, a4: bool) -> torch.Tensor:
    return F.linear(fake_nvfp4_activation(x) if a4 else x, w.to(x.dtype))


def layer_forward(index: int, w: dict[str, torch.Tensor], h: torch.Tensor,
                  override: dict[str, torch.Tensor] | None = None, a4: bool = False) -> torch.Tensor:
    """One decoder layer over h [S, L, H] BF16. `override` replaces projection weights by name;
    `a4` quantizes the inputs of the overridden MLP projections."""
    override = override or {}
    get = lambda name: override.get(name, projection(w, name))
    a4_of = lambda name: a4 and name in override
    full = TYPES[index] == "full_attention"
    g = GLOBAL if full else SLIDING
    s, length, _ = h.shape
    d, hkv = g["d"], g["hkv"]

    n = rmsnorm(h, w["input_layernorm.weight"])
    q = rmsnorm(linear(n, get("q_proj"), False).view(s, length, HQ, d), w["self_attn.q_norm.weight"])
    raw_k = linear(n, get("k_proj"), False).view(s, length, hkv, d)
    k = rmsnorm(raw_k, w["self_attn.k_norm.weight"])
    if full:
        v = rmsnorm(raw_k, None)
    else:
        v = rmsnorm(linear(n, get("v_proj"), False).view(s, length, hkv, d), None)
    q = rope(q.transpose(1, 2), g)
    k = rope(k.transpose(1, 2), g)
    v = v.transpose(1, 2).float()
    group = HQ // hkv
    m = mask(length, not full)
    out = torch.empty(s, HQ, length, d, device=DEV, dtype=torch.float32)
    for seq in range(s):
        for kvh in range(hkv):
            heads = slice(kvh * group, (kvh + 1) * group)
            scores = q[seq, heads] @ k[seq, kvh].transpose(0, 1) + m
            out[seq, heads] = torch.softmax(scores, dim=-1) @ v[seq, kvh]
    attended = out.transpose(1, 2).reshape(s, length, HQ * d).to(ACT)
    h = h + rmsnorm(linear(attended, get("o_proj"), False), w["post_attention_layernorm.weight"])

    x = rmsnorm(h, w["pre_feedforward_layernorm.weight"])
    gate = linear(x, get("gate_proj"), a4_of("gate_proj"))
    up = linear(x, get("up_proj"), a4_of("up_proj"))
    product = (F.gelu(gate.float(), approximate="tanh") * up.float()).to(ACT)
    del gate, up
    h = h + rmsnorm(linear(product, get("down_proj"), a4_of("down_proj")),
                    w["post_feedforward_layernorm.weight"])
    return (h * w["layer_scalar"].to(ACT)).to(ACT)


class Head:
    def __init__(self, ckpt: Checkpoint):
        self.table = ckpt.host["embed_tokens.weight"].to(DEV).to(ACT)
        self.final = ckpt.host["norm.weight"].to(DEV)
        # HF multiplies the BF16 row by sqrt(H) rounded to BF16; FP32 mode keeps both exact.
        self.scale = torch.tensor(math.sqrt(H), dtype=ACT, device=DEV)

    def embed(self, ids: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
        return (table if table is not None else self.table)[ids].to(ACT) * self.scale

    def logprobs(self, h: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """h [T, H] -> log-softmax of the soft-capped logits, FP32 [T, V]."""
        weight = weight if weight is not None else self.table
        logits = F.linear(rmsnorm(h, self.final), weight.to(ACT)).float()
        return torch.log_softmax(CAP * torch.tanh(logits / CAP), dim=-1)


# ---------------------------------------------------------------- calibration
def calibration_ids(sequences: int, length: int) -> torch.Tensor:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    domains = sorted(p for p in CORPUS.iterdir() if p.is_dir())
    rows = []
    for i in range(sequences):
        domain = domains[i % len(domains)]
        files = sorted(domain.glob("*.txt"))
        text = files[(i // len(domains)) % len(files)].read_text()[: length * 12]
        ids = tok.encode(text, add_special_tokens=False).ids[: length - 1]
        if len(ids) < length - 1:
            raise SystemExit(f"{domain.name}: text too short for {length} tokens")
        rows.append([2] + ids)  # <bos>
    print("calibration:", ", ".join(d.name for d in domains[:sequences]), f"x {length} tokens",
          flush=True)
    return torch.tensor(rows, device=DEV)


class Scorer:
    """KL divergence of candidate final states against the reference, chunked over tokens."""

    def __init__(self, head: Head, reference_final: torch.Tensor, ids: torch.Tensor, chunk: int = 512):
        self.head, self.chunk = head, chunk
        self.ref = reference_final.reshape(-1, H)  # host BF16
        self.targets = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1).reshape(-1)

    def score(self, finals: list[torch.Tensor], head_weights: list[torch.Tensor] | None = None):
        """finals: per candidate [S, L, H] on device (or None to reuse the reference state)."""
        count = len(finals) if finals else len(head_weights)
        per_token = [[] for _ in range(count)]
        kld = [0.0] * count
        dnll = [0.0] * count
        agree = [0] * count
        tokens = self.ref.shape[0]
        scored = 0
        for start in range(0, tokens, self.chunk):
            ref_h = self.ref[start:start + self.chunk].to(DEV)
            p_log = self.head.logprobs(ref_h)
            p = p_log.exp()
            targets = self.targets[start:start + self.chunk]
            valid = targets >= 0
            ref_nll = -p_log[valid].gather(1, targets[valid][:, None]).squeeze(1)
            ref_top = p_log.argmax(dim=-1)
            for c in range(count):
                h = finals[c].reshape(-1, H)[start:start + self.chunk] if finals else ref_h
                q_log = self.head.logprobs(h, head_weights[c] if head_weights else None)
                token_kld = (p * (p_log - q_log)).sum(dim=-1)
                per_token[c].append(token_kld.cpu())
                kld[c] += float(token_kld.sum())
                dnll[c] += float((-q_log[valid].gather(1, targets[valid][:, None]).squeeze(1)
                                  - ref_nll).sum())
                agree[c] += int((q_log.argmax(dim=-1) == ref_top).sum())
                del q_log
            scored += int(valid.sum())
        results = []
        for c in range(count):
            t = torch.cat(per_token[c]).double()
            results.append(dict(kld=kld[c] / tokens, dnll=dnll[c] / scored, top1=agree[c] / tokens,
                                kld_median=float(t.median()), kld_p90=float(t.quantile(0.9)),
                                kld_p99=float(t.quantile(0.99)), kld_max=float(t.max())))
        return results


# ---------------------------------------------------------------- driver
def reference_pass(ckpt: Checkpoint, head: Head, ids: torch.Tensor):
    h = head.embed(ids)
    inputs = []
    for index in range(LAYERS):
        inputs.append(h.cpu().pin_memory())
        h = layer_forward(index, ckpt.layer(index), h)
    return inputs, h.cpu().pin_memory()


def group_bytes(w: dict[str, torch.Tensor], group: str) -> int:
    return sum(projection(w, n).numel() for n in GROUPS[group] if projection(w, n) is not None)


def validate(ckpt: Checkpoint, head: Head, ids_path: str, hf_path: str):
    ids = torch.from_numpy(np.fromfile(ids_path, dtype=np.int32).astype(np.int64)).to(DEV)[None]
    _, final = reference_pass(ckpt, head, ids)
    lp = head.logprobs(final.reshape(-1, H).to(DEV))
    ours = lp[:-1].gather(1, ids[0, 1:, None]).squeeze(1).cpu().numpy()
    ours.astype(np.float32).tofile(hf_path + ".ours")
    hf = np.fromfile(hf_path, dtype=np.float32)[: len(ours)]
    d = np.abs(ours - hf)
    print(f"validate: {len(ours)} targets, NLL ours {-ours.mean():.4f} HF {-hf.mean():.4f}, "
          f"mean|d| {d.mean():.4f} p99 {np.percentile(d, 99):.4f} max {d.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--layers", default="all", help="'all' or a comma list of layer indices")
    parser.add_argument("--validate", nargs=2, metavar=("IDS", "HF_LOGPROBS"))
    parser.add_argument("--fp32-activations", action="store_true")
    parser.add_argument("--probe", nargs="+", metavar="GROUP[@LAYER]:FORMAT",
                        help="score only these candidates, e.g. embedding:q8 mlp_down@30:q4")
    parser.add_argument("--assign", nargs="+", metavar="LAYOUT.json",
                        help="score whole layouts end to end instead of single candidates")
    args = parser.parse_args()
    global ACT
    if args.fp32_activations:
        ACT = torch.float32

    ckpt = Checkpoint()
    head = Head(ckpt)
    if args.validate:
        validate(ckpt, head, *args.validate)
        return
    if args.out is None and not args.probe and not args.assign:
        sys.exit("--out or --probe is required")

    ids = calibration_ids(args.sequences, args.length)
    start = time.time()
    inputs, final = reference_pass(ckpt, head, ids)
    print(f"reference pass: {time.time() - start:.0f}s", flush=True)
    scorer = Scorer(head, final, ids)
    if args.assign:
        # A whole layout at once: {"head": f, "embedding": f, "layers": {"0": {group: f}}}; a group
        # or tensor left out stays BF16.
        for path in args.assign:
            layout = json.loads(Path(path).read_text())
            table = head.table
            h = head.embed(ids, fake_weight(table, layout["embedding"])
                           if layout.get("embedding", "bf16") != "bf16" else None)
            params = dict(bf16=0.0)
            total_bits = 0.0
            for index in range(LAYERS):
                w = ckpt.layer(index)
                chosen = layout.get("layers", {}).get(str(index), {})
                override, a4 = {}, False
                for group, fmt in chosen.items():
                    names = [n for n in GROUPS[group] if projection(w, n) is not None]
                    base = "nvfp4" if fmt == "nvfp4_a4" else fmt
                    a4 = a4 or fmt == "nvfp4_a4"
                    for n in names:
                        override[n] = fake_weight(projection(w, n), base)
                    total_bits += group_bytes(w, group) * BITS[fmt]
                for group in GROUPS:
                    if group not in chosen:
                        total_bits += group_bytes(w, group) * 16.0
                h = layer_forward(index, w, h, override, a4=a4)
                del w, override
            head_fmt = layout.get("head", "bf16")
            head_weight = fake_weight(table, head_fmt).to(ACT) if head_fmt != "bf16" else None
            r = scorer.score([h], [head_weight] if head_weight is not None else None)[0]
            total_bits += table.numel() * (BITS[head_fmt] + BITS[layout.get("embedding", "bf16")])
            print(f"layout {path}: {total_bits / 8 / 2**30:.2f} GiB  "
                  + " ".join(f"{k} {v:.5f}" for k, v in r.items()), flush=True)
        return
    if args.probe:
        for spec in args.probe:
            target, fmt = spec.split(":")
            group, _, layer = target.partition("@")
            layer = int(layer) if layer else -1
            if group == "embedding":
                h = head.embed(ids, fake_weight(head.table, fmt))
                first = 0
            else:
                w = ckpt.layer(layer)
                names = [n for n in GROUPS[group] if projection(w, n) is not None]
                base = "nvfp4" if fmt == "nvfp4_a4" else fmt
                override = {n: fake_weight(projection(w, n), base) for n in names}
                h = layer_forward(layer, w, inputs[layer].to(DEV), override, a4=(fmt == "nvfp4_a4"))
                del w, override
                first = layer + 1
            for later in range(first, LAYERS):
                h = layer_forward(later, ckpt.layer(later), h)
            r = scorer.score([h])[0]
            print(f"probe {spec:22s} " + " ".join(f"{k} {v:.5f}" for k, v in r.items()), flush=True)
        return
    results: dict = {"calibration": dict(sequences=args.sequences, length=args.length),
                     "bits": BITS, "candidates": []}
    out_path = args.out

    def flush():
        out_path.write_text(json.dumps(results, indent=1))

    # Head: the reference final state through each candidate head.
    formats = WEIGHT_FORMATS
    for f in formats:
        weight = fake_weight(head.table, f).to(ACT)
        r = scorer.score([], [weight])[0]
        del weight
        results["candidates"].append(dict(layer=-1, group="head", format=f,
                                          params=head.table.numel(), **r))
        print(f"head {f:9s} kld {r['kld']:.5f} dnll {r['dnll']:+.5f} top1 {r['top1']:.4f}", flush=True)
        flush()

    # Embedding: a quantized table feeds every layer.
    for f in formats:
        table = fake_weight(head.table, f)
        h = head.embed(ids, table)
        del table
        for index in range(LAYERS):
            h = layer_forward(index, ckpt.layer(index), h)
        r = scorer.score([h])[0]
        results["candidates"].append(dict(layer=-1, group="embedding", format=f,
                                          params=head.table.numel(), **r))
        print(f"embedding {f:9s} kld {r['kld']:.5f} dnll {r['dnll']:+.5f}", flush=True)
        flush()

    layers = range(LAYERS) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    for index in layers:
        t0 = time.time()
        w = ckpt.layer(index)
        h_in = inputs[index].to(DEV)
        candidates, states = [], []
        for group, group_formats in GROUP_FORMATS.items():
            names = [n for n in GROUPS[group] if projection(w, n) is not None]
            cache: dict[str, dict[str, torch.Tensor]] = {}
            for f in group_formats:
                base = "nvfp4" if f == "nvfp4_a4" else f
                if base not in cache:
                    cache.clear()  # nvfp4_a4 follows nvfp4, so one cached format is enough
                    cache[base] = {n: fake_weight(projection(w, n), base) for n in names}
                states.append(layer_forward(index, w, h_in, cache[base], a4=(f == "nvfp4_a4")))
                candidates.append(dict(layer=index, group=group, format=f,
                                       params=group_bytes(w, group)))
            del cache
        del w, h_in
        for later in range(index + 1, LAYERS):
            lw = ckpt.layer(later)
            states = [layer_forward(later, lw, h) for h in states]
            del lw
        for c, r in zip(candidates, scorer.score(states)):
            c.update(r)
            results["candidates"].append(c)
        del states
        flush()
        worst = max(candidates, key=lambda c: c["kld"])
        print(f"layer {index:2d} ({TYPES[index][:7]}) {time.time() - t0:5.0f}s  "
              + "  ".join(f"{c['group']}/{c['format']}={c['kld']:.4f}"
                          for c in candidates if c["format"] in ("q4", "nvfp4", "nvfp4_a4")),
              f" worst {worst['group']}/{worst['format']}", flush=True)
    print(f"done: {len(results['candidates'])} candidates in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
