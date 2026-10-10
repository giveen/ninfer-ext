"""Choose a mixed-precision Gemma 4 layout from measured per-tensor sensitivities.

Reads the JSON `gemma4_sensitivity.py` writes and, treating each candidate's KL divergence as an
independent additive cost, picks one format per tensor group that minimizes the summed KLD under a
byte budget (a Lagrangian sweep: each group takes argmin KLD + lambda * bytes, and lambda is bisected
to the budget). The additivity is an approximation, so the chosen layout must be scored end to end
with `gemma4_sensitivity.py --assign` before it is converted.

    .venv/bin/python tools/verify/gemma4_allocate.py profiles/sensitivity/gemma4_31b.json \
        --budget-gib 19.5 --out layout.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("--budget-gib", type=float, required=True)
    parser.add_argument("--formats", default="fp8,q8,q6,q5,q4,nvfp4",
                        help="formats a projection group may take")
    parser.add_argument("--head-formats", default="bf16,q8,fp8,q6")
    parser.add_argument("--embedding-formats", default="fp8,q8,q6,q5,q4,nvfp4")
    parser.add_argument("--attention-formats", help="formats the attention groups may take "
                        "(default: --formats)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = json.loads(args.sensitivity.read_text())
    bits = data["bits"]
    options: dict[tuple, list[tuple[str, float, float]]] = collections.defaultdict(list)
    allowed = set(args.formats.split(","))
    for c in data["candidates"]:
        key = (c["layer"], c["group"])
        if c["group"] == "head":
            if c["format"] not in args.head_formats.split(","):
                continue
        elif c["group"] == "embedding":
            if c["format"] not in args.embedding_formats.split(","):
                continue
        elif c["group"].startswith("attn_") and args.attention_formats:
            if c["format"] not in args.attention_formats.split(","):
                continue
        elif c["format"] not in allowed:
            continue
        options[key].append((c["format"], c["kld"], c["params"] * bits[c["format"]] / 8))
    head = next(c for c in data["candidates"] if c["group"] == "head")
    if "bf16" in args.head_formats.split(","):
        options[(-1, "head")].append(("bf16", 0.0, head["params"] * 2.0))
    # The engine keeps the embedding in pinned host memory, so an exact BF16 table costs no device
    # bytes; the budget is device memory.
    if "bf16" in args.embedding_formats.split(","):
        options[(-1, "embedding")].append(("bf16", 0.0, 0.0))
    budget = args.budget_gib * 2**30
    # Norms, scalars and resources are a few MiB; they are counted as fixed overhead.
    overhead = 64 * 2**20

    def solve(lam: float):
        pick = {k: min(v, key=lambda o: o[1] + lam * o[2]) for k, v in options.items()}
        return pick, sum(o[2] for o in pick.values()) + overhead, sum(o[1] for o in pick.values())

    lo, hi = 0.0, 1.0
    while solve(hi)[1] > budget:
        hi *= 4
        if hi > 1e6:
            raise SystemExit("the budget is below the smallest layout the formats allow")
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if solve(mid)[1] > budget else (lo, mid)
    pick, size, kld = solve(hi)
    print(f"budget {args.budget_gib:.2f} GiB: layout {size / 2**30:.2f} GiB, summed KLD {kld:.4f}")
    counts = collections.Counter(o[0] for o in pick.values())
    print("groups by format:", dict(counts))
    for (layer, group), o in sorted(pick.items()):
        if layer < 0:
            print(f"  {group}: {o[0]} (kld {o[1]:.5f}, {o[2] / 2**30:.2f} GiB)")
    layout = {"head": pick[(-1, "head")][0], "embedding": pick[(-1, "embedding")][0], "layers": {}}
    for (layer, group), o in pick.items():
        if layer >= 0:
            layout["layers"].setdefault(str(layer), {})[group] = o[0]
    lines = []
    for layer in sorted(layout["layers"], key=int):
        g = layout["layers"][layer]
        lines.append(f"  {int(layer):2d}: " + " ".join(f"{k}={g[k]}" for k in
                                                        ("attn_qkv", "attn_o", "mlp_gu", "mlp_down")))
    print("\n".join(lines))
    if args.out:
        args.out.write_text(json.dumps(layout, indent=1))


if __name__ == "__main__":
    main()
