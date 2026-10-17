"""Stage-0 probe for a ConvRot-style rotation on the W8A8 activation route.

Answers one question on real Gemma 4 31B data, before any kernel work: does rotating the projection
inputs (and the weights they multiply) inside groups of G reduce the error that per-token absmax INT8
activations inject into `x @ W^T`?

The mechanism under test is the one llama.cpp PRs #26914/#28481 implement: a fixed orthogonal rotation
R, block-diagonal along K in groups of G, applied to both operands (`x R` and `W R`), so the product is
unchanged in real arithmetic and the quantizer sees a Gaussianized row.

It reuses the sensitivity harness's own checkpoint, forward, and converter encoders, so its numbers sit
next to the recorded KLD measurements rather than beside a second implementation:

  /mnt/storage/ninfer/eval/.venv/bin/python -m tools.verify.gemma4_rotation_probe --depth 60

Reported per layer, group and rotation variant:

  * row shape of the input before and after rotation: absmax/RMS (a normed row has RMS ~ 1, so this is
    the outlier ratio the fixed-point range wastes) and excess kurtosis;
  * per-row relative L2 of the INT8 round trip, median and p90;
  * output error of `x @ W^T` against an FP32 oracle, relative L2 and per-token p90, for three
    separable contributions: the activation alone (exact weight), the weight format alone (exact
    activation), and the shipping route (both quantized, the weight rotated before its encoder);
  * `excess`: the A8-attributable part, `sqrt(route^2 - weight_only^2)`, i.e. what rotation has to
    remove for A8 to be free relative to the weight format already in the budget. Its reduction is the
    number that maps onto the recorded KLD arithmetic.

Self-check: the unquantized rotated route must reproduce the oracle to ~1e-6 relative, otherwise the
rotation is wrong (asserted, so a broken R cannot produce a flattering number).
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from tools.verify.gemma4_sensitivity import (DEV, Checkpoint, Head, LAYERS, calibration_ids,
                                             fake_groupwise, layer_forward, projection, rmsnorm)
from tools.verify.rotation import ROTATIONS, int8_rows, rotate_k

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False   # the oracle must be true FP32, not TF32


# ------------------------------------------------------------------ study
def row_metrics(x: torch.Tensor) -> tuple[float, float]:
    f = x.float()
    rms = f.pow(2).mean(dim=-1).sqrt()
    amax = f.abs().amax(dim=-1)
    var = f.var(dim=-1, unbiased=False)
    kurt = ((f - f.mean(dim=-1, keepdim=True)).pow(4).mean(dim=-1) / var.pow(2)).nan_to_num()
    return float((amax / rms).median()), float((kurt - 3.0).median())


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = out.float() - ref.float()
    per_row = d.pow(2).sum(dim=-1).sqrt() / ref.float().pow(2).sum(dim=-1).sqrt().clamp_min(1e-30)
    return float(d.norm() / ref.norm()), float(per_row.quantile(0.9))


# ------------------------------------------------------------------ study
def study(label: str, x: torch.Tensor, weight: torch.Tensor, rows: int, variants: list[str],
          weight_format: str, round_activation: bool = False) -> list[tuple[str, float, float]]:
    """Returns (variant, act_only_error, excess_error) for one projection."""
    x = x[:rows].float()
    w = weight.float()
    ref = x @ w.T
    out = []
    for name in ("none", *variants):
        if name == "none":
            xr, wr = x, w
        else:
            xr, wr = rotate_k(x, name), rotate_k(w, name)  # the pair keeps the product invariant
            if round_activation:
                # The engine's Op writes a BF16 tensor, so the rotated activation is rounded again
                # before the projection: a second, independent rounding of the vector whose
                # cancellation-amplified error dominates a wide dot product.
                xr = xr.to(torch.bfloat16).float()

        q8 = int8_rows(xr)
        scale = (q8 - xr).norm(dim=-1) / xr.norm(dim=-1).clamp_min(1e-30)
        amax_rms, kurt = row_metrics(xr)

        wq = fake_groupwise(wr.to(weight.dtype), weight_format).to(DEV)
        act_only, _ = rel_l2(q8 @ wr.T, ref)
        weight_only, _ = rel_l2(xr @ wq.T, ref)
        route, p90 = rel_l2(q8 @ wq.T, ref)
        exact, _ = rel_l2(xr @ wr.T, ref)
        excess = math.sqrt(max(route * route - weight_only * weight_only, 0.0))

        print(f"{label:>10} {name:>13} {amax_rms:9.2f} {kurt:9.1f} {float(scale.median()):9.2e} "
              f"{float(scale.quantile(0.9)):8.2e} | {act_only:9.2e} {weight_only:9.2e} "
              f"{route:9.2e} {p90:10.2e} {excess:9.2e}")
        out.append((name, act_only, excess))
        global _EXACT_WORST
        _EXACT_WORST = max(_EXACT_WORST, exact)
    return out


_EXACT_WORST = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=LAYERS, help="forward layers 0..depth-1")
    parser.add_argument("--tokens", type=int, default=1024, help="calibration length")
    parser.add_argument("--rows", type=int, default=256, help="rows entering the matmul study")
    parser.add_argument("--variants", default=",".join(ROTATIONS))
    parser.add_argument("--weight-format", default="q6", help="groupwise weight format to emulate")
    parser.add_argument("--groups", default="gate,down", help="gate (with up's input) and/or down")
    parser.add_argument("--round-activation", action="store_true",
                        help="round the rotated activation to BF16, as the engine's Op writes it")
    args = parser.parse_args()
    variants = args.variants.split(",")
    groups = args.groups.split(",")

    t0 = time.time()
    ckpt = Checkpoint()
    head = Head(ckpt)
    ids = calibration_ids(1, args.tokens)
    print(f"setup {time.time() - t0:.0f}s; depth {args.depth}, {args.rows} study rows, "
          f"{args.weight_format} weights\n", flush=True)

    names = ("none", *variants)
    header = (f"{'layer':>5} {'proj':>10} {'variant':>13} {'amax/rms':>9} {'kurt':>9} "
              f"{'i8 rowL2':>9} {'i8 p90':>8} | {'act-only':>9} {'w-only':>9} {'route':>9} "
              f"{'route p90':>10} {'excess':>9}")
    print(header)
    print("-" * len(header))

    per_layer: dict[int, dict[str, list]] = {}
    h = head.embed(ids)
    for index in range(args.depth):
        w = ckpt.layer(index)
        x_ff = rmsnorm(h, w["pre_feedforward_layernorm.weight"])
        gate = projection(w, "gate_proj")
        up = projection(w, "up_proj")
        down = projection(w, "down_proj")
        x_down = None
        if "down" in groups:
            g = x_ff.float() @ gate.float().T
            u = x_ff.float() @ up.float().T
            x_down = (torch.nn.functional.gelu(g, approximate="tanh") * u).to(x_ff.dtype)

        per_layer[index] = {}
        if "gate" in groups:
            per_layer[index]["gate"] = study(f"{index}|gate", x_ff.reshape(-1, x_ff.shape[-1]), gate,
                                             args.rows, variants, args.weight_format,
                                             args.round_activation)
        if "down" in groups:
            per_layer[index]["down"] = study(f"{index}|down", x_down.reshape(-1, x_down.shape[-1]),
                                             down, args.rows, variants, args.weight_format,
                                             args.round_activation)

        h = layer_forward(index, w, h)
        del w, gate, up, down, x_down
        (print(flush=True) if index % 10 == 9 else None)
    del h, head

    print(f"\nrotation self-check: worst exact-route error {_EXACT_WORST:.2e} (must be ~1e-6)")
    assert _EXACT_WORST < 1e-5, "rotation does not preserve the product; the probe is not valid"

    for proj in groups:
        print(f"\nA8-attributable excess, unrotated / variant, {proj} (higher is better):")
        print(f"{'layer':>5} " + " ".join(f"{n:>13}" for n in names))
        ratios = {n: [] for n in names}
        for index in sorted(per_layer):
            base = dict((n, e) for n, _, e in per_layer[index][proj])["none"]
            row = []
            for n in names:
                e = dict((nn, ee) for nn, _, ee in per_layer[index][proj])[n]
                ratio = base / max(e, 1e-30)
                ratios[n].append(ratio)
                row.append(f"{ratio:13.2f}x")
            print(f"{index:5d} " + " ".join(row))
        med = {n: sorted(v)[len(v) // 2] for n, v in ratios.items()}
        print("median: " + " ".join(f"{n}={med[n]:.2f}x" for n in names if n != "none") +
              f"   (worst {min(med[n] for n in names if n != 'none'):.2f}x)")


if __name__ == "__main__":
    main()
