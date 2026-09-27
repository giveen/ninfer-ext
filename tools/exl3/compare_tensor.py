"""Compare NInfer's EXL3 single-tensor quantizer with exllamav3 on the same weight and Hessian.

Run with the isolated exllamav3 reference environment (it needs torch and exllamav3), from the
repository root:

    /tmp/opencode/exllamav3-reference/.venv/bin/python tools/exl3/compare_tensor.py \
        --probe build/tests/ninfer_exl3_tensor_probe --bits 2 3 4

The Hessian is the real input of layer 0: input_layernorm(embed_tokens(tokens)) over packed trace
rows (only the stored lengths), so it needs no model forward. Both quantizers receive the same
mean XᵀX and weight; the reported proxy error tr(Eᵀ H E) / tr(Wᵀ H W) uses the damped Hessian in
the original domain after each side's scale refit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

DAMPING = 0.025


def load_tensor(model: Path, name: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(model / index[name], "pt") as shard:
        return shard.get_tensor(name)


def layer0_hessian(model: Path, rows_path: Path, rows: int, device: torch.device) -> torch.Tensor:
    embed = load_tensor(model, "model.language_model.embed_tokens.weight").to(device)
    norm = load_tensor(model, "model.language_model.layers.0.input_layernorm.weight").to(device)
    with safe_open(rows_path, "pt") as f:
        ids, lengths = f.get_tensor("input_ids"), f.get_tensor("lengths")
    k = embed.shape[1]
    h = torch.zeros((k, k), dtype=torch.float64, device=device)
    count = 0
    for r in range(min(rows, ids.shape[0])):
        tokens = ids[r, : int(lengths[r])].to(device)
        x = embed[tokens].float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + norm.float())
        x = x.bfloat16().double()
        h += x.T @ x
        count += x.shape[0]
    return (h / count).float()


def proxy_error(w: torch.Tensor, wq: torch.Tensor, h: torch.Tensor) -> float:
    hd = h.double().clone()
    hd.diagonal().add_(DAMPING * hd.diagonal().mean())
    e = w.double() - wq.double()
    return float((e * (hd @ e)).sum() / (w.double() * (hd @ w.double())).sum())


def run_exllamav3(w: torch.Tensor, h: torch.Tensor, bits: float, seed: int, out_scales: str) -> dict:
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    h_data = {"H": h.clone(), "count": 1, "finalized": False, "first_key": "probe"}
    quant_args = {
        "K": int(bits) if float(bits).is_integer() else bits,
        "devices": [w.device.index or 0],
        "mul1": True,
        "apply_out_scales": {"auto": None, "always": True, "never": False}[out_scales],
        "seed": seed,
        "sigma_reg": DAMPING,
    }
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wq, proxy_rotated, _ = quantize_exl3(w.clone(), h_data, quant_args, return_weight_q=True)
    end.record()
    torch.cuda.synchronize()
    return {
        "proxy_error_rotated": float(proxy_rotated),
        "proxy_error": proxy_error(w, wq.float(), h),
        "out_scales": bool(quant_args.get("apply_out_scales")),
        "global_scale": float(quant_args.get("g_scale", 0.0)),
        "seconds": start.elapsed_time(end) / 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/mnt/storage/models/qwen3.8/full"))
    parser.add_argument(
        "--rows-file", type=Path, default=Path("profiles/exl3/traces/evaluation.calibration.safetensors")
    )
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.linear_attn.in_proj_z.weight"
    )
    parser.add_argument("--bits", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-scales", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    h = layer0_hessian(args.model, args.rows_file, args.rows, device)
    w = load_tensor(args.model, args.tensor).to(device).float().T.contiguous()  # [k][n]
    k, n = w.shape
    print(f"{args.tensor}: k={k} n={n}", file=sys.stderr)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        w_path, h_path = Path(tmp) / "w.f32", Path(tmp) / "h.f32"
        w.cpu().numpy().tofile(w_path)
        h.cpu().numpy().tofile(h_path)
        for bits in args.bits:
            half_bits = int(round(bits * 2))
            ours = json.loads(
                subprocess.run(
                    [str(args.probe), str(k), str(n), str(half_bits), str(args.seed), str(w_path), str(h_path), args.out_scales],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            theirs = run_exllamav3(w, h, bits, args.seed, args.out_scales)
            results.append({"bits": bits, "ninfer": ours, "exllamav3": theirs})
            print(
                f"K={bits}: proxy NInfer {ours['proxy_error']:.6f} (rotated {ours['proxy_error_rotated']:.6f}, "
                f"{ours['seconds']:.1f}s) | exllamav3 {theirs['proxy_error']:.6f} "
                f"(rotated {theirs['proxy_error_rotated']:.6f}, {theirs['seconds']:.1f}s) | "
                f"ratio {ours['proxy_error'] / theirs['proxy_error']:.4f}",
                file=sys.stderr,
            )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
