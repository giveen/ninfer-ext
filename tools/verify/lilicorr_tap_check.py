"""Checks a tap dump against the target's own head.

A tap is only useful if it is the residual stream of the block it is filed under, at the position it
is filed under, laid out the way the draft's `fc` reads it. This checks all three at once, and
against the target's own weights rather than against another copy of our capture: take the *last*
block's residual, apply the target's final norm and its LM head, and the argmax must be the token
the dump's own greedy stream drew next.

    # capture the last block's residual
    NINFER_NEMOTRON_ARTIFACT=out/... NINFER_TAP_LAYERS=87 NINFER_TAP_DUMP=/tmp/taps-last \\
        ./build/tests/ninfer_nemotron_h_draft_taps_test
    python3 tools/verify/lilicorr_tap_check.py --dump /tmp/taps-last --target /mnt/storage/models/nemo/74B

The prompt region and the generated region are reported separately: the prompt's tokens are the
text's own, while every token past it is the argmax the engine drew, so a capture that files the
generated columns wrongly shows up as a prompt-region match and a tail-region mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lilicorr_reference as ref  # noqa: E402


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=87, help="the dumped block (the last one)")
    parser.add_argument("--eps", type=float, default=1e-5, help="the target's layer-norm epsilon")
    parser.add_argument("--chunk", type=int, default=64, help="positions scored per matmul")
    parser.add_argument("--print-mismatches", type=int, default=5)
    args = parser.parse_args()

    meta = json.loads((args.dump / "meta.json").read_text())
    hidden = int(meta["hidden"])
    positions = int(meta["positions"])
    prompt = int(meta["prompt_tokens"])
    tokens = np.fromfile(args.dump / "tokens.i32", dtype=np.int32)
    raw = np.fromfile(args.dump / f"tap_{args.layer}.bf16", dtype=np.uint16)
    if raw.size != positions * hidden:
        raise ValueError(f"tap_{args.layer}.bf16 is not [{positions}, {hidden}]")
    taps = (raw.astype(np.uint32) << 16).view(np.float32).reshape(positions, hidden)

    tables = ref.TargetTables(args.target, 131072, hidden, seed=0)
    norm_key = "language_model.backbone.norm_f.weight"
    weight_map = json.loads((args.target / "model.safetensors.index.json").read_text())["weight_map"]
    norm = ref._read_tensor(args.target / weight_map[norm_key], norm_key)
    print(f"final norm {tuple(norm.shape)} from {weight_map[norm_key]}; head {tables.source}")

    agree = np.zeros(positions - 1, dtype=bool)
    rank = np.zeros(positions - 1, dtype=np.int32)
    for start in range(0, positions - 1, args.chunk):
        stop = min(start + args.chunk, positions - 1)
        rows = torch.from_numpy(taps[start:stop]).unsqueeze(0)
        logits = tables.logits(rms_norm(rows, norm, args.eps))[0]
        want = torch.from_numpy(tokens[start + 1 : stop + 1].astype(np.int64))
        agree[start:stop] = (logits.argmax(dim=-1) == want).numpy()
        # How many vocabulary entries outrank the token the engine actually drew: a correct
        # residual puts it first, and a capture that mixes columns up puts it somewhere uniform.
        rank[start:stop] = (logits > logits.gather(1, want.unsqueeze(1))).sum(dim=-1).numpy()

    def region(name: str, first: int, last: int) -> None:
        if last <= first:
            return
        matches = agree[first:last]
        ranks = rank[first:last]
        print(f"  {name}: {matches.sum()}/{last - first} argmax matches "
              f"({100.0 * matches.mean():.1f}%) | rank of the drawn token: mean {ranks.mean():.1f}, "
              f"median {int(np.median(ranks))}, worst {ranks.max()}")

    print("the target's own head over the captured last-block residual, against the dumped stream:")
    region("prompt", 0, prompt - 1)
    region("generated", prompt - 1, positions - 1)
    region("all", 0, positions - 1)
    shown = 0
    for at in range(positions - 1):
        if agree[at] or shown >= args.print_mismatches:
            continue
        shown += 1
        print(f"  mismatch at position {at}: drew {int(tokens[at + 1])}, head ranks it "
              f"{int(rank[at])}")
    return 0 if agree.mean() > 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
