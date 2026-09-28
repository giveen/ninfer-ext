"""Sanity-check a KL reference against the text it was produced from.

Both producer bugs so far were silent: a 64-byte header shift still produced a finite KL, and an
off-by-one position still produced a finite KL. This checks the reference is actually predictive,
which is the property no down stream KL can be trusted without.

    python tools/perplexity/check_reference.py <reference.bin> <text.txt>

Reports how often the row's argmax is the token that actually follows (about 60% on natural text is
healthy; a few percent means the positions or the logits are wrong).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

MAGIC = b"NINFKL1\0"
HEADER_BYTES = 96


def load(path):
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic")
    version, vocab, rows, context, stride, _reserved = np.frombuffer(raw[8:32], dtype="<u4")
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    digest = raw[32:96].decode("ascii")
    payload = 96 + 4 * int(rows)
    expected = payload + int(rows) * int(vocab) * 2
    if len(raw) != expected:
        # The 64-byte header shift showed up exactly here.
        raise SystemExit(f"{path}: size {len(raw)} != expected {expected} "
                         f"(rows={rows} vocab={vocab})")
    positions = np.frombuffer(raw[96:payload], dtype="<u4").astype(np.int64)
    logits = np.frombuffer(raw[payload:], dtype="<u2").reshape(int(rows), int(vocab))
    return int(vocab), int(context), int(stride), digest, positions, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("text")
    parser.add_argument("--model", default="/mnt/storage/models/qwen3.8/full",
                        help="Hugging Face directory whose tokenizer matches the producer")
    args = parser.parse_args()

    vocab, context, stride, digest, positions, logits = load(args.reference)
    text = open(args.text, encoding="utf-8").read()
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"rows={len(positions)} vocab={vocab} context={context} stride={stride}")
    print(f"positions {positions[:4].tolist()} .. {positions[-2:].tolist()}")
    print(f"text digest {'matches' if digest == text_digest else 'DIFFERS from'} the file")

    from transformers import AutoTokenizer

    tokens = AutoTokenizer.from_pretrained(args.model)(text, add_special_tokens=False)["input_ids"]
    out_of_range = int((positions >= len(tokens)).sum())
    print(f"tokens={len(tokens)}  positions out of range: {out_of_range}")

    values = np.ascontiguousarray((logits.astype(np.uint32) << 16).view(np.float32))
    targets = np.array([tokens[p] for p in positions], dtype=np.int64)
    argmax = values.argmax(axis=1)
    top5 = np.argpartition(-values, 5, axis=1)[:, :5]
    hit = float((argmax == targets).mean())
    in5 = float(np.mean([t in top5[i] for i, t in enumerate(targets)]))
    print(f"argmax == next token: {hit * 100:.1f}%")
    print(f"next token in top-5:  {in5 * 100:.1f}%")
    if hit < 0.30:
        print("SUSPICIOUS: the reference does not predict the text", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
