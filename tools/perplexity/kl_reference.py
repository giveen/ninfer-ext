"""Write a NInfer KL reference, produced from a BF16 run of the same text and window protocol.

The format is what `apps/perplexity/reference.{h,cpp}` reads: a fixed 96-byte little-endian header
(magic, version, vocab, rows, context, stride, reserved, 64-byte hex text sha256), then `rows` u32
scored target indices, then `rows` x vocab BF16 logits with the vocab contiguous per row.

The window protocol must match `plan_windows` in `apps/perplexity/evaluation.cpp`: the first window
covers [0, min(tokens, context)) and scores [1, that), and every later window covers
[end - context, end) and advances by `stride` scored targets.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys

import numpy as np

MAGIC = b"NINFKL1\0"
VERSION = 1
HEADER_BYTES = 96


def plan_windows(tokens: int, context: int, stride: int):
    """(input_begin, input_end, target_begin, target_end, first_target) per window."""
    if tokens < 2:
        raise ValueError("stream must contain at least two tokens")
    if context < 2 or not 1 <= stride < context:
        raise ValueError("context>=2 and 1<=stride<context")
    windows = []
    previous_end = min(tokens, context)
    windows.append((0, previous_end, 1, previous_end, 1))
    while previous_end < tokens:
        end = min(tokens, previous_end + stride)
        begin = end - context if end > context else 0
        local_target = previous_end - begin
        windows.append((begin, end, previous_end, end, local_target))
        previous_end = end
    return windows


def write_reference(path, *, context, stride, text, positions, logits, vocab_size):
    """positions: [rows] int target indices; logits: [rows, vocab] floats, stored BF16."""
    positions = np.asarray(positions, dtype=np.uint32)
    logits = np.asarray(logits)
    if logits.shape != (positions.size, vocab_size):
        raise ValueError(f"logits shape {logits.shape} does not match {(positions.size, vocab_size)}")
    if np.unique(positions).size != positions.size:
        raise ValueError("duplicate scored position")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest().encode("ascii")
    if len(digest) != 64:
        raise ValueError("malformed digest")
    header = MAGIC + struct.pack("<IIIIII", VERSION, vocab_size, positions.size, context, stride, 0)
    with open(path, "wb") as handle:
        handle.write(header.ljust(HEADER_BYTES, b"\0"))
        handle.write(digest.ljust(64, b"\0"))
        handle.write(positions.tobytes())
        handle.write(to_bf16_bits(logits).tobytes())
    return digest.decode("ascii")


def to_bf16_bits(values):
    """BF16 bit patterns as uint16: round FP32 to nearest-even and keep the top 16 bits."""
    fp32 = np.asarray(values, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def produce(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = open(args.text, "r", encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map=args.device_map)
    model.eval()

    windows = plan_windows(len(tokens), args.context, args.stride)
    positions, rows = [], []
    with torch.no_grad():
        for index, (begin, end, target_begin, target_end, first_target) in enumerate(windows):
            if args.windows and index >= args.windows:
                break
            ids = torch.tensor([tokens[begin:end]], device=model.device)
            logits = model(ids).logits[0]
            # Local predictors [first_target, end - begin) predict the tokens after them.
            for local in range(first_target, end - begin):
                positions.append(target_begin + (local - first_target))
                rows.append(logits[local].float().cpu().numpy())
            print(f" -- window {index + 1}/{len(windows)}: {len(positions)} rows", file=sys.stderr)
    vocab_size = int(model.config.vocab_size)
    digest = write_reference(args.out, context=args.context, stride=args.stride, text=text,
                             positions=positions, logits=np.stack(rows), vocab_size=vocab_size)
    print(f" -- wrote {args.out}: {len(positions)} rows, vocab {vocab_size}, text {digest}",
          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="BF16 Hugging Face model directory")
    parser.add_argument("--text", required=True, help="UTF-8 text scored by both engines")
    parser.add_argument("--out", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (auto offloads)")
    parser.add_argument("--windows", type=int, default=0, help="stop after N windows (0 = all)")
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
