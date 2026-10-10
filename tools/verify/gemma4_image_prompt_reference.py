"""Compares an image prompt run by NInfer with transformers' Gemma4ForConditionalGeneration.

ninfer_gemma4_image_prompt_test writes the prompt ids its frontend built for a user turn of images and
a question, its greedy continuation, and the logits of every step. This script builds the same prompt
with transformers (the chat template, then the processor's <|image> + soft tokens + <image|>
expansion), checks the ids are identical, and runs the full BF16 model on CPU over the prompt and
NInfer's continuation (teacher forcing). Per step it reports the KL divergence of NInfer's distribution
from transformers', both top-1 tokens, and NInfer's chosen token's log-probability under each.

It also runs transformers without `mm_token_type_ids`, which makes every image causal. The
continuation barely depends on that, so the mask is judged where it decides what a token sees: the
test's logits at a sample of the first image's own positions, computed with the block bound and
without, are compared with transformers' at the same positions in both runs. NInfer's bound run must
be far closer to transformers' bidirectional one than its causal run is.

Run from the repository root with an interpreter that has torch, Pillow, transformers and safetensors:
    python -m tools.verify.gemma4_image_prompt_reference --model DIR --dump DIR IMAGE...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from tools.verify.gemma4_vision_reference import processor

IMAGE_TOKEN, BOI, EOI = "<|image|>", "<|image>", "<image|>"


def bf16_file(path: Path, rows: int) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(bits.view(np.float32).reshape(rows, -1).copy())


def build(model: Path, images: list[Path], question: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    out = processor(model)([Image.open(p).convert("RGB") for p in images], return_tensors="pt")
    pieces = text.split(IMAGE_TOKEN)
    assert len(pieces) == len(images) + 1, "the template wrote one placeholder per image"
    expanded = pieces[0]
    for count, piece in zip(out["num_soft_tokens_per_image"], pieces[1:]):
        expanded += BOI + IMAGE_TOKEN * int(count) + EOI + piece
    ids = tokenizer(expanded, add_special_tokens=False)["input_ids"]
    return tokenizer, ids, out["pixel_values"], out["image_position_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()

    question = (args.dump / "question.txt").read_text()
    ours_ids = np.fromfile(args.dump / "ids.i32", dtype=np.int32).tolist()
    steps = np.fromfile(args.dump / "steps.i32", dtype=np.int32).tolist()
    ours = bf16_file(args.dump / "logits.bf16", len(steps)).double()

    columns = np.fromfile(args.dump / "image_columns.i32", dtype=np.int32).tolist()
    tokenizer, ids, pixels, positions = build(args.model, args.images, question)
    if ids != ours_ids:
        first = next(i for i, (a, b) in enumerate(zip(ids + [-1], ours_ids + [-1])) if a != b)
        raise SystemExit(f"prompt ids differ at {first}: HF {ids[first:first+5]} "
                         f"NInfer {ours_ids[first:first+5]} (lengths {len(ids)}/{len(ours_ids)})")
    print(f"prompt ids identical: {len(ids)} tokens; NInfer continuation: "
          f"{tokenizer.decode(steps)!r}")

    model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    sequence = torch.tensor([ids + steps[:-1]])
    image_token = model.config.image_token_id
    with torch.no_grad():
        runs = {}
        for label, types in (("bidirectional", (sequence == image_token).long()), ("causal", None)):
            kwargs = {"mm_token_type_ids": types} if types is not None else {}
            logits = model(input_ids=sequence, pixel_values=pixels, image_position_ids=positions,
                           **kwargs).logits[0]
            runs[label] = logits[len(ids) - 1:].double()
            runs["image " + label] = logits[columns].double()
    ours_log = torch.log_softmax(ours, dim=-1)
    for label, logits in list(runs.items()):
        if label.startswith("image"):
            continue
        reference = torch.log_softmax(logits, dim=-1)
        kld = (reference.exp() * (reference - ours_log)).sum(dim=-1)
        top = (reference.argmax(dim=-1) == ours.argmax(dim=-1)).float()
        chosen = torch.tensor(steps)
        gap = (ours_log.gather(1, chosen[:, None]) - reference.gather(1, chosen[:, None]))[:, 0]
        print(f"vs HF {label:13s} KLD mean {kld.mean():.4f} max {kld.max():.4f} (step "
              f"{int(kld.argmax())}); top-1 agreement {int(top.sum())}/{len(steps)}; chosen-token "
              f"log-prob gap mean |{gap.abs().mean():.4f}| max |{gap.abs().max():.4f}|")
        print("  per-step KLD " + " ".join(f"{v:.3f}" for v in kld.tolist()))
    # The mask's own effect: how far the causal run is from the bidirectional one.
    bidirectional = torch.log_softmax(runs["bidirectional"], dim=-1)
    causal = torch.log_softmax(runs["causal"], dim=-1)
    effect = (bidirectional.exp() * (bidirectional - causal)).sum(dim=-1)
    print(f"HF bidirectional vs causal KLD mean {effect.mean():.4f} max {effect.max():.4f}")
    print("  per-step KLD " + " ".join(f"{v:.3f}" for v in effect.tolist()))

    # The first image's own positions, where the mask decides what each token sees.
    def kld(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        reference, candidate = torch.log_softmax(reference, -1), torch.log_softmax(candidate, -1)
        return (reference.exp() * (reference - candidate)).sum(dim=-1)

    print(f"first image, {len(columns)} of its positions:")
    for mode in ("bidirectional", "causal"):
        ninfer = bf16_file(args.dump / f"image_logits_{mode}.bf16", len(columns)).double()
        for label in ("image bidirectional", "image causal"):
            reference = runs[label]
            value = kld(reference, ninfer)
            top = (reference.argmax(-1) == ninfer.argmax(-1)).float().sum()
            print(f"  NInfer {mode:13s} vs HF {label[6:]:13s} KLD mean {value.mean():.4f} "
                  f"median {value.median():.4f} max {value.max():.4f}; top-1 {int(top)}/{len(columns)}")
            if label[6:] == mode:
                print("    by position " + " ".join(
                    f"{c - columns[0]}:{v:.2f}" for c, v in zip(columns, value.tolist())))
    value = kld(runs["image bidirectional"], runs["image causal"])
    print(f"  HF bidirectional vs HF causal          KLD mean {value.mean():.4f} "
          f"median {value.median():.4f} max {value.max():.4f}")


if __name__ == "__main__":
    main()
