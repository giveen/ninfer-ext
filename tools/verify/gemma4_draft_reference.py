"""Compares NInfer's Gemma 4 assistant drafter with transformers' on identical inputs.

ninfer_gemma4_draft_test dumps every input the drafter read (the anchor and draft tokens, the target's
post-final-norm state, the target's last sliding ring and last global compact rows) and what each step
wrote. This script rebuilds the shared KV transformers expects from those caches, runs
Gemma4AssistantForCausalLM in FP32 one step at a time on NInfer's own step inputs, and reports how
each step's logits and projected state agree.

Run with an interpreter that has torch, transformers and safetensors:
    python -I tools/verify/gemma4_draft_reference.py --dump DIR \\
        --target /mnt/storage/models/gemma/full-31b --drafter /mnt/storage/models/gemma/assistant-31b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import Gemma4AssistantForCausalLM


def bf16(path: Path, shape) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(bits.view(np.float32).reshape(shape).copy())


def target_tensor(root: Path, name: str) -> torch.Tensor:
    for file in sorted(root.glob("*.safetensors")):
        with safe_open(file, framework="pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(name)


def embedding_rows(root: Path, ids: list[int]) -> torch.Tensor:
    name = "model.language_model.embed_tokens.weight"
    for file in sorted(root.glob("*.safetensors")):
        with safe_open(file, framework="pt") as handle:
            if name in handle.keys():
                table = handle.get_slice(name)
                return torch.stack([table[i : i + 1][0] for i in ids])
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    args = parser.parse_args()

    meta = {}
    for line in (args.dump / "meta.txt").read_text().splitlines():
        key, *values = line.split()
        meta[key] = [int(v) for v in values]
    p, ring = meta["position"][0], meta["ring"][0]
    tokens = meta["tokens"]
    global_layer = meta["global_layer"][0]
    steps = len(tokens) - 1

    model = Gemma4AssistantForCausalLM.from_pretrained(args.drafter, torch_dtype=torch.float32)
    model.eval()
    # The per-layer config object refuses the global fields, so they come from the raw file.
    raw = json.loads((args.drafter / "config.json").read_text())
    text = raw["text_config"]
    hidden = raw["backbone_hidden_size"]
    vocabulary = text["vocab_size"]
    d, hkv = text["head_dim"], text["num_key_value_heads"]
    gd, ghkv = text["global_head_dim"], text["num_global_key_value_heads"]
    angles = gd // 2 // 4  # proportional RoPE: a quarter of the pairs rotate
    window = text["sliding_window"]

    # The sliding ring, in position order, restricted to the keys before the anchor's position.
    positions = np.fromfile(args.dump / "ring_positions.i32", dtype=np.int32)
    keys = bf16(args.dump / "ring_keys.bf16", (ring, hkv, d))
    values = bf16(args.dump / "ring_values.bf16", (ring, hkv, d))

    slots = [s for s in np.argsort(positions) if p - window <= positions[s] <= p - 1]
    assert len(slots) == min(window, p), (len(slots), p)
    sliding_k = keys[slots].permute(1, 0, 2).unsqueeze(0)
    sliding_v = values[slots].permute(1, 0, 2).unsqueeze(0)

    # The global rows hold [value | rotated key dims]; the key elsewhere is the value times the
    # source layer's key-norm weight.
    rows = bf16(args.dump / "global_rows.bf16", (p, ghkv, gd + 2 * angles))
    value = rows[:, :, :gd]
    k_norm = target_tensor(args.target,
                           f"model.language_model.layers.{global_layer}.self_attn.k_norm.weight")
    key = value * k_norm.float()
    key[:, :, :angles] = rows[:, :, gd : gd + angles]
    key[:, :, gd // 2 : gd // 2 + angles] = rows[:, :, gd + angles :]
    global_k = key.permute(1, 0, 2).unsqueeze(0)
    global_v = value.permute(1, 0, 2).unsqueeze(0)
    shared = {"sliding_attention": (sliding_k, sliding_v), "full_attention": (global_k, global_v)}

    scale = torch.tensor(math.sqrt(hidden), dtype=torch.bfloat16)
    embeddings = (embedding_rows(args.target, tokens[:steps]) * scale).float()
    state = bf16(args.dump / "hidden0.bf16", (hidden,))
    position = torch.tensor([[p]])
    worst = 0.0
    for step in range(steps):
        inputs = torch.cat([embeddings[step], state]).view(1, 1, 2 * hidden)
        with torch.no_grad():
            out = model(inputs_embeds=inputs, position_ids=position, shared_kv_states=shared)
        reference = out.logits[0, 0].double()
        ours = bf16(args.dump / f"step{step}_logits.bf16", (vocabulary,)).double()
        ours_state = bf16(args.dump / f"step{step}_hidden.bf16", (hidden,))
        cosine = torch.nn.functional.cosine_similarity(ours, reference, dim=0).item()
        state_cos = torch.nn.functional.cosine_similarity(
            ours_state.double(), out.last_hidden_state[0, 0].double(), dim=0).item()
        top_ref = torch.topk(reference, 5).indices.tolist()
        top_ours = torch.topk(ours, 5).indices.tolist()
        diff = (ours - reference).abs()
        print(f"step {step}: logits cosine {cosine:.6f}, max |d| {diff.max().item():.4f} "
              f"(scale {reference.abs().max().item():.2f}), argmax ours {top_ours[0]} ref "
              f"{top_ref[0]}, top-5 overlap {len(set(top_ours) & set(top_ref))}/5, "
              f"state cosine {state_cos:.6f}")
        worst = max(worst, 1.0 - cosine, 1.0 - state_cos)
        # The next step reads NInfer's own projected state, so each step is compared on its own.
        state = ours_state
    print(f"worst 1 - cosine: {worst:.2e}")


if __name__ == "__main__":
    main()
