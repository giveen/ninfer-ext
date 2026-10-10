"""Compares NInfer's Gemma 4 image encoder with transformers' on identical patches.

`prepare` runs transformers' Gemma4ImageProcessorPil (Pillow bicubic) on each image and writes, per
image, the BF16 patches the tower reads (2 * pixel - 1, patch-major, pixels in (row, column, channel) order) and the
patch grid. ninfer_gemma4_vision_test encodes them and writes its soft tokens beside them. `compare`
then runs transformers' Gemma4VisionModel and Gemma4MultimodalEmbedder (only the tower is loaded) on
the processor's own padded tensors in FP32, and reports per-token cosine and relative L2 of NInfer
against it, with transformers' BF16 tower against the same FP32 as the floor. A third reference runs the
FP32 tower on weights quantized and decoded by the converter's own encoders in the artifact's formats
(`--formats`), which separates what the formats cost from what the implementation adds.

Run from the repository root with an interpreter that has torch, Pillow, transformers and safetensors:
    python -m tools.verify.gemma4_vision_reference prepare --model DIR --dump OUT IMAGE...
    python -m tools.verify.gemma4_vision_reference compare --model DIR --dump OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Gemma4Config
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)

from tools.artifact.formats import get_format
from tools.convert.quantization import groupwise

# Candidate vision formats; the artifact's (tools/convert/official_recipes.py) is Q8 throughout.
FORMATS = {"q4": "q4_g64_fp16", "q5": "q5_g64_fp16", "q6": "q6_g64_fp16", "q8": "q8_g32_fp16"}
GATE_UP = ("mlp.gate_proj", "mlp.up_proj")
REST = ("mlp.down_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj")


def fake_groupwise(w: torch.Tensor, fmt: str) -> torch.Tensor:
    spec = get_format(FORMATS[fmt])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = groupwise.quantize_matrix(w, spec, device=device)
    n, k = w.shape
    values = q.codes.float() * q.scales.float().unsqueeze(-1)
    return values.reshape(n, -1)[:, :k].cpu()


@torch.no_grad()
def quantize_tower(vision, gate_up: str, rest: str) -> None:
    recipe = {**{name: gate_up for name in GATE_UP}, **{name: rest for name in REST}}
    for layer in vision.encoder.layers:
        for name, fmt in recipe.items():
            module = layer.get_submodule(name).linear
            module.weight.copy_(fake_groupwise(module.weight.float(), fmt).to(module.weight.dtype))


def processor(model: Path) -> Gemma4ImageProcessorPil:
    config = json.loads((model / "processor_config.json").read_text())["image_processor"]
    config = {k: v for k, v in config.items() if k != "image_processor_type"}
    return Gemma4ImageProcessorPil(**config)


def processed(model: Path, image: Path):
    out = processor(model)(Image.open(image).convert("RGB"), return_tensors="pt")
    return out["pixel_values"], out["image_position_ids"], int(out["num_soft_tokens_per_image"][0])


def to_bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def prepare(args) -> None:
    args.dump.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(args.images):
        pixels, positions, tokens = processed(args.model, image)
        real = (positions[0] >= 0).all(dim=-1)
        count = int(real.sum())
        width = int(positions[0, :count, 0].max()) + 1
        height = int(positions[0, :count, 1].max()) + 1
        assert width * height == count == tokens * 9, (width, height, count, tokens)
        # The patch embedder's own input mapping, 2 * (pixel - 0.5), rounded to the weight dtype.
        to_bf16_bits(2.0 * (pixels[0, :count] - 0.5)).tofile(args.dump / f"image{index}.bin")
        (args.dump / f"image{index}.grid").write_text(f"{width} {height}\n")
        (args.dump / f"image{index}.source").write_text(str(image.resolve()) + "\n")
        print(f"image{index}: {image.name} -> {width}x{height} patches, {tokens} soft tokens")


def tower(model: Path, dtype: torch.dtype):
    config = Gemma4Config.from_pretrained(model)
    config.vision_config._attn_implementation = "eager"
    vision = Gemma4VisionModel(config.vision_config)
    embed = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {}
    for name, file in index.items():
        if name.startswith("model.vision_tower."):
            wanted[name] = (file, vision, name[len("model.vision_tower."):])
        elif name.startswith("model.embed_vision."):
            wanted[name] = (file, embed, name[len("model.embed_vision."):])
    states = {id(vision): {}, id(embed): {}}
    for name, (file, module, key) in wanted.items():
        with safe_open(model / file, framework="pt") as handle:
            states[id(module)][key] = handle.get_tensor(name)
    for module in (vision, embed):
        missing, unexpected = module.load_state_dict(states[id(module)], strict=False)
        assert not unexpected and all("inv_freq" in m for m in missing), (missing, unexpected)
        module.to(dtype).eval()
    return vision, embed


@torch.no_grad()
def features(vision, embed, pixels, positions, dtype):
    hidden = vision(pixel_values=pixels.to(dtype), pixel_position_ids=positions).last_hidden_state
    return embed(hidden).float()


def compare(args) -> None:
    fp32 = tower(args.model, torch.float32)
    bf16 = tower(args.model, torch.bfloat16)
    quantized = tower(args.model, torch.float32)
    quantize_tower(quantized[0], args.gate_up, args.rest)
    worst = 1.0
    for grid in sorted(args.dump.glob("image*.grid")):
        stem = grid.stem
        source = Path((args.dump / f"{stem}.source").read_text().strip())
        pixels, positions, tokens = processed(args.model, source)
        reference = features(*fp32, pixels, positions, torch.float32)
        floor = features(*bf16, pixels, positions, torch.bfloat16)
        formats = features(*quantized, pixels, positions, torch.float32)
        bits = np.fromfile(args.dump / f"{stem}.out.bin", dtype=np.uint16).astype(np.uint32) << 16
        ours = torch.from_numpy(bits.view(np.float32).reshape(tokens, -1).copy())
        for label, value, base in (("ninfer", ours, reference), ("hf-bf16", floor, reference),
                                   ("hf-fmt", formats, reference),
                                   ("ninfer~fmt", ours, formats)):
            cosine = torch.nn.functional.cosine_similarity(value, base, dim=-1)
            relative = (value - base).norm(dim=-1) / base.norm(dim=-1)
            overall = float((value - base).norm() / base.norm())
            print(f"{stem} {label:10s} tokens={tokens} cosine min={cosine.min():.6f} "
                  f"mean={cosine.mean():.6f}  rel-L2 max={relative.max():.4f} "
                  f"mean={relative.mean():.4f} all={overall:.4f}")
            if label == "ninfer":
                worst = min(worst, float(cosine.mean()))
    print(f"worst NInfer mean token cosine {worst:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("images", type=Path, nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--dump", type=Path, required=True)
    c.add_argument("--gate-up", choices=sorted(FORMATS), default="q8",
                   help="the simulated format of the MLP's gate and up projections")
    c.add_argument("--rest", choices=sorted(FORMATS), default="q8",
                   help="the simulated format of q/k/v, the attention output and the MLP's down")
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else compare(args)


if __name__ == "__main__":
    main()
