#!/usr/bin/env python3
"""Export the tiny Nemotron-H model's routed-expert banks beside its stage dump.

`nemotron_h_reference.py --tiny` records every stage an Op has to reproduce, but an Op also needs
the operands it *consumes* and cannot reconstruct. For the routed experts those are the two banks
themselves -- `model.layers.N.mixer.experts.{up,down}_proj`, 3-D `(experts, intermediate, latent)`
tensors nested two levels inside the mixer -- and the stage tool deliberately records only stages,
the router's input and the router's own parameters.

Rather than change that tool, this one rebuilds the tiny model exactly as it does (its own
`tiny_config`, the same seed, the same `from_config`) and *checks* the rebuild against the dump
before writing anything: the router's three outputs, the latent projection's output and the
experts' own output must all come back. If they do not, the rebuild is a different model and this
tool fails rather than exporting operands that do not belong to the dump.

The banks are written flat, in the layout the Op reads them in -- `up` as
`(experts*intermediate, latent)` and `down` as `(experts*latent, intermediate)`, expert-major,
which is exactly the row-major order of the reference's own 3-D tensors -- with a manifest in the
stage dump's format, so a reader of that dump needs no second parser.

Usage:
    /usr/bin/python3 tools/verify/nemotron_h_expert_banks.py --dump /tmp/nemo_ref
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The tiny configuration and the model construction are the stage tool's own, so the two tools can
# never disagree about what the tiny model is.
import nemotron_h_reference as reference  # noqa: E402

# The stages a rebuild is accepted against, per MoE layer: the router's logits, weights and indices,
# the latent projection the experts consume, and the experts' own output. The first two suffixes
# name *modules* (the router returns a tuple, so its stages carry an index); the rest name stages.
MODULE_SUFFIXES = ("mixer.gate", "mixer.fc1_latent_proj", "mixer.experts")
CHECK_SUFFIXES = ("mixer.gate.0", "mixer.gate.1", "mixer.gate.2", "mixer.fc1_latent_proj",
                  "mixer.experts")

# The rebuild is a fresh construction of the same model, so it is the *same* weights and the same
# FP32 arithmetic; anything above this is a different model rather than a different reduction order.
TOLERANCE = 1.0e-4


def read_dump(directory: pathlib.Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    manifest = (directory / "manifest.txt").read_text().splitlines()
    for line in manifest:
        if not line:
            continue
        name, shape, file_name = line.split("\t")
        raw = np.frombuffer((directory / file_name).read_bytes(), dtype="<f4")
        tensors[name] = raw.reshape([int(extent) for extent in shape.split("x")])
    return tensors


def capture(model: "torch.nn.Module", tokens: "torch.Tensor") -> dict[str, np.ndarray]:
    """Run the model and keep the output of every checked module."""
    wanted = {name for name, _ in model.named_modules()
              if any(name.endswith(suffix) for suffix in MODULE_SUFFIXES)}
    captured: dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            outputs = output if isinstance(output, tuple) else (output,)
            for index, tensor in enumerate(outputs):
                if not isinstance(tensor, torch.Tensor):
                    continue
                stage = name if len(outputs) == 1 else f"{name}.{index}"
                captured[stage] = tensor.detach().to(torch.float32).cpu().numpy()

        return hook

    handles = [module.register_forward_hook(make_hook(name))
               for name, module in model.named_modules() if name in wanted]
    if not handles:
        sys.exit("no MoE stages matched: the tiny model does not look like a NemotronH stack")
    with torch.no_grad():
        model(tokens)
    return captured


def write_array(out: pathlib.Path, name: str, array: np.ndarray, manifest: list[str]) -> None:
    file_name = name.replace(".", "_") + ".f32"
    (out / file_name).write_bytes(array.astype("<f4", copy=False).tobytes())
    manifest.append(f"{name}\t{'x'.join(str(d) for d in array.shape)}\t{file_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=pathlib.Path, required=True,
                        help="a nemotron_h_reference.py --tiny dump to export beside")
    parser.add_argument("--out", type=pathlib.Path,
                        help="where to write (default: the dump directory itself)")
    parser.add_argument("--seed", type=int, default=0,
                        help="the stage dump's own seed (default 0)")
    arguments = parser.parse_args()

    out = arguments.out or arguments.dump
    out.mkdir(parents=True, exist_ok=True)
    dumped = read_dump(arguments.dump)
    tokens = torch.from_numpy(
        np.frombuffer((arguments.dump / "tokens.i32").read_bytes(), dtype="<i4").copy()
    ).reshape(1, -1)

    from transformers import AutoModelForCausalLM

    torch.manual_seed(arguments.seed)
    model = AutoModelForCausalLM.from_config(reference.tiny_config()).eval()
    captured = capture(model, tokens)

    # The rebuild is only trusted if it reproduces the dump's own stages.
    worst = 0.0
    for name, wanted in dumped.items():
        if not any(name.endswith(suffix) for suffix in CHECK_SUFFIXES):
            continue
        got = captured.get(name)
        if got is None:
            sys.exit(f"the rebuild produced no {name}")
        if got.shape != wanted.shape:
            sys.exit(f"{name}: the rebuild's shape {got.shape} is not the dump's {wanted.shape}")
        worst = max(worst, float(np.max(np.abs(got - wanted))))
    if worst > TOLERANCE:
        sys.exit(f"the rebuild does not reproduce the dump (worst absolute difference {worst:g}): "
                 f"the dump is from a different seed or a different config")
    print(f"rebuild matches the dump to {worst:g} at the largest (tolerance {TOLERANCE:g})")

    manifest: list[str] = []
    for layer, module in enumerate(model.model.layers):
        if not hasattr(module, "mixer") or not hasattr(module.mixer, "experts"):
            continue
        experts = module.mixer.experts
        prefix = f"model.layers.{layer}.mixer.experts"
        up = experts.up_proj.detach().to(torch.float32).cpu().numpy()
        down = experts.down_proj.detach().to(torch.float32).cpu().numpy()
        count, intermediate, latent = up.shape
        if down.shape != (count, latent, intermediate):
            sys.exit(f"{prefix}: up {up.shape} and down {down.shape} do not pair")
        # The Op's own bank layout: expert-major rows, which is the row-major order of the 3-D
        # tensors the reference stores.
        write_array(out, f"{prefix}.up_proj", up.reshape(count * intermediate, latent), manifest)
        write_array(out, f"{prefix}.down_proj", down.reshape(count * latent, intermediate),
                    manifest)
        print(f"layer {layer}: {count} experts, intermediate {intermediate}, latent {latent}")

    (out / "expert_banks_manifest.txt").write_text("\n".join(manifest) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
