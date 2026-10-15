#!/usr/bin/env python3
"""Dump the canonical Nemotron-H implementation's per-stage tensors as reference data.

NInfer implements Nemotron Bolt's mathematics in its own kernels, and the oracle for those kernels has
to be an independent implementation rather than a second kernel. For this architecture that oracle is
close at hand: the checkpoint's own `modeling_nemotron_h_puzzle.py` is 37 lines and subclasses
`NemotronHBlock`/`NemotronHForCausalLM` from transformers, so `transformers` *is* the reference. This
tool runs it and writes every intermediate stage to raw files, so the C++ ops can be compared against
the mathematics they are supposed to implement -- including the two rules that are easy to get wrong
and impossible to notice: the depthwise convolution's activation applies to B and C as well as to x,
and the output norm is gated *before* it normalises, within groups. Beside a module's output the dump
also carries the tensors an op consumes and cannot reconstruct -- the router's input, weight and
correction bias -- because an op cannot be run against a dump that holds only what it produces.

Two model sources:

    --tiny            a scaled-down config built here, so no checkpoint is needed and the stages stay
                      small enough to eyeball. It keeps the real architecture's shape rules
                      (chunk_size 128, conv_kernel 4, groups, per-layer block types) and is the source
                      for op-level qualification.
    --model DIR       a downloaded checkpoint, loaded with trust_remote_code so the checkpoint's own
                      config class threads the per-layer expert counts. This is the end-to-end source.

Requires the SYSTEM interpreter, not the project venv: /usr/bin/python3 here has transformers 5.14.1
and torch 2.13, while the venv predates NemotronH entirely (the same split the Gemma 4 references
document).

Usage:
    /usr/bin/python3 tools/verify/nemotron_h_reference.py dump --tiny --out /tmp/nemo_ref
    /usr/bin/python3 tools/verify/nemotron_h_reference.py dump --model /mnt/storage/models/nemo/74B \\
        --out /tmp/nemo_ref_full --tokens 130
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

# Module suffixes whose *output* is worth comparing. Everything here is a stage NInfer has its own
# implementation of; the set is deliberately narrow so the dump stays readable.
STAGE_SUFFIXES = (
    "embeddings",
    "norm_f",
    "lm_head",
    "mixer",
    "mixer.in_proj",
    "mixer.out_proj",
    "mixer.conv1d",
    "mixer.norm",
    "mixer.gate",
    "mixer.experts",
    "mixer.shared_experts",
    "mixer.fc1_latent_proj",
    "mixer.fc2_latent_proj",
    "mixer.q_proj",
    "mixer.k_proj",
    "mixer.v_proj",
    "mixer.o_proj",
)

# A stage is a module's *output*, and an op qualified against this dump also needs the tensors it
# consumes: the router's weight and correction bias are parameters rather than stages, and its input
# is the block's normalized hidden state. These two tables name them explicitly, so a parameter name
# can never shadow a stage name.
INPUT_SUFFIXES = ("mixer.gate",)
PARAMETER_SUFFIXES = {"mixer.gate": ("weight", "e_score_correction_bias")}


def tiny_config() -> "object":
    """A small model that keeps the real architecture's shape rules.

    Sequence length defaults to 130 tokens so the chunked scan path (chunk_size 128) is the one under
    test rather than the single-token decode path, and the block types cover all three mixers.
    """
    from transformers import NemotronHConfig

    return NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=8,
        n_groups=2,
        conv_kernel=4,
        chunk_size=128,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_latent_size=16,
        moe_shared_expert_intermediate_size=32,
        moe_shared_expert_overlap=True,
        use_mamba_kernels=False,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        hidden_act="relu2",
        mlp_hidden_act="relu2",
        layer_norm_epsilon=1e-5,
    )


def load_model(args) -> tuple["object", "object"]:
    from transformers import AutoModelForCausalLM

    if args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32
        )
        return model.eval(), model.config
    model = AutoModelForCausalLM.from_config(tiny_config())
    return model.eval(), model.config


def register_hooks(model) -> dict[str, torch.Tensor]:
    """Capture the output of every module whose name ends with one of the stage suffixes."""
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            # A module may return a tuple: the router returns (logits, weights, indices), and all three
            # are part of what its op has to reproduce, so each element is dumped with its position as
            # a suffix. Everything else returns a single tensor.
            outputs = output if isinstance(output, tuple) else (output,)
            for index, tensor in enumerate(outputs):
                if not isinstance(tensor, torch.Tensor):
                    continue
                stage = name if len(outputs) == 1 else f"{name}.{index}"
                captured[stage] = tensor.detach().to(torch.float32).cpu()
            if any(name == suffix or name.endswith("." + suffix) for suffix in INPUT_SUFFIXES):
                argument = _inputs[0] if _inputs else None
                if isinstance(argument, torch.Tensor):
                    captured[f"{name}.input"] = argument.detach().to(torch.float32).cpu()
        return hook

    handles = []
    for name, module in model.named_modules():
        tail = name.split(".", 1)[-1] if "." in name else name
        if any(tail == suffix or name.endswith("." + suffix) for suffix in STAGE_SUFFIXES):
            handles.append(module.register_forward_hook(make_hook(name)))
        for suffix, parameters in PARAMETER_SUFFIXES.items():
            if tail != suffix and not name.endswith("." + suffix):
                continue
            for parameter in parameters:
                tensor = getattr(module, parameter, None)
                if isinstance(tensor, torch.Tensor):
                    captured[f"{name}.{parameter}"] = tensor.detach().to(torch.float32).cpu()
    if not handles:
        sys.exit("no stages matched: the model does not look like a NemotronH stack")
    return captured


def dump_tensor(out: pathlib.Path, name: str, tensor: torch.Tensor, manifest: list[str]) -> None:
    file_name = name.replace(".", "_") + ".f32"
    array = tensor.numpy().astype(np.float32, copy=False)
    (out / file_name).write_bytes(array.tobytes())
    manifest.append(f"{name}\t{'x'.join(str(d) for d in array.shape)}\t{file_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["dump"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--tokens", type=int, default=130, help="sequence length (default past chunk_size)")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)

    # `from_config` initialises the tiny model from the global RNG, which torch seeds from the OS, so
    # without this the dump is a different model every run and no recorded comparison can be repeated.
    # A checkpoint load has no random initialisation, so this only ever pins the tiny path.
    torch.manual_seed(arguments.seed)
    model, config = load_model(arguments)
    captured = register_hooks(model)

    generator = torch.Generator().manual_seed(arguments.seed)
    tokens = torch.randint(0, config.vocab_size, (1, arguments.tokens), generator=generator)

    with torch.no_grad():
        model(tokens)

    manifest: list[str] = []
    (out / "tokens.i32").write_bytes(tokens.to(torch.int32).numpy().tobytes())
    manifest.append(f"input_tokens\t{'x'.join(str(d) for d in tokens.shape)}\ttokens.i32")
    for name, tensor in captured.items():
        dump_tensor(out, name, tensor, manifest)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")

    # The config is what the C++ side has to reproduce from the checkpoint's own JSON, so keep the
    # resolved view beside the tensors.
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))

    print(f"{arguments.tokens} tokens, {len(captured)} stages, {len(manifest)} files -> {out}")
    for line in manifest[:12]:
        print("  " + line)
    if len(manifest) > 12:
        print(f"  ... {len(manifest) - 12} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
