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
correction bias -- because an op cannot be run against a dump that holds only what it produces. For
composed work it also carries the decoder blocks themselves: every `model.layers.N` output beside the
residual stream it was handed, so a forward can be qualified stage by stage rather than only at the
logits. `--weights` adds every parameter under those layers (and the embedding, final norm and head
matrices), which is how a model-side test builds its parameters before a converter exists; it is off by
default because on the full checkpoint it would be the checkpoint.

Two model sources:

    --tiny            a scaled-down config built here, so no checkpoint is needed and the stages stay
                      small enough to eyeball. It keeps the real architecture's shape rules
                      (chunk_size 128, conv_kernel 4, groups, per-layer block types) and is the source
                      for op-level qualification.
    --model DIR       a downloaded checkpoint, loaded with trust_remote_code so the checkpoint's own
                      config class threads the per-layer expert counts. This is the end-to-end source.

The dump has two modes rather than being two tools. The default is the sequence above: one pass, every
stage of it. `--decode` is the stateful one, and it exists because a decoder's decode is not a shorter
prefill: it is a second call that hands the model the prefix's state. It prefills `--prefix` tokens,
runs the single-token step after them, and dumps *that step's* stages under `DIR/decode/`, under the
same names as the whole-sequence dump so a comparator walks the directory it already knows. One name
is absent there by construction: a single token with a cached state never calls `nn.Conv1d` -- the
reference convolves the cached history against the module's own weight -- so `mixer.conv1d` has no
output on a step, and the run prints the stage names that only a prefill pass has.

Beside the stages it dumps the reference's own state after the step, flat-named `state.*` and in the
same manifest: each mamba block's convolution buffer and recurrent plane, and each attention block's
cached K/V and its length. Those are indexed by the block's ordinal *within its kind*, because the
tree's state pool counts mamba and attention blocks separately rather than by absolute block index.
The step's token is dumped with them (`decode_token`), since a step means nothing without it, and a
control forward of the same token through an empty cache must differ from the dumped step: a decode
that never read the prefix's state still produces perfectly reasonable numbers, so reading it is shown
rather than assumed. The default prefix length is the default sequence length, so the decode dump's
prefix is the same tokens the whole-sequence dump carries, token for token.

Requires the SYSTEM interpreter, not the project venv: /usr/bin/python3 here has transformers 5.14.1
and torch 2.13, while the venv predates NemotronH entirely (the same split the Gemma 4 references
document).

Usage:
    /usr/bin/python3 tools/verify/nemotron_h_reference.py dump --tiny --out /tmp/nemo_ref
    /usr/bin/python3 tools/verify/nemotron_h_reference.py dump --model /mnt/storage/models/nemo/74B \\
        --out /tmp/nemo_ref_full --tokens 130
    /usr/bin/python3 tools/verify/nemotron_h_reference.py dump --tiny --decode --prefix 130 \\
        --out /tmp/nemo_ref --weights
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
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

# The decoder block is matched by name rather than by suffix, because its name ends in a layer index.
# Its output is the residual stream the next block consumes, and its input is what it was handed, so a
# composed forward can be qualified one block at a time instead of only at the logits.
LAYER_PATTERN = re.compile(r"^model\.layers\.\d+$")
# With `--weights`: the parameters outside the layers that a whole-model forward needs.
TOP_LEVEL_PARAMETERS = ("model.embeddings.weight", "model.norm_f.weight", "lm_head.weight")

# The block kinds that carry state, and the two names each of them answers to. transformers renames a
# config's legacy `layers_block_type` spellings when it builds the cache -- `mamba` becomes
# `linear_attention` and `attention` becomes `full_attention` -- while a checkpoint loaded with
# `trust_remote_code` may keep either, so both are accepted. `moe` is in neither set on purpose: its
# blocks carry no state at all, and neither does `mlp`.
MAMBA_BLOCK_KINDS = ("mamba", "linear_attention")
ATTENTION_BLOCK_KINDS = ("attention", "full_attention")


def state_block_indices(config) -> tuple[list[int], list[int]]:
    """(mamba block indices, attention block indices), each in the config's own block order.

    The ordinals a state file is named by are positions *within a kind*, not block indices: the tree's
    state pool counts one mamba state class and one attention state class, so `state.conv.1` is the
    second mamba block whatever its absolute index is.
    """
    kinds = getattr(config, "layer_types", None) or getattr(config, "layers_block_type", None) or []
    return (
        [index for index, kind in enumerate(kinds) if kind in MAMBA_BLOCK_KINDS],
        [index for index, kind in enumerate(kinds) if kind in ATTENTION_BLOCK_KINDS],
    )


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


def register_hooks(model, weights: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Capture every staged module's output, the blocks themselves, and with `weights` their parameters.

    The two come back as separate dictionaries because their lifetimes differ: a parameter is captured
    once, here, while a stage is whatever the most recent forward wrote, so the decode mode -- which
    runs one forward per call over a single registration, the prefix, the step it dumps and the control
    that shows the step read the prefix's state -- can empty the stages between calls without losing
    the parameters.
    """
    parameters: dict[str, torch.Tensor] = {}
    stages: dict[str, torch.Tensor] = {}

    def make_hook(name: str, capture_input: bool = False):
        def hook(_module, _inputs, output):
            # A module may return a tuple: the router returns (logits, weights, indices), and all three
            # are part of what its op has to reproduce, so each element is dumped with its position as
            # a suffix. Everything else returns a single tensor.
            outputs = output if isinstance(output, tuple) else (output,)
            for index, tensor in enumerate(outputs):
                if not isinstance(tensor, torch.Tensor):
                    continue
                stage = name if len(outputs) == 1 else f"{name}.{index}"
                stages[stage] = tensor.detach().to(torch.float32).cpu()
            if capture_input or any(
                name == suffix or name.endswith("." + suffix) for suffix in INPUT_SUFFIXES
            ):
                argument = _inputs[0] if _inputs else None
                if isinstance(argument, torch.Tensor):
                    stages[f"{name}.input"] = argument.detach().to(torch.float32).cpu()
        return hook

    handles = []
    for name, module in model.named_modules():
        tail = name.split(".", 1)[-1] if "." in name else name
        is_layer = LAYER_PATTERN.match(name) is not None
        if is_layer or any(tail == suffix or name.endswith("." + suffix) for suffix in STAGE_SUFFIXES):
            handles.append(module.register_forward_hook(make_hook(name, capture_input=is_layer)))
        for suffix, parameter_names in PARAMETER_SUFFIXES.items():
            if tail != suffix and not name.endswith("." + suffix):
                continue
            for parameter in parameter_names:
                tensor = getattr(module, parameter, None)
                if isinstance(tensor, torch.Tensor):
                    parameters[f"{name}.{parameter}"] = tensor.detach().to(torch.float32).cpu()
    if not handles:
        sys.exit("no stages matched: the model does not look like a NemotronH stack")
    if weights:
        for name, parameter in model.named_parameters():
            if name.startswith("model.layers.") or name in TOP_LEVEL_PARAMETERS:
                parameters[name] = parameter.detach().to(torch.float32).cpu()
    return parameters, stages


def dump_tensor(out: pathlib.Path, name: str, tensor: torch.Tensor, manifest: list[str]) -> None:
    file_name = name.replace(".", "_") + ".f32"
    # A live cache tensor is not a forward hook's copy: it can still be on the model's device, so the
    # detach and the cast belong here rather than at one of the callers.
    array = tensor.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    (out / file_name).write_bytes(array.tobytes())
    manifest.append(f"{name}\t{'x'.join(str(d) for d in array.shape)}\t{file_name}")


def dump_states(cache, config, out: pathlib.Path, manifest: list[str]) -> None:
    """The reference's own state classes after the step, flat-named the way the tree's pool indexes.

    These are the tensors a decoder reads back on its next token -- the convolution's raw history and
    the scan's recurrent plane for a mamba block, the K/V the attention block has already appended
    itself to for an attention block -- so a state the tree computes can be compared against the
    state the specification describes rather than only against the logits it leads to. The attention
    length is dumped as one FP32 element because every file in a manifest is the comparator's raw
    FP32, and a token position this small is exact there.
    """
    mamba_indices, attention_indices = state_block_indices(config)
    for ordinal, block in enumerate(mamba_indices):
        layer = cache.layers[block]
        conv_states = getattr(layer, "conv_states", None) or {}
        recurrent_states = getattr(layer, "recurrent_states", None) or {}
        if conv_states.get(0) is None or recurrent_states.get(0) is None:
            sys.exit(
                f"block {block} is a mamba block by the config but its cache holds no conv or "
                f"recurrent state: {type(layer).__name__}"
            )
        dump_tensor(out, f"state.conv.{ordinal}", conv_states[0], manifest)
        dump_tensor(out, f"state.recurrent.{ordinal}", recurrent_states[0], manifest)
    for ordinal, block in enumerate(attention_indices):
        layer = cache.layers[block]
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None or not hasattr(layer, "get_seq_length"):
            sys.exit(
                f"block {block} is an attention block by the config but its cache holds no K/V: "
                f"{type(layer).__name__}"
            )
        dump_tensor(out, f"state.keys.{ordinal}", keys, manifest)
        dump_tensor(out, f"state.values.{ordinal}", values, manifest)
        dump_tensor(
            out,
            f"state.attention_length.{ordinal}",
            torch.tensor([layer.get_seq_length()], dtype=torch.float32),
            manifest,
        )


def run_decode(model, config, arguments, out: pathlib.Path) -> int:
    """Prefill `--prefix` tokens, dump the single-token step after them, and dump the state it left."""
    from transformers import DynamicCache

    prefix = arguments.tokens if arguments.prefix is None else arguments.prefix
    if prefix < 1:
        sys.exit("--prefix must name at least one prefilled token")
    # The same seeded stream the whole-sequence dump draws from, so the default prefix is that dump's
    # own tokens and the step is the token immediately after them.
    generator = torch.Generator().manual_seed(arguments.seed)
    tokens = torch.randint(0, config.vocab_size, (1, prefix + 1), generator=generator)
    if arguments.step is not None:
        tokens[0, prefix] = arguments.step
    step_tokens = tokens[:, prefix : prefix + 1]

    parameters, stages = register_hooks(model, arguments.weights)
    cache = DynamicCache(config=config)
    with torch.no_grad():
        model(tokens[:, :prefix], past_key_values=cache, use_cache=True)
    # The hooks overwrite their last capture, so the prefix's stages are dropped here and only the
    # step's are ever written: they are the ones a decode comparator has to reproduce. Its stage
    # *names* are kept though, because the reference's own decode branch does not run every module a
    # prefill does -- a single token with a cached state convolves through `conv_states` and the
    # module's weight directly instead of calling `nn.Conv1d` -- and a comparator has to hear that from
    # the dump rather than discover it as a missing file.
    prefill_stages = set(stages)
    stages.clear()
    with torch.no_grad():
        model(step_tokens, past_key_values=cache, use_cache=True)
    step = dict(stages)

    # The step must have *read* the prefix's state rather than only written its own. The same token
    # through an empty cache shares nothing with it but the weights and the token, so a stage that
    # comes back bit-identical means the cache was ignored -- and the numbers would look perfectly
    # reasonable, which is exactly why the tool has to show the difference rather than assume it.
    stages.clear()
    control_cache = DynamicCache(config=config)
    with torch.no_grad():
        model(step_tokens, past_key_values=control_cache, use_cache=True)
    control = dict(stages)

    probe = "model.layers.0.mixer"
    mamba_indices, _ = state_block_indices(config)
    state_read = not mamba_indices or not torch.equal(
        cache.layers[mamba_indices[0]].recurrent_states[0],
        control_cache.layers[mamba_indices[0]].recurrent_states[0],
    )
    if torch.equal(step[probe], control[probe]) or not state_read:
        sys.exit("the step is bit-identical to the same token through an empty cache: no state was read")
    difference = float((step[probe] - control[probe]).norm() / step[probe].norm())

    decode_out = out / "decode"
    decode_out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for name, tensor in (("prefix_tokens", tokens[:, :prefix]), ("decode_token", step_tokens)):
        (decode_out / f"{name}.i32").write_bytes(tensor.to(torch.int32).numpy().tobytes())
        manifest.append(f"{name}\t{'x'.join(str(d) for d in tensor.shape)}\t{name}.i32")
    for name, tensor in {**parameters, **step}.items():
        dump_tensor(decode_out, name, tensor, manifest)
    dump_states(cache, config, decode_out, manifest)
    (decode_out / "manifest.txt").write_text("\n".join(manifest) + "\n")

    print(
        f"{prefix} prefilled tokens + 1 step token, {len(parameters) + len(step)} stages, "
        f"{len(manifest)} files -> {decode_out}"
    )
    print(f"  the step's block 0 mixer differs from a state-blind pass by {difference:.3e} relative L2")
    prefill_only = sorted(prefill_stages - set(step))
    if prefill_only:
        print(f"  stages a prefill pass has and this step has not: {', '.join(prefill_only)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["dump"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--tokens", type=int, default=130, help="sequence length (default past chunk_size)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="also dump every parameter under the decoder layers, plus the embedding, final norm and head",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="prefill --prefix tokens, dump the single-token step after them and the state it left, "
        "into OUT/decode",
    )
    parser.add_argument("--prefix", type=int, help="tokens prefilled before the step (default: --tokens)")
    parser.add_argument(
        "--step",
        type=int,
        help="the token fed to the step (default: the next token of the seeded stream)",
    )
    arguments = parser.parse_args()

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)

    # `from_config` initialises the tiny model from the global RNG, which torch seeds from the OS, so
    # without this the dump is a different model every run and no recorded comparison can be repeated.
    # A checkpoint load has no random initialisation, so this only ever pins the tiny path.
    torch.manual_seed(arguments.seed)
    model, config = load_model(arguments)

    # The config is what the C++ side has to reproduce from the checkpoint's own JSON, so keep the
    # resolved view beside the tensors.
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))

    if arguments.decode:
        return run_decode(model, config, arguments, out)

    parameters, stages = register_hooks(model, arguments.weights)

    generator = torch.Generator().manual_seed(arguments.seed)
    tokens = torch.randint(0, config.vocab_size, (1, arguments.tokens), generator=generator)

    with torch.no_grad():
        model(tokens)

    manifest: list[str] = []
    (out / "tokens.i32").write_bytes(tokens.to(torch.int32).numpy().tobytes())
    manifest.append(f"input_tokens\t{'x'.join(str(d) for d in tokens.shape)}\ttokens.i32")
    for name, tensor in {**parameters, **stages}.items():
        dump_tensor(out, name, tensor, manifest)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")

    total = len(parameters) + len(stages)
    print(f"{arguments.tokens} tokens, {total} stages, {len(manifest)} files -> {out}")
    for line in manifest[:12]:
        print("  " + line)
    if len(manifest) > 12:
        print(f"  ... {len(manifest) - 12} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
