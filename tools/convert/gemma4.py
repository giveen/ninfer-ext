"""Gemma 4 architecture adapter (Gemma 4 31B): config, logical parameters, groups.

The mathematics is defined by the model reference. Until it is written
(`docs/maintainer/gemma4-model.md`, P0), the plan's sections 3.1-3.5 are the authority: plain
RMSNorm (no `1 + w` offset), sandwich norms on the branch output, a per-layer `layer_scalar`, a
GeGLU MLP, alternating sliding (D256, window 1024) and global (D512, K = V) attention with
proportional RoPE, attention scale exactly 1.0, and a final logit soft-cap of 30.

The text tower is mapped, and with the `mtp` component the official assistant drafter
(`Gemma4AssistantForCausalLM`, e.g. google/gemma-4-31B-it-assistant) from a separate source. The
drafter has no key or value projections: each of its layers attends to the target's cache of the same
kind, the last sliding and the last global layer. The vision tower is a later phase.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .model import Model
from .qwen3_5 import _Builder, _f32, _fixed, _positive
from .sources.safetensors import SafetensorsSource
from .resources import load_resources

ARCHITECTURES = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")

# The structural constants of this checkpoint family. A source that disagrees is refused rather
# than silently mapped, because every one of them changes the mathematics.
_TEXT = {
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "sliding_num_key_value_heads": 16,
    "sliding_head_dim": 256,
    "sliding_window": 1024,
    "global_num_key_value_heads": 4,
    "global_head_dim": 512,
    "global_rotary_dim": 128,
    "global_rope_angles": 64,
    "sliding_rope_theta": 10_000.0,
    "global_rope_theta": 1_000_000.0,
    "max_position_embeddings": 262_144,
    "final_logit_softcapping": 30.0,
}
_GLOBAL_EVERY = 6  # global layers at 5, 11, ..., 59 (the last layer is forced global)
_FULL_ATTENTION = "full_attention"
_SLIDING_ATTENTION = "sliding_attention"


def is_gemma4(config: dict) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and bool(set(architectures) & set(ARCHITECTURES))


def _expected_layer_types(layers: int) -> list[str]:
    return [
        _FULL_ATTENTION if (index + 1) % _GLOBAL_EVERY == 0 else _SLIDING_ATTENTION
        for index in range(layers)
    ]


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not is_gemma4(source):
        raise ValueError(f"unsupported Gemma 4 architecture {architectures!r}")
    raw = source.get("text_config", source)

    result: dict = {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text"}
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
                "max_position_embeddings", "num_attention_heads"):
        result[key] = _positive(raw.get(key), "text." + key)
    for key, expected in (
        ("hidden_size", _TEXT["hidden_size"]),
        ("intermediate_size", _TEXT["intermediate_size"]),
        ("num_hidden_layers", _TEXT["num_hidden_layers"]),
        ("num_attention_heads", _TEXT["num_attention_heads"]),
        ("max_position_embeddings", _TEXT["max_position_embeddings"]),
    ):
        if result[key] != expected:
            raise ValueError(f"Gemma 4 31B expects text.{key} {expected}, got {result[key]}")

    _fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "attention_k_eq_v", True, "text")
    _fixed(raw, "enable_moe_block", False, "text")
    _fixed(raw, "head_dim", _TEXT["sliding_head_dim"], "text")
    _fixed(raw, "global_head_dim", _TEXT["global_head_dim"], "text")
    _fixed(raw, "sliding_window", _TEXT["sliding_window"], "text")
    _fixed(raw, "final_logit_softcapping", _TEXT["final_logit_softcapping"], "text")

    result["head_dim"] = _TEXT["sliding_head_dim"]
    result["num_key_value_heads"] = _positive(raw.get("num_key_value_heads"),
                                              "text.num_key_value_heads")
    if result["num_key_value_heads"] != _TEXT["sliding_num_key_value_heads"]:
        raise ValueError("Gemma 4 31B expects 16 sliding KV heads")
    result["num_global_key_value_heads"] = _positive(
        raw.get("num_global_key_value_heads"), "text.num_global_key_value_heads")
    if result["num_global_key_value_heads"] != _TEXT["global_num_key_value_heads"]:
        raise ValueError("Gemma 4 31B expects 4 global KV heads")
    result["global_head_dim"] = _TEXT["global_head_dim"]
    result["sliding_window"] = _TEXT["sliding_window"]
    result["attention_k_eq_v"] = True
    result["final_logit_softcapping"] = _TEXT["final_logit_softcapping"]
    result["hidden_act"] = "gelu_pytorch_tanh"
    result["attention_bias"] = False

    # Plain RMSNorm: the weight is used as is, with no `1 + w` offset (plan 3.2).
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    result["norm_unit_offset"] = False
    # The embedding is scaled by sqrt(H) and the head is tied to it.
    result["embedding_scale"] = math.sqrt(float(result["hidden_size"]))
    result["tie_word_embeddings"] = bool(raw.get("tie_word_embeddings", True))
    result["attention_scale"] = 1.0
    result["layer_scalar"] = True

    layers = raw.get("layer_types")
    expected = _expected_layer_types(result["num_hidden_layers"])
    if not isinstance(layers, list) or list(layers) != expected:
        raise ValueError("Gemma 4 31B expects global attention on every sixth layer, last included")
    result["layer_types"] = list(layers)
    result["global_layer_count"] = expected.count(_FULL_ATTENTION)

    # RoPE differs by layer kind: sliding rotates the full 256 dims at theta 10k; global is
    # proportional at theta 1e6 with 64 angles over a 512-wide denominator, so only dims
    # {0..63} and {256..319} rotate.
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict) or set(rope) != {_SLIDING_ATTENTION, _FULL_ATTENTION}:
        raise ValueError("text.rope_parameters must describe both layer kinds")
    _fixed(rope[_SLIDING_ATTENTION], "rope_type", "default", "text.rope_parameters")
    _fixed(rope[_FULL_ATTENTION], "rope_type", "proportional", "text.rope_parameters")
    _fixed(rope[_FULL_ATTENTION], "partial_rotary_factor", 0.25, "text.rope_parameters")
    sliding_theta = _f32(rope[_SLIDING_ATTENTION].get("rope_theta"),
                         "text.rope_parameters.sliding_attention.rope_theta")
    global_theta = _f32(rope[_FULL_ATTENTION].get("rope_theta"),
                        "text.rope_parameters.full_attention.rope_theta")
    if sliding_theta != _TEXT["sliding_rope_theta"] or global_theta != _TEXT["global_rope_theta"]:
        raise ValueError("Gemma 4 31B expects theta 10000 sliding and 1000000 global")
    result["rope_parameters"] = {
        _SLIDING_ATTENTION: {
            "rope_type": "default",
            "rope_theta": sliding_theta,
            "rotary_dim": _TEXT["sliding_head_dim"],
        },
        _FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": global_theta,
            "rotary_dim": _TEXT["global_rotary_dim"],
            "rope_angles": _TEXT["global_rope_angles"],
            "denominator_head_dim": _TEXT["global_head_dim"],
        },
    }
    return result


# The assistant drafter's own structure. Its attention geometry is the target's by construction,
# because every drafter layer reads the target's cache of the same kind.
_DRAFT = {
    "hidden_size": 1024,
    "intermediate_size": 8192,
    "num_hidden_layers": 4,
}


def draft_config(source: dict, target: dict) -> dict:
    """The `mtp` component config for the official Gemma 4 assistant drafter."""
    if source.get("architectures") != ["Gemma4AssistantForCausalLM"]:
        raise ValueError(f"unsupported Gemma 4 drafter {source.get('architectures')!r}")
    if source.get("use_ordered_embeddings", False):
        raise ValueError("the drafter's ordered (centroid) head is not supported; a full head is")
    if source.get("backbone_hidden_size") != target["hidden_size"]:
        raise ValueError("the drafter's backbone_hidden_size must be the target's hidden size")
    raw = source.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("the drafter needs a text_config")
    for key, expected in _DRAFT.items():
        if raw.get(key) != expected:
            raise ValueError(f"the Gemma 4 31B drafter expects {key} {expected}, got {raw.get(key)}")
    layers = raw["num_hidden_layers"]
    if raw.get("num_kv_shared_layers") != layers:
        raise ValueError("every drafter layer must share the target's KV")
    _fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "mtp")
    _fixed(raw, "attention_bias", False, "mtp")
    _fixed(raw, "attention_k_eq_v", True, "mtp")
    _fixed(raw, "final_logit_softcapping", None, "mtp")
    for key, target_key in (
        ("vocab_size", "vocab_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("num_global_key_value_heads", "num_global_key_value_heads"),
        ("head_dim", "head_dim"),
        ("global_head_dim", "global_head_dim"),
        ("sliding_window", "sliding_window"),
    ):
        if raw.get(key) != target[target_key]:
            raise ValueError(f"the drafter's {key} must equal the target's")
    if raw.get("rope_parameters") is None:
        raise ValueError("the drafter needs rope_parameters")
    for kind in (_SLIDING_ATTENTION, _FULL_ATTENTION):
        mine, theirs = raw["rope_parameters"].get(kind, {}), target["rope_parameters"][kind]
        if mine.get("rope_type") != theirs["rope_type"] or \
                _f32(mine.get("rope_theta"), "mtp.rope_theta") != theirs["rope_theta"]:
            raise ValueError(f"the drafter's {kind} RoPE must equal the target's")
    kinds = raw.get("layer_types")
    if kinds != [_SLIDING_ATTENTION] * (layers - 1) + [_FULL_ATTENTION]:
        raise ValueError("the drafter expects sliding layers then one global layer")
    return {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "hidden_size": raw["hidden_size"],
        "intermediate_size": raw["intermediate_size"],
        "num_hidden_layers": layers,
        "backbone_hidden_size": target["hidden_size"],
        "vocab_size": raw["vocab_size"],
        "rms_norm_eps": _f32(raw.get("rms_norm_eps", 1e-6), "mtp.rms_norm_eps"),
        "hidden_act": "gelu_pytorch_tanh",
        "layer_types": list(kinds),
        "tie_word_embeddings": True,
    }


class _Gemma4Builder(_Builder):
    """Maps the Gemma 4 text tower: four norms per layer, a layer scalar, and two attention kinds."""

    def weight_vector(self, name, store, source_name, width):
        # Plain RMSNorm (no `1 + w`) and K-norm-of-V are stored as plain weights; `direct`
        # keeps them BF16.
        self.add(name, store, source_name, (width,), direct="bf16")

    def draft_block(self, prefix, source_prefix, store, draft, target, kind):
        """One drafter layer: the target layer's body without key and value projections."""
        h = draft["hidden_size"]
        intermediate = draft["intermediate_size"]
        attention = prefix + "attention/"
        head_dim = target["global_head_dim"] if kind == _FULL_ATTENTION else target["head_dim"]
        q_rows = target["num_attention_heads"] * head_dim
        for role, source in (("input_norm", "input_layernorm"),
                             ("post_attention_norm", "post_attention_layernorm"),
                             ("pre_feedforward_norm", "pre_feedforward_layernorm"),
                             ("post_feedforward_norm", "post_feedforward_layernorm")):
            self.weight_vector(prefix + role, store, source_prefix + source + ".weight", h)
        self.add(prefix + "layer_scalar", store, source_prefix + "layer_scalar", (1,),
                 direct="fp32")
        self.add(attention + "query", store, source_prefix + "self_attn.q_proj.weight",
                 (q_rows, h), inputs=(prefix + "mixer_input",))
        self.weight_vector(attention + "query_norm", store,
                           source_prefix + "self_attn.q_norm.weight", head_dim)
        self.add(attention + "output", store, source_prefix + "self_attn.o_proj.weight",
                 (h, q_rows), inputs=(attention + "gated_output",))
        for role in ("gate", "up"):
            self.add(prefix + "mlp/" + role, store, source_prefix + "mlp." + role + "_proj.weight",
                     (intermediate, h), inputs=(prefix + "mlp_input",))
        self.add(prefix + "mlp/down", store, source_prefix + "mlp.down_proj.weight",
                 (h, intermediate), inputs=(prefix + "mlp/product",))

    def block(self, prefix, source_prefix, store, config, kind):
        h = config["hidden_size"]
        intermediate = config["intermediate_size"]
        attention = prefix + "attention/"
        global_layer = kind == _FULL_ATTENTION
        kv_heads = (config["num_global_key_value_heads"] if global_layer
                    else config["num_key_value_heads"])
        head_dim = config["global_head_dim"] if global_layer else config["head_dim"]
        q_rows = config["num_attention_heads"] * head_dim
        kv_rows = kv_heads * head_dim

        self.weight_vector(prefix + "input_norm", store, source_prefix + "input_layernorm.weight", h)
        self.weight_vector(prefix + "post_attention_norm", store,
                  source_prefix + "post_attention_layernorm.weight", h)
        self.weight_vector(prefix + "pre_feedforward_norm", store,
                  source_prefix + "pre_feedforward_layernorm.weight", h)
        self.weight_vector(prefix + "post_feedforward_norm", store,
                  source_prefix + "post_feedforward_layernorm.weight", h)
        # One BF16 scalar per layer, applied to the whole residual stream.
        self.add(prefix + "layer_scalar", store, source_prefix + "layer_scalar", (1,),
                 direct="fp32")

        self.add(attention + "query", store, source_prefix + "self_attn.q_proj.weight",
                 (q_rows, h), inputs=(prefix + "mixer_input",))
        self.add(attention + "key", store, source_prefix + "self_attn.k_proj.weight",
                 (kv_rows, h), inputs=(prefix + "mixer_input",))
        if global_layer:
            # K = V: the global layers store no value projection, and no value norm weight.
            pass
        else:
            self.add(attention + "value", store, source_prefix + "self_attn.v_proj.weight",
                     (kv_rows, h), inputs=(prefix + "mixer_input",))
        self.weight_vector(attention + "query_norm", store, source_prefix + "self_attn.q_norm.weight",
                  head_dim)
        # Every layer stores a K weight, including the global ones where it scales the
        # shared K = V vector after the weightless norm.
        self.weight_vector(attention + "key_norm", store,
                           source_prefix + "self_attn.k_norm.weight", head_dim)
        self.add(attention + "output", store, source_prefix + "self_attn.o_proj.weight",
                 (h, q_rows), inputs=(attention + "gated_output",))
        if global_layer:
            self.group(attention + "query", attention + "key")
        else:
            roles = [attention + role for role in ("query", "key", "value")]
            self.group(*roles)

        for role in ("gate", "up"):
            self.add(prefix + "mlp/" + role, store,
                     source_prefix + "mlp." + role + "_proj.weight",
                     (intermediate, h), inputs=(prefix + "mlp_input",))
        self.add(prefix + "mlp/down", store, source_prefix + "mlp.down_proj.weight",
                 (h, intermediate), inputs=(prefix + "mlp/product",))
        # The gate and up halves stay separate objects rather than one packed parent. Packing them
        # would bind each half as a row slice of a BlockScaleK16M128x4 object, and such a slice
        # cannot re-derive the swizzled scale plane it shares with its sibling, so no consumer could
        # execute it. Separate objects are also what the reference engine's converter produces, and
        # each half then carries its own per-tensor scale.


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
    draft: SafetensorsSource | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text", "mtp"}:
        raise ValueError("Gemma 4 supports the text and mtp components")
    if ("mtp" in selected) != (draft is not None):
        raise ValueError("the mtp component needs the drafter checkpoint as --source mtp=PATH")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    if draft is not None:
        records["mtp"] = {"config": draft_config(draft.config, config)}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=None,
        overrides=resource_overrides, family="gemma",
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Gemma4Builder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    text_prefix = "model.language_model."
    builder.add("text/token_embedding", base, text_prefix + "embed_tokens.weight", (r, h))
    # The head is tied to the embedding, so it names the same stored tensor.
    builder.add("text/output_head", base, text_prefix + "embed_tokens.weight", (r, h),
                inputs=("text/final_hidden",))
    builder.weight_vector("text/final_norm", base, text_prefix + "norm.weight", h)
    for index, kind in enumerate(config["layer_types"]):
        builder.block(f"text/layers/{index}/", text_prefix + f"layers.{index}.", base, config,
                      kind)
    if draft is not None:
        mtp = records["mtp"]["config"]
        d = mtp["hidden_size"]
        # The drafter reads [target embedding; target hidden] in, and hands a target-width hidden
        # state to its next step. Its head is tied to its own (otherwise unused) embedding table.
        builder.add("mtp/pre_projection", draft, "pre_projection.weight", (d, 2 * h),
                    inputs=("mtp/input",))
        builder.add("mtp/post_projection", draft, "post_projection.weight", (h, d),
                    inputs=("mtp/final_hidden",))
        builder.add("mtp/output_head", draft, "model.embed_tokens.weight", (r, d),
                    inputs=("mtp/final_hidden",))
        builder.weight_vector("mtp/final_norm", draft, "model.norm.weight", d)
        for index, kind in enumerate(mtp["layer_types"]):
            builder.draft_block(f"mtp/layers/{index}/", f"model.layers.{index}.", draft, mtp,
                                config, kind)
    return model
