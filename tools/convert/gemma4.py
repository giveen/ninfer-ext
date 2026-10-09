"""Gemma 4 architecture adapter (Gemma 4 31B): config, logical parameters, groups.

The mathematics is defined by the model reference. Until it is written
(`docs/maintainer/gemma4-model.md`, P0), the plan's sections 3.1-3.5 are the authority: plain
RMSNorm (no `1 + w` offset), sandwich norms on the branch output, a per-layer `layer_scalar`, a
GeGLU MLP, alternating sliding (D256, window 1024) and global (D512, K = V) attention with
proportional RoPE, attention scale exactly 1.0, and a final logit soft-cap of 30.

Only the text tower is mapped here; the vision tower and the assistant drafter are later phases.
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


def text_config(source: dict, *, mtp: bool = False) -> dict:
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not is_gemma4(source):
        raise ValueError(f"unsupported Gemma 4 architecture {architectures!r}")
    if mtp:
        raise ValueError("Gemma 4 has no MTP component in this plan (the drafter is a later phase)")
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


class _Gemma4Builder(_Builder):
    """Maps the Gemma 4 text tower: four norms per layer, a layer scalar, and two attention kinds."""

    def weight_vector(self, name, store, source_name, width):
        # Plain RMSNorm (no `1 + w`) and K-norm-of-V are stored as plain weights; `direct`
        # keeps them BF16.
        self.add(name, store, source_name, (width,), direct="bf16")

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
        self.group(prefix + "mlp/gate", prefix + "mlp/up")


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text"}:
        raise ValueError("Gemma 4 supports the text component only in this phase")
    config = text_config(base.config)
    records = {"text": {"config": config}}
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
    return model
