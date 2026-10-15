"""Nemotron-H architecture adapter (Nemotron 3.5 Bolt 74B A9B): config, logical parameters, groups.

The checkpoint is `NemotronH_Omni_Reasoning_V3` with a vision tower, a projector and a language
tower; this adapter maps the **language tower alone** (`language_model.backbone.*`, `lm_head`), which
is the 74B hybrid the engine serves in phase 1. Its text config is the embedded `llm_config`, and
every structural constant of the family is pinned here rather than read, because the release has one
shape: 88 blocks interleaving 40 Mamba-2, 40 LatentMoE and 8 attention blocks, a per-*layer* expert
bank (240/304/368/432/496 experts at four intermediate widths around one 1024-wide latent), and the
config's own `num_experts`/`num_local_experts` of 512 is a bound rather than a bank size.

Three facts the checkpoint cannot state and the artifact must carry:

  - `mixer.A_log`, `mixer.D` and `mixer.dt_bias` are stored BF16 and are consumed **FP32**. The
    reference's own state arithmetic runs fp32 and the engine's scan Op requires all three in fp32, so
    the conversion upcasts them; a BF16 copy would lose exactly the resolution the decay is made of.
    The router's `e_score_correction_bias` is the same case and the checkpoint already stores it F32.
  - The depthwise convolution's weight is stored **tap-major** (`j * conv_channels + c`), which is the
    order `ops::causal_conv1d_silu` indexes (`weight[j * C + c]`). The checkpoint stores it
    `[conv_channels, 1, conv_kernel]`, whose flat order is `c * conv_kernel + j`; storing it as it
    lies gives a shape-valid weight whose every tap belongs to another channel.
  - Every routed expert is its own checkpoint tensor, so a bank is assembled from E sources into one
    parent per matrix: expert `e` owns the row range `[e*I, (e+1)*I)` of the `up` parent and
    `[e*latent, (e+1)*latent)` of the `down` parent, which is how the MoE Op reads a bank and how a
    host-offload cache addresses one.

The first eight blocks are `mamba moe mamba moe mamba moe mamba attention`, and the block list is
what decides which of the three mixers a layer has -- the layer count alone does not.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch

from .model import Model, Parameter
from .qwen3_5 import _Builder, _f32, _positive
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.safetensors import tensor_source

# The wrapper architecture and the language tower the wrapper embeds.
ARCHITECTURES = ("NemotronH_Omni_Reasoning_V3", "NemotronHPuzzleForCausalLM")

# The family's structure, read from the checkpoint's own `llm_config` and confirmed against its
# 28,725 tensors. A source that disagrees with any of it is refused rather than mapped.
_HIDDEN = 4096
_VOCAB = 131072
_MAX_POSITIONS = 262144
_LAYERS = 88
_BLOCK_COUNTS = {"mamba": 40, "moe": 40, "attention": 8}
# Mamba-2: 128 heads of 64, B/C grouped 8 ways, a state of 128 and a 4-tap depthwise convolution.
_MAMBA_HEADS = 128
_MAMBA_HEAD_DIM = 64
_SSM_STATE = 128
_N_GROUPS = 8
_CONV_KERNEL = 4
_CHUNK_SIZE = 128
_TIME_STEP_MIN = _f32(0.001, "time_step_min")
# Attention: plain GQA, no bias, no sinks, no window, and no positional encoding at all.
_ATTENTION_HEADS = 32
_ATTENTION_KV_HEADS = 2
_HEAD_DIM = 128
# MoE: one shared expert per layer, top-9 of the layer's own bank, no group machinery in effect.
_EXPERT_COUNTS = (240, 304, 368, 432, 496)
_EXPERT_WIDTHS = (1536, 2048, 2304, 2560)
_EXPERTS_PER_TOKEN = 9
_LATENT = 1024
_SHARED_WIDTH = 5376
_NORM_EPSILON = _f32(1e-5, "layer_norm_epsilon")
_ROUTED_SCALING = _f32(5.0, "routed_scaling_factor")


def is_nemotron_h(config: dict) -> bool:
    """Whether `config` is a Bolt release wrapper or the language tower it embeds."""
    for container in (config, config.get("llm_config")):
        if not isinstance(container, dict):
            continue
        architectures = container.get("architectures")
        if isinstance(architectures, list) and set(architectures) & set(ARCHITECTURES):
            return True
    return False


def _expect_int(config: dict, key: str, value: int, label: str) -> int:
    actual = config.get(key)
    if type(actual) is not int or actual != value:
        raise ValueError(f"{label}.{key}: expected {value!r}, got {actual!r}")
    return actual


def _expect_flag(config: dict, key: str, value: bool, label: str) -> bool:
    actual = config.get(key)
    if type(actual) is not bool or actual is not value:
        raise ValueError(f"{label}.{key}: expected {value!r}, got {actual!r}")
    return actual


def _expect_f32(config: dict, key: str, value: float, label: str) -> float:
    actual = _f32(config.get(key), f"{label}.{key}")
    if actual != value:
        raise ValueError(f"{label}.{key}: expected {value!r}, got {actual!r}")
    return actual


def _expect_text(config: dict, key: str, value: str, label: str) -> str:
    actual = config.get(key)
    if actual != value:
        raise ValueError(f"{label}.{key}: expected {value!r}, got {actual!r}")
    return actual


def text_config(source: dict) -> dict:
    """The language tower's config in the form the artifact's own parser reads.

    The emitted key set is exactly `nemotron_h/config.cpp`'s: it validates every member it is given,
    so a field from another level is a refusal rather than a field it ignores.
    """
    if not is_nemotron_h(source):
        raise ValueError(f"unsupported Nemotron-H architecture {source.get('architectures')!r}")
    raw = source.get("llm_config")
    if not isinstance(raw, dict):
        raw = source
    # This release has no MTP: `num_nextn_predict_layers` is zero and both MTP block lists are empty.
    # A source that declares one would need those blocks mapped, so it is refused here.
    for key in ("mtp_block_configs", "mtp_layers_block_type"):
        if raw.get(key):
            raise ValueError(f"llm_config.{key}: MTP blocks are not mapped")
    if raw.get("mtp_num_hidden_layers") or raw.get("num_nextn_predict_layers"):
        raise ValueError("llm_config: MTP layers are not mapped")

    _expect_flag(raw, "tie_word_embeddings", False, "llm_config")
    _expect_flag(raw, "use_bias", False, "llm_config")
    _expect_flag(raw, "mlp_bias", False, "llm_config")
    _expect_flag(raw, "use_qk_norm", False, "llm_config")
    _expect_flag(raw, "attention_bias", False, "llm_config")
    _expect_flag(raw, "norm_topk_prob", True, "llm_config")
    _expect_flag(raw, "rescale_prenorm_residual", True, "llm_config")
    _expect_text(raw, "hidden_act", "relu2", "llm_config")
    _expect_text(raw, "mlp_hidden_act", "relu2", "llm_config")
    _expect_text(raw, "mamba_hidden_act", "silu", "llm_config")
    _expect_text(raw, "scoring_func", "sigmoid", "llm_config")
    _expect_int(raw, "hidden_size", _HIDDEN, "llm_config")
    _expect_int(raw, "vocab_size", _VOCAB, "llm_config")
    _expect_int(raw, "max_position_embeddings", _MAX_POSITIONS, "llm_config")
    _expect_int(raw, "moe_latent_size", _LATENT, "llm_config")
    _expect_int(raw, "moe_shared_expert_intermediate_size", _SHARED_WIDTH, "llm_config")
    _expect_int(raw, "num_attention_heads", _ATTENTION_HEADS, "llm_config")
    _expect_int(raw, "num_key_value_heads", _ATTENTION_KV_HEADS, "llm_config")
    _expect_int(raw, "head_dim", _HEAD_DIM, "llm_config")
    _expect_int(raw, "mamba_num_heads", _MAMBA_HEADS, "llm_config")
    _expect_int(raw, "mamba_head_dim", _MAMBA_HEAD_DIM, "llm_config")
    _expect_int(raw, "ssm_state_size", _SSM_STATE, "llm_config")
    _expect_int(raw, "conv_kernel", _CONV_KERNEL, "llm_config")
    _expect_int(raw, "chunk_size", _CHUNK_SIZE, "llm_config")
    # `n_groups` is the number of B/C groups inside the Mamba mixer and has nothing to do with the
    # router's `n_group`; both appear in the checkpoint with different values (8 and 1).
    _expect_int(raw, "n_groups", _N_GROUPS, "llm_config")
    _expect_int(raw, "n_group", 1, "llm_config")
    _expect_int(raw, "topk_group", 1, "llm_config")
    _expect_int(raw, "n_shared_experts", 1, "llm_config")
    _expect_f32(raw, "layer_norm_epsilon", _NORM_EPSILON, "llm_config")
    _expect_f32(raw, "time_step_min", _TIME_STEP_MIN, "llm_config")
    _expect_f32(raw, "routed_scaling_factor", _ROUTED_SCALING, "llm_config")
    # `time_step_max` clamps nothing on the torch path and `n_groups`-style booleans above are the
    # only flags the mathematics reads, so neither is carried into the artifact.
    _expect_flag(raw, "use_conv_bias", True, "llm_config")
    _expect_flag(raw, "mamba_proj_bias", False, "llm_config")

    blocks = raw.get("block_configs")
    if not isinstance(blocks, list) or len(blocks) != _LAYERS:
        raise ValueError(f"llm_config.block_configs must list {_LAYERS} blocks")
    kinds: list[str] = []
    emitted: list[dict] = []
    for layer, block in enumerate(blocks):
        label = f"llm_config.block_configs[{layer}]"
        if not isinstance(block, dict):
            raise ValueError(f"{label}: expected an object")
        kind = block.get("block_type")
        if kind not in _BLOCK_COUNTS:
            raise ValueError(f"{label}.block_type: unsupported block type {kind!r}")
        if kind == "moe":
            experts = _positive(block.get("n_routed_experts"), label + ".n_routed_experts")
            per_token = _positive(block.get("num_experts_per_tok"), label + ".num_experts_per_tok")
            width = _positive(block.get("moe_intermediate_size"), label + ".moe_intermediate_size")
            if experts not in _EXPERT_COUNTS:
                raise ValueError(f"{label}.n_routed_experts: unsupported bank width {experts}")
            if width not in _EXPERT_WIDTHS:
                raise ValueError(f"{label}.moe_intermediate_size: unsupported width {width}")
            if per_token != _EXPERTS_PER_TOKEN:
                raise ValueError(f"{label}.num_experts_per_tok: expected {_EXPERTS_PER_TOKEN}")
            emitted.append(
                {
                    "block_type": "moe",
                    "n_routed_experts": experts,
                    "num_experts_per_tok": per_token,
                    "moe_intermediate_size": width,
                }
            )
        else:
            # The parser requires the MoE fields on exactly the MoE blocks, so a non-MoE block that
            # carries one would be refused at load; refusing it here names the layer instead.
            stray = sorted(set(block) - {"block_type"})
            if stray:
                raise ValueError(f"{label}: a {kind} block declares {stray}")
            emitted.append({"block_type": kind})
        kinds.append(kind)
    counts = Counter(kinds)
    if counts != Counter(_BLOCK_COUNTS):
        raise ValueError(f"llm_config.block_configs: expected {_BLOCK_COUNTS}, got {dict(counts)}")
    declared = raw.get("layers_block_type")
    if declared is not None and list(declared) != kinds:
        raise ValueError("llm_config.layers_block_type disagrees with block_configs")

    return {
        "hidden_size": _HIDDEN,
        "vocab_size": _VOCAB,
        "max_position_embeddings": _MAX_POSITIONS,
        "tie_word_embeddings": False,
        "layer_norm_epsilon": _NORM_EPSILON,
        "block_configs": emitted,
        "mamba": {
            "mamba_num_heads": _MAMBA_HEADS,
            "mamba_head_dim": _MAMBA_HEAD_DIM,
            "ssm_state_size": _SSM_STATE,
            "n_groups": _N_GROUPS,
            "conv_kernel": _CONV_KERNEL,
            "chunk_size": _CHUNK_SIZE,
            "time_step_min": _TIME_STEP_MIN,
            "use_conv_bias": True,
        },
        "attention": {
            "num_attention_heads": _ATTENTION_HEADS,
            "num_key_value_heads": _ATTENTION_KV_HEADS,
            "head_dim": _HEAD_DIM,
        },
        "router": {
            "routed_scaling_factor": _ROUTED_SCALING,
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            # The parser reads the *presence* of this key as "the checkpoint carries the buffer",
            # which decides whether the routed selection is corrected.
            "e_score_correction_bias": True,
        },
        "moe_latent_size": _LATENT,
        "moe_shared_expert_intermediate_size": _SHARED_WIDTH,
    }


def _tap_major(source: LogicalSource, shape: tuple[int, int]) -> LogicalSource:
    """`source` (`[channels, 1, kernel]`, flat `c * kernel + j`) as `shape`, flat `j * channels + c`.

    The convolution Op reads `weight[j * C + c]`, so the tap is the outer axis of the stored payload
    while the declaration stays `[conv_channels, conv_kernel]`.
    """
    channels, kernel = shape

    def read(begin: int, end: int) -> torch.Tensor:
        values = source.values().reshape(channels, kernel).transpose(0, 1).contiguous()
        return values.reshape(-1)[begin:end]

    return LogicalSource(shape, f"tap_major({source.label})", read)


class _NemotronBuilder(_Builder):
    """Maps one Bolt language tower: one block norm, then exactly one of three mixers."""

    def tap_major_matrix(self, name, store, source_name, shape, source_shape):
        """A depthwise convolution weight stored in the tap-major order its Op indexes."""
        def factory(selected, format=None):
            if format is not None:
                raise ValueError(f"{name}: a depthwise weight has no encoded source")
            return _tap_major(tensor_source(selected, source_name, source_shape), shape)

        self.model.add(
            Parameter(name, shape, factory(store), factory, (), "bf16",
                      residency=name.split("/", 1)[0])
        )

    def mamba(self, prefix, source_prefix, store, config):
        h = config["hidden_size"]
        m = config["mamba"]
        inner = m["mamba_num_heads"] * m["mamba_head_dim"]
        conv = inner + 2 * m["n_groups"] * m["ssm_state_size"]
        # `in_proj` emits z, x/B/C and dt in that order, and `out_proj` reads the gated grouped norm
        # applied to the scan's output, which stays inside the Op.
        mixer_input = prefix + "mixer_input"
        self.add(prefix + "mamba/in_projection", store, source_prefix + "mixer.in_proj.weight",
                 (inner + conv + m["mamba_num_heads"], h), inputs=(mixer_input,))
        self.tap_major_matrix(prefix + "mamba/convolution", store,
                              source_prefix + "mixer.conv1d.weight", (conv, m["conv_kernel"]),
                              (conv, 1, m["conv_kernel"]))
        self.add(prefix + "mamba/convolution_bias", store, source_prefix + "mixer.conv1d.bias",
                 (conv,))
        # The scan's per-head parameters: `A_log` and `D` are the decay and the skip, `dt_bias` the
        # time step's bias. All three are FP32 here whatever the checkpoint stores.
        for role, source in (("a_log", "mixer.A_log"), ("d_skip", "mixer.D"),
                             ("dt_bias", "mixer.dt_bias")):
            self.add(prefix + "mamba/" + role, store, source_prefix + source,
                     (m["mamba_num_heads"],), direct="fp32")
        self.add(prefix + "mamba/norm", store, source_prefix + "mixer.norm.weight", (inner,))
        self.add(prefix + "mamba/out_projection", store, source_prefix + "mixer.out_proj.weight",
                 (h, inner), inputs=(prefix + "mamba/gated_norm_output",))

    def attention(self, prefix, source_prefix, store, config):
        h = config["hidden_size"]
        a = config["attention"]
        query_rows = a["num_attention_heads"] * a["head_dim"]
        kv_rows = a["num_key_value_heads"] * a["head_dim"]
        mixer_input = prefix + "mixer_input"
        for role, source, rows in (("query", "mixer.q_proj.weight", query_rows),
                                   ("key", "mixer.k_proj.weight", kv_rows),
                                   ("value", "mixer.v_proj.weight", kv_rows)):
            self.add(prefix + "attention/" + role, store, source_prefix + source, (rows, h),
                     inputs=(mixer_input,))
        self.add(prefix + "attention/output", store, source_prefix + "mixer.o_proj.weight",
                 (h, query_rows), inputs=(prefix + "attention/attended",))

    def moe(self, prefix, source_prefix, store, block, config):
        h = config["hidden_size"]
        latent = config["moe_latent_size"]
        experts = block["n_routed_experts"]
        width = block["moe_intermediate_size"]
        shared = config["moe_shared_expert_intermediate_size"]
        p = prefix + "moe/"
        # The router and the latent projection read the block's pre-FFN residual; the shared expert
        # reads the same residual rather than the latent space, because the reference adds it after
        # the round trip.
        ffn_input = prefix + "ffn_input"
        self.add(p + "router", store, source_prefix + "mixer.gate.weight", (experts, h),
                 inputs=(ffn_input,))
        self.add(p + "correction_bias", store,
                 source_prefix + "mixer.gate.e_score_correction_bias", (experts,), direct="fp32")
        self.add(p + "latent_in", store, source_prefix + "mixer.fc1_latent_proj.weight",
                 (latent, h), inputs=(ffn_input,))
        self.add(p + "latent_out", store, source_prefix + "mixer.fc2_latent_proj.weight",
                 (h, latent), inputs=(p + "expert_product",))
        self.add(p + "shared/up", store, source_prefix + "mixer.shared_experts.up_proj.weight",
                 (shared, h), inputs=(ffn_input,))
        self.add(p + "shared/down", store, source_prefix + "mixer.shared_experts.down_proj.weight",
                 (h, shared), inputs=(p + "shared/product",))
        # Every expert is its own tensor; the pair of groups below is what makes each layer's bank a
        # single parent per matrix, so an expert is a row range rather than a separate object.
        expert_input = p + "expert_input"
        ups, downs = [], []
        for expert in range(experts):
            ep = p + f"experts/{expert}/"
            self.add(ep + "up", store,
                     source_prefix + f"mixer.experts.{expert}.up_proj.weight", (width, latent),
                     inputs=(expert_input,))
            self.add(ep + "down", store,
                     source_prefix + f"mixer.experts.{expert}.down_proj.weight", (latent, width),
                     inputs=(ep + "product",))
            ups.append(ep + "up")
            downs.append(ep + "down")
        self.group(*ups)
        self.group(*downs)

    def block(self, prefix, source_prefix, store, config, block):
        self.add(prefix + "block_norm", store, source_prefix + "norm.weight",
                 (config["hidden_size"],))
        kind = block["block_type"]
        if kind == "mamba":
            self.mamba(prefix, source_prefix, store, config)
        elif kind == "attention":
            self.attention(prefix, source_prefix, store, config)
        else:
            self.moe(prefix, source_prefix, store, block, config)


def build_model(
    base,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str] | None = None,
) -> Model:
    """Map the Bolt language tower. Vision, the projector and MTP are outside this component."""
    if tuple(components) != ("text",):
        raise ValueError("Nemotron-H maps the text component alone")
    config = text_config(base.config)
    references, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        overrides=resource_overrides,
        family="nemotron",
    )
    records = {"text": {"config": config, "resources": references["text"]}}
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _NemotronBuilder(model)
    h, vocabulary = config["hidden_size"], config["vocab_size"]
    stem = "language_model.backbone."
    builder.add("text/token_embedding", base, stem + "embeddings.weight", (vocabulary, h))
    builder.add("text/output_head", base, "language_model.lm_head.weight", (vocabulary, h),
                inputs=("text/final_hidden",))
    builder.add("text/final_norm", base, stem + "norm_f.weight", (h,))
    # The config decides each layer's shape and kind; the checkpoint's tensors confirm both, because
    # every parameter's declared shape is checked against the tensor it is read from and every
    # expert's own tensor has to exist.
    for index, block in enumerate(config["block_configs"]):
        builder.block(f"text/layers/{index}/", stem + f"layers.{index}.", base, config, block)
    return model
