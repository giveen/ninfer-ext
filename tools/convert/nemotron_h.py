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
from .qwen3_5 import _Builder, _f32, _positive, draft_config
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.modelopt import nvfp4_source
from .sources.safetensors import tensor_source

# The wrapper architecture and the language tower the wrapper embeds.
ARCHITECTURES = ("NemotronH_Omni_Reasoning_V3", "NemotronHPuzzleForCausalLM",
                 "NemotronHForCausalLM")

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


def _flat_text_config(source: dict) -> dict:
    """The flat `NemotronHForCausalLM` layout (Lightning 30B).

    No wrapper, so no `llm_config`, no `block_configs` and none of the Omni flags: the geometry and
    the block kinds are read from the checkpoint itself and rendered into the artifact parser's key
    set. Every number is taken from the source rather than compared against the 74B's constants --
    this model is 52 blocks (mamba 23 / moe 23 / attention 6), hidden 2688, top-6 of 128 experts, a
    direct (non-latent) bank and no MTP mapping (the pair's spec path is the separate DFlash
    drafter).
    """

    def integer(key: str, minimum: int = 1) -> int:
        value = source.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key}: expected an integer >= {minimum}, got {value!r}")
        return value

    kinds = source.get("layers_block_type")
    if not isinstance(kinds, list) or not kinds:
        raise ValueError("layers_block_type: expected a non-empty list")
    if any(kind not in ("mamba", "moe", "attention") for kind in kinds):
        raise ValueError(f"layers_block_type: unsupported block kinds {sorted(set(kinds))}")
    if integer("num_hidden_layers") != len(kinds):
        raise ValueError("layers_block_type length differs from num_hidden_layers")

    experts = integer("n_routed_experts")
    per_token = integer("num_experts_per_tok")
    # The engine's expert kernel requires the intermediate width to be a multiple of 128; a model
    # that stores a narrower one (1856 here) carries it zero-padded, which is exact.
    moe_width = (integer("moe_intermediate_size") + 127) // 128 * 128
    if per_token > experts:
        raise ValueError("num_experts_per_tok exceeds n_routed_experts")
    shared = source.get("moe_shared_expert_intermediate_size") or 0
    if type(shared) is not int or shared < 0:
        raise ValueError("moe_shared_expert_intermediate_size must be a non-negative integer")
    latent = source.get("moe_latent_size") or 0
    if type(latent) is not int or latent < 0:
        raise ValueError("moe_latent_size must be a non-negative integer")

    emitted = [
        {
            "block_type": "moe",
            "n_routed_experts": experts,
            "num_experts_per_tok": per_token,
            "moe_intermediate_size": moe_width,
        }
        if kind == "moe"
        else {"block_type": kind}
        for kind in kinds
    ]
    return {
        "hidden_size": integer("hidden_size"),
        "vocab_size": integer("vocab_size"),
        "max_position_embeddings": integer("max_position_embeddings"),
        "tie_word_embeddings": False,
        "layer_norm_epsilon": _f32(source.get("layer_norm_epsilon"), "layer_norm_epsilon"),
        "block_configs": emitted,
        "mamba": {
            "mamba_num_heads": integer("mamba_num_heads"),
            "mamba_head_dim": integer("mamba_head_dim"),
            "ssm_state_size": integer("ssm_state_size"),
            "n_groups": integer("n_groups"),
            "conv_kernel": integer("conv_kernel"),
            "chunk_size": integer("chunk_size"),
            "time_step_min": _f32(source.get("time_step_min"), "time_step_min"),
            "use_conv_bias": bool(source.get("use_conv_bias", True)),
        },
        "attention": {
            "num_attention_heads": integer("num_attention_heads"),
            "num_key_value_heads": integer("num_key_value_heads"),
            "head_dim": integer("head_dim"),
        },
        "router": {
            "routed_scaling_factor": _f32(source.get("routed_scaling_factor"), "routed_scaling_factor"),
            "norm_topk_prob": bool(source.get("norm_topk_prob", True)),
            "n_group": 1,
            "topk_group": 1,
            "e_score_correction_bias": True,
        },
        "moe_latent_size": latent,
        "moe_shared_expert_intermediate_size": shared,
    }


def text_config(source: dict) -> dict:
    """The language tower's config in the form the artifact's own parser reads.

    The emitted key set is exactly `nemotron_h/config.cpp`'s: it validates every member it is given,
    so a field from another level is a refusal rather than a field it ignores.
    """
    if not is_nemotron_h(source):
        raise ValueError(f"unsupported Nemotron-H architecture {source.get('architectures')!r}")
    raw = source.get("llm_config")
    if not isinstance(raw, dict):
        return _flat_text_config(source)
    # The Omni wrapper has no MTP: `num_nextn_predict_layers` is zero and both MTP block lists are
    # empty; a source that declares one would need those blocks mapped, so it is refused. The flat
    # `NemotronHForCausalLM` layout does declare one (Lightning 30B: a single MTP layer), and this
    # engine's speculative path for that pair is the separate DFlash drafter, so the layer is
    # deliberately not mapped -- stated, not silently dropped.
    flat_layout = not isinstance(source.get("llm_config"), dict)
    if not flat_layout:
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
        # A latent of zero means a direct bank: the experts read the block's residual, so there are
        # no latent projections and the bank's width is the hidden width (Lightning 30B).
        bank = latent if latent else h
        if latent:
            self.add(p + "latent_in", store, source_prefix + "mixer.fc1_latent_proj.weight",
                     (latent, h), inputs=(ffn_input,))
            self.add(p + "latent_out", store, source_prefix + "mixer.fc2_latent_proj.weight",
                     (h, latent), inputs=(p + "expert_product",))
        self.add(p + "shared/up", store, source_prefix + "mixer.shared_experts.up_proj.weight",
                 (shared, h), inputs=(ffn_input,))
        self.add(p + "shared/down", store, source_prefix + "mixer.shared_experts.down_proj.weight",
                 (h, shared), inputs=(p + "shared/product",))
        # Every expert is its own tensor; the pair of groups below is what makes each layer's bank a
        # single parent per matrix, so an expert is a row range rather than a separate object. A
        # direct bank (latent 0) has no latent projection to produce `expert_input`, so the experts
        # read the block's residual -- the same name the router reads.
        expert_input = (p + "expert_input") if latent else ffn_input
        # The engine's expert kernel requires the intermediate width to be a multiple of 128. The
        # declared width is that rounded value; a model storing a narrower one is zero-padded here,
        # which is exact (zero rows activate to zero, zero columns contribute nothing).
        stored_width = width
        probe        = source_prefix + "mixer.experts.0.up_proj.weight"
        if store.has(probe):
            stored_width = store.describe(probe).shape[0]
        pad_up   = (width, bank) if width != stored_width else None
        pad_down = (bank, width) if width != stored_width else None
        ups, downs = [], []
        for expert in range(experts):
            ep = p + f"experts/{expert}/"
            self.add(ep + "up", store,
                     source_prefix + f"mixer.experts.{expert}.up_proj.weight", (width, bank),
                     inputs=(expert_input,), pad=pad_up,
                     source_shape=(stored_width, bank) if pad_up else None)
            self.add(ep + "down", store,
                     source_prefix + f"mixer.experts.{expert}.down_proj.weight", (bank, width),
                     inputs=(ep + "product",), pad=pad_down,
                     source_shape=(bank, stored_width) if pad_down else None)
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


# ---------------------------------------------------------------------------------------------
# The LiLiCorr speculative draft: a five-layer DFlash2 body over this tower's own residual taps,
# plus the correlation head that reorders the body's per-position candidates.
#
# The checkpoint (`/mnt/storage/models/nemo/74B/LiLiCorr`) is already quantized -- ModelOpt NVFP4
# W4A16, `U8` codes with `F8_E4M3` per-16 scales and one `F32` divisor per tensor, whose *stored*
# widths are half the logical ones because two codes share a byte. `sources/modelopt.py` reads that
# triple and hands back decoded values, so this emitter declares logical shapes throughout and the
# recipe leaves the draft at BF16: the decoded 4-bit values re-quantized through a fresh
# scale sweep would be a second lossy rounding for a drafter that costs 3% of a round, and the
# 2.4 GB it adds to a 45 GB artifact is not worth it. Importing the stored scales directly is the
# upgrade path if the drafter ever shows up in a profile.

_LILICORR_BLOCK = 8
_LILICORR_LAYERS = 5
_LILICORR_HEADS = 32
_LILICORR_KV_HEADS = 2
_LILICORR_HEAD_DIM = 128
_LILICORR_INTERMEDIATE = 12288
_LILICORR_TAPS = (1, 18, 35, 51, 68, 85)
_LILICORR_MASK = 990
_LILICORR_WINDOW = 1024
_LILICORR_EPSILON = _f32(1e-6, "rms_norm_eps")
_LILICORR_ROPE_THETA = 10000.0
_LILICORR_MAX_POSITIONS = 262144
_LILICORR_HEAD_WIDTH = 1024
_LILICORR_HEAD_LAYERS = 2
_LILICORR_HEAD_HEADS = 8
_LILICORR_HEAD_TOPK = 8
_LILICORR_FACTOR = 1024
_LILICORR_VECTOR_EPS = _f32(1e-4, "lilicorr_vector_eps")
_LILICORR_LOGIT_SCALE = 8.0


def is_lilicorr_draft(config: dict) -> bool:
    return "LiLiCorrDraftModel" in (config.get("architectures") or ())


def lilicorr_config(raw: dict, target: dict) -> dict:
    """The draft's config in the artifact's own form, validated against the tower it drafts for.

    Every structural constant is pinned: the release has one shape, and a source that disagrees with
    any of it is a different drafter rather than a variant of this one. The taps are *absolute*
    target block indices, so they are checked against the tower's block count. The nested
    `dflash_config` keeps the shape the generic draft path already emits, so the artifact carries one
    schema for drafts whatever family they belong to.
    """
    if not is_lilicorr_draft(raw):
        raise ValueError(f"unsupported draft architecture {raw.get('architectures')!r}")
    label = "dflash2"
    _expect_text(raw, "model_type", "qwen3", label)
    _expect_int(raw, "block_size", _LILICORR_BLOCK, label)
    _expect_int(raw, "hidden_size", target["hidden_size"], label)
    _expect_int(raw, "num_hidden_layers", _LILICORR_LAYERS, label)
    _expect_int(raw, "num_attention_heads", _LILICORR_HEADS, label)
    _expect_int(raw, "num_key_value_heads", _LILICORR_KV_HEADS, label)
    _expect_int(raw, "head_dim", _LILICORR_HEAD_DIM, label)
    _expect_int(raw, "intermediate_size", _LILICORR_INTERMEDIATE, label)
    _expect_int(raw, "vocab_size", target["vocab_size"], label)
    _expect_f32(raw, "rms_norm_eps", _LILICORR_EPSILON, label)
    _expect_f32(raw, "rope_theta", _LILICORR_ROPE_THETA, label)
    _expect_int(raw, "max_position_embeddings", _LILICORR_MAX_POSITIONS, label)
    _expect_flag(raw, "tie_word_embeddings", False, label)
    _expect_flag(raw, "attention_bias", False, label)
    _expect_text(raw, "hidden_act", "silu", label)
    if raw.get("layer_types") != ["sliding_attention"] * _LILICORR_LAYERS:
        raise ValueError(f"{label}.layer_types: expected {_LILICORR_LAYERS} sliding-attention blocks")
    _expect_int(raw, "sliding_window", _LILICORR_WINDOW, label)
    _expect_flag(raw, "use_sliding_window", True, label)
    _expect_flag(raw, "attention_sink_bias", True, label)

    draft = raw.get("dflash_config")
    if not isinstance(draft, dict):
        raise ValueError(f"{label}.dflash_config: expected an object")
    taps = draft.get("target_layer_ids")
    if (
        not isinstance(taps, list)
        or tuple(taps) != _LILICORR_TAPS
        or any(type(i) is not int or not 0 <= i < len(target["block_configs"]) for i in taps)
    ):
        raise ValueError(f"{label}.dflash_config.target_layer_ids: expected {list(_LILICORR_TAPS)}")
    _expect_int(draft, "mask_token_id", _LILICORR_MASK, label + ".dflash_config")
    _expect_flag(draft, "use_swa", True, label + ".dflash_config")
    _expect_int(draft, "swa_window_size", _LILICORR_WINDOW, label + ".dflash_config")
    _expect_flag(draft, "causal", False, label + ".dflash_config")
    _expect_int(draft, "conv_kernel_size", 2, label + ".dflash_config")
    _expect_int(draft, "conv_group_size", 16, label + ".dflash_config")
    _expect_text(draft, "projector_type", "lilicorr", label + ".dflash_config")
    _expect_flag(draft, "lilicorr_enabled", True, label + ".dflash_config")
    _expect_int(draft, "lilicorr_candidate_topk", _LILICORR_HEAD_TOPK, label + ".dflash_config")
    _expect_int(draft, "lilicorr_hidden_size", _LILICORR_HEAD_WIDTH, label + ".dflash_config")
    _expect_int(draft, "lilicorr_num_layers", _LILICORR_HEAD_LAYERS, label + ".dflash_config")
    _expect_int(draft, "lilicorr_num_heads", _LILICORR_HEAD_HEADS, label + ".dflash_config")
    _expect_f32(draft, "lilicorr_mlp_ratio", 2.0, label + ".dflash_config")
    _expect_int(draft, "lilicorr_factor_dim", _LILICORR_FACTOR, label + ".dflash_config")
    _expect_f32(draft, "lilicorr_vector_eps", _LILICORR_VECTOR_EPS, label + ".dflash_config")
    _expect_f32(draft, "lilicorr_logit_scale", _LILICORR_LOGIT_SCALE, label + ".dflash_config")

    return {
        "architectures": list(raw["architectures"]),
        "model_type": "qwen3",
        "block_size": _LILICORR_BLOCK,
        "hidden_size": target["hidden_size"],
        "num_hidden_layers": _LILICORR_LAYERS,
        "num_attention_heads": _LILICORR_HEADS,
        "num_key_value_heads": _LILICORR_KV_HEADS,
        "head_dim": _LILICORR_HEAD_DIM,
        "intermediate_size": _LILICORR_INTERMEDIATE,
        "vocab_size": target["vocab_size"],
        "rms_norm_eps": _LILICORR_EPSILON,
        # The artifact's draft parser reads the rotation out of `rope_parameters` and requires a
        # declared positional capacity, so both are written in that shape rather than flat.
        "rope_parameters": {"rope_theta": _LILICORR_ROPE_THETA},
        "max_position_embeddings": _LILICORR_MAX_POSITIONS,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "layer_types": ["sliding_attention"] * _LILICORR_LAYERS,
        "sliding_window": _LILICORR_WINDOW,
        "dflash_config": {
            "target_layer_ids": list(_LILICORR_TAPS),
            "mask_token_id": _LILICORR_MASK,
            "use_swa": True,
            "swa_window_size": _LILICORR_WINDOW,
            "causal": False,
            "attention_sink_bias": True,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "projector_type": "lilicorr",
            "lilicorr_enabled": True,
            "lilicorr_candidate_topk": _LILICORR_HEAD_TOPK,
            "lilicorr_hidden_size": _LILICORR_HEAD_WIDTH,
            "lilicorr_num_layers": _LILICORR_HEAD_LAYERS,
            "lilicorr_num_heads": _LILICORR_HEAD_HEADS,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": _LILICORR_FACTOR,
            "lilicorr_vector_eps": _LILICORR_VECTOR_EPS,
            "lilicorr_logit_scale": _LILICORR_LOGIT_SCALE,
        },
    }


class _LilicorrBuilder(_NemotronBuilder):
    """Maps the draft: the DFlash2 body over the tower's taps, then the correlation head."""

    def reshaped(self, name, store, source_name, shape, stored_shape):
        """A parameter whose checkpoint tensor carries singleton axes the artifact does not.

        The head's structural embeddings are stored `[1,1,1,slots,width]` and `[1,1,slots,1,width]`
        because the reference indexes them through a broadcast; the artifact carries the two live
        axes, and the flat element order is already slot-major.
        """
        def factory(selected, format=None):
            if format is not None:
                raise ValueError(f"{name}: a structural embedding has no encoded source")
            stored = tensor_source(selected, source_name, stored_shape)

            def read(begin: int, end: int) -> torch.Tensor:
                return stored.values().reshape(-1)[begin:end]

            return LogicalSource(shape, f"reshaped({stored.label})", read)

        self.model.add(
            Parameter(name, shape, factory(store), factory, (), "bf16",
                      residency="dflash2")
        )

    def quantized(self, name, store, source_name, shape, inputs=()):
        """A parameter whose source is a ModelOpt NVFP4 triple, read as decoded values."""
        prefix = "dflash2/lilicorr/"
        factory = lambda selected: nvfp4_source(selected, source_name, shape)  # noqa: E731
        self.model.add(
            Parameter(name, shape, factory(store), factory, tuple(inputs), "bf16",
                      residency="dflash2")
        )

    def lilicorr(self, store, draft):
        h = draft["hidden_size"]
        d = draft["head_dim"]
        query, kv = draft["num_attention_heads"] * d, draft["num_key_value_heads"] * d
        taps = draft["dflash_config"]
        p = lambda rest: "dflash2/" + rest  # noqa: E731
        # The taps are fused by `fc`; the checkpoint stores that projection quantized, and its
        # logical width is the tap count times the tower's width.
        self.quantized(p("feature_projection"), store, "fc",
                       (h, len(taps["target_layer_ids"]) * h), inputs=(p("target_features"),))
        self.add(p("context_norm"), store, "hidden_norm.weight", (h,))
        self.add(p("final_norm"), store, "norm.weight", (h,))
        for index in range(draft["num_hidden_layers"]):
            layer, source = p(f"layers/{index}/"), f"layers.{index}."
            self.add(layer + "input_norm", store, source + "input_layernorm.weight", (h,))
            self.add(layer + "post_attention_norm", store,
                     source + "post_attention_layernorm.weight", (h,))
            for role, field, rows in (("query", "q_proj", query),
                                      ("key", "k_proj", kv),
                                      ("value", "v_proj", kv)):
                # The same projection supplies the block rows and the cached-context rows, so the
                # parameter is declared once and read from both inputs.
                inputs = (layer + "query_projection_input",)
                if role in ("key", "value"):
                    self.add(layer + "attention/context_" + role, store,
                             source + "self_attn." + field + ".weight", (rows, h),
                             inputs=(p("context_input"),))
                self.add(layer + "attention/" + role, store,
                         source + "self_attn." + field + ".weight", (rows, h), inputs=inputs)
            self.group(*(layer + "attention/" + r for r in ("query", "key", "value")))
            for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
                self.add(layer + "attention/" + role, store,
                         source + "self_attn." + field + ".weight", (d,))
            # The sink the reference adds unscaled after the logit scale, one per query head.
            self.add(layer + "attention/sink_bias", store,
                     source + "self_attn.attention_sink_bias",
                     (draft["num_attention_heads"],))
            self.add(layer + "attention/output", store, source + "self_attn.o_proj.weight",
                     (h, query), inputs=(layer + "attention_output",))
            self.dense(layer, source, store, h, draft["intermediate_size"], draft=True)
            for branch in ("attention_conv", "mlp_conv"):
                # The checkpoint stores the base kernel as `[side, tap, channel]`; the artifact
                # carries it flat over the tap and channel axes, which is the order the convolution
                # indexes (`side * taps * H + tap * H + e`) and the only rank a dense parameter has.
                self.reshaped(layer + branch + "/base_kernel", store,
                              source + branch + ".base_kernel",
                              (2, taps["conv_kernel_size"] * h),
                              (2, taps["conv_kernel_size"], h))
                self.add(layer + branch + "/kernel_projection", store,
                         source + branch + ".kernel_projection.weight",
                         (2 * taps["conv_kernel_size"] * (h // taps["conv_group_size"]), h),
                         inputs=(layer + branch + "_input",))
        self.lilicorr_head(store, draft)

    def lilicorr_head(self, store, draft):
        """The correlation head: 64 tensors, all at the head's own 1024 width."""
        head = draft["dflash_config"]
        width = head["lilicorr_hidden_size"]
        factor = head["lilicorr_factor_dim"]
        layer_width = int(width * head["lilicorr_mlp_ratio"])
        h = draft["hidden_size"]
        p = lambda rest: "dflash2/lilicorr/" + rest  # noqa: E731
        self.add(p("anchor_norm"), store, "lilicorr.anchor_norm.weight", (width,))
        self.add(p("output_norm"), store, "lilicorr.output_norm.weight", (width,))
        self.add(p("feature_norm/weight"), store, "lilicorr.feature_norm.weight", (5,))
        self.add(p("feature_norm/bias"), store, "lilicorr.feature_norm.bias", (5,))
        self.add(p("in_head"), store, "lilicorr.in_head.weight", (width, width))
        self.add(p("in_head_bias"), store, "lilicorr.in_head.bias", (width,))
        self.add(p("out_head"), store, "lilicorr.out_head.weight", (width, width))
        self.add(p("out_head_bias"), store, "lilicorr.out_head.bias", (width,))
        self.add(p("anchor_out_head"), store, "lilicorr.anchor_out_head.weight", (width, width))
        self.add(p("anchor_out_head_bias"), store, "lilicorr.anchor_out_head.bias", (width,))
        # The three views of the head's input space and the feature path are quantized in the
        # checkpoint; `quantized` decodes them.
        self.quantized(p("token_projection"), store, "lilicorr.token_proj", (width, h),
                       inputs=(p("token_input"),))
        self.add(p("token_bias"), store, "lilicorr.token_proj.bias", (width,))
        self.quantized(p("context_projection"), store, "lilicorr.context_proj", (width, h),
                       inputs=(p("context_input"),))
        self.add(p("context_bias"), store, "lilicorr.context_proj.bias", (width,))
        self.quantized(p("pass_hidden_projection"), store, "lilicorr.pass_hidden_proj",
                       (width, h), inputs=(p("pass_hidden_input"),))
        self.add(p("pass_hidden_bias"), store, "lilicorr.pass_hidden_proj.bias", (width,))
        self.add(p("feature_mlp/up"), store, "lilicorr.feature_mlp.up_proj.weight", (width, 5))
        self.add(p("feature_mlp/up_bias"), store, "lilicorr.feature_mlp.up_proj.bias", (width,))
        self.add(p("feature_mlp/down"), store, "lilicorr.feature_mlp.down_proj.weight",
                 (width, width))
        self.add(p("feature_mlp/down_bias"), store, "lilicorr.feature_mlp.down_proj.bias",
                 (width,))
        self.add(p("factor_input_projection"), store, "lilicorr.factor_input_proj.weight",
                 (factor, 3 * width))
        self.add(p("factor_input_bias"), store, "lilicorr.factor_input_proj.bias", (factor,))
        # Structural embeddings: one row per block slot (`block_size` ranks, `block_size - 1`
        # relative slots) and the two bias tables the head adds over candidate pairs.
        self.reshaped(p("rank_embedding"), store, "lilicorr.rank_embedding",
                      (draft["block_size"], width), (1, 1, 1, draft["block_size"], width))
        self.reshaped(p("slot_embedding"), store, "lilicorr.slot_embedding",
                      (draft["block_size"] - 1, width), (1, 1, draft["block_size"] - 1, 1, width))
        self.add(p("relative_slot_bias"), store, "lilicorr.relative_slot_bias",
                 (draft["block_size"], 2 * draft["block_size"] - 1))
        self.add(p("same_slot_bias"), store, "lilicorr.same_slot_bias", (draft["block_size"],))
        for index in range(head["lilicorr_num_layers"]):
            layer, source = p(f"layers/{index}/"), f"lilicorr.layers.{index}."
            self.add(layer + "norm", store, source + "attn_norm.weight", (width,))
            self.add(layer + "mlp_norm", store, source + "mlp_norm.weight", (width,))
            # The head's attention is one fused QKV matrix at the head width.
            self.add(layer + "attention/in_projection", store, source + "attn.in_proj_weight",
                     (3 * width, width))
            self.add(layer + "attention/in_projection_bias", store,
                     source + "attn.in_proj_bias", (3 * width,))
            self.quantized(layer + "attention/output", store, source + "attn.out_proj",
                           (width, width), inputs=(layer + "attention_output",))
            self.add(layer + "attention/output_bias", store, source + "attn.out_proj.bias",
                     (width,))
            self.quantized(layer + "mlp/up", store, source + "mlp.up_proj", (layer_width, width),
                           inputs=(layer + "mlp_input",))
            self.add(layer + "mlp/up_bias", store, source + "mlp.up_proj.bias", (layer_width,))
            self.quantized(layer + "mlp/down", store, source + "mlp.down_proj",
                           (width, layer_width), inputs=(layer + "mlp_product",))
            self.add(layer + "mlp/down_bias", store, source + "mlp.down_proj.bias", (width,))


def build_model(
    base,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, "object"] | None = None,
    resource_overrides: Mapping[str, str] | None = None,
) -> Model:
    """Map the Bolt language tower and, when asked, the LiLiCorr draft that drafts for it."""
    selected = set(components)
    if "text" not in selected or selected - {"text", "dflash", "dflash2"}:
        raise ValueError("Nemotron-H maps the text component and a dflash or dflash2 draft")
    companions = {} if companions is None else companions
    config = text_config(base.config)
    references, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        overrides=resource_overrides,
        family="nemotron",
    )
    records = {"text": {"config": config, "resources": references["text"]}}
    if "dflash" in selected:
        if "dflash" not in companions:
            raise ValueError("selected dflash requires its source")
        # A standalone DFlash export omits two keys the shared normalizer asserts; both are fixed
        # by the target it drafts for.
        raw = dict(companions["dflash"].config)
        raw.setdefault("is_causal", False)
        raw.setdefault("num_target_layers", len(config["block_configs"]))
        # This export declares YaRN (factor 128 over an 8192-token original window). Inside that
        # window YaRN is the identity, and the engine's draft rope is a plain theta rotation over
        # the absolute positions, so the base theta is carried and drafts past 8192 positions are
        # the one regime this artifact does not reproduce.
        rope = dict(raw.get("rope_parameters") or {})
        if rope.get("rope_type") not in (None, "default"):
            rope = {
                "rope_type": "default",
                "rope_theta": rope.get("rope_theta", raw.get("rope_theta", 10000.0)),
            }
        raw["rope_parameters"] = rope
        # The normalizer reads the target's layer count by that name; the emitted config carries
        # the block list instead, so it is supplied here rather than added as a key the artifact's
        # parser would refuse.
        target_for_draft = dict(config, num_hidden_layers=len(config["block_configs"]))
        dflash_config = draft_config(raw, target_for_draft, "dflash")
        records["dflash"] = {"config": dflash_config, "target": "text"}
    if "dflash2" in selected:
        if "dflash2" not in companions:
            raise ValueError("selected dflash2 requires its source")
        # `target` names the component the draft conditions on, which is how the runtime finds the
        # taps without every draft repeating the tower's own description.
        records["dflash2"] = {
            "config": lilicorr_config(companions["dflash2"].config, config),
            "target": "text",
        }
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _LilicorrBuilder(model)
    if "dflash" in selected:
        builder.draft(companions["dflash"], dflash_config, config, "dflash")
    h, vocabulary = config["hidden_size"], config["vocab_size"]
    # The Omni wrapper nests the tower under `language_model.`; the flat `NemotronHForCausalLM`
    # layout (Lightning 30B) stores `backbone.*` and a top-level `lm_head`.
    flat = not isinstance(base.config.get("llm_config"), dict)
    stem = "backbone." if flat else "language_model.backbone."
    builder.add("text/token_embedding", base, stem + "embeddings.weight", (vocabulary, h))
    builder.add("text/output_head", base, ("lm_head.weight" if flat else "language_model.lm_head.weight"),
                (vocabulary, h), inputs=("text/final_hidden",))
    builder.add("text/final_norm", base, stem + "norm_f.weight", (h,))
    # The config decides each layer's shape and kind; the checkpoint's tensors confirm both, because
    # every parameter's declared shape is checked against the tensor it is read from and every
    # expert's own tensor has to exist.
    for index, block in enumerate(config["block_configs"]):
        builder.block(f"text/layers/{index}/", stem + f"layers.{index}.", base, config, block)
    if "dflash2" in selected:
        builder.lilicorr(companions["dflash2"], records["dflash2"]["config"])
    return model
