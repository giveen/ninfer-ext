"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

import re

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    q4_rows_maxabs,
    q4_rows_qat,
    nvfp4_absmax,
    nvfp4_mse,
)
from .sources.compressed_tensors import compressed_matrix_source
from .sources.exl3 import exl3_matrix_source
from .sources.logical import select_rows
from .sources.modelopt import fp8_tensor_rows_source, nvfp4_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"
Q4ROWS = "q4_g32_fp16_rows"


def _assign(recipe, name, format, *, source=None, activation_policy=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(
        name,
        format=format,
        method=method,
        source=source,
        activation_policy=activation_policy,
    )


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/", "eagle3/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary, gate_up=Q4):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(("/mlp/gate", "/mlp/up")):
            # The fused gate/up projection has a Q4 W4A8 route, so the prefill-heavy
            # activation is allowed to run int8; the other projections keep A16.
            _assign(
                recipe,
                name,
                gate_up,
                activation_policy="AllowA8" if gate_up == Q4 else None,
            )
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def _dense_bf16(model, recipe):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    for name, parameter in model.parameters.items():
        if not parameter.projection and name not in ("text/token_embedding", "text/output_head"):
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
        _assign(recipe, name, "bf16")
    # DFlash/DFlash2 reuse the target's key/value matrices when the component is present.
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_8_27b_bf16(model, recipe, sources):
    """Full-precision BF16 weights, the native EXL3 quantizer's input artifact."""
    _dense_bf16(model, recipe)


def qwen3_8_27b_exl3(model, recipe, sources):
    """Qwen3.8-27B with EXL3 Text/MTP projections produced by `ninfer-quantize`.

    `--source quantized=PATH` is the quantizer's `exl3.safetensors`. Each shared-input parent the
    model executes as one weight (attention q/key/gate/value, GDN q/key/value/z, MLP gate/up) is
    stored as one EXL3 matrix sharing its `suh`; the recipe binds its members to their row ranges so
    the automatic grouping reproduces the parent. Everything absent from the store (embedding,
    norms, GDN a/b, Vision, uncalibrated MTP) keeps its default precision.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    store = sources["quantized"]
    parameters = model.parameters
    # The GDN control projections stay BF16 as separate parents, exactly as the groupwise recipe
    # keeps them; their dedicated gating Op consumes one 48-row weight per side.
    for name in parameters:
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)

    def assign(name, source):
        recipe.assign(name, format="exl3_mul1", method=import_encoded, source=source)

    assigned: set[str] = set()
    # Longest packing group first, so a fully stored group wins over its stored sub-groups.
    for names in sorted(model.packing_groups, key=len, reverse=True):
        if not names or any(name in assigned for name in names):
            continue
        if any(name not in parameters for name in names):
            continue
        first = names[0]
        if not store.has(first + ".trellis"):
            continue
        shapes = [parameters[name].shape for name in names]
        if any(len(shape) != 2 for shape in shapes) or len({shape[1] for shape in shapes}) != 1:
            continue
        parent_shape = (sum(shape[0] for shape in shapes), shapes[0][1])
        info = store.describe(first + ".trellis")
        if tuple(info.shape[:2]) != (parent_shape[0] // 16, parent_shape[1] // 16):
            continue
        parent = exl3_matrix_source(store, first, parent_shape)
        begin = 0
        for name, shape in zip(names, shapes):
            assign(name, select_rows(parent, ((begin, begin + shape[0]),)))
            assigned.add(name)
            begin += shape[0]

    for name, parameter in parameters.items():
        # The vocabulary head is a terminal parameter rather than a projection; it is stored and
        # executed as a weight like any other, so it is assigned here too.
        if name in assigned or not store.has(name + ".trellis"):
            continue
        if not parameter.projection and not name.endswith("output_head"):
            continue
        assign(name, exl3_matrix_source(store, name, parameter.shape))
        assigned.add(name)

    # Vision is not EXL3: the tower's MLP intermediate (4304) is not a multiple of the 128-point
    # Hadamard block the format's tiles need, so it keeps the groupwise formats every other official
    # recipe uses (Q6 patch embedding, Q8 merger, Q4 qkv and fc1, Q5 the rest).
    for name, parameter in parameters.items():
        if not name.startswith("vision/") or not parameter.projection:
            continue
        if name == "vision/patch_embedding":
            format = Q6
        elif name.startswith("vision/merger/"):
            format = Q8
        elif name.endswith(
            ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_8_27b_q6(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8, gate_up=Q6)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_quasar_nvfp4(model, recipe, sources):
    """Qwen3.8-27B from the QUASAR QAT checkpoint: every Text linear imported as stored NVFP4.

    The checkpoint is its own main source: embedding, LM head, norms, MTP and Vision are
    unquantized there. GDN a/b are stored NVFP4 too but have no NVFP4 route, so they stay
    separate and decoded to their logical values, as `qwen3_6_27b_nvfp4` keeps them.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    for name in ("text/token_embedding", "text/output_head"):
        recipe.assign(name, format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/"):
            continue
        if not parameter.projection:
            recipe.assign(name, source=model.source(name, quantized))
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            recipe.assign(name, source=model.source(name, quantized))
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


_MOE_EXPERT_SOURCE = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def _moe_encoded_source(store, parameter, name):
    """Map one routed or shared expert matrix onto its per-expert NVFP4 prefix.

    The MoE description reads routed experts out of a single fused `experts.gate_up_proj`
    tensor with an offset, so the generic factory cannot reach an encoded source for them
    and says so. The checkpoint stores every expert as its own prefix, which is a plain
    matrix, so the mapping is stated here once instead. Both the compressed-tensors naming
    (`weight_packed`) and the NVIDIA ModelOpt naming (`weight`/`weight_scale_2`) are accepted,
    so the same recipe converts either export of the model.
    """
    head, _, tail = name.partition("/moe/")
    layer = head.rsplit("/", 1)[-1]
    if tail.startswith("experts/"):
        expert, _, role = tail[len("experts/") :].partition("/")
        prefix = f"mlp.experts.{expert}."
    else:
        prefix, role = "mlp.shared_expert.", tail.rsplit("/", 1)[-1]
    if role not in _MOE_EXPERT_SOURCE:
        raise ValueError(f"{name}: unknown expert matrix {role!r}")
    leaf = prefix + _MOE_EXPERT_SOURCE[role]
    for root in ("model.language_model.layers.", "model.layers."):
        prefix = f"{root}{layer}.{leaf}"
        if store.has(prefix + ".weight_packed"):
            return compressed_matrix_source(store, prefix, parameter.shape, "nvfp4")
        if store.has(prefix + ".weight_scale_2"):
            return nvfp4_source(store, prefix, parameter.shape)
    raise ValueError(f"{name}: no NVFP4 source for {leaf} in {store.path}")


def qwen3_6_35b_a3b_nvfp4(model, recipe, sources):
    """`qwen3_6_35b_a3b` with routed and shared experts imported as NVFP4 instead of Q4/Q5.

    Everything outside the experts keeps the representation of the groupwise recipe, so the
    two artifacts differ in exactly one mechanism and are comparable to each other.
    """
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    quantized = sources["quantized"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name or "/moe/shared/" in name:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=_moe_encoded_source(quantized, parameter, name),
                activation_policy="AllowA4",
            )
        else:
            _assign(recipe, name, Q8)
    # The op takes the routed experts as one plane per bank, so the banks the model declares have
    # to become one parent each. Default packing keeps parents apart when their sources were
    # quantised against different divisors, which is right where the consumer reads one divisor and
    # wrong here: this plane is a stack by construction, and its kernels take the divisor from the
    # row they are reading.
    for names in model.packing_groups:
        # Text layers only: the MTP component declares the same role names and keeps its groupwise
        # Q8 experts, which this recipe does not assign and must not group.
        if all(
            name.startswith("text/layers/")
            and ("/moe/experts/" in name or "/moe/shared/" in name)
            for name in names
        ):
            recipe.group(names)


# Qwen4Exp dense projections with a compiled FP8 A8 route: the shape list in
# src/ops/linear/fp8/shapes/ and the two-term activation route both exist for these (n, k) pairs.
# They carry 74% of the model's dense prefill FLOPs at a 16,384-token chunk. The hyper-connection
# stems (n=324, k=10240 and n=10240, k=320), the indexer projection (n=640) and the shared-expert
# down projection (k=640) keep Q8: their shapes divide neither the FP8 row tile nor the GEMV's
# 32 x ValuesPerLane, so no FP8 route resolves for them.
QWEN4_DENSE_A8_SHAPES = frozenset(
    {(16384, 2560), (13312, 2560), (2560, 6144), (1280, 2560)}
)


def _qwen4_dense_a8(shape) -> bool:
    """Whether this projection shape has a compiled FP8 two-term route."""
    return len(shape) >= 2 and (int(shape[0]), int(shape[1])) in QWEN4_DENSE_A8_SHAPES


def _flash_next_nvfp4_text(model, recipe, sources, *, dense_a8: bool):
    """Qwen3.8-Flash-Next (Qwen4Exp) with one NVFP4 block-scaled route for every routed expert.

    An encoded source (ModelOpt) keeps its NVFP4 codes, block scales and divisors; a full-precision
    source is quantized with nvfp4_absmax. A full-precision source carries no calibrated activation
    divisor, so its bank reports a zero A4 divisor and executes the BF16-activation route; an
    encoded source keeps its calibrated divisors and runs W4A4. The n-gram table keeps its FP8 codes
    when the source stores them and is row-quantized otherwise. Dense projections become Q8 (the
    output head Q6); routers, shared-expert gates, norms and small vectors stay direct.
    """
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    # Vision's mixed groupwise assignment; the loop below restates every Text and MTP choice.
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not (parameter.projection or name.endswith("/ple/table")):
            continue
        encoded = parameter.source.read_encoded is not None
        if name.endswith("/ple/table"):
            if encoded:
                recipe.assign(name, format=FP8, method=import_encoded, source=parameter.source)
            else:
                recipe.assign(name, format=FP8, method=fp8_row_maxabs)
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded if encoded else nvfp4_absmax,
                source=parameter.source,
                activation_policy="AllowA4",
            )
            continue
        if dense_a8 and _qwen4_dense_a8(parameter.shape):
            # FP8 weights on the two-term activation route: the pair kernel carries the activation
            # as two E4M3 terms, so the result lands at the A16 floor (the FP8 A8-pair test asserts
            # that) while the op runs 1.5-1.9x faster at a prefill width.
            recipe.assign(name, format=FP8, method=fp8_row_maxabs,
                          activation_policy="AllowA8Pair")
        else:
            _assign(recipe, name, Q8)
    # Each layer's gate/up and down banks become one parent so an expert is a fixed row range of
    # its bank, addressed by the expert cache as a few contiguous spans.
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """The shipped route: Q8 dense projections (A16 activations)."""
    _flash_next_nvfp4_text(model, recipe, sources, dense_a8=False)


def qwen3_8_flash_next_nvfp4_dense_a8(model, recipe, sources):
    """Q8 everywhere except the dense projections that have a compiled FP8 two-term route.

    Those become FP8 with `AllowA8Pair`, which prefill runs ~1.8x faster (the shapes carry 74% of
    the model's dense prefill FLOPs). Decode widths are unaffected: the pair route is admitted only
    above kFp8A8PairMinTokens, so every narrower call keeps the A16 ladder.
    """
    _flash_next_nvfp4_text(model, recipe, sources, dense_a8=True)


def qwen3_8_flash_next_exl3(model, recipe, sources):
    """Qwen3.8-Flash-Next with EXL3 routed experts produced by `ninfer-quantize`.

    `--source quantized=PATH` is the quantizer's expert store (a directory with a safetensors
    index). Every routed expert gate, up and down is its own EXL3 matrix with its own input and
    output scales; each layer's gate/up and down banks keep the NVFP4 recipe's expert-major order,
    and because the matrices were quantized apart each bank object stores one input-scale set per
    expert. An expert missing from the store is an error: leaving it at another precision would
    make the bank unexecutable. The n-gram table is quantized to 4-bit group codes (row_group_v1,
    group of 32 with a binary16 scale, about 4.5 bits per value); its noise cost was measured at KL
    0.016 against the FP8 table. Dense projections become Q6 (Q8 where K is not a multiple of 128; the output head Q6, the token
    embedding Q8); routers, shared-expert gates, norms and small vectors stay direct.
    """
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    store = sources["quantized"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not (parameter.projection or name.endswith("/ple/table")):
            continue
        encoded = parameter.source.read_encoded is not None
        if name.endswith("/ple/table"):
            recipe.assign(name, format=Q4ROWS, method=q4_rows_maxabs)
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            if not store.has(name + ".trellis"):
                raise ValueError(f"{name}: no EXL3 matrix in the quantizer store")
            recipe.assign(
                name,
                format="exl3_mul1",
                method=import_encoded,
                source=exl3_matrix_source(store, name, parameter.shape),
            )
            continue
        # The Q6 kernels need K aligned to 128; the low-rank hyper-connection stems (K=320) stay Q8.
        _assign(recipe, name, Q6 if parameter.shape[-1] % 128 == 0 else Q8)
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def qwen3_8_flash_next_bf16(model, recipe, sources):
    """Qwen3.8-Flash-Next at full BF16 precision: the native EXL3 quantizer's input artifact.

    Every Text/MTP projection, routed experts included, is stored as BF16 and the experts keep the
    NVFP4 recipe's per-layer gate/up and down parents, so an expert is a row range of its bank.
    The n-gram table keeps its row-scaled FP8 form; the quantizer does not read it.
    """
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not (parameter.projection or name.endswith("/ple/table")):
            continue
        if name.endswith("/ple/table"):
            recipe.assign(name, format=FP8, method=fp8_row_maxabs)
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")):
            continue
        _assign(recipe, name, "bf16")
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def gemma4_31b_base(model, recipe, sources):
    """Gemma 4 31B from the base BF16 checkpoint: our own NVFP4 MLPs and FP8 rows elsewhere.

    NVIDIA's NVFP4 checkpoint is an instruction-tuned artifact, so it cannot be imported here;
    the plan's L1 layout is reached with the converter's own encoders instead (plan 6.2/6.3).
    """
    _optional(model, recipe)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name == "text/token_embedding":
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")
            continue
        if not parameter.projection:
            continue
        if "/mlp/" in name:
            recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="AllowA4")
        elif "/attention/" in name:
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")


def gemma4_31b_g0(model, recipe, sources):
    """Gemma 4 31B with gewell's G0 precision layout, which spends precision where it pays.

    G0 keeps embeddings and norms BF16, holds Q/K/V at FP8 throughout, and uses FP8 rather than NVFP4
    for the O and MLP projections in the first six layers, every global layer and the local layer before
    it. The cheaper layout this project shipped first - FP8 embedding, NVFP4 MLP everywhere - leaves the
    head faithful but the logit tails further from HuggingFace than the per-layer agreement predicts, and
    the tails are what a KLD ranking reads. See docs/maintainer/gemma4-model.md §9.
    """
    _optional(model, recipe)
    # Layers whose O and MLP projections stay at FP8: the first six, then every global layer (5, 11, ...
    # 59) together with the local layer that precedes it.
    precise = {0, 1, 2, 3, 4, 5}
    for layer in range(6, 60, 6):
        precise.add(layer)
        precise.add(layer - 1)

    def layer_of(name):
        parts = name.split("/")
        return int(parts[2]) if len(parts) > 2 and parts[1] == "layers" else -1

    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name in ("text/token_embedding", "text/output_head"):
            recipe.assign(name, format="bf16")
            continue
        if not parameter.projection:
            continue
        layer = layer_of(name)
        if "/mlp/" in name:
            if layer in precise:
                recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")
            else:
                recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="AllowA4")
        elif "/attention/" in name:
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")


_GEMMA4_GROUPS = {
    "attn_qkv": ("attention/query", "attention/key", "attention/value"),
    "attn_o": ("attention/output",),
    "mlp_gu": ("mlp/gate", "mlp/up"),
    "mlp_down": ("mlp/down",),
}
_GEMMA4_FORMATS = {"q4": Q4, "q5": Q5, "q6": Q6, "q8": Q8, "fp8": FP8, "nvfp4": "nvfp4"}


def _gemma4_layout(model, recipe, layout_name):
    """Assigns a measured Gemma 4 precision layout (tools/convert/layouts/<name>.json).

    The layout names one format per projection group and layer, and the head's and embedding's. It
    is the output of tools/verify/gemma4_allocate.py, scored end to end with
    tools/verify/gemma4_sensitivity.py --assign before it was committed.
    """
    import json
    from pathlib import Path

    layout = json.loads((Path(__file__).parent / "layouts" / f"{layout_name}.json").read_text())
    _optional(model, recipe)

    def assign(name, short):
        if short == "bf16":
            recipe.assign(name, format="bf16")
        elif short == "nvfp4":
            recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="AllowA4")
        elif short == "fp8":
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")
        else:
            _assign(recipe, name, _GEMMA4_FORMATS[short])

    assign("text/token_embedding", layout["embedding"])
    assign("text/output_head", layout["head"])
    # The assistant drafter, when converted, is Q6 throughout, head included. The target verifies every
    # draft, so the format moves only acceptance and the drafter's step cost: against BF16, adaptive MTP
    # ran 4.3% faster on average over five prompts greedy and sampled (Q8 3.9%), worst -1.9%, and the
    # drafter shrinks from 0.94 to about 0.4 GB.
    for name, parameter in model.parameters.items():
        if name.startswith("mtp/") and parameter.projection:
            _assign(recipe, name, Q6)
    # The vision tower, when converted: Q6 for every layer's linears, and BF16 for the patch and output
    # projections. The tower amplifies weight error: against its FP32 output, Q4/Q5 (the Qwen tower's
    # formats) leave a 0.956 mean token cosine on a photo, Q6 0.989 (the BF16 tower's own level) and Q8
    # 0.999 (tools/verify/gemma4_vision_reference.py), and nothing downstream checks image features.
    for name, parameter in model.parameters.items():
        if not name.startswith("vision/") or not parameter.projection:
            continue
        if name in ("vision/patch_embedding", "vision/embedding_projection"):
            recipe.assign(name, format="bf16", method=cast_direct)
        else:
            _assign(recipe, name, Q6)
    for layer, groups in layout["layers"].items():
        for group, short in groups.items():
            for role in _GEMMA4_GROUPS[group]:
                name = f"text/layers/{layer}/{role}"
                if name in model.parameters:
                    assign(name, short)


def gemma4_31b_m1(model, recipe, sources):
    """Gemma 4 31B, mixed layout M1: Q8/Q6 attention, NVFP4/Q-format MLPs, Q6 head, BF16 embedding."""
    _gemma4_layout(model, recipe, "gemma4_31b_m1")


def _gemma4_26b_a4b(model, recipe, assign_expert, assign_dense):
    """Gemma 4 26B-A4B from the QAT BF16 checkpoint: `assign_expert` for the routed experts,
    `assign_dense` for attention, the dense MLP and the head, Q8 for the token table.

    The MoE block's router, its two scales and the layer's norms stay at the source precision. A
    router's output is a probability that a small error moves across the top-k boundary, and the
    per-expert scale multiplies what an expert produces, so neither is a place to spend bits.

    The token table is Q8 in every layout: it is a gather, not a projection, and the embedding Op's
    gather reads the row-split int8 table, not the int4 row format.

    Each expert bank becomes one parent object, so an expert is a fixed row range of its bank and a
    decode round streams the top-8 experts' spans.
    """
    _optional(model, recipe)
    # The assistant drafter, when the artifact carries it, is Q6 throughout, head included -- the
    # choice the 31B's layout makes, and for the same measured reason (against BF16, adaptive MTP ran
    # 4.3% faster where Q8 ran 3.9%, worst -1.9%, and the drafter shrinks from 0.94 to ~0.4 GB). Q8
    # is the fallback `_optional` gives it, so this override is deliberate. It is also why the
    # drafter's shapes need their own Q6 registry entries: a quantized linear route resolves its
    # geometry from a fixed table, and this drafter's attention is the 26B's 16 heads, not the 31B's
    # 32, so its widths are geometry the 31B's entries do not cover.
    for name, parameter in model.parameters.items():
        if name.startswith("mtp/") and parameter.projection:
            _assign(recipe, name, Q6)
    # The vision tower is the 31B's, at the same geometry, so it takes the 31B's measured formats
    # with the linear route one step finer: Q8 for every layer's linears and BF16 for the patch and
    # output projections. The tower amplifies weight error -- against its FP32 output, Q4/Q5 (the
    # formats the generic fallback would assign here) leave a 0.956 mean token cosine on a photo,
    # Q6 0.989 (the BF16 tower's own level) and Q8 0.999 -- and the tower is only about 0.42 B
    # parameters, so Q8 costs roughly 0.1 GB more than Q6 on a 16.5 GB artifact. Q8 also needs no
    # per-geometry registration: its row-split int8 route resolves any shape through its generic
    # selector, where Q6 would need a registry entry for each one. The patch and output projections
    # stay at the source precision the 31B measured them at, and together they are about 8 MB.
    for name, parameter in model.parameters.items():
        if not name.startswith("vision/") or not parameter.projection:
            continue
        if name in ("vision/patch_embedding", "vision/embedding_projection"):
            recipe.assign(name, format="bf16", method=cast_direct)
        else:
            _assign(recipe, name, Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name == "text/token_embedding":
            _assign(recipe, name, Q8)
            continue
        if name == "text/output_head":
            assign_dense(name)
            continue
        if "/moe/" in name and "/moe/experts/" not in name:
            continue
        if not parameter.projection:
            continue
        if "/moe/experts/" in name:
            # The experts are read by the MoE Op's own decoder, which computes in the represented
            # activations: no linear route consumes them, so nothing grants A4 and none of them
            # carries a calibration divisor.
            assign_expert(name)
        else:
            assign_dense(name)
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def gemma4_26b_a4b_base(model, recipe, sources):
    """The 26B's bring-up layout: NVFP4 routed experts, Q8 projections and head.

    Q8 because a quantized linear route resolves its geometry from a fixed registry of hand-tuned
    shape files, neither the FP8 nor the NVFP4 registry has this model's hidden 2816, and the
    row-split int8 route has a generic selector. NVFP4's E2M1 grid cannot hold the int4 group-32
    grid this checkpoint was trained on, so it re-quantizes the experts (plan P5).
    """
    _gemma4_26b_a4b(
        model,
        recipe,
        lambda name: recipe.assign(
            name, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only"
        ),
        lambda name: _assign(recipe, name, Q8),
    )


def gemma4_26b_a4b_qat(model, recipe, sources):
    """The 26B on the grid it was trained on: every projection, the routed experts and the head as
    int4 group-32 rows whose binary16 step is recovered per group (q4_rows_qat), exact up to that
    step's rounding. The QAT checkpoint carries that grid in every one of them (plan P5)."""
    on_grid = lambda name: recipe.assign(name, format=Q4ROWS, method=q4_rows_qat)  # noqa: E731
    _gemma4_26b_a4b(model, recipe, on_grid, on_grid)
    # A layer's gate and up become one parent, so a decode runs them as one launch. The mapping
    # leaves them apart because a BlockScale parent has a plane a row slice cannot address
    # (gemma4.py's `block`); a row-grouped parent has none, so the group is declared here rather
    # than in the block, where it would reach the layouts that cannot read a slice. The sweep is
    # restricted to the text layers: the vision tower keeps its gate and up apart, which is the
    # layout the 31B converts with, and the fusion is a decode-scheduling choice that was never
    # measured on the tower. Without the `text/` test it also catches `vision/layers/N/mlp/gate`.
    for name in model.parameters:
        if not name.startswith("text/") or not name.endswith("/mlp/gate"):
            continue
        pair = (name, name[: -len("gate")] + "up")
        if pair[1] in model.parameters:
            recipe.group(pair)


def nemotron_h_74b_nvfp4(model, recipe, sources):
    """Nemotron 3.5 Bolt 74B A9B from the BF16 checkpoint, in NVIDIA's own per-tensor assignment.

    The routed and shared experts are NVFP4 -- E2M1 codes, E4M3 per-16-block scales, one FP32
    divisor per bank -- quantized from full precision with ModelOpt's weight-MSE scale sweep. That
    is **W4A16**: a full-precision source carries no calibrated activation divisor, and the engine
    refuses an A4 use without one, so the banks report `A16Only` exactly as the released
    `W4A16_NVFP4` Lightning checkpoint does. The Mamba mixer projections, the attention projections
    and the head are the engine's FP8 W8A8 route (row-scaled, which is finer than the recipe's
    per-tensor scale); the latent projections, the router, the embeddings, the norms and the Mamba
    per-head vectors stay BF16, and the four scalars the scan and the router read stay FP32.

    Each layer's `up` and `down` banks become one parent each, so an expert is a fixed row range of
    its bank: the MoE Op reads expert `e`'s rows in place, and a host-offload cache addresses them as
    a few contiguous spans.
    """
    if "block_configs" not in model.config:
        raise ValueError("this official recipe requires Nemotron-H mathematics")
    fp8_roles = ("/attention/", "/mamba/in_projection", "/mamba/out_projection")
    for name, parameter in model.parameters.items():
        if name.startswith("dflash2/"):
            # The speculative draft is stored exactly as its checkpoint holds it -- BF16 after the
            # ModelOpt decode. It is about 3% of a round and half the size of a single expert bank,
            # so a fresh FP8/FP4 sweep here would be a second lossy rounding for no measured gain;
            # the tower's role rules must not reach into it either (a quantized *bias* is not even
            # representable at the declared shape).
            continue
        if "/moe/experts/" in name or "/moe/shared/" in name:
            recipe.assign(
                name, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only"
            )
            continue
        if name == "text/output_head" or any(role in name for role in fp8_roles):
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def nemotron_lightning_30b(model, recipe, sources):
    """Lightning 30B from NVIDIA's already-quantised ModelOpt checkpoint, codes preserved.

    Unlike `nemotron_h_74b_nvfp4` (a BF16 source quantised by the recipe), this checkpoint stores
    the codes: the routed experts as E2M1 planes with E4M3 per-16-block scales and one FP32 bank
    divisor (`weight_scale_2`), and the Mamba mixer projections as per-tensor-scaled E4M3. Both are
    carried through with `import_encoded`, which requires the source's own format to equal the
    target's -- `matrix_source` resolves `modelopt_nvfp4` / `modelopt_fp8` from the stored
    companions, so no explicit source is needed here.

    The mixer projections' `input_scale` is deliberately not carried: the engine's route is A16,
    which is more accurate than the checkpoint's own A8 and needs no activation divisor. The 72
    excluded modules (embeddings, `mixer.conv1d`, `mixer.gate`, `mixer.k_proj`, norms, router, the
    attention projections) stay BF16, as the checkpoint stores them.
    """
    if "block_configs" not in model.config:
        raise ValueError("this official recipe requires Nemotron-H mathematics")
    # The companion's two projections whose widths this engine registers as NVFP4 keep the codes
    # the checkpoint stores: a draft round reads them once per round and they are its largest
    # single term, so decoding them to BF16 (which the rest of the companion gets) would double it.
    companion = sources.get("dflash")
    for name in model.parameters:
        if name == "dflash/feature_projection" or name.endswith(
                ("/mlp/down", "/mlp/gate", "/mlp/up")
        ):
            if not name.startswith("dflash/"):
                continue
            recipe.assign(name, format="nvfp4", method=import_encoded,
                          source=model.source(name, companion),
                          activation_policy="A16Only")
    fp8_roles = ("/attention/", "/mamba/in_projection", "/mamba/out_projection",
                 "/moe/shared/", "text/output_head")
    for name in model.parameters:
        if name.startswith("dflash/") or name.startswith("dflash2/"):
            # The speculative companion is stored exactly as its checkpoint holds it. The role
            # rules below name substrings ("/attention/") that also appear inside the draft, and a
            # draft norm given a matrix format is not representable -- so the companion is skipped
            # wholesale, as the 74B recipe skips its own.
            continue
        if "/moe/experts/" in name:
            # Only the routed banks go NVFP4: they are 18 GiB and the dedicated MoE Op reads them.
            # The shared expert and the head are ordinary linears whose shapes the NVFP4 route
            # registry does not carry (and a generic fallback is deliberately absent), so they stay
            # BF16 -- exact, and the dense route covers the shape.
            recipe.assign(name, format="nvfp4", method=nvfp4_mse,
                          activation_policy="A16Only")
            continue
        if any(role == name or role in name for role in fp8_roles):
            # W8A16 at the 30B's widths: the checkpoint scales its two mamba mixer projections per
            # tensor with an A8 activation model, and re-encoding them (plus the attention, shared
            # expert and head from their BF16 values) onto the engine's row-scaled FP8 route is what
            # makes the dense read half the size. `A16Only` is what the route implements.
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, activation_policy="A16Only")
            continue
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


RECIPES = {
    "nemotron_h_74b_nvfp4": nemotron_h_74b_nvfp4,
    "nemotron_lightning_30b": nemotron_lightning_30b,
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_bf16": qwen3_8_27b_bf16,
    "qwen3_8_27b_exl3": qwen3_8_27b_exl3,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_quasar_nvfp4": qwen3_8_27b_quasar_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_6_35b_a3b_nvfp4": qwen3_6_35b_a3b_nvfp4,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_nvfp4_dense_a8": qwen3_8_flash_next_nvfp4_dense_a8,
    "qwen3_8_flash_next_bf16": qwen3_8_flash_next_bf16,
    "qwen3_8_flash_next_exl3": qwen3_8_flash_next_exl3,
    "gemma4_31b_base": gemma4_31b_base,
    "gemma4_31b_g0": gemma4_31b_g0,
    "gemma4_31b_m1": gemma4_31b_m1,
    "gemma4_26b_a4b_base": gemma4_26b_a4b_base,
    "gemma4_26b_a4b_qat": gemma4_26b_a4b_qat,
}
