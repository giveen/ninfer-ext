"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_absmax,
)
from .sources.compressed_tensors import compressed_matrix_source
from .sources.exl3 import exl3_matrix_source
from .sources.logical import select_rows

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


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
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
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
            format = gate_up
        elif name.endswith(
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


_MOE_EXPERT_SOURCE = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def _moe_encoded_source(store, parameter, name):
    """Map one routed or shared expert matrix onto its per-expert NVFP4 prefix.

    The MoE description reads routed experts out of a single fused `experts.gate_up_proj`
    tensor with an offset, so the generic factory cannot reach an encoded source for them
    and says so. The compressed-tensors checkpoint stores every expert as its own prefix,
    which is a plain matrix, so the mapping is stated here once instead.
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


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """Qwen3.8-Flash-Next (Qwen4Exp) from the ModelOpt NVFP4 checkpoint.

    Routed Text experts keep their NVFP4 codes, block scales and divisors; the MTP bank is
    re-encoded from block FP8 to NVFP4 so every routed expert uses one execution path. The n-gram
    table keeps its FP8 codes under the shared multiplier. Dense projections become Q8 (the output
    head Q6); routers, shared-expert gates, norms and small vectors stay direct.
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
        if name.endswith("/ple/table"):
            recipe.assign(
                name, format=FP8, method=import_encoded, source=parameter.source
            )
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            text = name.startswith("text/")
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded if text else nvfp4_absmax,
                source=parameter.source,
                activation_policy="AllowA4",
            )
            continue
        _assign(recipe, name, Q8)
    # Each layer's gate/up and down banks become one parent so an expert is a fixed row range of
    # its bank, addressed by the expert cache as a few contiguous spans.
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def qwen3_8_flash_next_q4(model, recipe, sources):
    """Qwen3.8-Flash-Next (Qwen4Exp) with the routed experts re-encoded to groupwise.

    The NVFP4 recipe keeps the ModelOpt expert codes and runs the W4A4 path. This one quantizes the
    experts from full precision instead: Q4 gate/up with Q5 down, the `sparse_moe` main profile, and
    Q8 shared experts. The expert banks are the bulk of the file and g64's per-64 FP16 scale is
    cheaper than NVFP4's block-16 FP8 scale plus its per-tensor divisor, so this is smaller at the
    cost of 4-bit activations. Everything else matches the NVFP4 recipe.
    """
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not parameter.projection:
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            _assign(recipe, name, Q4 if name.endswith(("/gate", "/up")) else Q5)
            continue
        _assign(recipe, name, Q8)
    # Each layer's gate/up and down banks become one parent so an expert is a fixed row range of
    # its bank, addressed by the expert cache as a few contiguous spans.
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_bf16": qwen3_8_27b_bf16,
    "qwen3_8_27b_exl3": qwen3_8_27b_exl3,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_6_35b_a3b_nvfp4": qwen3_6_35b_a3b_nvfp4,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_q4": qwen3_8_flash_next_q4,
}
