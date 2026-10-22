from __future__ import annotations

import torch

from dataclasses import replace

from tools.convert.methods import (
    cast_direct,
    fp8_row_maxabs,
    import_encoded,
    nvfp4_absmax,
    nvfp4_mse,
    q4_rows_qat,
)
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    _qwen4_dense_a8,
    _qwen4_dense_a8_parameters,
    fp8_row_maxabs,
    qwen3_8_flash_next_nvfp4_dense_a8,
    RECIPES,
    gemma4_26b_a4b_base,
    gemma4_26b_a4b_qat,
    qwen3_6_27b,
    qwen3_8_27b,
    qwen3_8_27b_bf16,
    qwen3_8_27b_exl3,
    qwen3_8_27b_q6,
    qwen3_8_flash_next_exl3,
    qwen3_8_flash_next_nvfp4,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"

LAYER = "text/layers/0"


def _dense_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/key",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name in ("text/token_embedding", "text/output_head") else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    return model


def _formats(recipe_function) -> dict[str, set[str]]:
    model = _dense_model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def _single(formats: dict[str, set[str]], name: str) -> str:
    values = formats[name]
    assert len(values) == 1, f"{name} is split across formats {values}"
    return next(iter(values))


def test_q6_recipe_is_registered() -> None:
    assert RECIPES["qwen3_8_27b_q6"] is qwen3_8_27b_q6


def _flash_next_model(*, encoded: bool = False) -> Model:
    model = Model({"text": {"config": {"model_type": "qwen4_exp_text"}}})

    def add(name: str, shape: tuple[int, ...], inputs: tuple[str, ...]) -> None:
        source = array_source(torch.ones(shape, dtype=torch.bfloat16), name)
        if encoded:
            source = replace(source, read_encoded=lambda begin, rows: None)
        model.add(Parameter(name, shape, source, inputs=inputs))

    add("text/token_embedding", (8, 4), ())
    add("text/output_head", (8, 4), ())
    add("text/layers/0/moe/router", (8, 4), ("x",))
    add("text/layers/0/moe/experts/gate", (4, 4), ("x",))
    add("text/layers/0/moe/experts/up", (4, 4), ("x",))
    add("text/layers/0/moe/experts/down", (4, 4), ("p",))
    add("text/layers/0/moe/shared/gate", (4, 4), ("x",))
    # A dense group whose packed shape has a compiled FP8 two-term route
    # (src/ops/linear/fp8/shapes/n1280_k2560.cu): four 320-row projections pack into 1280 x 2560.
    # Small enough to materialise in a unit test.
    # The real grouping stacks unequal members (query/key 2048 rows each, value/z 6144 each), so the
    # fixture does too: 160 + 160 + 480 + 480 = 1280 rows.
    for role, rows in (("query", 160), ("key", 160), ("value", 480), ("z", 480)):
        add(f"text/layers/0/gdn/{role}", (rows, 2560), ("x",))
    # An unpacked projection whose own shape is compiled: 1280 x 2560, stored as its own object.
    add("text/layers/0/gdn/output", (1280, 2560), ("x",))
    model.packing_groups.append(tuple(f"text/layers/0/gdn/{role}"
                                      for role in ("query", "key", "value", "z")))
    # The n-gram table is a value source: it has no inputs, so it is not a "projection" and a
    # recipe that only visits projections would leave all 51 B parameters at BF16.
    add("text/layers/1/ple/table", (4, 4), ())
    model.packing_groups.append(
        ("text/layers/0/moe/experts/gate", "text/layers/0/moe/experts/up")
    )
    return model


def _method(model: Model, function, name: str):
    recipe = Recipe(model)
    function(model, recipe, {})
    selections = recipe.selections[name]
    assert len(selections) == 1, f"{name} is split across {len(selections)} selections"
    return selections[0]


def test_flash_next_recipes_are_registered() -> None:
    # Exercising it needs a full Qwen4Exp sparse-MoE model, so this pins the wiring.
    assert RECIPES["qwen3_8_flash_next_nvfp4"] is qwen3_8_flash_next_nvfp4
    assert RECIPES["qwen3_8_flash_next_nvfp4_dense_a8"] is qwen3_8_flash_next_nvfp4_dense_a8
    assert RECIPES["qwen3_8_flash_next_exl3"] is qwen3_8_flash_next_exl3


def test_flash_next_dense_a8_group_shapes_are_the_compiled_ones() -> None:
    # The set is matched against a packing group's summed rows, which is what the converter stores.
    model = _flash_next_model()
    groups = _qwen4_dense_a8_parameters(model)
    assert groups == {f"text/layers/0/gdn/{role}" for role in ("query", "key", "value", "z")} | {
        "text/layers/0/gdn/output"
    }


def test_flash_next_dense_a8_shape_set_matches_the_compiled_fp8_shapes() -> None:
    # The set names the projections src/ops/linear/fp8/shapes/ has an FP8 (and two-term) route for.
    # A shape outside it must keep Q8, because no FP8 route resolves for that geometry.
    for shape in ((16384, 2560), (13312, 2560), (2560, 6144), (1280, 2560)):
        assert _qwen4_dense_a8(shape), f"{shape} should take the FP8 two-term route"
    for shape in ((324, 10240), (10240, 320), (2560, 640), (640, 2560), (96, 2560), (4, 4)):
        assert not _qwen4_dense_a8(shape), f"{shape} has no FP8 route"


def test_flash_next_dense_a8_recipe_grants_the_pair_route_only_where_fp8_exists() -> None:
    model = _flash_next_model()
    supported = "text/layers/0/gdn/query"
    unsupported = "text/layers/0/moe/shared/gate"  # (4, 4): no FP8 route resolves

    recipe = Recipe(model)
    qwen3_8_flash_next_nvfp4_dense_a8(model, recipe, {})
    for role in ("query", "key", "value", "z"):
        name = f"text/layers/0/gdn/{role}"
        assert recipe.selections[name][0].format == "fp8_e4m3fn_row_bf16"
        assert recipe.selections[name][0].method is fp8_row_maxabs
        assert recipe.policies[(name, "x")] == "AllowA8Pair"
    assert recipe.selections[supported][0].method is fp8_row_maxabs
    assert recipe.policies[(supported, "x")] == "AllowA8Pair"
    # A shape without a compiled FP8 route keeps Q8 and the default (A16) activation policy.
    assert recipe.selections[unsupported][0].format == "q8_g32_fp16"
    assert recipe.policies[(unsupported, "x")] == "A16Only"

    shipped = Recipe(model)
    qwen3_8_flash_next_nvfp4(model, shipped, {})
    assert shipped.selections[supported][0].format == "q8_g32_fp16"
    assert shipped.policies[(supported, "x")] == "A16Only"


def test_flash_next_nvfp4_quantizes_a_full_precision_source() -> None:
    # A full-precision checkpoint has no NVFP4 codes to import, so the experts are quantized and
    # run the BF16-activation route (no calibrated A4 divisor), and the table is row-quantized.
    model = _flash_next_model()
    experts = _method(model, qwen3_8_flash_next_nvfp4, "text/layers/0/moe/experts/gate")
    assert experts.format == "nvfp4"
    assert experts.method is nvfp4_absmax
    table = _method(model, qwen3_8_flash_next_nvfp4, "text/layers/1/ple/table")
    assert table.format == "fp8_e4m3fn_row_bf16"
    assert table.method is fp8_row_maxabs


def test_flash_next_nvfp4_imports_an_encoded_source() -> None:
    # ModelOpt ships NVFP4 codes and calibrated divisors, which are kept so the bank runs W4A4.
    model = _flash_next_model(encoded=True)
    experts = _method(model, qwen3_8_flash_next_nvfp4, "text/layers/0/moe/experts/gate")
    assert experts.format == "nvfp4"
    assert experts.method is import_encoded
    table = _method(model, qwen3_8_flash_next_nvfp4, "text/layers/1/ple/table")
    assert table.method is import_encoded


def test_bf16_recipe_is_registered_and_matches_the_groupwise_layout() -> None:
    assert RECIPES["qwen3_8_27b_bf16"] is qwen3_8_27b_bf16
    formats = _formats(qwen3_8_27b_bf16)
    assert set(formats) == set(_formats(qwen3_8_27b))
    for name, values in formats.items():
        assert values == {"bf16"}, f"{name} is not stored at full precision: {values}"


def test_exl3_recipe_reads_the_native_quantizer_store(tmp_path) -> None:
    from safetensors.torch import save_file

    from tools.convert.official_recipes import qwen3_8_27b_exl3
    from tools.convert.sources.safetensors import SafetensorsSource

    n = k = 128
    # The MLP gate/up pair is one shared-input parent, so the store holds one [256, 128] matrix
    # keyed by the first member (gate) and the recipe binds both members to its row ranges.
    parent_rows = 256
    half_bits = 3
    generator = torch.Generator().manual_seed(5)
    trellis = torch.randint(
        0,
        256,
        (parent_rows // 16, k // 16, 16 * half_bits),
        dtype=torch.uint8,
        generator=generator,
    )
    su = (torch.rand(k, generator=generator) + 0.5).float()
    sv = (torch.rand(parent_rows, generator=generator) + 0.5).float()
    save_file(
        {
            "text/layers/0/mlp/gate.trellis": trellis,
            "text/layers/0/mlp/gate.su": su,
            "text/layers/0/mlp/gate.sv": sv,
        },
        str(tmp_path / "exl3.safetensors"),
    )

    model = Model({"text": {"config": {}}})
    for name in ("text/layers/0/mlp/gate", "text/layers/0/mlp/up", "text/layers/0/mlp/down"):
        model.add(
            Parameter(
                name,
                (128, 128),
                array_source(torch.ones((128, 128), dtype=torch.bfloat16), name),
                inputs=("input",),
            )
        )
    model.packing_groups.append(("text/layers/0/mlp/gate", "text/layers/0/mlp/up"))
    recipe = Recipe(model)
    with SafetensorsSource(tmp_path / "exl3.safetensors") as store:
        qwen3_8_27b_exl3(model, recipe, {"quantized": store})
        formats = {
            name: {selection.format for selection in selections}
            for name, selections in recipe.selections.items()
        }
        assert formats["text/layers/0/mlp/gate"] == {"exl3_mul1"}
        assert formats["text/layers/0/mlp/up"] == {"exl3_mul1"}
        assert formats["text/layers/0/mlp/down"] == {"bf16"}
        # The stored members are bound as row ranges of one shared parent, not separately stored.
        assert "text/layers/0/mlp/gate" not in recipe.separate_parameters
        assert "text/layers/0/mlp/up" not in recipe.separate_parameters
        gate_label = recipe.selections["text/layers/0/mlp/gate"][0].source.label
        up_label = recipe.selections["text/layers/0/mlp/up"][0].source.label
        assert "rows(" in gate_label and "rows(" in up_label
        assert gate_label == up_label


def test_registered_recipe_gives_the_mlp_pair_q4() -> None:
    formats = _formats(qwen3_8_27b)
    assert _single(formats, f"{LAYER}/mlp/gate") == Q4
    assert _single(formats, f"{LAYER}/mlp/up") == Q4
    assert _single(formats, f"{LAYER}/mlp/down") == Q5
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_groupwise_recipes_allow_a8_on_the_mlp_pair_only() -> None:
    """The Q4 gate/up pair has a W4A8 prefill route; every other use stays A16Only."""
    for recipe_function in (qwen3_6_27b, qwen3_8_27b):
        model = _dense_model()
        recipe = Recipe(model)
        recipe_function(model, recipe, {})
        assert recipe.policies[(f"{LAYER}/mlp/gate", "input")] == "AllowA8"
        assert recipe.policies[(f"{LAYER}/mlp/up", "input")] == "AllowA8"
        for name in (
            f"{LAYER}/mlp/down",
            f"{LAYER}/attention/query",
            f"{LAYER}/attention/output",
        ):
            assert recipe.policies[(name, "input")] == "A16Only"


def test_q6_recipe_moves_only_the_mlp_pair() -> None:
    registered = _formats(qwen3_8_27b)
    q6 = _formats(qwen3_8_27b_q6)

    assert set(registered) == set(q6)
    moved = {name for name in q6 if q6[name] != registered[name]}
    assert moved == {f"{LAYER}/mlp/gate", f"{LAYER}/mlp/up"}

    for name in moved:
        assert registered[name] == {Q4}
        assert q6[name] == {Q6}


def test_q6_recipe_keeps_the_vocabulary_at_q8() -> None:
    """Q8 is 8.5 bits per weight, so the vocabulary endpoints already outrank Q6."""
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_leaves_the_other_projections_alone() -> None:
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, f"{LAYER}/attention/query") == Q4
    assert _single(formats, f"{LAYER}/attention/key") == Q4
    assert _single(formats, f"{LAYER}/gdn/query") == Q4
    assert _single(formats, f"{LAYER}/attention/output") == Q5
    assert _single(formats, f"{LAYER}/gdn/output") == Q5
    assert _single(formats, f"{LAYER}/mlp/down") == Q5


def test_35b_nvfp4_recipe_reads_a_modelopt_expert_prefix(tmp_path) -> None:
    """`_moe_encoded_source` must import either the compressed-tensors or the ModelOpt layout."""
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from tools.convert.official_recipes import _moe_encoded_source
    from tools.convert.sources.safetensors import SafetensorsSource

    prefix = "model.language_model.layers.3.mlp.experts.7.gate_proj"
    save_file(
        {
            prefix + ".weight": torch.tensor(
                [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
            ),
            prefix + ".weight_scale": torch.tensor([[0x38]], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            prefix + ".weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
            prefix + ".input_scale": torch.tensor(1.5, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = _moe_encoded_source(
            store, SimpleNamespace(shape=(1, 16)), "text/layers/3/moe/experts/7/gate"
        )
        assert torch.equal(
            source.values().reshape(1, 16),
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                  0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0]]
            ),
        )


def _gemma4_26b_model(w: int = 4, *, drafter: bool = False, vision: bool = False) -> Model:
    """One 26B-A4B layer: the dense MLP, the router with its two scales, and two experts of a bank.

    `w` is the hidden width every projection contracts over. The row-grouped 4-bit format needs K to
    be a multiple of 32, so a caller that wants to prepare the QAT recipe asks for 32.

    `drafter` adds the assistant's own projections -- the input and output projections, the tied head
    and one drafter layer -- under the names the mapping declares them.

    `vision` adds one image-tower layer with the tower's patch and output projections. It is a flag
    rather than part of the default model: the QAT layout's gate/up sweep matches any layer name
    ending in `/mlp/gate`, so a tower MLP would join that layout's explicit groups, which its own
    test pins.
    """
    components = {"text": {"config": {"model_type": "gemma4_text"}}}
    if drafter:
        components["mtp"] = {"config": {"model_type": "gemma4_assistant"}}
    model = Model(components)
    layer = "text/layers/0/"

    def add(name: str, shape: tuple[int, ...], inputs: tuple[str, ...]) -> None:
        source = array_source(torch.ones(shape, dtype=torch.bfloat16), name)
        model.add(Parameter(name, shape, source, inputs=inputs))

    add("text/token_embedding", (8, w), ())
    add("text/output_head", (8, w), ("hidden",))
    for role in ("input_norm", "post_attention_norm", "pre_feedforward_norm",
                 "post_feedforward_norm", "post_feedforward_norm_1", "pre_feedforward_norm_2",
                 "post_feedforward_norm_2"):
        add(layer + role, (w,), ())
    add(layer + "layer_scalar", (1,), ())
    for role in ("query", "key", "value"):
        add(layer + "attention/" + role, (w, w), ("normed",))
    add(layer + "attention/output", (w, w), ("attended",))
    for role in ("gate", "up"):
        add(layer + "mlp/" + role, (w, w), ("mlp_input",))
    add(layer + "mlp/down", (w, w), ("mlp_product",))
    add(layer + "moe/router", (2, w), ("residual",))
    add(layer + "moe/router_scale", (w,), ())
    add(layer + "moe/per_expert_scale", (2,), ())
    gate_up, downs = [], []
    for expert in range(2):
        prefix = layer + f"moe/experts/{expert}/"
        for role in ("gate", "up"):
            add(prefix + role, (w, w), ("expert_input",))
            gate_up.append(prefix + role)
        add(prefix + "down", (w, w), (prefix + "product",))
        downs.append(prefix + "down")
    if drafter:
        # The assistant's own projections, at the shapes its own config gives: an input projection
        # over the target's hidden appended twice, an output projection back to the target's hidden,
        # the tied head, and one layer whose attention width is the target's head count.
        add("mtp/pre_projection", (4, 2 * w), ())
        add("mtp/post_projection", (w, 4), ("mtp/final_hidden",))
        add("mtp/output_head", (8, 4), ("mtp/final_hidden",))
        draft = "mtp/layers/0/"
        add(draft + "attention/query", (4, 4), (draft + "mixer_input",))
        add(draft + "attention/output", (4, 4), (draft + "attention/gated_output",))
        for role in ("gate", "up"):
            add(draft + "mlp/" + role, (4, 4), (draft + "mlp_input",))
        add(draft + "mlp/down", (4, 8), (draft + "mlp/product",))
    # The image tower's own projections, under the names the mapping gives them: the patch
    # embedding reads a patch's pixels, the output projection returns the soft tokens to the text
    # width, and one layer carries the attention and MLP linears the layout quantizes.
    if vision:
        add("vision/patch_embedding", (w, 3 * 16 * 16), ("vision/patch_input",))
        add("vision/embedding_projection", (w, w), ("vision/soft_tokens",))
        tower = "vision/layers/0/"
        for role in ("query", "key", "value"):
            add(tower + "attention/" + role, (w, w), (tower + "attention_input",))
        add(tower + "attention/output", (w, w), (tower + "attended",))
        for role in ("gate", "up"):
            add(tower + "mlp/" + role, (w, w), (tower + "mlp_input",))
        add(tower + "mlp/down", (w, w), (tower + "mlp/product",))
    # One bank per projection, declared exactly as the gemma4 mapping declares them, and the
    # attention input projections as one group, which is what the mapping's `group(query, key,
    # value)` call produces.
    model.packing_groups.append(tuple(layer + "attention/" + role
                                      for role in ("query", "key", "value")))
    model.packing_groups.append(tuple(gate_up))
    model.packing_groups.append(tuple(downs))
    return model


def test_gemma4_26b_recipe_is_registered() -> None:
    assert RECIPES["gemma4_26b_a4b_base"] is gemma4_26b_a4b_base
    assert RECIPES["gemma4_26b_a4b_qat"] is gemma4_26b_a4b_qat


def test_gemma4_26b_recipe_quantizes_both_branches_and_keeps_the_router_exact() -> None:
    model = _gemma4_26b_model()
    recipe = Recipe(model)
    gemma4_26b_a4b_base(model, recipe, {})
    layer = "text/layers/0"

    def selection(name: str):
        selections = recipe.selections.get(name, [])
        assert len(selections) == 1, f"{name} has {len(selections)} selections"
        return selections[0]

    # The routed experts are the only quantized part no linear route sees, so they are the only one
    # that can spend four bits.
    for name in (f"{layer}/moe/experts/0/gate", f"{layer}/moe/experts/1/up",
                 f"{layer}/moe/experts/1/down"):
        assert selection(name).format == "nvfp4"
        assert selection(name).method is nvfp4_mse
    # Everything a linear route does see takes the generic int8 route: a quantized linear route
    # resolves its geometry from a fixed registry of hand-tuned shape files, and the row-split int8
    # registry is the only one with a selector for a shape that is not one of them.
    for name in (f"{layer}/mlp/gate", f"{layer}/mlp/up", f"{layer}/mlp/down",
                 f"{layer}/attention/query", f"{layer}/attention/output", "text/token_embedding",
                 "text/output_head"):
        assert selection(name).format == "q8_g32_fp16", f"{name} must take the generic int8 route"

    # The router decides the top-k and the per-expert scale multiplies what an expert produces, so
    # both, and the three norms the block adds, stay at the source precision.
    for name in (f"{layer}/moe/router", f"{layer}/moe/router_scale",
                 f"{layer}/moe/per_expert_scale", f"{layer}/post_feedforward_norm_1",
                 f"{layer}/pre_feedforward_norm_2", f"{layer}/post_feedforward_norm_2"):
        assert selection(name).format == "bf16", f"{name} must stay at the source precision"

    # Every expert bank becomes one parent, so an expert is a fixed row range of its bank.
    grouped = {name for names, _ in recipe.explicit_groups for name in names}
    assert grouped == {name for name in model.parameters if "/moe/experts/" in name}


def test_gemma4_26b_qat_recipe_puts_every_projection_on_its_trained_grid() -> None:
    model = _gemma4_26b_model(32)
    recipe = Recipe(model)
    gemma4_26b_a4b_qat(model, recipe, {})
    layer = "text/layers/0"
    for name in (f"{layer}/moe/experts/0/gate", f"{layer}/moe/experts/1/down"):
        (selection,) = recipe.selections[name]
        assert selection.format == "q4_g32_fp16_rows"
        assert selection.method is q4_rows_qat
    for name in (f"{layer}/mlp/gate", f"{layer}/attention/query", f"{layer}/attention/output",
                 "text/output_head"):
        (selection,) = recipe.selections[name]
        assert selection.format == "q4_g32_fp16_rows", name
        assert selection.method is q4_rows_qat
    # The token table is a gather the embedding Op reads as row-split int8.
    (table,) = recipe.selections["text/token_embedding"]
    assert table.format == "q8_g32_fp16"
    (router,) = recipe.selections[f"{layer}/moe/router"]
    assert router.format == "bf16"
    grouped = {name for names, _ in recipe.explicit_groups for name in names}
    assert grouped == ({name for name in model.parameters if "/moe/experts/" in name} |
                       {f"{layer}/mlp/gate", f"{layer}/mlp/up"})
    # The attention input projections must land in one stored parent: the model runs the union in a
    # single launch whenever their planes are contiguous (gemma4 forward's
    # `row_slices_are_contiguous`), and only a grouped parent makes them contiguous. A quantizing
    # method missing from the recipe's `standard` tuple silently emits three objects instead, which
    # is how the QAT artifact lost the fusion until `q4_rows_qat` was listed. A layer's gate and up
    # are grouped explicitly for the same launch.
    parents = [job.parameters for job in recipe.prepare(device="cpu").weights]
    assert (f"{layer}/attention/query", f"{layer}/attention/key",
            f"{layer}/attention/value") in parents
    assert (f"{layer}/mlp/gate", f"{layer}/mlp/up") in parents


def test_gemma4_26b_recipes_put_the_drafter_on_the_31b_grid() -> None:
    """The assistant drafter is Q6 throughout, head included, in both 26B layouts.

    The choice the 31B's layout makes, and for the same measured reason: against BF16, adaptive MTP
    ran 4.3% faster where Q8 ran 3.9%, worst -1.9%, and the drafter shrinks to about 0.4 GB. Q8 is
    what `_optional` gives an `mtp/` projection, so the override is what this checks. It is also why
    the artifact needs the drafter's shapes in the Q6 registry rather than the Q8 one: the 26B's
    drafter has 16 heads where the 31B's has 32, so its attention widths are new geometry.
    """
    model = _gemma4_26b_model(32, drafter=True)
    drafter = [name for name, parameter in model.parameters.items()
               if name.startswith("mtp/") and parameter.projection]
    assert drafter, "the fixture must carry the drafter's projections"
    for build in (gemma4_26b_a4b_base, gemma4_26b_a4b_qat):
        recipe = Recipe(model)
        build(model, recipe, {})
        for name in drafter:
            (selection,) = recipe.selections[name]
            assert selection.format == "q6_g64_fp16", f"{build.__name__}: {name}"


def test_gemma4_26b_recipes_put_the_vision_tower_on_the_int8_route() -> None:
    """The tower's linears are Q8 and its patch and output projections BF16, in both 26B layouts.

    The 31B's tower at the same geometry, with its linear route one step finer: the tower amplifies
    weight error -- against its FP32 output, Q4/Q5 (what `_optional` would leave here) score a 0.956
    mean token cosine on a photo, Q6 0.989 and Q8 0.999 -- and the tower is only about 0.42 B
    parameters. Q8 also needs no per-geometry registration: its row-split int8 route resolves any
    shape through its generic selector, where Q6 needs a registry entry for each one.
    """
    model = _gemma4_26b_model(32, vision=True)
    vision = [name for name, parameter in model.parameters.items()
              if name.startswith("vision/") and parameter.projection]
    assert vision, "the fixture must carry the tower's projections"
    cast = {"vision/patch_embedding", "vision/embedding_projection"}
    assert cast <= set(vision), "the fixture must carry both unquantized projections"
    for build in (gemma4_26b_a4b_base, gemma4_26b_a4b_qat):
        recipe = Recipe(model)
        build(model, recipe, {})
        for name in vision:
            (selection,) = recipe.selections[name]
            if name in cast:
                assert selection.format == "bf16", f"{build.__name__}: {name}"
                assert selection.method is cast_direct, f"{build.__name__}: {name}"
            else:
                assert selection.format == "q8_g32_fp16", f"{build.__name__}: {name}"
        # The tower's gate and up stay two parents: the 31B converts the tower that way, and the
        # one-launch fusion is a decode-scheduling choice measured on the text layers only. The QAT
        # layout's gate/up sweep is text-only for that reason; a wider sweep regroups them quietly.
        grouped = {name for names, _ in recipe.explicit_groups for name in names}
        assert not {name for name in grouped if name.startswith("vision/")}, \
            f"{build.__name__}: the tower's linears must not be grouped"
