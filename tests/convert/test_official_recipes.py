from __future__ import annotations

import torch

from dataclasses import replace

from tools.convert.methods import fp8_row_maxabs, import_encoded, nvfp4_absmax
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    RECIPES,
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
    assert RECIPES["qwen3_8_flash_next_exl3"] is qwen3_8_flash_next_exl3


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
