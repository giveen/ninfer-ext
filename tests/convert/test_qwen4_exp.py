"""Qwen4Exp (Qwen3.8-Flash-Next) mapping and the ModelOpt NVFP4 official recipe."""

from __future__ import annotations

import json
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
from tools.artifact.layouts import block_scale_geometry
from tools.artifact.reader import Artifact
from tools.convert.official_recipes import qwen3_8_flash_next_nvfp4
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import divisor_for, quantize_rows
from tools.convert.qwen4_exp import build_model, ngram_table_rows, text_config
from tools.convert.recipe import Recipe
from tools.convert.sources.modelopt import decode_nvfp4
from tools.convert.sources.safetensors import SafetensorsSource

H, IR, E, VOCAB = 128, 128, 2, 16


def _config():
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": H,
            "vocab_size": VOCAB,
            "num_hidden_layers": 2,
            "max_position_embeddings": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 16,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "hidden_act": "silu",
            "output_gate_type": "sigmoid",
            "num_experts": E,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": IR,
            "shared_expert_intermediate_size": 32,
            "hc_count": 2,
            "hc_lowrank": 8,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 16,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [1],
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 11,
            "make_ngram_vocab_size_divisible_by": 8,
            "eos_token_id": 3,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
            "rms_norm_eps": 1e-6,
        },
    }


def _nvfp4_expert(tensors, prefix, shape, generator):
    values = torch.randn(shape, generator=generator)
    divisor = divisor_for(float(values.abs().max()))
    rows = quantize_rows(values, divisor)
    tensors[prefix + ".weight"] = rows.codes
    tensors[prefix + ".weight_scale"] = rows.scales.view(torch.float8_e4m3fn)
    tensors[prefix + ".weight_scale_2"] = torch.tensor(1.0 / struct.unpack("<f", divisor)[0])
    tensors[prefix + ".input_scale"] = torch.tensor(0.01)
    return rows


def _block_fp8(tensors, prefix, shape, generator):
    values = torch.randn(shape, generator=generator)
    tensors[prefix + ".weight"] = values.to(torch.float8_e4m3fn)
    tiles = ((shape[0] + 127) // 128, (shape[1] + 127) // 128)
    tensors[prefix + ".weight_scale_inv"] = torch.full(tiles, 0.5, dtype=torch.bfloat16)
    return values.to(torch.float8_e4m3fn).float() * 0.5


def _checkpoint(tmp_path):
    config = _config()
    text = text_config(config, mtp=True)
    g = torch.Generator().manual_seed(0)
    tensors = {}
    shapes = {}

    def dense(name, shape):
        tensors[name] = torch.randn(shape, generator=g).to(torch.bfloat16)
        shapes[name] = shape

    wide, rank = 2 * H, 8

    def mixer(prefix, inject=True):
        dense(prefix + "hc_norm.weight", (wide,))
        dense(prefix + "input_mix_weight_down.weight", (rank, wide))
        dense(prefix + "input_mix_weight_up.weight", (wide, rank))
        if inject:
            dense(prefix + "block_inject_weight.weight", (2, wide))

    def moe(prefix, expert):
        dense(prefix + "mlp.gate.weight", (E, H))
        dense(prefix + "mlp.shared_expert_gate.weight", (1, H))
        for role, shape in (("gate", (32, H)), ("up", (32, H)), ("down", (H, 32))):
            dense(prefix + f"mlp.shared_expert.{role}_proj.weight", shape)
        for e in range(E):
            for role, shape in (("gate", (IR, H)), ("up", (IR, H)), ("down", (H, IR))):
                expert(tensors, f"{prefix}mlp.experts.{e}.{role}_proj", shape, g)

    def attention(prefix):
        dense(prefix + "self_attn.q_proj.weight", (128, H))
        dense(prefix + "self_attn.k_proj.weight", (32, H))
        dense(prefix + "self_attn.v_proj.weight", (32, H))
        dense(prefix + "self_attn.o_proj.weight", (H, 64))
        dense(prefix + "self_attn.q_norm.weight", (32,))
        dense(prefix + "self_attn.k_norm.weight", (32,))
        dense(prefix + "self_attn.indexer.index_qk_proj.weight", (48, H))
        dense(prefix + "self_attn.indexer.q_layernorm.weight", (16,))
        dense(prefix + "self_attn.indexer.k_layernorm.weight", (16,))

    lm = "model.language_model."
    dense(lm + "embed_tokens.weight", (VOCAB, H))
    dense("lm_head.weight", (VOCAB, H))
    mixer(lm + "hyper_connection_mixer.", inject=False)
    p = lm + "layers.0."
    mixer(p + "attn_hyper_connection.")
    mixer(p + "mlp_hyper_connection.")
    gdn = p + "linear_attn."
    dense(gdn + "in_proj_qkv.weight", (64, H))
    dense(gdn + "in_proj_z.weight", (32, H))
    dense(gdn + "in_proj_a.weight", (2, H))
    dense(gdn + "in_proj_b.weight", (2, H))
    dense(gdn + "conv1d.weight", (64, 1, 4))
    dense(gdn + "norm.weight", (16,))
    dense(gdn + "out_proj.weight", (H, 32))
    tensors[gdn + "A_log"] = torch.randn(2, generator=g)
    tensors[gdn + "dt_bias"] = torch.randn(2, generator=g)
    moe(p, _nvfp4_expert)
    rows = ngram_table_rows(text)
    table = torch.randn(rows, 8, generator=g).to(torch.float8_e4m3fn)
    ple = p + "ple."
    half = rows // 2
    tensors[ple + "ple_embedding.ngram_embedding.shard_0.weight"] = table[:half]
    tensors[ple + "ple_embedding.ngram_embedding.shard_1.weight"] = table[half:]
    tensors[ple + "ple_embedding.ngram_embedding.weight_scale"] = torch.tensor(
        [0.25], dtype=torch.bfloat16
    )
    tensors[ple + "ple_embedding.layer_multipliers"] = torch.tensor([1, 3, 5])
    dense(ple + "key_proj.weight", (wide, 32))
    dense(ple + "value_proj.weight", (H, 32))
    for norm in ("norm_key", "norm_query", "norm_conv"):
        dense(ple + norm + ".weight", (wide,))
    dense(ple + "conv1d.weight", (wide, 1, 4))
    p = lm + "layers.1."
    mixer(p + "attn_hyper_connection.")
    mixer(p + "mlp_hyper_connection.")
    attention(p)
    moe(p, _nvfp4_expert)
    mtp_expected = {}

    def mtp_expert(tensors, prefix, shape, generator):
        mtp_expected[prefix] = _block_fp8(tensors, prefix, shape, generator)

    dense("mtp.pre_fc_norm_embedding.weight", (H,))
    dense("mtp.pre_fc_norm_hidden.weight", (wide,))
    dense("mtp.fc_embedding.weight", (H, H))
    dense("mtp.fc_hidden.weight", (H, H))
    mixer("mtp.hyper_connection_mixer.", inject=False)
    mixer("mtp.layers.0.attn_hyper_connection.")
    mixer("mtp.layers.0.mlp_hyper_connection.")
    attention("mtp.layers.0.")
    moe("mtp.layers.0.", mtp_expert)

    path = tmp_path / "source"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {str(i): i for i in range(VOCAB)}}})
    )
    for role in ("tokenizer_config.json", "generation_config.json"):
        (path / role).write_text("{}")
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return path, tensors, table, mtp_expected


def test_config_keeps_qwen4_fields_and_checks_fixed_mathematics():
    config = text_config(_config(), mtp=True)
    assert config["architectures"] == ["Qwen4ExpForCausalLM"]
    assert config["model_type"] == "qwen4_exp_text"
    assert config["hc_count"] == 2 and config["ple_layer_ids"] == [1]
    assert config["output_gate_type"] == "sigmoid" and config["eos_token_id"] == 3
    bad = _config()
    bad["text_config"]["ple_layer_ids"] = [2]  # one-indexed: block 1 is attention
    with pytest.raises(ValueError, match="GDN block"):
        text_config(bad, mtp=False)
    bad = _config()
    bad["text_config"]["norm_topk_prob"] = False
    with pytest.raises(ValueError, match="norm_topk_prob"):
        text_config(bad, mtp=False)


def test_ngram_table_rows_match_official_shards():
    official = dict(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
    )
    assert ngram_table_rows(official) == 128 * 2_500_012


def test_nvfp4_quantizer_is_exact_on_representable_values():
    divisor = struct.pack("<f", 2.0)
    codes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4).reshape(2, 16)
    values = torch.cat([codes, -codes], dim=1) / 2.0  # block scale 1 after the divisor
    rows = quantize_rows(values, divisor)
    torch.testing.assert_close(decode_nvfp4(rows.codes, rows.scales, divisor), values)


def test_nvfp4_quantizer_error_is_bounded_by_half_a_code_step():
    values = torch.randn(4, 64, generator=torch.Generator().manual_seed(1))
    divisor = divisor_for(float(values.abs().max()))
    rows = quantize_rows(values, divisor)
    decoded = decode_nvfp4(rows.codes, rows.scales, divisor)
    block_max = values.reshape(4, 4, 16).abs().amax(-1).repeat_interleave(16, dim=1)
    # E4M3 rounding of the block scale plus the widest E2M1 gap (2 at magnitude 4..6).
    assert bool(((decoded - values).abs() <= block_max * (1 / 6) * 1.07 + 1e-6).all())


def test_official_recipe_imports_experts_and_table_exactly(tmp_path):
    path, tensors, table, mtp_expected = _checkpoint(tmp_path)
    with SafetensorsSource(path) as source:
        model = build_model(source, components=("text", "mtp"))
        recipe = Recipe(model)
        qwen3_8_flash_next_nvfp4(model, recipe, {"base": source})
        out = tmp_path / "model.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        directory = artifact.directory
        text = directory.components["text"]["config"]
        assert text["model_type"] == "qwen4_exp_text" and text["hc_count"] == 2
        assert directory.components["mtp"]["config"]["architectures"] == ["Qwen4ExpMTP"]

        def parts(name):
            binding = directory.bindings[name]
            return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]

        # Expert 1's up matrix is the fourth IR-row range of its layer's gate/up bank.
        (part,) = parts("text/layers/1/moe/experts/1/up")
        bank = artifact.object(part["object"])
        assert bank.format == "nvfp4" and bank.shape == (2 * E * IR, H)
        assert bank.divisors == 2 * E
        assert part["range"] == [3 * IR * H, 4 * IR * H]
        raw = torch.frombuffer(bytearray(artifact.read_object(bank.id)), dtype=torch.uint8)
        geometry = block_scale_geometry("nvfp4", bank.shape, bank.divisors)
        codes = raw[: geometry.code_plane_bytes].reshape(bank.shape[0], H // 2)
        prefix = "model.language_model.layers.1.mlp.experts.1.up_proj"
        assert torch.equal(codes[3 * IR : 4 * IR], tensors[prefix + ".weight"])
        stored = raw[
            geometry.scale_plane_offset : geometry.scale_plane_offset
            + geometry.scale_plane_bytes
        ]
        scales = unswizzle_nvfp4_scales(stored, bank.shape)
        assert torch.equal(
            scales[3 * IR : 4 * IR], tensors[prefix + ".weight_scale"].view(torch.uint8)
        )
        divisor_words = raw[geometry.weight_divisor_offset :].numpy().tobytes()
        want = struct.pack("<f", 1.0 / float(tensors[prefix + ".weight_scale_2"]))
        assert divisor_words[12:16] == want

        # The FP8 n-gram table keeps its codes and repeats the shared multiplier per row.
        (part,) = parts("text/layers/0/ple/table")
        obj = artifact.object(part["object"])
        assert obj.format == "fp8_e4m3fn_row_bf16" and obj.shape == tuple(table.shape)
        raw = artifact.read_object(obj.id)
        rows, width = table.shape
        assert raw[: rows * width] == table.view(torch.uint8).numpy().tobytes()

        # MTP experts are re-encoded as one NVFP4 bank with a single divisor.
        (part,) = parts("mtp/layers/0/moe/experts/0/gate")
        bank = artifact.object(part["object"])
        assert bank.format == "nvfp4" and bank.divisors == 1
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        assert uses[("text/layers/0/moe/experts/0/gate", "text/layers/0/ffn_input")][
            "activation_policy"
        ] == "AllowA4"
        assert text["ple_layer_ids"] == [1]
