"""Qualification of the Qwen4Exp FP64 oracle.

The oracle is the mathematical authority for Qwen4Exp Op and model qualification, so it is checked
against the published checkpoint constants, against its own state semantics, and, when PyTorch and
a Transformers build with ``qwen4_exp`` are installed, against the upstream reference model on a
small random configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference as ref  # noqa: E402

# Official Qwen3.8-Flash-Next text_config fields that decide the n-gram hash and table.
OFFICIAL = dict(
    vocab_size=248320,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    eos_token_id=248044,
)


def tiny_text_config(**overrides) -> dict:
    config = dict(
        hidden_size=32,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        vocab_size=97,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        output_gate_type="sigmoid",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=12,
        norm_topk_prob=True,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[2],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=53,
        make_ngram_vocab_size_divisible_by=8,
        seed=1234,
        eos_token_id=1,
        bos_token_id=1,
        pad_token_id=None,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
    )
    config.update(overrides)
    return config


def random_weights(config: ref.TextConfig, rng: np.random.Generator) -> dict:
    """Logical weights with the official names and shapes; scales keep activations O(1)."""
    c = config
    h, hc_dim = c.hidden_size, c.hc_dim

    def mat(rows, cols, scale=None):
        return rng.normal(0.0, scale if scale is not None else cols**-0.5, (rows, cols))

    def vec(n, scale=0.2):
        return rng.normal(0.0, scale, n)

    w = {"embed_tokens.weight": mat(c.vocab_size, h, 1.0), "lm_head.weight": mat(c.vocab_size, h)}

    def mixer(prefix, combine=True):
        w[f"{prefix}.hc_norm.weight"] = vec(hc_dim)
        w[f"{prefix}.input_mix_weight_down.weight"] = mat(c.hc_lowrank, hc_dim)
        w[f"{prefix}.input_mix_weight_up.weight"] = mat(hc_dim, c.hc_lowrank)
        if combine:
            w[f"{prefix}.block_inject_weight.weight"] = mat(c.hc_count, hc_dim)

    def attention(p):
        d = c.head_dim
        w[f"{p}.q_proj.weight"] = mat(2 * c.num_attention_heads * d, h)
        w[f"{p}.k_proj.weight"] = mat(c.num_key_value_heads * d, h)
        w[f"{p}.v_proj.weight"] = mat(c.num_key_value_heads * d, h)
        w[f"{p}.o_proj.weight"] = mat(h, c.num_attention_heads * d)
        w[f"{p}.q_norm.weight"] = vec(d)
        w[f"{p}.k_norm.weight"] = vec(d)
        w[f"{p}.indexer.index_qk_proj.weight"] = mat((c.indexer_n_heads + 1) * c.indexer_head_dim, h)
        w[f"{p}.indexer.q_layernorm.weight"] = vec(c.indexer_head_dim)
        w[f"{p}.indexer.k_layernorm.weight"] = vec(c.indexer_head_dim)

    def moe(p):
        e, i = c.num_experts, c.moe_intermediate_size
        w[f"{p}.gate.weight"] = mat(e, h, 1.0)
        w[f"{p}.experts.gate_up_proj"] = rng.normal(0.0, h**-0.5, (e, 2 * i, h))
        w[f"{p}.experts.down_proj"] = rng.normal(0.0, i**-0.5, (e, h, i))
        s = c.shared_expert_intermediate_size
        w[f"{p}.shared_expert.gate_proj.weight"] = mat(s, h)
        w[f"{p}.shared_expert.up_proj.weight"] = mat(s, h)
        w[f"{p}.shared_expert.down_proj.weight"] = mat(h, s)
        w[f"{p}.shared_expert_gate.weight"] = mat(1, h)

    for layer, kind in enumerate(c.layer_types):
        p = f"layers.{layer}"
        mixer(f"{p}.attn_hyper_connection")
        mixer(f"{p}.mlp_hyper_connection")
        moe(f"{p}.mlp")
        if kind == "linear_attention":
            g = f"{p}.linear_attn"
            kd = c.linear_num_key_heads * c.linear_key_head_dim
            vd = c.linear_num_value_heads * c.linear_value_head_dim
            w[f"{g}.in_proj_qkv.weight"] = mat(2 * kd + vd, h)
            w[f"{g}.in_proj_z.weight"] = mat(vd, h)
            w[f"{g}.in_proj_b.weight"] = mat(c.linear_num_value_heads, h)
            w[f"{g}.in_proj_a.weight"] = mat(c.linear_num_value_heads, h)
            w[f"{g}.conv1d.weight"] = rng.normal(0.0, 0.5, (2 * kd + vd, 1, c.linear_conv_kernel_dim))
            w[f"{g}.dt_bias"] = rng.uniform(0.5, 1.5, c.linear_num_value_heads)
            w[f"{g}.A_log"] = np.log(rng.uniform(0.5, 4.0, c.linear_num_value_heads))
            w[f"{g}.norm.weight"] = 1.0 + vec(c.linear_value_head_dim)
            w[f"{g}.out_proj.weight"] = mat(h, vd)
        else:
            attention(f"{p}.self_attn")
        if layer + 1 in c.ple_layer_ids:
            q = f"{p}.ple"
            _, _, rows = ref.ngram_head_tables(c, c.ple_layer_ids.index(layer + 1))
            heads = (c.ngram_size - 1) * c.heads_per_ngram
            w[f"{q}.ple_embedding.ngram_embedding.weight"] = rng.normal(0.0, 1.0, (rows, c.ple_embed_dim // heads))
            w[f"{q}.key_proj.weight"] = mat(hc_dim, c.ple_embed_dim)
            w[f"{q}.value_proj.weight"] = mat(h, c.ple_embed_dim)
            w[f"{q}.norm_key.weight"] = vec(hc_dim)
            w[f"{q}.norm_query.weight"] = vec(hc_dim)
            w[f"{q}.norm_conv.weight"] = vec(hc_dim)
            w[f"{q}.conv1d.weight"] = rng.normal(0.0, 0.5, (hc_dim, 1, c.ple_conv_kernel_size))
    mixer("hyper_connection_mixer", combine=False)

    w["mtp.pre_fc_norm_embedding.weight"] = vec(h)
    w["mtp.pre_fc_norm_hidden.weight"] = vec(hc_dim)
    w["mtp.fc_embedding.weight"] = mat(h, h)
    w["mtp.fc_hidden.weight"] = mat(h, h)
    mixer("mtp.layers.0.attn_hyper_connection")
    mixer("mtp.layers.0.mlp_hyper_connection")
    attention("mtp.layers.0.self_attn")
    moe("mtp.layers.0.mlp")
    mixer("mtp.hyper_connection_mixer", combine=False)
    return w


@pytest.fixture(scope="module")
def tiny():
    config = ref.TextConfig.from_hf(tiny_text_config())
    return config, random_weights(config, np.random.default_rng(7))


def tokens_with_eos(rng, count, config):
    tokens = rng.integers(2, config.vocab_size, count).tolist()
    tokens[count // 3] = config.eos_token_id
    return tokens


# ---------------------------------------------------------------------------------------------
# Published checkpoint constants


def official_config() -> ref.TextConfig:
    return ref.TextConfig.from_hf(tiny_text_config(**OFFICIAL))


def test_ngram_multipliers_match_checkpoint_buffer():
    # layers.1.ple.ple_embedding.layer_multipliers in Qwen/Qwen3.8-Flash-Next.
    assert ref.ngram_layer_multipliers(official_config(), 0) == [
        23703573157769,
        20109073645365,
        8052911324071,
    ]


def test_ngram_table_rows_match_checkpoint_shards():
    sizes, offsets, rows = ref.ngram_head_tables(official_config(), 0)
    assert sizes[0] == 20_000_003 and sizes[-1] == 20_000_171 and len(sizes) == 16
    assert offsets[1] == sizes[0]
    # 128 shards of [2500012, 160] rows.
    assert rows == 128 * 2_500_012


def test_ngram_hash_products_stay_in_signed_64_bit_range():
    config = official_config()
    top = max(ref.ngram_layer_multipliers(config, 0)) * (config.vocab_size - 1)
    assert top < 2**63


def test_ngram_eos_closes_every_window(tiny):
    config, _ = tiny
    eos = config.eos_token_id
    history = [eos, eos, 10, 11, eos, 12, 13]
    rows = ref.ngram_row_ids(config, 0, history, 5)
    # After an EOS the context positions are EOS again, exactly like a fresh sequence start.
    fresh = ref.ngram_row_ids(config, 0, [eos, eos, 12, 13], 2)
    np.testing.assert_array_equal(rows[3:], fresh)
    assert not np.array_equal(rows[1], ref.ngram_row_ids(config, 0, [eos, eos, 11], 1)[0])


# ---------------------------------------------------------------------------------------------
# State semantics


def test_incremental_decode_equals_single_prefill(tiny):
    config, weights = tiny
    rng = np.random.default_rng(1)
    tokens = tokens_with_eos(rng, 23, config)
    whole = ref.TextState(config)
    logits, hidden = ref.forward(config, weights, whole, tokens)

    pieces = ref.TextState(config)
    parts, cursor = [], 0
    for size in (5, 1, 1, 9, 1, 6):
        out, _ = ref.forward(config, weights, pieces, tokens[cursor : cursor + size])
        parts.append(out)
        cursor += size
    np.testing.assert_allclose(np.concatenate(parts), logits, rtol=1e-10, atol=1e-10)


def test_qsa_is_dense_until_budget_then_prunes(tiny):
    config, weights = tiny
    rng = np.random.default_rng(2)
    budget_tail = config.indexer_budget + config.indexer_compress_ratio - 1
    short = tokens_with_eos(rng, budget_tail, config)
    sparse, _ = ref.forward(config, weights, ref.TextState(config), short)
    dense, _ = ref.forward(config, weights, ref.TextState(config), short, sparse=False)
    np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-12)

    longer = short + tokens_with_eos(rng, 16, config)
    sparse, _ = ref.forward(config, weights, ref.TextState(config), longer)
    dense, _ = ref.forward(config, weights, ref.TextState(config), longer, sparse=False)
    np.testing.assert_allclose(sparse[: len(short)], dense[: len(short)], rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(sparse[len(short) :] - dense[len(short) :])) > 1e-6


def test_qsa_selection_width_and_tail(tiny):
    config, weights = tiny
    rng = np.random.default_rng(3)
    visible = 23
    di = config.indexer_head_dim
    selected = ref.qsa_select(
        config,
        weights,
        "layers.2.self_attn.indexer",
        rng.normal(size=config.indexer_n_heads * di),
        *ref.rope_tables(config, ref.text_positions(visible - 1, 1)),
        visible,
        rng.normal(size=(visible, di)),
        ref.text_positions(0, visible),
    )
    assert len(selected) == config.indexer_budget + visible % config.indexer_compress_ratio
    assert set(range(20, 23)) <= set(selected.tolist())
    assert len(set(selected.tolist())) == len(selected)


def test_mtp_incremental_matches_aligned_prefill(tiny):
    config, weights = tiny
    rng = np.random.default_rng(4)
    tokens = rng.integers(2, config.vocab_size, 14).tolist()
    _, hidden = ref.forward(config, weights, ref.TextState(config), tokens)
    positions = ref.text_positions(0, len(tokens) - 1)
    aligned = ref.TextState(config)
    whole, _ = ref.mtp_forward(config, weights, aligned, tokens[1:], hidden[:-1], positions)
    stepped = ref.TextState(config)
    steps = [
        ref.mtp_forward(config, weights, stepped, [tokens[i + 1]], hidden[i : i + 1], positions[:, i : i + 1])[0]
        for i in range(len(tokens) - 1)
    ]
    np.testing.assert_allclose(np.concatenate(steps), whole, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# Upstream reference


def _upstream():
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    configuration = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    return torch, modeling, configuration


def _upstream_model(tiny_config: dict, weights: dict):
    torch, modeling, configuration = _upstream()
    hf_config = configuration.Qwen4ExpTextConfig(**tiny_config)
    hf_config._attn_implementation = "eager"
    hf_config._experts_implementation = "eager"
    model = modeling.Qwen4ExpForCausalLM(hf_config).to(torch.float64).eval()
    state = model.state_dict()
    loaded = {}
    for name, value in weights.items():
        if name.startswith("mtp."):
            continue
        key = name if name == "lm_head.weight" else f"model.{name}"
        assert key in state, key
        loaded[key] = torch.from_numpy(np.asarray(value, dtype=np.float64).reshape(state[key].shape))
    missing = [k for k in state if k not in loaded and not k.endswith(("layer_multipliers", "vocab_sizes", "offsets"))]
    assert not missing, missing
    model.load_state_dict(loaded, strict=False)
    return torch, model


def test_oracle_matches_upstream_prefill_and_decode(tiny):
    config, weights = tiny
    torch, model = _upstream_model(tiny_text_config(), weights)
    from transformers import DynamicCache

    rng = np.random.default_rng(5)
    tokens = tokens_with_eos(rng, 27, config)
    expected = ref.TextState(config)
    prompt, decode = tokens[:20], tokens[20:]
    want = [ref.forward(config, weights, expected, prompt)[0]]
    want += [ref.forward(config, weights, expected, [t])[0] for t in decode]
    want = np.concatenate(want)

    cache = DynamicCache(config=model.config)
    got = []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt]), past_key_values=cache, use_cache=True)
        got.append(out.logits[0].numpy())
        for t in decode:
            out = model(input_ids=torch.tensor([[t]]), past_key_values=out.past_key_values, use_cache=True)
            got.append(out.logits[0].numpy())
    got = np.concatenate(got)
    # Upstream evaluates RoPE tables, the gated norm and pooled indexer keys in FP32.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_oracle_matches_upstream_multimodal_positions(tiny):
    config, weights = tiny
    torch, model = _upstream_model(tiny_text_config(), weights)
    rng = np.random.default_rng(6)
    tokens = rng.integers(2, config.vocab_size, 18).tolist()
    positions = np.stack([np.arange(18), np.arange(18) // 2 + 3, np.arange(18) % 5 + 1])
    want, _ = ref.forward(config, weights, ref.TextState(config), tokens, positions=positions)
    with torch.no_grad():
        got = model(
            input_ids=torch.tensor([tokens]), position_ids=torch.from_numpy(positions)[:, None, :]
        ).logits[0].numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
