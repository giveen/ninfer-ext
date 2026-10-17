# Nemotron Lightning 30B + DFlash: plan

The 74B port settled one thing empirically: speculative decoding loses when the target's routed
experts live in host memory, because a proposal block's verify pays PCIe staging per novel expert
(`docs/maintainer/lilicorr-draft.md`). This pair is the configuration where that constraint
disappears, and it should be treated as the real spec-decode target.

## The artifacts

**Target — `/mnt/storage/models/nemo/lightning-30b-nvfp4`** (`model_type: nemotron_h`)

| | |
|---|---|
| layers / hidden | 52 / 2688 |
| attention | 32 heads, 2 KV heads, head_dim 128, max_positions 1,048,576 |
| experts | 128 routed, top-6, `moe_intermediate_size` 1856 |
| precision | mixed: FP8 on the mixer in/out projections, W4A16_NVFP4 group 16 on the experts |
| bytes | **20.08 GiB total, 18.05 GiB of experts** |

18 GiB of experts is the whole point: it fits a 32 GB card, so the expert cache can be *fully
resident* and the verify's marginal row cost should collapse from ~7.4 ms (measured on the 74B) to a
plain weight read.

**Drafter — `/mnt/storage/models/nemo/30b-dflash`** (DFlash, `qwen3` block shape)

| | |
|---|---|
| layers / hidden | 6 / **2688 — the target's width** |
| attention | 32 heads, 2 KV heads, head_dim 128 |
| intermediate | 6144 |
| taps | `target_layer_ids = [1, 5, 19, 29, 41, 51]` |
| config | `mask_token_id`, `pard_token`, `sample_from_anchor`, `dflash_query_causal`, `dspark_fixup_head_type`, `attention_sink_bias` |
| tensors | 108: `embed_tokens`, `fc`, `hidden_norm`, `layers.*`, `norm` — **no LiLiCorr head** |
| precision | W4A16_NVFP4 group 16, excluding embeddings and attention |

## Why this one can pay

A round wins if `draft + verify + commit < (1 + accepted) × decode_step`. On the 74B that failed
because the verify was 4.6 decode steps. Here, with the experts resident:

| term | 74B (measured) | 30B (expected, to verify) |
|---|---|---|
| decode step | 19.7 ms | ~15 GB read ≈ 15 ms (`ninfer_bench`) |
| verify, 8 rows | 74.8 ms | one resident pass ≈ 20-25 ms |
| draft | 22.9 ms | 6 layers, ~300 MB ≈ 3-5 ms |
| commit | 21.5 ms | 21.5 ms until the scan replay lands |

Round ≈ 5 + 22 + 21.5 = 48.5 ms for ~2.3 tokens ≈ 21 ms/token — still 1.4× over decode **because of
the commit**. With the commit replay (the blocked item in the 74B doc, same machinery) it is
≈ 27 ms ≈ 11.7 ms/token, **~1.3× faster than decode**. That is the concrete win this pair is for,
and it makes the commit replay the highest-value engine item rather than a cost shave.

## Work, in order

1. **Conversion.** New recipe `nemotron_lightning_30b`: mixed precision — FP8 on the two mixer
   projections, W4A16_NVFP4 group 16 on the experts, as `hf_quant_config.json` records. The
   converter's existing FP8 and NVFP4 policies cover both; the recipe must select per input. Emit
   `lightning_30b_nvfp4.ninfer` + its report.
2. **Loader/config.** `nemotron_h` must accept 52 layers, hidden 2688, 128 experts top-6, head_dim
   128, 2 KV heads. Verify against `derive()`'s assumptions (the 74B is hidden 4096, top-8).
3. **Residency.** Size the expert cache to hold all 18 GiB (`--expert-cache auto` must grant it, and
   the run report must show buckets/clamped = none). This is the measurement that decides the pair:
   if it clips, revisit.
4. **Drafter binding.** A new draft component family for this DFlash: 108 tensors, 6 layers, taps into
   a 52-layer target, no LiLiCorr head. The 74B port already binds a dflash2 component in the same
   place, so this is a binding + config derivation, plus a head path that is plain DFlash proposals
   (mask token + `sample_from_anchor` + the fixup head) rather than the LiLiCorr lattice.
5. **Round.** Reuse the nemotron_h draft round with the DFlash head; keep the same acceptance and
   bonus publication. Gate on: verify row 0 reproduces decode, bonus agreement, and the per-round
   acceptance table.
6. **Commit replay.** The per-column Mamba scan replay, now worth ~21 ms of a 48 ms round.
7. **Measure.** `ninfer_bench -p 512 -n 64` plain versus `--spec dflash --draft-tokens N`, plus the
   round test for acceptance. Report both, and the decode step separately.

## Conversion: what the source already does (measured, not assumed)

`tools/convert/sources/modelopt.py` is the preserve-source the plan called for, and its contract
already covers both halves:

- `nvfp4_source(store, prefix, shape)` yields the **E2M1 code plane and the E4M3 block scales
  bit-exactly**, with the `weight_scale_2` multiplier converted once to the v3 divisor. The 18.05 GiB
  of experts is therefore preserved, not re-quantised.
- `fp8_tensor_rows_source(...)` maps a per-tensor FP8 checkpoint "exactly to the row-scaled FP8
  format with that multiplier on every row": the **E4M3 codes are untouched** and only the scale
  representation changes. So the FP8 half is a re-encode of the scale, not a second rounding. Its
  `input_scale` is **not** carried because our route is A16 -- a decision to state in the report, not
  a preservation.

What is actually missing is two edits, not a subsystem:

1. `tools/convert/nemotron_h.py` imports only `nvfp4_source` (line 41) and its `quantized()` helper
   (lines 540-543) is hard-wired to it. The tower reader needs per-parameter source selection that
   admits the per-tensor FP8 form; the idiom already exists in `qwen4_exp.py:344-357`
   (`fp8_tensor_rows_source(store, shards, prefix + ".weight_scale", (rows, width))`) and the
   dispatch-by-companion pattern in `_expert_source` beside it.
2. A `nemotron_lightning_30b` recipe keyed on the measured names, not the 74B's:
   - FP8 `fp8_e4m3fn_row_bf16`: `backbone.layers.N.mixer.in_proj [10304, 2688]`,
     `backbone.layers.N.mixer.out_proj [2688, 4096]`;
   - NVFP4 g16 preserved: `backbone.layers.N.mixer.experts.E.{down,up,gate}_proj` (U8 packed along
     K, `weight_scale` 1 per 16 K values, `weight_scale_2` F32 divisor);
   - BF16 for the 72 excluded modules the checkpoint names (embeddings, `mixer.conv1d`,
     `mixer.gate`, `mixer.k_proj`, norms, router).

`hf_quant_config.json` is **not** the per-parameter authority for this checkpoint (72 exclusions, no
per-layer entries); the stored dtype plus companion names/shapes are. The emitted report must carry a
per-input row of `(source dtype, decision in {preserved, re-encoded, BF16}, emitted format)` and call
out `input_scale` explicitly.

**The source admission already exists — the work is the recipe, not the builder.** `matrix_source`
(`tools/convert/sources/compressed_tensors.py:138`) resolves the stored encoding itself:
`weight_scale_2` -> `modelopt_nvfp4`; a `F8_E4M3` weight with an `F32` scale of `prod == 1` ->
**`modelopt_fp8`** (the per-tensor case); otherwise `fp8_e4m3fn_row_bf16`. So a quantized ModelOpt
checkpoint is readable as-is, and the earlier "tower has no FP8 admission" concern was wrong: the
admission is in the shared dispatcher, and the 74B recipe simply never exercised it because its
checkpoint was BF16.

What a recipe must then do is name the *stored* format rather than the quantisation target, using the
preserve pattern the EXL3 recipes already use:

```python
recipe.assign(name, format="nvfp4",  method=import_encoded, activation_policy="A16Only",
              source=nvfp4_source(store, prefix, shape))        # experts, codes preserved
recipe.assign(name, format=FP8,      method=<per-row replication>,  activation_policy="AllowA8",
              source=fp8_tensor_rows_source(store, shards, prefix + ".weight_scale", shape))
```

`import_encoded` is the method that carries stored codes through without re-quantising
(`official_recipes.py:165-166`), and the FP8 form replicates the per-tensor multiplier across rows
with the E4M3 codes untouched. Confirm the exact FP8 preserve-method name in `tools/convert/methods.py`
before writing the recipe; if no such method exists, the FP8 half needs one.

`import_encoded` is the method that carries stored codes through without re-quantising
(`tools/convert/methods.py:367`). Its contract, read from source, is exact and it is all the recipe
needs to satisfy:

- `request.target.format` must be `nvfp4`, `fp8_e4m3fn_row_bf16` or `exl3_mul1`;
- the source's `read_encoded(0, 1).format` must **equal** that target format, otherwise it raises
  ("source <fmt> differs from target <fmt>").

So the two assignments are:

```python
# experts: nvfp4_source(...) yields read_encoded().format == "nvfp4"
recipe.assign(name, format="nvfp4", method=import_encoded, activation_policy="A16Only",
              source=nvfp4_source(store, prefix, shape))
# mixer projections: fp8_tensor_rows_source(...) yields "fp8_e4m3fn_row_bf16"
recipe.assign(name, format=FP8, method=import_encoded, activation_policy="AllowA8",
              source=fp8_tensor_rows_source(store, shards, prefix + ".weight_scale", shape))
```

Both preserve paths are therefore already implemented; the remaining work in the recipe is only to
construct those two sources from the store (the EXL3 recipe at `official_recipes.py:165-166` is the
template for `assign(..., source=...)`) and to keep `input_scale` out with a comment, since the route
is A16.

## Conversion: the builder gates (measured by running it)

Running the converter against `/mnt/storage/models/nemo/lightning-30b-nvfp4` now reaches the right
builder and stops on three successive guards, in this order:

1. `unsupported Qwen text architecture ['NemotronHForCausalLM']` -- fixed by adding the name to
   `ARCHITECTURES` in `tools/convert/nemotron_h.py:45`.
2. `llm_config.mtp_layers_block_type: MTP blocks are not mapped` -- the target **declares one MTP
   layer** (`num_nextn_predict_layers: 1`, `mtp_layers_block_type: ['attention','moe']`). Our spec
   path uses the separate 6-layer DFlash drafter, so the MTP layer is deliberately not mapped; the
   guard needs to allow a declared-but-unmapped MTP layer for this architecture, with the drop
   stated, rather than refuse.
3. The 74B validation constants (`_HIDDEN 4096`, `_MAX_POSITIONS 262144`, `_LAYERS 88`, `_LATENT`,
   `_SHARED_WIDTH`, `_ATTENTION_HEADS`) -- this model is hidden 2688, 52 layers,
   `layers_block_type` over `{moe, mamba, attention}` and `moe_intermediate_size` 1856. They need to
   become a per-variant table keyed on the architecture (or read from the config and cross-checked
   against a recorded value), not 74B constants.

The recipe is already written and admitted (`nemotron_lightning_30b` in `official_recipes.py`), and
`text_config` already falls back to the flat config when `llm_config` is absent, so these three are
the whole remaining distance to a running conversion.

## Status (measured)

- **`out/lightning_30b_nvfp4.ninfer`** (text only, 405 objects) loads and runs: **prefill 1108 t/s,
  decode 46.0 t/s**, 20.02 GiB of weights wholly resident — no expert offload, no PCIe staging. This
  is the target the pair was for.
- **`out/lightning_30b_nvfp4_dflash.ninfer`** (468 objects) adds the 6-layer DFlash companion built
  from `/mnt/storage/models/nemo/30b-dflash` and loads identically (1117 prefill / 46.2 decode); the
  Program currently ignores the component.

What the conversion had to establish, each as a gate that failed loudly first: the flat
`NemotronHForCausalLM` layout (no `llm_config`, no `block_configs`, 23 mamba / 23 moe / 6 attention),
a direct (non-latent) bank, zero-padded expert banks (`moe_intermediate_size` 1856 -> 1920, exact),
BF16 for every shape the FP8/NVFP4 route registries do not carry, and the draft's YaRN rope reduced
to its base theta (identical inside the 8192-token original window; drafts beyond it are the one
regime this artifact does not reproduce).

## Remaining: the C++ half of the drafter port

1. A `dflash` component config parser (the existing `models::draft::Spec` is LiLiCorr-only with
   closed member sets) and its binding in `src/models/nemotron_h/load.cpp`.
2. The drafter execution beside the Program: `fc` over the six tap residuals, six qwen3 blocks
   conditioned on the context and on the mask tokens' embeddings (the companion carries neither
   embeddings nor a head -- both come from the target), then the projection that turns its hidden
   state into the target's proposal logits.
3. The round: propose/verify/accept/publish/commit, reusing `program.cpp`'s machinery, plus the
   Mamba scan replay that removes the commit pass.
4. Acceptance and the bench, then the final comparison against 46 t/s.

## The DFlash port (next engine slice)

**Pieces 1-2 landed and verified.** `models::draft::parse_dflash` validates the companion's closed
key set (taking the width and vocabulary from the tower, because the converter checks those against
it rather than storing them), `parse_config` reads the `dflash` component, and `bind_dflash` binds
its 63 tensors under the `dflash/` prefix -- `feature_projection [hidden, 6*hidden]`,
`context_norm`, `final_norm`, and per layer `input_norm`, `post_attention_norm`, the query/key/value
trio, the pair of context projections, both norms, the output projection and the gated
`mlp/{gate,up,down}` -- selected by the spec's flavor, since a plain draft stores no block length.
The artifact loads with the companion bound and the bench is unchanged: 1604 prefill / 148.7 decode
at 8 tokens.

Still to come, in order, each independently checkable:

The artifact already carries the companion (`out/lightning_30b_nvfp4_dflash.ninfer`, 468 objects)
and the Program ignores it. Four pieces, in order, each independently checkable:

1. **Spec parser.** A `parse_dflash(...)` beside `models::draft::parse_lilicorr`, validating the
   closed key set the converter emits: `architectures` (`DFlashDraftModel`), `model_type` (`qwen3`),
   `hidden_size` (the target's), `vocab_size` (the target's), `intermediate_size`,
   `num_attention_heads`, `num_key_value_heads`, `head_dim`, `num_hidden_layers` (6),
   `max_position_embeddings`, `rms_norm_eps`, `rope_parameters.rope_theta` (10000, YaRN reduced to
   its base theta by the converter), `layer_types` (all `full_attention`), and
   `dflash_config{target_layer_ids [1,5,19,29,41,51], mask_token_id 990}`. No conv keys, no head:
   a plain DFlash has neither.
2. **Binding.** Mirror `bind_draft` with the `dflash/` prefix: `feature_projection` `[hidden, 6*hidden]`,
   `context_norm`, `final_norm`, and per layer `input_norm`, `post_attention_norm`,
   `attention/{query,key,value,context_key,context_value,query_norm,key_norm,output}`, `mlp/{up,down}`
   (silu, `intermediate_size` 6144). The companion carries no embeddings and no head: the drafter
   reads the target's token embeddings and proposes through the target's `text/output_head`.
3. **Execution.** `fc` over the six tapped residuals -> six qwen3 blocks (self-attention over the
   context keys/values the companion's `context_key`/`context_value` provide, rope at the absolute
   positions, silu MLP) -> `final_norm` -> the target's head over the mask tokens. Positions and the
   conditioning taps come from the same `DraftContext` plumbing the LiLiCorr body uses.
4. **Round.** Reuse `program.cpp`'s propose/verify/accept/publish/commit, with the Mamba scan replay
   replacing the commit pass. Gates unchanged: verify row 0 reproduces decode, bonus agreement,
   acceptance table; then `ninfer_bench --spec dflash` against plain decode at 188 t/s.

## The companion's "23% cost" was two stale/recipe bugs, not the drafter

The earlier comparison (plain 193.8 vs companion 148.8 t/s) turned out to be two separate defects,
both now understood:

1. **The companion artifact predated the FP8 dense change.** A kernel profile settled it: the plain
   artifact's top kernels are `fp8_a16_mma`, the older companion artifact's are `bf16_latent_row` --
   its text tower was still BF16, which is exactly the 2x per-token read the FP8 work removed. The
   drafter was never the cost.
2. **The recipe's role rules reached into the companion.** `fp8_roles` matches substrings
   ("/attention/"), which also appear inside `dflash/layers/N/attention/...`, so the first
   reconversion gave the draft's **norms** an FP8 matrix format; the loader refused with "direct
   tensor requires a resident contiguous weight". The Lightning recipe now skips `dflash/` and
   `dflash2/` wholesale, the way the 74B recipe skips its own companion: a speculative drafter is
   stored exactly as its checkpoint holds it.


## MoE: the cheap levers are exhausted, and the measurement says why

Three attempts at the 2.05 ms MoE term, each measured against the same artifact:

| change | effect | note |
|---|---|---|
| warp per output row (32 lanes on a row, warp reduction) | +2% decode | op test green, 74B acceptance unchanged |
| block size 64 -> 32 threads (double the blocks) | 0% | reverted |
| E4M3 scale decoded once per 8-byte group instead of per byte | +1.5% prefill, 0% decode | bit-identical (op test green) |

So the term is not parallelism, not block shape and not redundant scale work: it is the per-element
INT4 decode itself (~713M element decodes per token), which only a tensor-core path removes. The
engine already has the Blackwell `mxfp4` MMA for dense NVFP4 linears; using it for the routed bank
needs a per-expert GEMM over the bank reader's code planes and block scales, a schedule for M in
[1, 8], a replacement numeric contract (the SIMT route is bit-identical to the per-token kernel) and
the op-test tolerance. That is the remaining work for prefill (2,313 against Qwen's 17,705) and the
larger half of decode.

## Risks

- **Mixed-precision conversion** is the one unproven step: the recipe must reproduce FP8 on linears
  and NVFP4 on experts, and the report must show it per input.
- **Drafter semantics**: `dspark_fixup_head_type`, `sample_from_anchor` and `pard_token` need to be
  matched exactly; a wrong head flatters or destroys acceptance silently, which is what the
  verify-row-0 and bonus gates exist to catch.
- **Taps**: six tap layers out of 52, at 2688 hidden — re-derive the tap planes rather than reuse
  the 74B's.
- The 74B's LiLiCorr head is not part of this pair; do not carry its lattice code into the DFlash
  path.

## Plain DFlash: the execution path, and the mask-token embedding (measured)

The companion in `out/lightning_30b_nvfp4_dflash.ninfer` is a **plain** DFlash (`DFlashDraftModel`,
6 layers, 81 tensors, no conv, no sink, no head, no stored block length). The execution path in
`models/nemotron_h/execution/draft.*` was LiLiCorr's, so it refused this companion before it ever
proposed. It is now flavour-gated (`Spec::Flavor`): a plain DFlash carries the block width and the
context window from the caller (`Program::draft_width_`, the request's capacity), skips the
convolution/sink/head tensors, and proposes through the **tower's** output head with one greedy argmax
per column. LiLiCorr's path is unchanged.

The defect that made acceptance zero: **the companion's `embed_tokens` is the tower's matrix with
exactly one row of its own -- the mask token's -- and that row is the entire input of every proposal
column.** Measured over the 131,072 rows, one differs (id 990; cosine 0.002 with the tower's row).
Feeding the tower's row produced context-dependent but unrelated proposals (acceptance 0.000). The
converter now emits `dflash/mask_embedding` `[1, 2688]` and the body writes it over the mask columns;
the anchor row keeps the tower's embedding, which the two stores share.

Measured after the fix, at the reference prompt length (187 tokens, 32 rounds, `draft_tokens=7`):
**2.875 accepted drafts per round, slot 0 at 0.75** -- against the LiLiCorr reference's 2.122 and
0.707. The 0.375 measured on a 128-token prompt is a property of the text, not of the port: the
drafter is conditioned on the committed prefix, so a short prefix drafts worse. Acceptance is a
sentence-level number, never a single-run one.

Two more defects the port had to fix, both found by measurement:

- **A captured body's width is part of its identity.** The body graph is captured per lane and read
  the round's `1 + slots` ids and hidden columns, so a later, narrower round left the graph's tail
  rows reading whatever the next arena allocation wrote there -- an illegal access the moment the
  engine's request budget shortened a round. The body now runs the *widest* block every round (the
  reference's own rule) and the head scores the round's leading columns; `block_size` is the only
  width the graph ever sees.
- **The draft's BF16 shapes were on the dense fallback.** `fc`, `gate/up` and `down` now have
  registered routes in `shapes/nemotron_h_latent.cu` (bit-identical to the fallback, asserted in
  `test_bf16_a16.cpp`), which took the draft forward from **13.7 ms to 3.9 ms** per round.

The round is now `draft 3.9 + verify 10.6 + commit 8.5 ms` at 3.875 committed tokens per round.

## Engine wiring and the bench (measured)

Speculation runs through the Engine: `nemotron_engine_options` no longer zeroes the backend (it
refuses MTP/EAGLE3 by name), the load path checks the requested flavour against the artifact's
companion, the draft's bytes are part of the residency reservation, `NemotronProgram` engages the
draft at the first transaction and `decode` publishes a multi-token `PendingBatch` when the batch is
one row and the budget leaves room for a round. `ninfer_bench -pg 512,128`:

| arm | decode t/s | rounds | fallbacks |
|---|---:|---:|---:|
| plain | 253.5 | - | - |
| `--spec dflash --draft-tokens 7` | **267.9** | 36 | 0 |

+5.7% for 2.56 accepted drafts per round: the verify block (10.6 ms) and the commit pass (8.5 ms)
are still ~2.7 decode steps of the 3.875 tokens a round commits. The expert-binned MoE (the verify's
8 rows re-read the bank once per (token,rank)) and the Mamba scan replay (which removes the commit
pass) are what turn that into the 2-3x the acceptance supports; until then speculation is wired,
correct and marginally ahead rather than ahead by a round's worth.

## Decode, after the FP8 GEMV route (measured)

`shapes/nemotron_lightning.cu` gated the one-row GEMV route on `K % 256 == 0`, so K = 2688 and
K = 3712 (the mamba in-projection, the shared expert, attention q/k/v) ran the MMA schedule with 31
of 32 rows empty: 390-875 GB/s against the GEMV route's 1.3 TB/s. The kernel now walks one guarded
tail phase and the route needs only `K % ValuesPerLane`. Decode **195.5 -> 255.3 t/s** (5.11 ->
3.92 ms/token), prefill unchanged at 1929 t/s; the change is covered by
`tests/ops/linear/test_fp8_a16.cpp`'s Lightning shapes at T = 1, 2, 8 against the FP64 oracle.

## Where a speculative round's time goes (measured, after the replay)

The round is now one draft body, one target verify pass and a **state replay** instead of a second
target pass. `draft_round` snapshots the Mamba planes, runs the verify and -- once the accepted
prefix is known -- advances the state by replaying only the convolution and the scan over the
accepted columns from the verify's own captured operands (`MambaReplay`, `forward.cpp`), rewinds
the attention frontier, moves the head's logits and the taps to the committed column and advances
the cursor. It is numerically the state the pass computed, so acceptance is unchanged (2.875
drafts/round, slot 0 0.75) and the commit phase went from 8.46 ms to **1.60 ms**.

Two instrumentation traps this campaign hit, worth keeping:

- **nsys does not trace kernels launched by a CUDA graph unless asked.** Without
  `--cuda-graph-trace=node` the draft body (graph-captured) and the now-graph-captured verify pass
  vanish from the kernel summary: the first attribution here put the draft at 1.2 ms/round when it
  is **4.1 ms**. Use the flag and diff two round counts; the difference is the honest number.
- A captured pass must have an invariant width. The body was captured with the first round's
  `1 + slots`, and a later narrower round left the graph's tail rows reading the next arena
  allocation -- an illegal access. The body now runs the *widest* block every round and the head
  scores its leading columns.

Per-round kernels, `--cuda-graph-trace=node`, 2 vs 34 rounds divided by 32 (wall 16.4 ms):

| family | ms/round |
|---|---:|
| MoE row-tiled up + down (verify) | **5.54** |
| draft body (`bf16_latent_row`) | **4.10** |
| FP8 dense MMA (verify, incl. the head) | **3.52** |
| router logits + select | 0.43 |
| rmsnorm | 0.29 |
| attention | 0.27 |
| mamba scan + conv | 0.28 |
| rest (residual/rank-sum/rope/relu) | 0.31 |
| **kernels** | **14.87** |
| host gaps | ~1.5 |

| arm | decode t/s |
|---|---:|
| plain | 254.4 |
| `--spec dflash --draft-tokens 7`, before the replay | 267.9 |
| `--spec dflash --draft-tokens 7`, after the replay | **417.2** |

## What is left, and what it is worth

The draft's own read is now solved: its four matrix families keep the checkpoint's NVFP4 codes
(`fc`, `mlp/down` on the registered chunked route; `mlp/gate`/`up` on a **dedicated group route**
added for this model, `nvfp4_group_gemv.cuh`, because the shared route's 256-wide phase cannot tile
K = 2688). The op test checks all three widths against the FP64 oracle, acceptance is unchanged
(2.875/round, slot 0 0.75), and the draft went from **4.10 to 2.97 ms** of the round.

The round's remaining budget (14.92 ms wall, 7.11 committed tokens per round) is the verify pass:

| term | ms | note |
|---|---:|---|
| verify wall | 10.36 | MoE 5.5 + FP8 dense 3.5 + head 0.4 + rest 1.0 |
| draft forward | 2.97 | NVFP4; the BF16 attention matrices are the rest |
| replay commit | 1.38 | snapshot + conv/scan replay + frontier/logits/taps |

The verify's bytes are fixed by the block width -- 8 rows route 48 (token, rank) pairs per layer
over ~41 distinct experts, i.e. ~4.9 GB of 4-bit expert weights, plus 2.1 GB of FP8 dense weights.
That is a **4.9 ms floor at this card's bandwidth and 8.3 ms at the 885 GB/s the routes measure**;
the wall is 10.36 including the head and the norms. So the remaining ~2 ms is route efficiency, not
traffic, and two tile experiments have already come back neutral (16-row/8-token FP8; wider expert
tiles), which is the signature of a load-path limit rather than a schedule one.

Where that time actually goes, measured on the isolated kernels (`ninfer_linear_bench`,
`ncu` on the profile call) rather than inferred from the wall:

| kernel | width | time | DRAM | warps active | what limits it |
|---|---|---:|---:|---:|---|
| fp8_a16_mma (16384,5376) | 8 | 100.4 us | 49% | 13.8% | 8x activation staging (`BlockTokens` 64 for 8 rows), 512 blocks |
| same, token-exact 16/8 tile | 8 | 77.8 us | 63% | - | still 2 warps of the 4 the wide tile had |
| same, 16-row blocks, model shapes (10304/3712/2688 x 2688) | 8 | -8..12% | 20-45% | - | 84-322 blocks, 1-2 warps/SM |
| nemotron_h_moe_experts row-tiled | 8 | 240 us/layer | ~50% | - | 1392/4032 blocks is fine; 17% duplicate expert reads and 1-warp CTAs |

The dense verify kernels are **memory-latency bound, not bandwidth bound**: `smsp` long-scoreboard
is 4.3%, short-scoreboard 21.5%, and occupancy is 2-14% of the warp budget because a block's only
reuse is 8-16 rows of activation, so it owns 16-32 output rows and the card gets 84-644 blocks of
1-4 warps. The weight stage is single-buffered (`kSharedBytes` in `Fp8A16MmaSchedule` holds one
`BlockRows x BlockK` buffer), so a warp has one K-step load in flight and no second to hide behind.
Two token-exact schedules are now in place for the shapes this family uses (the `tokens <= 16`
branch of `shapes/nemotron_lightning.cu`, 16-row blocks where 32 leaves half the card idle); they
are worth 1.4% of the round (14.92 -> 14.74 ms, 464.3 t/s), and no further schedule parameter
moved the wall.

The route that would move it is a small-width memory path of its own, branched as a new module so
the wide-token schedules stay untouched: double-buffered FP8 weight staging (two outstanding K-step
loads per warp, which is what 2-4 warps/SM need), or a K-split across blocks with a reduction to
buy back the block count on the 2688-row shapes. Either is a new kernel, not a schedule; the
existing `Fp8A16MmaSchedule` and the MoE row-tiled kernels stay as they are for the wide calls.

Reaching 700 t/s therefore needs the verify's routes to move their bytes at ~90% of peak (TMA or a
wider load path in `fp8_a16_mma` and `nemotron_h_moe_experts_row_tiled_*`), or a cheaper verify in
*bytes*: 8-bit-exact experts are already 4-bit; the only remaining byte cut is verifying fewer rows
(and the K sweep says the width is already at its optimum for this corpus). At 90% efficiency the
round would be ~10 ms for 7.11 tokens, i.e. ~710 t/s.

Measured arms, `ninfer_bench -pg 512,128`:

| arm | decode t/s |
|---|---:|
| plain | 257.7 |
| `--spec dflash --draft-tokens 7`, before this turn | 267.9 |
| after the replay + graphed verify | 417.2 |
| after the NVFP4 draft | 457.8 |
| after the small-width token-exact route | **464.3** |
