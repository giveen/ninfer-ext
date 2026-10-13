# Speculative decoding

MTP on every model, DFlash on Qwen3.6-35B-A3B, DFlash2 on Qwen3.8-27B artifacts that carry the
companion weights, EAGLE3 on artifacts that carry an EAGLE3 draft head, and a model-free
prompt-lookup draft source that rides any MTP model.

## Recommended settings

The fastest measured settings per model. "Concurrency" is how many requests decode at once; set
`--max-concurrency` to that number, because every extra lane reserves KV and graph memory whether it
is used or not. Where a cell says the workload decides, the numbers behind it are on the
[performance record](../docs/performance.md) and in the [stock comparison](vs-stock-ninfer.md).

| Model | 1 request | 2–4 concurrent | 8 concurrent |
|---|---|---|---|
| Qwen3.8-27B `nvfp4` | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | same | same |
| Qwen3.8-27B `groupwise-int` with DFlash2 weights | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | same | same |
| Qwen3.8-27B `groupwise-int` (official artifact) | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` | `--spec mtp --lm-head-draft` | `--spec mtp --draft-tokens 5 --fixed-draft --lm-head-draft` |
| Qwen3.8-27B `exl3` 4.0 / 3.5 bpw | `--spec mtp --draft-tokens 3 --fixed-draft` | not measured | not measured |
| Qwen3.6-35B-A3B | `--spec mtp --lm-head-draft` | same at 2; at 4, same for long reasoning and no `--spec` for short prose | `--spec dflash --draft-tokens 7 --lm-head-draft` for long reasoning; no `--spec` for short prose |
| Qwen3.6-27B (both) | not measured; start from the Qwen3.8-27B row of the same weights | | |
| Qwen3.8-Flash-Next | `--spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts auto` | same | same |
| Gemma 4 31B / 26B-A4B | `--spec mtp` (full proposal head; `--lm-head-draft` is refused) | draft length decides: see [Gemma 4](gemma4.md) | | 

Why these:

- **27B models.** Speculative decoding pays at every concurrency. On the long-reasoning load a fixed
  5-token MTP draft beat the adaptive default on `nvfp4` by 3–17%. On `groupwise-int` the adaptive
  default wins at 2–4 requests, and fixed K=5 at 1 and 8. DFlash2 is the fastest mode on
  `groupwise-int` whenever the artifact carries its weights.
- **EXL3.** Only the single-request settings are measured: fixed 3-token MTP drafts, the same
  speculative shape the other 27B artifacts use. Concurrency is not measured yet.
- **35B-A3B.** For one or two requests, adaptive MTP is within 5% of the best fixed draft on long
  reasoning and ahead on short prompts. Speculation hurts short prose from 4 requests up: on
  512-token essays, plain decode is 14% faster than MTP at C=4 and 32% faster at C=8. On long
  reasoning at C=8, DFlash with 7 drafts is the fastest mode.
- **Copy-heavy work (edits, refactors, quoting, repeated tool arguments).** On any MTP model use
  `--spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts auto`. It is the fastest mode measured
  on that traffic and, where nothing repeats, the cost gate falls back to ordinary decode, so it
  matches plain decode rather than regressing.
- **Flash-Next.** Decode is bound by fetching experts over PCIe, so an MTP verify routes up to four
  columns and touches more experts than the accepted drafts save: on 512-token essays plain decode
  was 20% faster than MTP for a single request and 2% faster at C=8. The lookup window is the
  exception — it only runs a wide verify where the text repeats earlier context — so use it for
  copy-heavy work and plain decode otherwise.
- **The adaptive default (`--spec mtp` alone)** is a reasonable choice when the workload is unknown
  or mixed; see [Adaptive MTP draft length](#adaptive-mtp-draft-length).

## Adaptive MTP draft length

`--spec mtp` alone selects an adaptive policy with a longest draft of 7 tokens:

- A single request drafts 2, 3, 4 or 7 tokens per round, chosen from its recent acceptance and from
  round times measured at startup.
- Rounds where several requests decode together choose their draft length from measured batch
  rounds.
- On host-resident-expert models, batches of two or more run as ordinary rounds plus an MTP KV
  append.
- `--draft-tokens N` lowers the ceiling, and `--fixed-draft` pins exactly N.

Against fixed K=3 over nine short code-edit, code-generation, prose and reasoning prompts, adaptive
scored a geometric mean of +8.2% on Qwen3.8-27B `groupwise-int` (edits +19%) and +2.7% on
Qwen3.6-35B-A3B. No single MTP setting wins on every model and load: on the long-reasoning load a
fixed K=5 was faster on 27B `nvfp4` and at C=8 on 35B-A3B.

## Prompt-lookup draft source

A host-side suffix index drafts the tokens that followed an earlier occurrence of the current
suffix — a copied file, a quoted document, a repeated tool argument. It is model-free (a trigram
key with four recent positions and a backward extension), and every draft is verified by the target,
so it cannot change the output distribution.

- **`--lookup-drafts off|auto|always`** (MTP only) adds it to the MTP round; with
  **`--draft-tokens 15 --fixed-draft`** the round becomes a wide lookup-only verify that skips the
  MTP draft phases and falls back to ordinary decode when nothing repeats.
- **Cost gate.** The engine measures the wall time of a wide lookup round and of an ordinary round
  and takes lookup only when it is predicted to commit more than their ratio, so `auto` is safe by
  construction. Round times and per-match acceptance are learned per run.
- **Measured**, 400 tokens, greedy, RTX 5090: on copy-heavy edits 35B-A3B goes 390 → 1,610 tok/s
  (×4.1 over plain, +66% over MTP K=7) and Flash-Next 76 → 158 (+33% over MTP K=7); on prose the
  wide window is byte-identical to plain decode and the cost gate never fires. Request logs and the
  CLI summary carry per-request `lookup_*`/`chain_rounds` counters.
- **Rejected alternatives** (recorded so they are not repeated): the last-follower n-gram pool has
  worse recall than the trigram index, chaining MTP with the pool is dominated by the pool-only wide
  window, and the 15-draft verify ceiling is structural (target verify and GDN conv-record cap
  `T` at 16).

[Prompt-lookup suffix drafter](maintainer/lookup-drafter.md) is the authority.

## Converting a DFlash2 artifact

No published artifact carries DFlash2. Convert one from `Qwen/Qwen3.8-27B` and the DFlash2 companion
weights:

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-27B --recipe qwen3_8_27b \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal --name qwen3.8-27b --out models/qwen3_8_27b_dflash2.ninfer
```

## Converting an EAGLE3 artifact

No published artifact carries EAGLE3 either. Convert one from a Qwen3.6-35B-A3B checkpoint and a
SpecForge `LlamaForCausalLMEagle3` head:

```bash
python3 -m tools.convert --model /path/to/Qwen3.6-35B-A3B-NVFP4 \
  --recipe qwen3_6_35b_a3b_nvfp4 \
  --source quantized=/path/to/Qwen3.6-35B-A3B-NVFP4 \
  --source eagle3=/path/to/Qwen3.6-35B-A3B-Eagle3-Specforge \
  --components text,vision,mtp,eagle3 \
  --resource chat_template.jinja=tools/chat_templates/qwen.jinja \
  --proposal --name qwen3.6-35b-a3b-eagle3 \
  --out models/qwen3_6_35b_a3b_nvfp4_eagle3.ninfer
```

The head used here is the SpecForge EAGLE3 draft for Qwen3.5-35B-A3B
(`jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge`). The converter normalizes its `midlayer.*` weights,
derives the three target layers it fuses, stores the draft-to-target token map, and inherits the
target's token embedding and tokenizer. Heads in the `speculators`/vLLM layout (nested
`transformer_layer_config`, `layers.0.*` weights, a declared `eagle_aux_hidden_state_layer_ids` and a
partial rotary factor) convert the same way; note that a published head is not automatically better
for your traffic — the EAGLE3 reference measures the choices it knows. Serve it with
`--spec eagle3 --draft-tokens N`. Keep `--proposal`: EAGLE3 does not need the optimized proposal head
(it drafts through its own 32k head and rejects `--lm-head-draft`), but without it the file cannot
run the documented `--spec mtp --lm-head-draft` configuration at all, so EAGLE3 could only be
compared against — or fall back to — the slower full-head MTP.

See [EAGLE3](maintainer/eagle3.md) for the draft graph, the target conditioning, the draft-KV cost
and the measured comparison against MTP.
