# Prompt-lookup speculative decoding (suffix drafter)

Status: **Phases 1-3 implemented.** This is the active work record for porting the prompt-lookup
(suffix) draft source that Strata uses (`src/spec/suffix_drafter.cpp`, `draft_policy.cpp`,
`controller.cpp`). It states the design, the phases, and the evidence each phase must produce. It is
not yet a stable contract; promote the final design into the model/runtime reference when the work
settles.

## 1. Goal and scope

Add a model-free draft source that proposes the tokens which followed an earlier occurrence of the
current suffix. It is nearly free on the Host (a trigram index and a hash probe), it needs no draft
weights, no draft KV, and no GPU work, and its proposals are one-hot, so the target's rejection
sampling keeps the output distribution exactly the model's — the same guarantee MTP already has.

The question this work answers is **economic, not algorithmic**: on our models and hardware, does a
lookup draft commit more tokens per round than the alternative it replaces?

Scope:

- In: Host suffix index, per-sequence lifecycle, a learned acceptance model, the per-lane choice,
  the substitution inside the MTP round, and the measurement to decide whether to continue.
- Out for now: a dedicated lookup-only backend (Phase 2), the joint `{none, lookup, MTP}` controller
  with per-source cost model (Phase 3), and any change to DFlash/DFlash2.

## 2. What the port is

Strata's pieces and our decision on each:

| Strata piece | Decision |
|---|---|
| `SuffixDrafter`: open-addressed trigram table, 4 recent positions per key, `min_match` 3, extend backwards to `max_match`, propose the earlier occurrence's continuation | **Port** essentially unchanged (Host C++). |
| Per-match-bucket acceptance EMA (`{<6, <12, <24, >=24}`) | **Port**, with conservative priors and an in-engine learner. |
| `DraftPolicy`/`Controller` choosing between sources by expected committed tokens per ms | **Defer to Phase 3.** Phase 1 reuses the MTP round, so both sources cost the same and the choice is pure expected tokens. |
| Expert-cost model (`distinct_ratio`) to price verify width | **Phase 3.** Phase 1 uses a fixed round width (`--fixed-draft`) so the comparison is clean. |
| Proposals independent of a trained draft head | **Keep.** This is the point. |

## 3. Phase 1 design

### 3.1 Index (`SuffixDrafter`)

Host-only, in `src/models/qwen3_5/program/speculative/`. Faithful to Strata:

- key = mixed hash of the trigram ending at position `i`; open addressing, linear probe, empty = 0.
- each key keeps `Ways = 4` most recent end positions.
- `propose(max_drafts)`: find the current suffix's trigram, extend each candidate backwards to
  `max_match` (default 32), keep the longest (newer wins ties), then emit history up to the current
  end and up to `max_drafts`.
- `sync(span<TokenId>)` appends tokens after the indexed prefix, so the caller can sync the whole
  sequence ledger incrementally once per round; `reset()` clears it.
- Bounded memory: history is the sequence ledger (4 B/token), the table is fixed at construction to
  the next power of two covering the indexed capacity. When the table fills, new keys are dropped,
  not evicted (a trigram that was never inserted simply yields no proposal).

### 3.2 Acceptance model and choice (`LookupPolicy`)

- `LookupAcceptance`: per-bucket **conditional** probability `q` that a draft is accepted given its
  prefix was, estimated from `(drafted, accepted)` per round as Strata's `DraftPolicy` does
  (`q ~ accepted / (accepted + partial_rounds)`, so `q/(1-q)` matches the realized accepted per
  round). Using `accepted / drafted` instead would understate `q` by the window width; the real logs
  show a ~3x error from that, which is why the estimator is stated explicitly. Priors
  `{0.35, 0.6, 0.8, 0.92}` (Strata's controller corner).
- `expected_tokens(drafts, q) = 1 + sum_{i=1..drafts} q^i`.
- `LookupPolicy::choose(available, match, mtp_expected, force)`: returns the lookup extent to verify,
  `0` to keep the MTP drafts. `available` is the proposal length, `mtp_expected` the tokens the MTP
  drafts are predicted to commit (`MtpAcceptanceEstimate::expected_tokens`). Lookup is taken when its
  expected tokens exceed the MTP's by `margin` (default 0, so ties keep the already-paid MTP path),
  or unconditionally when `force` (the `always` measurement mode). A promising bucket the engine has
  not measured yet (`kProbes` = 3 rounds, `kProbeRate` = 0.80) is tried anyway, so a conservative
  prior cannot permanently veto a bucket it would win.

### 3.3 Integration

Phase 1 reuses `SpeculativeBackend::Mtp`; no new backend, no new kernel, no new graph shape.

- `SequenceState` gains the drafter (lazily allocated when lookup is enabled) and a `LookupPolicy`.
- At request begin (`install_sampling`) the drafter is reset; each MTP round syncs it from
  `sequence.ledger`, which already holds the full prompt plus committed tokens.
- In `decode_mtp_batch`, before the per-lane ingress fill, each lane asks the policy. A lane that
  chooses lookup fills `current_drafts` and `current_extents` from the proposal instead of
  `sequence.mtp_drafts`/`mtp_draft_count`; every other ingress field is unchanged, and the graph,
  `target_verify_accept`, and `mtp_prepare_next_round` run exactly as before.
- After the round, a lane that used lookup feeds its (drafted, accepted) into `LookupAcceptance` and
  **does not** update `sequence.mtp_acceptance`, so the MTP estimate is not polluted by lookup
  outcomes. MTP lanes behave exactly as today.
- Because one CUDA graph runs the whole MTP round, a lookup round still runs the MTP forward and
  proposal head. That is deliberate: Phase 1 measures the *draft source*, not a cheaper round. A
  lookup-only round is Phase 2.

### 3.4 Configuration

- `--lookup-drafts off|auto|always` (default `off`), valid only with `--spec mtp`.
- `--lookup-min-match N` (default 8, range 3..32): a proposal shorter than this is not used.
- Recommended A/B: `--fixed-draft` so the round width (and its cost) is constant; the adaptive MTP
  policy is not meaningful while the MTP estimate is deliberately not updated on lookup rounds.

### 3.5 Measurement

Phase 1 must produce, per model and workload, the acceptance of lookup drafts by match bucket and
the resulting tokens per round, next to the same figures for the MTP it replaced. These counters are
carried on `SpeculativeStats` and exposed through the existing speculative metrics:

- `lookup_rounds`, `lookup_drafted_tokens`, `lookup_accepted_tokens`
- `lookup_drafted_by_match_bucket`, `lookup_accepted_by_match_bucket` (4 buckets)

A/B harness: the edit/quote load (a file returned with a rename — where the drafter is designed to
win), a code load, and the existing long-reasoning and prose loads, at C=1..8, on Qwen3.8-Flash-Next
and Qwen3.6-35B-A3B, each with and without `--lookup-drafts`.

### 3.6 Wide lookup-only rounds (Phase 2)

The MTP layer drafts at most seven tokens, so a round that runs its draft phases cannot verify a
wider lookup proposal. When `--draft-tokens` is above 7 (which requires `--fixed-draft` and
`--lookup-drafts`), the program is **lookup-only**:

- The MTP round frame is planned at the requested width (up to `kLookupDecodeMaximumDrafts`, 15) and
  the ladder has one rung, so the graph family is a single wider verify.
- The round body skips `mtp_prepare_next_round`, the MTP layer forward and the proposal head, and
  reports no next drafts. The MTP layer's KV is therefore never read; its pages are still mapped and
  committed so the existing KV bookkeeping stays valid.
- When no lane has a proposal at or above `--lookup-min-match`, the round runs as an ordinary
  one-token round instead, so non-repeating text does not pay for an empty wide verify.

This reuses the entire MTP execution path (frame, graph capture, dispatch, ReplaySSM settle) and
adds no new backend. The cost is that lookup-only requests cannot fall back to MTP drafts, and the
target verify of a wide round is more expensive per round; the measurement below shows where that is
worth it.

### 3.7 Cost-aware selection (Phase 3)

A lookup round commits `E` tokens where an ordinary round commits one, but it also costs more. The
break-even is therefore `E > lookup_ms / ordinary_ms`, not `E > 1`:

- The engine measures the single-request wall time of an ordinary round and of a wide lookup round
  as EMAs (`lookup_round_seconds_`, `plain_round_seconds_`). Their ratio is model- and
  context-dependent: about 8 on host-resident Flash-Next, about 4 on device-resident 35B-A3B, and
  it is learned from the run rather than assumed.
- `LookupPolicy::choose(available, match, alternative_tokens, cost_ratio, force)` takes the lookup
  only when `E_lookup(n, q) > alternative_tokens * cost_ratio * (1 + margin)`. For the MTP hybrid
  the alternative is the MTP draft in the same round (`alternative_tokens` = MTP expected,
  `cost_ratio` = 1); for a lookup-only program it is an ordinary round (`alternative_tokens` = 1,
  `cost_ratio` = the measured ratio).
- The first round of a lookup-only request is an ordinary round, so the denominator is measured
  before anything is priced against it. Until the wide round's cost is also measured once, one
  proposal is taken per request to measure it; then the gate decides. A round with nothing worth
  verifying runs ordinary.
- Probing an unmeasured match bucket is itself a round, so it is skipped when the lookup round is
  more than `kMaxProbeCostRatio` (2) times an ordinary one.

What this fixes comes straight from the real request logs: on Flash-Next agent traffic lookup fires
on 0.1-2.5% of rounds with 5-22% acceptance and commits 1.75-4.27 tokens per wide round, which is a
net loss against a ratio near 8; on 35B-A3B the same gate accepts the deep matches (bucket 3 at
87-89% acceptance) that make the feature worth having.

## 4. Phases

| Phase | Deliverable | Decision it informs |
|---|---|---|
| **1 (done)** | Host index + policy + substitution in the MTP round + counters + tests | Whether lookup beats the MTP draft on any real load; acceptance by match bucket |
| **2 (done)** | Wider lookup-only rounds inside the MTP frame (window up to 15, MTP draft phases skipped, ordinary fallback) + tests | Whether a wider verify collects the deep matches, and for which models |
| **3 (done)** | Cost-aware lookup selection (corrected conditional acceptance + measured lookup/ordinary round-cost ratio) + tests | Whether to ship it on by default: reject rounds that cannot out-commit a cheaper round |

## 5. Work checklist (Phase 1)

- [x] Design plan (this document)
- [x] `SuffixDrafter` + unit tests
- [x] `LookupPolicy` + unit tests
- [x] Options: `SpeculativeOptions`, CLI, validation, plan plumbing
- [x] `SequenceState` index/policy lifecycle and per-round sync
- [x] `decode_mtp_batch` substitution and observation
- [x] `SpeculativeStats` counters surfaced in the request log
- [x] Build and run the affected test targets (full non-model suite passes)
- [x] A/B measurement on the edit/quote and reasoning/prose loads (see §9)
- [x] Phase 2: wide lookup-only rounds, ordinary fallback, validation and tests (see §3.6, §9.2)
- [x] Phase 3: conditional acceptance model + measured lookup/ordinary cost ratio gate (§3.7, §9.3)

Landing shape: `--spec mtp --draft-tokens K --fixed-draft --lookup-drafts auto` (or `always`) with
`--lookup-min-match N`. The counters appear per request under `speculative.lookup_rounds`,
`speculative.lookup_drafted_tokens`, `speculative.lookup_accepted_tokens`,
`speculative.lookup_drafted_by_match_bucket` and `speculative.lookup_accepted_by_match_bucket` in the
request log. `off` is bit-identical to the pre-change MTP path.

## 6. Verification

- Unit: index finds exact repeats, honors `min_match`/`max_match`/`max_drafts`, ignores the current
  suffix itself, is stable across `sync` boundaries; policy chooses by expected tokens and learns
  from outcomes.
- Behavioral: with lookup on, a lane's licensed tokens equal what the target would commit for those
  drafts — the accept path is the existing, already-qualified op.
- Regression: `--lookup-drafts off` must be bit-identical to today on the same seed; a request with
  no repeat must produce no proposals and no stats.
- Economics: the counters above on the A/B loads. This is the phase's acceptance evidence.

## 7. Risks and non-goals

- **Host-resident experts (Flash-Next):** a wider verify touches more distinct experts, so lookup
  needs a longer/denser match than on a device-resident model. If no load clears break-even, the
  finding is negative and the feature stays off; that is an acceptable Phase 1 outcome.
- **Graph width:** one width per round is shared by all lanes; a lane uses up to that many lookup
  drafts. Per-lane extent already exists (`current_extents[b]`).
- **Index lifetime:** the index must be reset with the sequence, and `sync` must be idempotent with
  respect to the ledger, or a reused continuation would draft from another request's tokens.
- **Non-goals:** changing sampling semantics (none — rejection sampling preserves the distribution),
  masking/quality changes, DFlash, and any device allocation.

## 8. Open questions

- Interleaving MTP and wide lookup in one request: lookup-only requests cannot use MTP drafts today
  because the MTP layer cannot run at the wide width, so a request with both sources is the
  remaining design work.
- The cost gate compares a whole wide round with a whole ordinary round. It does not yet price the
  extra distinct experts a wide verify fetches from the host expert cache as a function of the
  window, which Strata's controller models directly; the measured ratio captures it in aggregate.
- Should the index be bounded further for very long contexts (memory vs. coverage tradeoff)?

## 9. Results

### 9.1 Phase 1 (window within the MTP layer's seven drafts)

RTX 5090, CUDA graphs, FP8 KV, `--fixed-draft --draft-tokens 7`, greedy, thinking off, one
1791-token prompt (a header file returned with an identifier renamed), 400 new tokens. Flash-Next
used `--expert-cache auto` (8090 experts, 20.8 GiB). Same prompt and settings for 35B-A3B, which
has device-resident experts.

| Model / load | mode | decode tok/s | rounds | tok/round | lookup rounds | lookup acc. |
|---|---|---:|---:|---:|---:|---:|
| Flash-Next, edit | off | 118.7 / 121.7 | 56 | 7.11 (MTP) | – | – |
| Flash-Next, edit | `always` | 143.7 / 142.9 | 51 | 7.89 | 46 | 100% |
| Flash-Next, edit | `auto` before probing | 132.1 | 53 | 7.53 | 42 | 100% |
| Flash-Next, edit | `auto` with probing | 143.3 | 51 | 7.89 | 46 | 100% |
| Flash-Next, prose | off / `always` | 57.3 / 57.3 | 100 | 2.53 (MTP) | 0 | – |
| Flash-Next, prose | `auto` with probing | 58.7 | 100 | 2.53 (MTP) | 0 | – |
| 35B-A3B, edit | off | 971.7 | 53 | 7.53 (MTP) | – | – |
| 35B-A3B, edit | `always` | 1.01k | 51 | 7.96 | 46 | 100% |
| 35B-A3B, edit | `auto` before probing | 967.5 | 53 | 7.53 | 0 | – |
| 35B-A3B, edit | `auto` with probing | 994.8 | 51 | 7.96 | 46 | 100% |

Findings:

- **Correctness holds.** Greedy output is byte-identical across `off`/`auto`/`always` on both models
  and both prompts, so the substituted drafts do not change what the model commits.
- **Lookup wins on verbatim/echo text.** Flash-Next edit: 118.7/121.7 -> 143.7/142.9 tok/s
  (`always`, two runs each) and 56 -> 51 rounds; 35B-A3B edit: 971.7 -> 1.01k tok/s and 53 -> 51
  rounds. Lookup acceptance was 100%, all from match buckets 2 and 3 (12-23 and >=24 tokens).
- **No cost on novel text.** The prose load produced no proposal at all (no repeated suffix);
  `off`/`auto`/`always` are identical there, and probing caused no attempt. Enabling lookup cannot
  regress a non-repeating load.
- **A conservative prior needs probing, and cannot veto a bucket it never tries.** Before probing,
  `auto` scored 132.1 (Flash-Next) and 967.5 (35B-A3B): on 35B-A3B it never fired although `always`
  was a clean win, and on Flash-Next it used only bucket 3, because its bucket-3 prior (0.92)
  understated the measured ~1.0 and bucket 2 was never sampled. `LookupPolicy::kProbes` now verifies
  an unmeasured bucket at or above `kProbeRate` (0.80) for three rounds before the measured rate may
  veto it. `auto` then matches `always` on both models, still with no attempt on prose. This mirrors
  Strata's `kProbes` for the same reason.
- **The MTP frame is the ceiling.** Lookup matched 24+ tokens and accepted every draft, yet the
  round still verified at most seven. A lookup-only backend with a wider window (DFlash's domain is
  15) is where the deep-match advantage would actually be collected; Phase 1's gain is the part that
  survives sharing MTP's seven-draft frame.

Reproduce:

```bash
./build/apps/ninfer models/Qwen3.8-Flash-NVFP4/qwen3_8_flash_next_nvfp4.ninfer \
  --prompt "$(cat prompt.txt)" --max-context 32768 --kv-capacity auto --kv-dtype fp8 \
  --expert-cache auto --spec mtp --draft-tokens 7 --fixed-draft \
  --lookup-drafts off|auto|always --max-new 400 --greedy --no-thinking
```

### 9.2 Phase 2 (window above the MTP layer's seven drafts)

Same machine, prompt and settings, `--draft-tokens 15 --fixed-draft --lookup-drafts always`
(lookup-only). The prompt is returned verbatim by the model, so a 15-draft window accepts ~15.2-15.8
tokens per round; the depth probe on the Phase 1 output predicted this (the drafter proposed 15/15
bytes at 217/228 positions and 35% of positions would accept more than seven).

| Model / load | mode | decode tok/s | rounds | tok/round | ordinary fallback |
|---|---|---:|---:|---:|---:|
| Flash-Next, edit | plain (no spec) | 76.4 | 400 | 1.0 | – |
| Flash-Next, edit | MTP K7 | 118.7 / 121.7 | 56 | 7.11 | – |
| Flash-Next, edit | lookup K7 `always` | 143.7 / 142.9 | 51 | 7.89 | – |
| Flash-Next, edit | lookup K15 | 155.8 / 157.8 / 165.0 | 24 | 15.75 | – |
| Flash-Next, prose | plain | 95.9 | – | 1.0 | n/a |
| Flash-Next, prose | lookup K15 | 96.4 | 0 lookup | 1.0 | all rounds |
| 35B-A3B, edit | plain (no spec) | 390.1 | 400 | 1.0 | – |
| 35B-A3B, edit | MTP K7 | 971.7 | 53 | 7.53 | – |
| 35B-A3B, edit | lookup K7 `always` | 1.01k | 51 | 7.96 | – |
| 35B-A3B, edit | lookup K15 | 1.61k / 1.63k | 25 | 15.36 | – |

Findings:

- **Correctness on this load.** The greedy output is byte-identical to plain decode on both models:
  Flash-Next K15 == plain, 35B-A3B K15 == plain. Lookup drafts are verified by the same accept path,
  so they cannot change the model's distribution; a near-tie-heavy input can still round differently
  from a one-column decode (§9.3).
- **Device-resident experts want the wide window; host-resident experts do not much.** 35B-A3B
  (experts in Device memory) gains most of all: 390 -> 1,610 tok/s, **x4.1** over plain and **+66%**
  over MTP K7, because its round cost grows with compute while the committed tokens grow with the
  window. Flash-Next (experts in Host memory behind the PCIe expert cache) gains 118.7 -> 157.8
  (**+33%** over MTP K7, +107% over plain): the wider verify also widens the distinct-expert set it
  must fetch, so the round cost grows nearly with the window and the gain is bounded.
- **A wide window is harmless on non-repeating text** because of the ordinary fallback: Flash-Next
  prose with K15 = 96.4 tok/s, byte-identical to plain decode (95.9). Without the fallback the same
  request ran 80.6 tok/s (empty wide verifies).
- **The MTP path itself is not bit-identical to plain decode on prose.** Flash-Next prose with MTP
  K7 diverges from plain decode at byte 210 while the lookup-only run (all ordinary) matches plain
  exactly. That is the MTP backend's own batched-verify near-tie, not a lookup result: the Phase 2
  runs exercised zero lookup rounds on that text. It is pre-existing and worth a separate look.

Reproduce:

```bash
./build/apps/ninfer models/Qwen3.8-Flash-NVFP4/qwen3_8_flash_next_nvfp4.ninfer \
  --prompt "$(cat prompt.txt)" --max-context 32768 --kv-capacity auto --kv-dtype fp8 \
  --expert-cache auto --spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts always \
  --max-new 400 --greedy --no-thinking
```

### 9.3 Phase 3: real-task logs and the cost gate

Two real agentic logs (`ninfer-serve`, OpenAI chat completions, tools, thinking mixed,
`--spec mtp --draft-tokens 15 --fixed-draft --lookup-drafts always`); the 35B-A3B server ran
`--max-concurrency 4`.

| Model | lookup rounds/request | acceptance | E per lookup round | decode tok/s |
|---|---:|---:|---:|---:|
| Flash-Next (35k prompt, 77 tools) | 8-62 | 5-22% (bucket 1) | 1.75-4.27 | 70-77 |
| 35B-A3B | 8-628 | 22-66% (bucket 3: 87-89%) | 4.09-11.0 | 359-576 |

What the logs establish:

- **The acceptance estimator was wrong by ~3x.** `accepted / drafted` gave 0.16 where the realized
  mean accepted per round implies `q ~ 0.7`. The estimator is now the conditional rate
  (`accepted / (accepted + partial_rounds)`), so `E` matches the measured tokens per round; without
  this the gate would have rejected profitable 35B rounds.
- **The gate is model-appropriate.** Flash-Next's measured ratio is near 8, so bucket-1 rounds
  (E 1.75-4.27) are rejected; 35B-A3B's is near 4, so its bucket-2/3 rounds (E up to 11) are taken.
- **Validated A/B:** on the 35B edit load `auto` matches `always` (1.56k vs 1.61k tok/s); on prose
  forced to short matches (`--lookup-min-match 3`) `auto` runs one bootstrap lookup, then rejects
  (91.5 vs 95.5 tok/s plain, versus 86.1 for `always`, which kept burning wide rounds).

**Correctness scope.** Speculative decoding preserves the model's *distribution*, not always the
byte-identical greedy text: a 15-column target verify can round a near-tie differently from a
one-column decode. On the unambiguous edit load `auto`/`always`/plain are byte-identical (§9.2); on
prose with forced short matches the lookup round flips an early near-tie, exactly as the MTP path
already does (§9.2, last finding). The feature does not change the distribution.

**Confirmation with `auto`.** Re-running the same coding task with `--lookup-drafts auto` on both
models:

| Model | lookup rounds | share of rounds | acceptance | E per lookup round | rounds saved vs plain |
|---|---:|---:|---:|---:|---:|
| Flash-Next | 49 | 0.25% | 62.7% | 10.47 | 464 (2.4%) |
| 35B-A3B | 814 | 3.68% | 69.0% | 11.34 | 8416 (38%) |

The gate changed *which* rounds run, not just how many: acceptance rose from 5-22% to 62.7% on
Flash-Next because `auto` rejects the short-match rounds (E ~3 was below its ratio near 8) and keeps
the bucket-3 ones (E ~10), so the 49 wide rounds now pay for themselves where `always` was a net
loss. On 35B-A3B it keeps the same deep matches and the 38% round reduction stands.

`auto` is now the recommended mode; `always` remains the measurement override.

### 9.4 Pool design evaluation

The n-gram chain plan (`ngram-chain-mtp-plan.md`, kept on the desktop) proposes replacing the
suffix index with the fork's pool: `hash(last n tokens) -> the token that most recently followed
that window` (last-writer-wins, fixed table, 14-bit tag, no probing). `tools/spec_sim/ngram.py`
replays token ledgers through both designs so the choice is made before engine work. Self-replay
makes tokens/round an upper bound; the comparison, and the gate-profitability counts, are the point.

Ledgers are built by tokenizing repository sources with the local Qwen3.8-27B tokenizer (248,044
entries) — the closest available to Flash-Next's 248,320, so a design comparison and not a numeric
oracle. `edit` is a header followed by its lightly-renamed copy, `code` is several sources, `prose`
is the docs.

| Ledger | design | firings | accepted/firing | gate-profitable firings E>8 | tokens from E>8 |
|---|---|---:|---:|---:|---:|
| edit | trigram (shipped) | 221 | 6.46 | 87 | **1347** |
| edit | pool n=8 | 108 | **11.61** | 83 | 1276 |
| code | trigram | 9147 | 2.21 | 731 | **9403** |
| code | pool n=8 | 2466 | **3.59** | 385 | 5003 |
| prose | trigram | 2660 | 1.11 | 74 | **985** |
| prose | pool n=8 | 233 | **3.92** | 46 | 627 |

Sweeping the pool key length n = 4..12 does not close the gap: no n beats the trigram on E>8 tokens
(edit 972-1276 vs 1347; code 3692-6638 vs 9403). Every pool miss is `empty` (the n-gram was never
observed), not a tag collision (1-161 out of tens of thousands), so a longer key trades recall for
precision that the gate already supplies: the trigram's noisy firings are rejected by the cost gate
and cost only a host check, while the pool's lost firings are simply gone.

**Decision: keep the trigram index; do not adopt the last-follower pool as a replacement.** It is a
subset of the trigram's firings with no additional recall, and its precision advantage does not
translate into more committed tokens.

The decision holds on a **real ledger**. `--generation-token-trace-jsonl` now records
`prompt_token_ids` alongside the generated ids (schema version 2, `prompt.token_ids()` on
`PreparedPrompt`), so `ngram.py --trace FILE` replays an actual served request instead of a
tokenized fixture. A live 35B-A3B edit request (1789 prompt + 300 generated tokens) gives trigram
188 firings / 1.84 accepted per firing / 171 E>8 tokens against the pool's 25 / 4.80 / 84 — the same
ordering, on the engine's own tokenization.

**Chaining was implemented and rejected (2026-09-30).** The plan's other half — a round whose
drafts are `[MTP][suffix continuation]` — was built behind `--chain-drafts` and measured on
35B-A3B, 400 tokens, greedy: chain K=7 gave +2-4.5% over MTP K=7, but the wide lookup window
dominated everywhere (edit 1560 lookup vs 1440 chain-15 vs 944 MTP; mixed 1540 / 1440 / 960; prose
394 / 290 / 346). The wide chain needed the MTP alignment forward and `mtp_prepare_next_round`
raised from 8 to 16 columns, and the MTP part capped at seven. It was reverted because on copied
text the pool alone beats MTP-then-pool and on novel text a wide rung is wasted; a chained round
would need interleaved copy/novel text inside a window *and* a rung that falls back to narrow when
the chain is short. Output was greedy-identical to MTP on the copy and mixed loads. Do not rebuild
it without a workload that shows the pool-only wide mode losing.
