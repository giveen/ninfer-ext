# Prompt-lookup speculative decoding (suffix drafter)

Status: **Phase 1 and Phase 2 implemented.** This is the active work record for porting the
prompt-lookup (suffix) draft source that Strata uses (`src/spec/suffix_drafter.cpp`,
`draft_policy.cpp`, `controller.cpp`). It states the design, the phases, and the evidence each phase
must produce. It is not yet a stable contract; promote the final design into the model/runtime
reference when the work settles.

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

- `LookupAcceptance`: per-bucket probability `q` that a draft from a match of that length is
  accepted, EMA-updated from (drafted, accepted) per round. Priors `{0.35, 0.6, 0.8, 0.92}`
  (Strata's controller priors; a conservative corner of the measured range).
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

## 4. Phases

| Phase | Deliverable | Decision it informs |
|---|---|---|
| **1 (done)** | Host index + policy + substitution in the MTP round + counters + tests | Whether lookup beats the MTP draft on any real load; acceptance by match bucket |
| **2 (done)** | Wider lookup-only rounds inside the MTP frame (window up to 15, MTP draft phases skipped, ordinary fallback) + tests | Whether a wider verify collects the deep matches, and for which models |
| 3 | Joint `{none, lookup, MTP}` source selection with measured per-source round costs | Whether to ship it on by default, and per-model gating |

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

- Phase 3 should choose between MTP and lookup per round; lookup-only requests cannot use MTP
  drafts today, and a request with both is the remaining design work.
- The wide round's cost on host-resident experts is not yet modeled: the policy should predict the
  distinct-expert cost of the window, as Strata's controller does, instead of always preferring the
  deepest match.
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

- **Correctness holds.** The greedy output is byte-identical to plain decode on both models:
  Flash-Next K15 == plain, 35B-A3B K15 == plain. Lookup drafts are verified by the same accept path,
  so the committed text does not change.
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
