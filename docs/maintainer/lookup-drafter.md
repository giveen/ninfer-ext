# Prompt-lookup speculative decoding (suffix drafter)

Status: **Phase 1 in progress.** This is the active work record for porting the prompt-lookup
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

- `LookupAcceptance`: per-bucket probability `q` that a draft from a match of that length is
  accepted, EMA-updated from (drafted, accepted) per round. Priors `{0.35, 0.6, 0.8, 0.92}`
  (Strata's controller priors; a conservative corner of the measured range).
- `expected_tokens(drafts, q) = 1 + sum_{i=1..drafts} q^i`.
- `LookupPolicy::choose(available, match, mtp_expected, force)`: returns the lookup extent to verify,
  `0` to keep the MTP drafts. `available` is the proposal length, `mtp_expected` the tokens the MTP
  drafts are predicted to commit (`MtpAcceptanceEstimate::expected_tokens`). Lookup is taken when its
  expected tokens exceed the MTP's by `margin` (default 0, so ties keep the already-paid MTP path),
  or unconditionally when `force` (the `always` measurement mode).

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

## 4. Phases

| Phase | Deliverable | Decision it informs |
|---|---|---|
| **1 (active)** | Host index + policy + substitution in the MTP round + counters + tests | Whether lookup beats the MTP draft on any real load; acceptance by match bucket |
| 2 | Lookup-only verify backend (no MTP forward/head), its own graph family | Whether the round can be made cheaper than MTP |
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
- [ ] A/B measurement on the edit/quote and reasoning/prose loads (needs a real artifact)

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

- Does the adaptive MTP policy need to keep updating while lookup is on? Phase 2 should decide.
- Should the index be bounded further for very long contexts (memory vs. coverage tradeoff)?
