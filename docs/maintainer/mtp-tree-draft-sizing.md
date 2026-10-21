# MTP draft-tree sizing

Sizing for replacing the MTP draft chain with a tree, on Qwen3.8-Flash-Next (Qwen4Exp) on one
RTX 5090. It is a decision document, not a plan of record: the gate in "Decide first" has to pass
before the work in "What is missing" is worth doing.

## Where the round stands

Measured warm on the development machine, greedy, 32k context, `--spec mtp --draft-tokens 3`
(2026-10-21):

| | ms/round | notes |
|---|---:|---|
| verify pass (T=4) | ~14.6 | 51 MoE layers, expert staging inside |
| three draft steps | ~4.6 | one MTP layer each, ~1.5 ms per step |
| head, commit, sample | ~1.5 | the shortlist head is 0.07 of it |
| **round** | **20.7** | 2.27 accepted tokens, 109.6 tok/s |

Plain decode (no speculation) is 10.7 ms per token, so a round is worth it while it commits
2.27 tokens; break-even is 1.93.

Acceptance is what limits the round, and it decays with position: 0.59, 0.33, 0.18 from the
242-round trace. The third draft costs ~1.5 ms for 0.18 tokens, i.e. 8.3 ms per accepted token,
which is still inside the 9.1 ms/token the round already pays, so the chain is at its own optimum
(K=1/2/4 all measured worse: 105.5 / 101.9 / 91.4 tok/s).

## What already exists

The repository already carries the pieces a tree needs, for EAGLE3:

- `--tree-width` in `SpeculativeOptions`, refused for every backend but EAGLE3
  (`src/product/speculative_options.h:58`, `program/planning/startup.cpp:1248`), and refused
  outright today: "EAGLE3 draft-tree widths above 1 are not implemented yet"
  (`startup.cpp:1383`).
- `ops::mask_selected_logits` — repeated argmax with masking, documented as the way a draft tree's
  top-k first tokens are produced (`include/ninfer/ops/speculative_round.h`).
- `TextContext::eagle3_propose_topk_batch` — top-k first tokens straight into `[top * batch]`.
- `ops::select_tree_winners` — "picks each sequence's winning draft chain by accepted count" over
  `[batch * tree_width]`.
- `ops::speculative_accept_sparse_drafts` — a 16-candidate accept with `candidate_ids`/`proposal_q`.

The shape this implies is **`tree_width` chains in the batch dimension**, not one shared tree with
an ancestor mask. That matters because `causal_softmax_attention` masks a live prefix per row
(`include/ninfer/ops/softmax_attention.h`); a per-node ancestor set would need a new attention
mask, whereas independent chains reuse the existing per-row prefix mask and the existing
per-row cache positions.

## Decide first: is the top of the distribution worth branching?

The tree pays only if the target's argmax at a position is often the draft's *second* choice. The
gate measurement is one instrumented run: capture the draft head's top-2 (post-shortlist) at draft
position 1 alongside the verify's target argmax, over a few hundred rounds, and report

- `P(target == top1)` — already known in aggregate: 0.59;
- `P(target == top2)` — unknown, and the whole question;
- the same pair at position 2 for the second level.

Expected outcome and what it decides:

| `P(target == top2)` at position 1 | verdict |
|---|---|
| < 0.12 | stop. Two chains cannot return the ~2.2 ms per chain they add. |
| 0.12–0.20 | marginal; only a depth-2, width-2 tree has a chance, and it needs its own A/B. |
| > 0.20 | worth implementing; acceptance at position 1 could move 0.59 → ~0.75. |

The instrumentation is a temporary capture of the proposal logits' top-2 (the shortlist rows are
already there) and the verify's `target_tokens`, neither of which needs a new op.

## Cost model

Two chains of depth 2 (verify width 2x3, two draft steps per chain), using the measured per-pass
costs and assuming the second chain adds ~70% of a pass (latency-bound, experts already staged):

| shape | round | accepted | tok/s |
|---|---:|---:|---:|
| chain K=3 (today) | 20.7 | 2.27 | 110 |
| W=2, depth 2 | ~24 | 2.1–2.4 | 88–100 |
| W=2, depth 3 | ~29 | 2.4–2.8 | 83–97 |

That is the honest arithmetic: **at today's per-pass costs a tree does not obviously win**, because
the verify pass is not the cheap part — it is 70% of the round, and duplicating columns keeps the
MoE staging per layer at 51 calls while adding draft steps. The tree wins only where a wider
verify is nearly free, and that is not this model at this point in its optimization.

A tree becomes attractive if the expert-staging cost per round falls (fewer calls per round or
cheaper staging), because then the verify width stops being the dominant term.

## What is missing (if the gate passes)

1. **Gate change** — allow `tree_width > 1` for MTP in `speculative_options.h:58` and
   `startup.cpp:1248`, with the plan carrying W into the frame capacity.
2. **Draft frontier** — `execution/mtp.cpp` and `program/speculative/mtp.cpp` draft one column per
   step. A tree expands a frontier: position 1 from `mask_selected_logits` top-W, later levels from
   each surviving chain's argmax. Each chain needs its own draft KV; the MTP KV is per row today and
   would be sized `B * W`.
3. **Verify at batch W** — the round driver in `program/decode.cpp` (1296 lines: lanes, rung
   selection, host egress, stats) runs one chain per lane. It would run W chains per lane with
   per-chain cache positions, and the policy's acceptance estimate would need a per-chain form.
4. **Accept and commit the winner** — `select_tree_winners` picks the chain; the commit then has to
   gather that chain's accepted KV rows instead of its own, and the losing chains' KV rows must be
   released. This is the correctness-critical part: `commit_sequence_kv` currently commits the
   lane's own speculative rows.
5. **Statistics and the policy** — `accepted_per_position` becomes per-position over the winning
   chain; `MtpDraftPolicy` selects a rung over (K, W) pairs rather than K alone.
6. **Tests** — the FP64/oracle form of the accept path, a chains-equal-single-chain test (W=1 must
   reproduce today's output bit for bit), and a winner-selection test where the better chain is not
   first.

## Risks

- **KV commit of a selected chain** is where a silent wrong answer lives: a gather of the wrong rows
  produces plausible text with corrupted state. The W=1 bit-exact test is the guard.
- **Graph capture width** — the frame is captured per width; W chains multiply the captured
  columns and the plan's frame capacity, which can cost residency (a smaller expert cache), eating
  the gain.
- **Effort** — items 1–6 touch the plan, the draft, the round driver and the commit path; on this
  codebase that is a week-plus, and the cost model above does not promise a win at the end.

## Recommendation

Run the gate measurement first. If `P(target == top2)` is under ~0.12, stop: the chain is at its
optimum and the remaining speed is in expert staging, not drafting. If it is above ~0.2, implement
W=2 depth-2 behind a flag, keep the W=1 bit-exact test, and accept only if the A/B on the same
workload beats 110 tok/s by more than the ±3% run-to-run spread.
