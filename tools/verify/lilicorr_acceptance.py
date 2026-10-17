"""Measures the accepted length a LiLiCorr drafter would reach on this target.

It reads a tap dump produced by `ninfer_nemotron_h_draft_taps_test` -- the target's own residual
stream at the drafter's six tapped blocks, one file per block, plus the target's greedy stream -- and
runs the draft's reference implementation (`tools/verify/lilicorr_reference.py`) once per round a
speculative loop would take. Each round's proposal is compared against the target's own greedy
stream, so what comes out is the accepted length of *this* pair: this engine's converted NVFP4
target, conditioning on these taps, with this drafter's weights.

Why this exists: the pairing is the risk, not the arithmetic. The drafter ships as the companion of
the 24 GB target, whose experts are 2-bit `IQ2_XXS` `QAT`, while this engine's target is an NVFP4
W4A16 conversion from the BF16 checkpoint. A drafter trained against a differently post-trained or
differently quantized target still runs, still proposes plausible tokens and accepts almost nothing
-- which is what happened to the Gemma 4 drafter here. Measuring the accepted length answers that
before any of the draft's CUDA exists.

The probe is greedy: the drafter's proposal is a point mass per slot (the paper's own mode), and the
target's stream is what a greedy decode committed. The card's own 4.30 average is a
temperature-1, top-p-0.95 SPEED-Bench figure over rejection sampling, so it is not the same quantity
-- read the reported tokens-per-round as the number to compare, and see the summary's own note.

    python3 tools/verify/lilicorr_acceptance.py --dump /tmp/taps \\
        --drafter /mnt/storage/models/nemo/74B/LiLiCorr --target /mnt/storage/models/nemo/74B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lilicorr_reference as ref  # noqa: E402  (the reference implementation, imported as the oracle)


def load_dump(directory: Path) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    """The dump's metadata, its token stream, and one `[positions, hidden]` array per tap."""
    meta = json.loads((directory / "meta.json").read_text())
    tokens = np.fromfile(directory / "tokens.i32", dtype=np.int32)
    if tokens.size != meta["positions"]:
        raise ValueError(f"tokens.i32 holds {tokens.size} ids, meta says {meta['positions']}")
    taps = []
    for layer in meta["layers"]:
        raw = np.fromfile(directory / f"tap_{layer}.bf16", dtype=np.uint16)
        if raw.size != meta["positions"] * meta["hidden"]:
            raise ValueError(f"tap_{layer}.bf16 is not [{meta['positions']}, {meta['hidden']}]")
        # The composition's own residual order: channels contiguous, the token axis strided.
        bits = raw.astype(np.uint32) << 16
        taps.append(bits.view(np.float32).reshape(meta["positions"], meta["hidden"]))
    return meta, tokens, taps


def make_inputs(meta: dict, tokens: np.ndarray, taps: list[np.ndarray], frontier: int,
                slots: int, window: int, mask_id: int) -> tuple[list[torch.Tensor], torch.Tensor, dict]:
    """The drafter's inputs for one round.

    `frontier` is the last *committed* position, and the round's anchor is the bonus token the
    target has already produced at `frontier + 1`: the block's row 0 is that real token id, and the
    `slots` mask rows that follow sit at `frontier + 2 ... frontier + 1 + slots`, each predicting the
    token at its own position. The context is the committed positions up to `frontier` -- the bonus
    row is a *query* row, so its K/V comes from the block's own projections -- and only the live
    window of it is passed, because an older row is invisible to the block anyway.
    """
    first = max(0, frontier + 1 - window)
    context = [tap[first : frontier + 1, :] for tap in taps]
    block = np.concatenate(
        [
            np.array([tokens[frontier + 1]], dtype=np.int32),
            np.full(slots, mask_id, dtype=np.int32),
        ]
    )
    inputs = {
        "taps": [torch.from_numpy(rows).unsqueeze(0).to(torch.float32) for rows in context],
        "block_ids": torch.from_numpy(block).unsqueeze(0).to(torch.long),
        "context_positions": torch.arange(first, frontier + 1).unsqueeze(0),
        "block_positions": torch.arange(frontier + 1, frontier + 2 + slots).unsqueeze(0),
        "context": frontier + 1 - first,
        "slots": slots,
        "window": window,
    }
    return inputs["taps"], inputs["block_ids"], inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path, help="a tap dump directory")
    parser.add_argument("--drafter", required=True, type=Path, help="the LiLiCorr checkpoint")
    parser.add_argument("--target", type=Path, default=None,
                        help="the target checkpoint's directory (its embedding table and LM head)")
    parser.add_argument("--slots", type=int, default=7, help="draft length (the card's is 7)")
    parser.add_argument("--window", type=int, default=None,
                        help="the draft's sliding window; default: the checkpoint's own")
    parser.add_argument("--start", type=int, default=None,
                        help="the first frontier position; default: the dump's prompt boundary")
    parser.add_argument("--rounds", type=int, default=None, help="stop after this many rounds")
    parser.add_argument("--progress", type=int, default=25, help="print every N rounds; 0 is silent")
    parser.add_argument("--samples", type=int, default=0,
                        help="print this many rounds' proposals beside the target's own tokens")
    parser.add_argument("--diagnose", type=int, default=0,
                        help="print the first N rounds' candidate ranks under the drafter's hidden")
    parser.add_argument("--propose-from-target", action="store_true",
                        help="skip the drafter and propose the target's own tokens: the round loop "
                             "checked against a proposer that cannot be wrong")
    parser.add_argument("--self-check", action="store_true",
                        help="run the loop with --propose-from-target and assert maximal acceptance")
    args = parser.parse_args()

    meta, tokens, taps = load_dump(args.dump)
    config = json.loads((args.drafter / "config.json").read_text())
    dflash = config["dflash_config"]
    window = args.window if args.window is not None else int(dflash["swa_window_size"])
    tap_ids = [int(x) for x in dflash["target_layer_ids"]]
    if tap_ids != [int(x) for x in meta["layers"]]:
        # The dump's own order is what the concatenation uses; a different *set* of blocks is an
        # experiment (the boundary convention is a reading of the reference, not a fact), so it is
        # reported rather than refused.
        print(f"note: the dump taps {meta['layers']}, the checkpoint declares {tap_ids}")
    if int(config["hidden_size"]) != int(meta["hidden"]):
        raise ValueError("the dump's tap width is not the drafter's hidden size")
    slots = args.slots
    if not 1 <= slots < int(config["block_size"]):
        raise ValueError(f"a draft length of {slots} is outside the checkpoint's block size "
                         f"{config['block_size']}")

    # The drafter is only needed by the arm that runs it: the loop's own check proposes from the
    # target and loads nothing, which is what makes it instant.
    if args.propose_from_target or args.self_check:
        weights = None
        tables = None
        head = None
        body = None
    else:
        weights_path = args.drafter / "model.safetensors" if args.drafter.is_dir() else args.drafter
        weights = ref.Checkpoint(weights_path)
        tables = ref.TargetTables(args.target, int(config["vocab_size"]),
                                  int(config["hidden_size"]), seed=0)
        head = ref.CorrelatorHead(weights, config, ref.Trace())
        body = ref.DraftBody(weights, config, ref.Trace())
        print(f"drafter: {args.drafter} | head parameters resolved | target tables: "
              f"{tables.source}")

    positions = int(meta["positions"])
    first_frontier = args.start if args.start is not None else int(meta["prompt_tokens"]) - 1
    accepted_slots = np.zeros(slots, dtype=np.int64)
    reached_slots = np.zeros(slots, dtype=np.int64)
    accepted_total = 0
    rounds = 0
    committed = 0
    started = time.time()

    frontier = first_frontier
    while frontier + 1 < positions:
        if args.rounds is not None and rounds >= args.rounds:
            break
        if args.propose_from_target or args.self_check:
            # The round loop's own check: a proposer that returns the target's tokens must be
            # accepted in full, every round, so what the accounting reports is the loop's and not
            # the drafter's. No weights, no forward -- this arm is instant.
            proposed = tokens[frontier + 2 : frontier + 2 + slots].tolist()
        else:
            taps_in, block_ids, inputs = make_inputs(meta, tokens, taps, frontier, slots, window,
                                                     int(dflash["mask_token_id"]))
            embeddings = tables.embedding_rows(block_ids)
            stage = body.forward(taps_in, embeddings, inputs["context_positions"],
                                 inputs["block_positions"], slots)
            logits = tables.logits(stage["pass_hidden"])
            ids, log_probs = head.candidates(logits)
            scored = head.score(tables.embedding_rows(ids), log_probs, stage["pass_hidden"],
                                stage["anchor_hidden"], stage["anchor_valid"])
            proposed = head.walk(scored["scores"], ids)["ids"][0].tolist()

        accepted = 0
        for index, token in enumerate(proposed):
            if frontier + 2 + index >= positions:
                break
            if int(token) != int(tokens[frontier + 2 + index]):
                break
            accepted += 1
        if args.diagnose and rounds < args.diagnose:
            # Where the target's own tokens sit under the *drafter's* hidden state, through the
            # target's own head: a drafter whose conditioning works ranks them near the top, and one
            # whose hidden carries nothing ranks them uniformly.
            want = torch.from_numpy(tokens[frontier + 2 : frontier + 2 + slots].astype(np.int64))
            rows = logits[0].detach().float()
            if rows.shape[0] != want.numel():
                raise ValueError(f"the head produced {rows.shape[0]} rows for {want.numel()} slots")
            drawn = rows.gather(1, want.unsqueeze(1)).squeeze(1)
            ranks = (rows > drawn.unsqueeze(1)).sum(dim=1).numpy()
            top1 = rows.argmax(dim=1).numpy()
            in_top = [bool((ids[0, index] == want[index]).any()) for index in range(slots)]
            print(f"  frontier {frontier} | target {want.tolist()}")
            print(f"    drafter top-1 {top1.tolist()}")
            print(f"    rank of the target's token {ranks.tolist()} | in the top-8 {in_top}")
            print(f"    walked {proposed} "
                  f"(top-8 at slot 0: {ids[0, 0].tolist()})")
        if args.samples and rounds < args.samples:
            print(f"  frontier {frontier} | bonus {int(tokens[frontier + 1])} | proposed "
                  f"{proposed[:4]} | target {tokens[frontier + 2 : frontier + 6].tolist()} | "
                  f"accepted {accepted}")
        for index in range(slots):
            reached_slots[index] += 1
            if index < accepted:
                accepted_slots[index] += 1
        accepted_total += accepted
        committed += accepted + 1
        rounds += 1
        frontier += accepted + 1
        if args.progress and rounds % args.progress == 0:
            print(f"  round {rounds}: frontier {frontier} | accepted {accepted} | "
                  f"mean accepted so far {accepted_total / rounds:.2f} | "
                  f"{time.time() - started:.0f} s")

    if rounds == 0:
        print("no round ran: the dump is shorter than the first frontier")
        return 1

    if args.self_check:
        # What the loop owes: every position after the dump's first frontier consumed exactly once
        # (that frontier's own token is already committed, which is what its bonus row feeds), one
        # round per commit, and `accepted + 1` committed per round -- the accepted drafts plus the
        # target's own correction or bonus.
        expected = positions - (first_frontier + 1)
        full_slots = all(accepted_slots[index] == reached_slots[index] for index in range(slots - 1))
        ok = committed == expected and accepted_total == committed - rounds and full_slots
        print(f"self-check: committed {committed} of {expected} positions in {rounds} rounds | "
              f"accepted {accepted_total}, which must be committed - rounds = {committed - rounds} | "
              f"every slot but the last accepted in full: {full_slots} | {'ok' if ok else 'FAIL'}")
        return 0 if ok else 1

    print(f"\nrounds {rounds} | positions used {frontier - first_frontier} | "
          f"wall {time.time() - started:.1f} s")
    print(f"accepted drafts per round: {accepted_total / rounds:.3f}")
    print(f"committed tokens per round: {committed / rounds:.3f}  "
      "(accepted + the correction or bonus the target always supplies)")
    print("per-slot acceptance (slot i, given the round reached it):")
    for index in range(slots):
        rate = accepted_slots[index] / reached_slots[index] if reached_slots[index] else 0.0
        print(f"  slot {index}: {rate:.3f} over {reached_slots[index]} rounds")
    print("\nread the tokens-per-round figure, not the accepted-drafts one, against the card's "
          "4.30: that number is a temperature-1 SPEED-Bench average over rejection sampling, while "
          "this run is greedy, so the two are different quantities. This run's pair is also this "
          "engine's own quantization of the target, not the 2-bit QAT variant the drafter shipped "
          "against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
