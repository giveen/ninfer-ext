#!/usr/bin/env python3
"""Offline payoff simulation for the n-gram / suffix draft source.

Replays token ledgers through two host drafter designs so the pool design is chosen before any
engine work:

  trigram        the shipped SuffixDrafter: trigram key, up to 4 recent end positions, backward
                 extension to ``max_match``.
  last_follower  the fork's pool: a fixed table mapping the hash of the last ``n`` tokens to the
                 token that most recently followed that window (last-writer-wins, 14-bit tag,
                 no probing).

The replay is self-referential: the ledger is the model's own output, so "accepted" means the
drafter reproduced text the model actually emitted. That makes tokens/round an upper bound, not a
prediction; the point is the *comparison* between the two designs under the same ledgers.

Usage:
  .venv/bin/python tools/spec_sim/ngram.py [--ledger edit|code|prose|all] [--max-k N] [--limit N]
  .venv/bin/python tools/spec_sim/ngram.py --trace served-tokens.jsonl

`--trace` replays the prompt and generated token IDs recorded by
`ninfer-serve --generation-token-trace-jsonl`, one ledger per request.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# The only local Qwen tokenizer is Qwen3.8-27B (248,044 entries); Flash-Next's is 248,320. Close
# enough for a design comparison, not for a correctness oracle.
DEFAULT_TOKENIZER = (
    "/mnt/storage/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json"
)

MASK64 = (1 << 64) - 1
M = 6364136223846793005
TAG_BITS = 14
TOKEN_BITS = 18  # token domain <= 2^18 - 1


def fmix64(x: int) -> int:
    x ^= x >> 33
    x = (x * 0xff51afd7ed558ccd) & MASK64
    x ^= x >> 33
    x = (x * 0xc4ceb9fe1a85ec53) & MASK64
    x ^= x >> 33
    return x


class TrigramIndex:
    """The shipped drafter (src/models/qwen3_5/program/speculative/suffix_drafter.cpp)."""

    WAYS = 4
    MIN_MATCH = 3
    MAX_MATCH = 32

    def __init__(self):
        self.table: dict[int, list[int]] = {}
        self.history: list[int] = []

    def _key(self, end: int) -> int:
        a, b, c = self.history[end - 2], self.history[end - 1], self.history[end]
        return ((a * 0x9E3779B97F4A7C15) ^ fmix64(b + 0x632BE59BD9B4E019) ^ (c << 1)) | 1

    def observe(self, token: int) -> None:
        self.history.append(token)
        end = len(self.history) - 1
        if end < 2:
            return
        lst = self.table.setdefault(self._key(end), [])
        lst.insert(0, end)
        del lst[self.WAYS:]

    def propose(self, max_k: int) -> tuple[list[int], int]:
        n = len(self.history)
        if n < 4 or max_k == 0:
            return [], 0
        cur = n - 1
        lst = self.table.get(self._key(cur))
        if not lst:
            return [], 0
        best_end = best_len = 0
        for p in lst:
            if p >= cur:
                continue
            ln = 0
            while ln < self.MAX_MATCH and ln <= p and self.history[p - ln] == self.history[cur - ln]:
                ln += 1
            if ln > best_len:
                best_len, best_end = ln, p
        if best_len < self.MIN_MATCH:
            return [], 0
        out = []
        for q in range(best_end + 1, min(cur, best_end + max_k) + 1):
            out.append(self.history[q])
        return out, best_len


class LastFollowerPool:
    """The fork's pool: hash(last n tokens) -> most recent following token."""

    def __init__(self, n: int = 8, entries: int = 1 << 22):
        self.n = n
        self.entries = entries
        self.mask = entries - 1
        self.table = [0] * entries  # 0 = empty
        self.window: collections.deque[int] = collections.deque(maxlen=n)
        self.hash = 0
        self.filled = 0
        self.mn = pow(M, n, 1 << 64)

    def _roll(self, entering: int) -> None:
        if len(self.window) == self.n:
            leaving = self.window[0]
            self.hash = (self.hash * M + entering - leaving * self.mn) & MASK64
        else:
            self.hash = (self.hash * M + entering) & MASK64
        self.window.append(entering)

    def _slot(self) -> int:
        return fmix64(self.hash) & self.mask

    def observe(self, token: int) -> None:
        # Record (window -> token) for the window that ended just before `token`.
        if len(self.window) == self.n:
            h = self.hash
            tag = (h >> (64 - TAG_BITS)) & ((1 << TAG_BITS) - 1)
            slot = fmix64(h) & self.mask
            if self.table[slot] == 0:
                self.filled += 1
            self.table[slot] = (tag << TOKEN_BITS) | (token + 1)
        self._roll(token)

    def propose(self, max_k: int):
        if len(self.window) < self.n or max_k == 0:
            return [], 0, None
        # Copy the rolling state so a proposal does not mutate the observed history.
        h = self.hash
        window = collections.deque(self.window, maxlen=self.n)
        out: list[int] = []
        miss = None
        while len(out) < max_k:
            slot = fmix64(h) & self.mask
            entry = self.table[slot]
            tag = (h >> (64 - TAG_BITS)) & ((1 << TAG_BITS) - 1)
            if entry == 0:
                miss = "empty"
                break
            if (entry >> TOKEN_BITS) != tag:
                miss = "collision"
                break
            token = (entry & ((1 << TOKEN_BITS) - 1)) - 1
            out.append(token)
            leaving = window[0]
            h = (h * M + token - leaving * pow(M, self.n, 1 << 64)) & MASK64
            window.append(token)
        return out, (len(out) if out else 0), miss


def bucket(match: int) -> int:
    return 0 if match < 6 else 1 if match < 12 else 2 if match < 24 else 3


def simulate(drafter, tokens, max_k, observe_first=None):
    """Replay `tokens` through `drafter`; returns counters."""
    observe_first = observe_first if observe_first is not None else 0
    for t in tokens[:observe_first]:
        drafter.observe(t)
    steps = 0
    steps_with_proposal = 0
    sum_accepted = 0
    sum_drafted = 0
    profitable = {2: 0, 4: 0, 8: 0}
    profitable_tokens = {2: 0, 4: 0, 8: 0}
    proposed = collections.Counter()
    accepted = collections.Counter()
    accepted_by_bucket = collections.Counter()
    drafted_by_bucket = collections.Counter()
    misses = collections.Counter()
    i = observe_first
    n = len(tokens)
    while i < n:
        steps += 1
        result = drafter.propose(max_k)
        props = result[0]
        miss = result[2] if len(result) > 2 else None
        actual = tokens[i:i + len(props)]
        a = 0
        while a < len(props) and a < len(actual) and props[a] == actual[a]:
            a += 1
        if props:
            steps_with_proposal += 1
            sum_accepted += a
            sum_drafted += len(props)
            for be in profitable:
                if a + 1 > be:
                    profitable[be] += 1
                    profitable_tokens[be] += a + 1
            proposed[len(props)] += 1
            accepted[a] += 1
            b = bucket(len(props))
            drafted_by_bucket[b] += len(props)
            accepted_by_bucket[b] += a
        elif miss is not None:
            misses[miss] += 1
        step = a + 1
        for t in tokens[i:i + step]:
            drafter.observe(t)
        i += step
    return {
        "steps": steps,
        "tokens": n - observe_first,
        "fired": steps_with_proposal,
        "sum_accepted": sum_accepted,
        "sum_drafted": sum_drafted,
        "profit_rounds": profitable,
        "profit_tokens": profitable_tokens,
        "mean_proposed": (sum_drafted / steps_with_proposal) if steps_with_proposal else 0.0,
        "accepted_per_firing": (sum_accepted / steps_with_proposal) if steps_with_proposal else 0.0,
        "mean_accepted": sum_accepted / steps,
        "tokens_per_round": 1.0 + sum_accepted / steps,
        "by_bucket": {b: (accepted_by_bucket[b], drafted_by_bucket[b]) for b in range(4)},
        "misses": dict(misses),
    }


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return Tokenizer.from_file(str(p))


def build_ledgers(tok, limit):
    def enc(text):
        return tok.encode(text, add_special_tokens=False).ids

    ledgers = {}
    # Edit: a source file followed by a lightly-renamed copy (what an edit/refactor returns).
    src = (REPO / "src/models/qwen3_5/program/speculative/lookup_policy.h").read_text()
    renamed = src.replace("LookupPolicy", "DraftLookupPolicy").replace("LookupAcceptance", "DraftAcceptance")
    ledgers["edit"] = enc(src + "\n" + renamed)
    # Code: several sources concatenated.
    code = []
    for rel in ["src/models/qwen3_5/program/decode.cpp",
                "src/models/qwen3_5/program/program_impl.h",
                "src/ops/offload_moe/kernels.cu"]:
        code.append((REPO / rel).read_text())
    ledgers["code"] = enc("\n".join(code))
    # Prose: the docs.
    prose = []
    for rel in ["docs/maintainer/lookup-drafter.md", "docs/serving.md"]:
        prose.append((REPO / rel).read_text())
    ledgers["prose"] = enc("\n".join(prose))
    return {k: v[:limit] for k, v in ledgers.items()}


def iter_traces(path: str):
    """Yield (request_id, ledger) from a --generation-token-trace-jsonl file."""
    import json

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("schema") != "ninfer_generated_token_trace":
                continue
            ledger = list(record.get("prompt_token_ids", [])) + list(
                record.get("generated_token_ids", [])
            )
            if ledger:
                yield record.get("request_id"), ledger


def merge(stats_list):
    merged = {
        "steps": 0, "tokens": 0, "fired": 0, "sum_accepted": 0, "sum_drafted": 0,
        "profit_rounds": {2: 0, 4: 0, 8: 0}, "profit_tokens": {2: 0, 4: 0, 8: 0},
        "misses": collections.Counter(),
    }
    for s in stats_list:
        for key in ("steps", "tokens", "fired", "sum_accepted", "sum_drafted"):
            merged[key] += s[key]
        for be in (2, 4, 8):
            merged["profit_rounds"][be] += s["profit_rounds"][be]
            merged["profit_tokens"][be] += s["profit_tokens"][be]
        for k, v in s["misses"].items():
            merged["misses"][k] += v
    merged["accepted_per_firing"] = (
        merged["sum_accepted"] / merged["fired"] if merged["fired"] else 0.0
    )
    merged["tokens_per_round"] = (
        1.0 + merged["sum_accepted"] / merged["tokens"] if merged["tokens"] else 0.0
    )
    return merged


def report(label, stats):
    pr = stats["profit_rounds"]
    print(f"  {label:20s} firings={stats['fired']:6d} "
          f"accepted/firing={stats['accepted_per_firing']:5.2f} "
          f"tokens/round={stats['tokens_per_round']:5.2f}")
    print(f"  {'':20s} gate-profitable firings (E>2/4/8): {pr[2]}/{pr[4]}/{pr[8]} "
          f"tokens {stats['profit_tokens'][2]}/{stats['profit_tokens'][4]}/"
          f"{stats['profit_tokens'][8]}"
          + (f"  misses={dict(stats['misses'])}" if stats["misses"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="all", choices=["edit", "code", "prose", "all"])
    ap.add_argument("--trace", default=None,
                    help="a --generation-token-trace-jsonl file; replays its prompt+generated ledgers")
    ap.add_argument("--max-k", type=int, default=15)
    ap.add_argument("--limit", type=int, default=120000)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--pool-n", type=int, default=8)
    ap.add_argument("--pool-entries", type=int, default=1 << 22)
    args = ap.parse_args()

    designs = (("trigram", lambda: TrigramIndex()),
               (f"last_follower n={args.pool_n}",
                lambda: LastFollowerPool(args.pool_n, args.pool_entries)))

    if args.trace:
        ledgers = [ledger for _, ledger in iter_traces(args.trace)]
        print(f"=== trace {args.trace}: {len(ledgers)} request ledger(s) ===")
        for label, make in designs:
            report(label, merge([simulate(make(), ledger, args.max_k) for ledger in ledgers]))
        print()
        return 0

    tok = load_tokenizer(args.tokenizer)
    if tok is None:
        print("tokenizer unavailable; falling back to bytes", file=sys.stderr)
        ledgers = {
            "edit": (REPO / "src/models/qwen3_5/program/speculative/lookup_policy.h").read_bytes(),
            "code": (REPO / "src/models/qwen3_5/program/decode.cpp").read_bytes(),
            "prose": (REPO / "docs/serving.md").read_bytes(),
        }
        ledgers = {k: list(v[:args.limit]) for k, v in ledgers.items()}
    else:
        ledgers = build_ledgers(tok, args.limit)

    names = list(ledgers) if args.ledger == "all" else [args.ledger]
    for name in names:
        tokens = ledgers[name]
        print(f"=== ledger {name}: {len(tokens)} tokens ===")
        for label, make in designs:
            report(label, simulate(make(), tokens, args.max_k))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
