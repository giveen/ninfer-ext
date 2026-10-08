# Constrained decoding

This document is the authority for how NInfer constrains generated tokens. It covers the mechanism
that is implemented today — GBNF, its request contract, the vocabulary-wide legal set, the
per-position masks that reach the sampling and speculative-acceptance kernels, and the exact limits
of the current coverage. JSON, choice, regex and tool constraints are later entry points on the same
mechanism; they are not implemented.

## 1. What is implemented

| Piece | Where | Notes |
|---|---|---|
| Grammar source base | `third_party/xgrammar` | pinned CPU core, compiled into `ninfer_xgrammar`; no CUDA, Python or TVM dependency |
| Generic text adapter | `src/text/grammar.{h,cpp}` | vocabulary/compiler per model, transactional matcher per request |
| Request contract | `RequestOptions::grammar`, `EngineOptions::grammar_cache_bytes` | public API; `ResolvedRequestOptions::grammar` carries it to the session |
| Compiler ownership | `Frontend::Impl::grammars()` | one immutable compiler per model vocabulary, built on the first constrained request |
| Matcher ownership | `OutputSession` | the request owns its matcher; the Engine never reaches into grammar state directly |
| Mask provider | `runtime::TokenMaskProvider`, `EngineCore::RoundMasks` | borrowed for one synchronous Program call, mapping compact rows to requests |
| Mask storage and staging | `ProgramImpl::grammar_masks_device` / `grammar_masks_host` | device `I32[words][draft positions][lanes]` plus pinned staging |
| Mask consumption | `ops::SamplingConfig::mask` | vocabulary masks in the sampling and speculative-acceptance kernels |
| Input surfaces | `--grammar-file`, `structured_outputs.grammar` | CLI file, or one nonempty grammar string on all three HTTP protocols |

The public entry points (CLI flag, protocol object, its refusals and error codes) are documented in
[cli.md](../cli.md) and [serving.md](../serving.md); this document is about the mechanism.

## 2. Semantics

A constraint fixes the set of token sequences the request may publish. It replaces the model's
per-token distribution with the same distribution restricted to the legal set; it is **not** the
model's globally conditioned distribution over sequences that happen to match. Speculative decoding
preserves that same per-token target, because a draft is only committed when the target's restricted
distribution agrees with it.

Three facts stay distinct, and conflating them produces wrong output:

- the current prefix can still be continued;
- the current content is already at an accepted end state (`GrammarSession` reports this through the
  mask: the stop token becomes legal, without becoming forced);
- the request ended through the constraint rather than through a length limit, a caller stop, or a
  cancellation.

A token's **whole** byte sequence has to be legal at its position. A token may span terminals, rules
and the reasoning→content boundary, and it may contain only part of a UTF-8 sequence. The vocabulary
therefore carries the model's raw token bytes, with special and invalid ids left empty, and EOS is
supplied separately.

### 2.1 Per-position masks

A round computes one mask per position it may reach: the accepted state, then one per draft token.
`TokenMaskProvider::fill(row, drafts, words)` writes them and returns the positions whose legal set
was **empty**. An empty set is a dead end, not a completion: the provider leaves a safe placeholder
mask so the kernels always have something to sample, and reports the position separately.

How the row reacts depends on whether verification reaches it:

- a dead end the round actually consumes fails the request (`ConstraintDeadEnd`), publishing no
  output from that round;
- a dead end beyond the row's licensed prefix is irrelevant, and the row proceeds normally.

A constrained row **drafts nothing**: the round's single accepted token is sampled from mask
position zero, because one mask per verify position inside the same round is not implemented for the
draft-producing backends. `decode_ordinary_batch` and `decode_mtp_batch` bind the mask into their
lane's sampling configuration; `decode_eagle3_batch` and `decode_dflash_batch` refuse a constrained
row, and the request is refused at preparation for those backends, so the refusal is reported as an
`InvalidGrammar` request error rather than as a round failure.

### 2.2 Transaction boundary

`GrammarSession` is transactional, which is what makes speculative and cancellation paths safe:

- `masks` advances a *copy* of the state across a draft chain and rolls it back before returning, so
  a lookahead never disturbs the accepted state;
- `accept` records tokens the round *proposes* to publish;
- `confirm` makes them permanent, at the same boundary where the Engine commits published output;
- `discard` rolls them back, which is what a cancelled request, a failed commit, or a rejected
  preview does.

The Engine's rollback paths call `OutputSession::discard_preview()`, so a round that fails anywhere
between preview and commit leaves the matcher exactly at the last committed token.

### 2.3 Reasoning framing

A chat reply may reason before it answers. With a constraint, the framing is the model's **canonical**
reasoning-close serialization: `src/text/grammar.cpp` prepends a small automaton that consumes
anything through the first complete occurrence of that marker, and `OutputSession` matches the same
bytes exactly (`exact_reasoning_framing`), without the separator refinement it applies to
unconstrained output and without stripping the leading bytes of the content behind it. Compiler and
decoder therefore agree byte for byte; a token that crosses the boundary is handled by the grammar,
not by special cases in the decoder.

Raw continuation requests do not get this framing: their grammar starts at the first generated byte,
and the prompt is not treated as a grammar prefix. A `ContinueFinalAssistant` request initializes its
matcher from the assistant text that already exists, so the constraint covers the whole final answer.

### 2.4 Rejected combinations

`Frontend::make_output_session` refuses a grammar that cannot be honored, and the Engine refuses a
backend that cannot consume masks. Rejections are request errors (`InvalidGrammar`), never silent
downgrades:

| Refused with a grammar | Reason |
|---|---|
| empty grammar source | nothing to compile |
| active tools | the tool call itself would be unconstrained text |
| caller stop strings or stop token ids, or non-default EOS | the stop would end the request outside the language |
| `publish_stop_token`, raw output, `preserve_special_tokens` | published bytes would not be the language |
| `dflash`, `dflash2`, `eagle3` backends | their verify positions consume no masks yet |

### 2.5 Compilation and caching

`GrammarCompiler` is immutable and per vocabulary; `GrammarSession` is per request. Compilation is
cached inside the vendor core by grammar source and framing, sharing one budget
(`EngineOptions::grammar_cache_bytes`, default 256 MiB) with bounded concurrent cold compilations.
Compiled grammars outlive cache eviction while a request still uses them, and a compilation happens
on the submitting thread before the request can be queued — so a cold compile is charged to request
preparation, and the pending deadline is rechecked afterwards.

GBNF is the language itself. The vendor's rule attributes that change decoding behavior (token and
character budgets, captures, lazy matching, temperature) and its tag-dispatch, regex and substring
extensions are rejected, as are `Token()`/`ExcludeToken()` references to ids the model cannot
publish.

## 3. Coverage

Implemented and verified end-to-end on a real artifact:

| Path | State |
|---|---|
| Prefill round sampling the first generated token | constrained |
| Ordinary decode round (plain backend, `--draft-tokens 0`) | constrained |
| MTP round with drafts | constrained; the constrained row drafts nothing |
| MTP round in lookup mode | constrained through the same lane configuration |
| Reasoning → content framing | constrained from the exact boundary |
| Cancellation, failed commit, rejected preview | matcher rolled back to the committed prefix |
| Grammar dead end | request error, no partial publication |

Not implemented:

| Not implemented | Consequence |
|---|---|
| JSON object, JSON Schema, choice and regex entry points | `response_format`/`text.format` JSON output is still refused, now pointing at GBNF |
| Tool constraints (`strict` schemas, constrained tool calls) | `strict:true` remains refused |
| Masks on the draft-producing verify positions (draft trees) | `dflash`, `dflash2`, `eagle3` refuse constrained requests |
| Jump-forward decoding | not planned here; a mask-only contract is enough for correctness |
| Grammar-aware sampling-side fast paths | a constrained round takes the general sampler path, which costs more than the unmasked greedy path for the same round |

## 4. Verification

- `tests/text/test_grammar.cpp` — language oracle: the complete language of two equivalent grammars
  is enumerated independently of the parser and the tokenizer, and every bit of every mask is
  compared against it, for every prefix. Also covers rollback fidelity, compiled-grammar lifetime
  across compiler destruction and cache eviction, embedded NULs, Unicode split across tokens,
  bounded repetition with a continuation prefix, a token crossing the framing boundary, concurrent
  compilation, and the contract rejections.
- `tests/ops/test_sampling.cpp` — masked greedy and masked distribution against the FP64 oracle,
  including a vocabulary where the legal tokens rank below the unmasked top-k, which is what proves
  the mask applies before top-k and normalization rather than after.
- `ninfer_xgrammar_core_test` — the vendored core's observable contract.
- `tests/models/qwen3_5/test_mtp_draft_policy.cpp` — an unintended interaction is covered: a
  zero-extent round records no per-position acceptance evidence, so a constrained row cannot bias
  the draft-length estimate.
- End-to-end: a GBNF grammar over a JSON object shape, greedy, on `models/qwen3_8_27b.ninfer`,
  through the CLI (MTP round, drafts enabled) and through `ninfer-serve` with `--spec mtp
  --draft-tokens {0,1}`; the published content parses and matches the grammar in every case, while
  the unconstrained control run emits a value the grammar forbids.
