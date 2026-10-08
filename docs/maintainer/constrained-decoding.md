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
| Schema validation | `src/text/json_schema.{h,cpp}` | validates the supported dialect, keeps property order, normalizes source for the vendor converter, reports a JSON Pointer |
| Choice and regex | `src/text/grammar.cpp` | literal alternation and a scalar-value regex automaton |
| Tool contracts | `src/models/qwen3_5/frontend/tool_contract.{h,cpp}` | per-declaration parameters, types and encodings, plus this request's selection |
| Tool language | `src/models/qwen3_5/frontend/tool_grammar.{h,cpp}` | composes the model's own tool framing with the selected tools' argument languages |
| Request contract | `RequestOptions::grammar`, `EngineOptions::grammar_cache_bytes` | public API; `ResolvedRequestOptions::grammar` carries it to the session |
| Compiler ownership | `Frontend::Impl::grammars()` | one immutable compiler per model vocabulary, built on the first constrained request |
| Matcher ownership | `OutputSession` | the request owns its matcher; the Engine never reaches into grammar state directly |
| Mask provider | `runtime::TokenMaskProvider`, `EngineCore::RoundMasks` | borrowed for one synchronous Program call, mapping compact rows to requests |
| Mask storage and staging | `ProgramImpl::grammar_masks_device` / `grammar_masks_host` | device `I32[words][draft positions][lanes]` plus pinned staging |
| Mask consumption | `ops::SamplingConfig::mask` | vocabulary masks in the sampling and speculative-acceptance kernels |
| Input surfaces | `--grammar-file`, `structured_outputs.grammar` | CLI file, or one nonempty grammar string on all three HTTP protocols |

`OutputConstraint` carries the kind (Grammar, JsonObject, JsonSchema) and its owning source text.
The public entry points — `--grammar-file`/`--json-object`/`--json-schema-file`,
`response_format` and its Responses and Anthropic equivalents, `structured_outputs.grammar`, the
refusals and the error codes — are documented in [cli.md](../cli.md) and
[serving.md](../serving.md); this document is about the mechanism.

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

MTP drafts are produced at the end of the previous round and live in `sequence.mtp_drafts` on the
host, so a constrained MTP row drafts normally: `decode_mtp_batch` passes the very draft span it is
about to verify to `fill_grammar_mask`, which fills the accepted state plus one mask per draft
position, and the acceptance kernels read mask position `col` for column `col`. The ordinary batch
(`decode_ordinary_batch`) fills position zero only, because it verifies a single token.

The tree and block backends draft *inside* the round, so their positions cannot be masked before it
runs. `decode_eagle3_batch` and `decode_dflash_batch` therefore refuse a constrained row, and the
request is refused at preparation for those backends, so the refusal is reported as an
`InvalidGrammar` request error rather than as a round failure. Masking them needs the draft handoff
and the forward/finish phase split: the drafts are copied to pinned host memory after the forward
phase, the provider fills the masks from them, and the verify phase consumes them.

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

### 2.4 JSON object and JSON Schema

`json_object` compiles to a root object language, not to arbitrary JSON. `json_schema` uses the root
type the caller declared, which may be an object, an array, or a supported scalar. Both compile
through the vendor's schema converter with compact separators, no indentation and non-strict mode,
after `prepare_json_schema` validates the source and rejects what the dialect does not support with
a JSON Pointer.

Semantics that follow from that path:

- property order in the source is the generation order, so the schema text is part of the compile
  cache key rather than being normalized away;
- string, bounded-number, array-length and enum constraints become part of the same automaton as a
  hand-written grammar, so `pattern` and `minLength` can apply together;
- the vendor does not fetch external documents: document-local `$ref`/`$defs` resolve, remote ones
  are refused.

A schema error surfaces as a request error whose `param` is the wire field the client sent plus the
failing JSON Pointer, and whose `code` distinguishes invalid, unsupported and unsatisfiable schemas.

### 2.5 Choice and regex

A choice compiles to an alternation of literal byte strings: candidates keep their case, whitespace
and Unicode exactly, duplicates are dropped, and the order carries no weight, so the compiled result
is keyed by the normalized set rather than by how the client listed it. A short candidate that
prefixes a longer one still allows the stop token after the short form, and also allows continuing
into the longer one.

A regex is matched against the whole content, not searched: an empty pattern admits only the empty
content. Literals, Unicode characters, character classes, groups, alternation and `*`, `+`, `?`,
`{m,n}` repetitions are supported; greedy and lazy spellings describe the same language, and no
capture values are returned. Classes follow ECMAScript semantics (`\d`, `\w` are ASCII, `\s`
includes Unicode whitespace, `.` excludes `\n`, `\r`, U+2028 and U+2029), escapes such as `\xNN`
and `\uNNNN` are accepted, and the output contains Unicode scalar values only: a scalar-value pass
rewrites every class so surrogates can never be published. Anchors are accepted only at the ends of
the pattern or a top-level branch; backreferences, lookaround, word boundaries, property classes,
flags, surrogate escapes and unknown escapes are refused as `InvalidRegex`.

### 2.6 Tools

Tool declarations belong to the prompt; **selection and cardinality belong to the request**
(`RequestOptions::tool_choice`). The declarations are always rendered into the prompt, and a narrowed
choice never edits that list — it constrains the generated calls instead, so the prompt and the
cache markers it carries cannot change behind the caller's back.

Three choices compose:

- **structure** — the model's own wrapper and function names;
- **arguments** — per-tool schemas, when a tool is declared `strict`;
- **selection** — `Auto` (zero or more calls), `Required` (at least one), an allowed-name set, and
  whether more than one call may be published.

`ToolConstraintMode::Basic` (the default) constrains every request that has tools, including an
ordinary `Auto` request with no strict tools, so a call the model does start is well formed.
`Automatic` constrains only what the request asked for, leaving ordinary free generation untouched.
`Auto` may emit ordinary content before the call sequence; `Required` starts with a call; once the
call sequence starts, only further calls and EOS are legal, with a single newline between them. A
non-strict tool's arguments keep the established best-effort normalization (any order, the
vocabulary's representable names, values quoted or parsed as JSON), while a strict tool's
`arguments_json` satisfies its declared schema, generating properties in declaration order and
forbidding duplicates. A raw-string parameter ends at `\n</parameter>` — that text cannot appear in
the value, and the grammar and the parser share the rule so a length or `pattern` assertion is never
quietly weakened. Nested JSON uses the model template's separators.

One output language per request: a content constraint and a tool constraint are mutually exclusive,
and each is refused with the others' guarantees still intact.

### 2.7 Rejected combinations

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
| Plain decode round with a JSON object or schema constraint | constrained; the published content parses as the declared type |
| Tool call under `strict`, `Required`, named or single-call choice | constrained; the call is complete and its arguments satisfy the declared schema |
| Plain and drafted rounds with a choice or regex constraint | constrained; the published content is one of the literals, or matches the pattern in full |
| Prefill round sampling the first generated token | constrained |
| Ordinary decode round (plain backend, `--draft-tokens 0`) | constrained |
| MTP round with drafts | constrained; the row drafts normally, one mask per verify position |
| MTP round in lookup mode | constrained through the same lane configuration |
| Reasoning → content framing | constrained from the exact boundary |
| Cancellation, failed commit, rejected preview | matcher rolled back to the committed prefix |
| Grammar dead end | request error, no partial publication |

Not implemented:

| Not implemented | Consequence |
|---|---|
| Content constraints composed with tool constraints | a request may have one output language, so combining them is refused |
| Tool constraints (`strict` schemas, constrained tool calls) | `strict:true` remains refused, and a constraint cannot be combined with active tools |
| Identifier and unused-keyword composition (`anyOf`/`oneOf`/`allOf` reduction, `$ref` across documents) | schemas using them are refused with a pointer rather than approximated |
| Masks on draft-producing verify positions (trees and blocks) | `dflash`, `dflash2`, `eagle3` refuse constrained requests; needs the draft handoff and the forward/finish split |
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
- `tests/text/test_json_schema.cpp` — native schema acceptance/rejection, including the source
  order that becomes the generation order.
- `tests/text/test_json_schema.py` — the same schemas checked against the independent `jsonschema`
  library, driven through a probe binary so the native validator and the library are compared on the
  same inputs.
- `tests/text/test_regex_choice.cpp` and `tests/text/test_regex_choice.py` — native choice/regex
  acceptance, with the same candidates and patterns checked against Python's `re`, including escape
  forms, anchors, alternation and repetition.
- `ninfer_qwen3_5_grammar_real_test` — the Engine end to end with a real artifact, over content,
  sampling, thinking, continuation, mixed batches, truncation, raw input and JSON/schema
  constraints, with and without CUDA Graphs.
- `tests/models/qwen3_5/test_tool_constraints.cpp` and `tests/models/qwen3_5/test_tool_schema.py` —
  the tool language natively, and the same schemas validated through the independent `jsonschema`
  library via a probe binary, including `const`/`enum` values, ranges, nesting and separator text.
- `ninfer_qwen3_5_tools_real_test` — the Engine end to end on a real artifact with declared tools:
  basic, eager and CUDA-Graph modes, speculative drafting on and off, and multiple concurrent lanes,
  asserting that published calls satisfy their schemas (including a `const` value with a leading
  space and a non-ASCII body, and a value that contains the parameter terminator text).
- `ninfer_xgrammar_core_test` — the vendored core's observable contract.
- `tests/models/qwen3_5/test_mtp_draft_policy.cpp` — an unintended interaction is covered: a
  zero-extent round records no per-position acceptance evidence, so a constrained row cannot bias
  the draft-length estimate.
- End-to-end: a GBNF grammar over a JSON object shape, greedy, on `models/qwen3_8_27b.ninfer`,
  through the CLI (MTP round) and through `ninfer-serve` with `--spec mtp --draft-tokens {0,1,3}`;
  the published content parses and matches the grammar in every case, while the unconstrained
  control run emits a value the grammar forbids.
- Cost, measured on the same fixture (greedy, `--spec mtp --draft-tokens 3`, 512 new tokens, two
  runs each, CLI): unconstrained 148.6 and 139.3 tok/s against 141.2 and 140.3 tok/s with a
  permissive grammar over the whole byte range. The difference is inside run-to-run spread, so the
  mechanism costs no measurable throughput on this fixture, while the row keeps its drafting: 2.68-2.82
  accepted tokens per round and `K2`/`K3` rounds, against the 1.0 token per round a row that drafts
  nothing commits. Acceptance is marginally lower under a constraint (56.5% against 61.6% here)
  because the legal set excludes the vocabulary's special tokens, so drafts made of them can no
  longer be accepted.
