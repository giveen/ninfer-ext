# P1 design: borrow the idle prefill workspace from the expert cache

Status: **implemented**. Goal met: at the 256k budget the wide idle step (16384) now leaves decode at
0.99-1.00 of the 4096 baseline across K3/K7 and C=1-8 (K7 C8 0.98), cache -1.5 % instead of -12 %,
prefill unchanged (8k 3600, 64k 2353, 256k 867 tok/s). The lend region is the whole wide arena; the
permanent narrow arena stays a separate owning allocation. Wide steps are gated on the remaining
prompt exceeding the ordinary chunk, so short requests never disturb the cache. The cache floor was
removed: the lend makes the decode cache independent of the idle width, so the floor's premise is
gone; 16384 is capped rather than 32768 because the wider lend still costs ~3 % cache (K7 C8 -7 %)
for only 1-5 % more prefill.

## Measured motivation

At the 256k budget, `auto` picks idle 16384 and the cache drops 6,804 -> 5,974 slots (-12 %),
costing 6-9 % decode (K3) and up to 24 % (K7 C8). The lost slots are exactly the prefill workspace
the planner withheld from the cache.

## Why a constant bump cannot work

NInfer sizes the prefill workspace once, permanently, and `resolve_expert_cache_slots` subtracts it
from the automatic cache. Strata instead carves the prefill buffers out of the cache's tail per
request and refills them afterwards (`strata::prefill::Prefill::init(borrow=...)` / `relayout`;
`--no-prefill-borrow` is NInfer's current behaviour). Borrowing does not create memory; it
time-multiplexes the same bytes between "cache" and "prefill workspace".

## Layout

Put everything in one persistent pool (slot-aligned) so the prefill arena and the lend slots share
bytes:

```
[ cache slots C0 | lend L | permanent workspace Wn | banks 1024 ]   (all in Qwen4PersistentLayout.pool)
```

- `C0` = the auto cache slots at the *narrow* (`prefill_chunk`) workspace. Reported cache.
- `L = W(wide) - W(narrow)` bytes of slots: cache during decode, workspace during a wide idle step.
- `Wn` = permanent workspace for decode rounds and the narrow prefill; never cache.
- Wide arena = `pool + C0*slotBytes`, capacity `Wn + L` = `W(wide)`.
- Narrow/decode arena = `pool + (C0 + L/slotBytes)*slotBytes`, capacity `Wn`.
- Banks stay at the top of the pool and are addressed by a **stored** `bank_slot = C0 + L/slots +
  Wn/slots`; `Qwen4ExpertPager` must take that offset instead of `cache.slots - kBankSlots`.

## Runtime

- During decode and narrow prefill: `ExpertCacheState.slots = C0 + L/slots` (lend is cache), the
  narrow arena is `work`.
- During a wide idle step (`Pace::Idle`, width > `prefill_chunk`): `expert_cache_reclaim(cache, C0)`
  then `slots = C0`, and the step runs on the wide arena. Restore `slots` and leave the lend slots
  absent so decode refills them on demand (optionally proactively refill the hottest).
- Concurrency: wide steps only happen at `Pace::Idle` (no decode waiting). A decode admitted between
  wide steps forces the next step narrow, so the lend is reclaimed at most once per idle-prefill
  episode, not per step. This is the part Strata does not have to solve (single request).

## Accounting

The trap: the permanent workspace must not be counted twice. Put `Wn + L` inside
`Qwen4PersistentLayout.pool` and make `device_reservation_bytes` for Qwen4Exp exclude
`workspace.capacity` (it is already in `persistent.bytes`). Then the auto-cache solve yields
`C0 = (free - floor)/slotBytes - banks - (Wn+L)/slotBytes`, i.e. the same `C0` as the current narrow
width, and effective decode cache = `C0 + L/slotBytes` = the old 4096-width cache. The plan still
carries the wide layout sizes (used to bind tensors and to `mark_workspace_usage`), but only the
narrow capacity is reserved through `workspace`.

## Staged implementation

1. Layout: add `lend` and `workspace` regions to `Qwen4PersistentLayout`; build the workspace plan
   for the narrow width and expose `lend_bytes`; fix `resolve_expert_cache_slots`/reservation so the
   pool includes `Wn+L` and `workspace.capacity` is not double-counted.
2. Program: own narrow (borrowed over `pool`) and wide arenas; select by step width in
   `advance_prefill`/decode; store `bank_slot` in the pager.
3. Scheduler: reclaim/restore around wide steps; verify decode returns to baseline and prefill is
   unchanged via `tools/bench/qwen4_prefill_width/`.

Each stage builds and is smoke-tested against Flash-Next NVFP4 before the next.
