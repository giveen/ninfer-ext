import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_default = [os.path.join(HERE, "results", "bench-auto-none.json"),
            os.path.join(HERE, "results", "bench-rest.json")]
paths = sys.argv[1:] if len(sys.argv) > 1 else _default
meta = []
results = []
for p in paths:
    with open(p) as f:
        data = json.load(f)
    meta.extend(data["meta"])
    results.extend(data["results"])
meta_by_tag = {m["tag"]: m for m in meta}
data = {"meta": meta, "results": results}

# Servers run in meta order; each contributes a fixed number of result rows.
assignments = []  # (tag, spec, row)
i = 0
for m in data["meta"]:
    tag = m["tag"]
    spec = tag.rsplit("-", 1)[1]
    count = 12 if spec == "none" else 8
    chunk = results[i:i + count]
    i += count
    for row in chunk:
        assignments.append((tag, spec, row))
if i != len(results):
    print(f"WARNING: consumed {i} of {len(results)} results", file=sys.stderr)


def mode_of(tag):
    return tag.rsplit("-", 1)[0]


order = []
for m in data["meta"]:
    mode = mode_of(m["tag"])
    if mode not in order:
        order.append(mode)
fixtures = ["8k", "64k", "128k", "256k"]

print("== server configuration (256k context budget, C up to 8, fp8 KV) ==")
print(f"{'tag':>10} {'chunk':>6} {'idle':>7} {'cacheGiB':>9} {'slots':>7} {'kv_cap':>8} {'load_s':>7}")
for tag in meta_by_tag:
    e = meta_by_tag[tag]["engine"]
    print(
        f"{tag:>10} {e.get('prefill_chunk'):>6} {e.get('idle_prefill_chunk'):>7} "
        f"{e.get('expert_cache_bytes', 0) / 2**30:>9.2f} {e.get('expert_cache_slots'):>7} "
        f"{e.get('kv_capacity'):>8} {meta_by_tag[tag]['load_s']:>7.1f}"
    )

print()
print("== prefill C=1, MTP off (tok/s, mean of 3) ==")
header = f"{'fixture':>8} {'prompt':>8} " + " ".join(f"{m:>12}" for m in order) + f" {'auto/base':>10}"
for name in fixtures:
    rates = []
    prompt = 0
    for mode in order:
        rows = [r for (t, s, r) in assignments if s == "none" and mode_of(t) == mode
                and r.get("fixture") == name]
        if rows:
            prompt = rows[0]["prompt"]
        rate = statistics.mean(r["computed_prefill"] / r["prefill_s"]
                               for r in rows if r.get("prefill_s"))
        rates.append(rate)
    ratio = rates[0] / rates[1] if len(rates) > 1 else 0
    print(f"{name:>8} {prompt:>8} " + " ".join(f"{x:>12.1f}" for x in rates) + f" {ratio:>10.2f}x")

print()
print("== decode, MTP fixed draft, greedy, 256 output (aggregate tok/s = decode tokens / wall) ==")
for spec in ["k3", "k7"]:
    print(f"-- {spec} --")
    print(f"{'C':>3} " + " ".join(f"{m:>26}" for m in order))
    for c in [1, 2, 4, 8]:
        cells = []
        for mode in order:
            rows = [r for (t, s, r) in assignments if s == spec and mode_of(t) == mode
                    and r.get("c") == c]
            if not rows:
                cells.append("n/a")
                continue
            agg = statistics.mean(r["aggregate_tok_s"] for r in rows)
            per = statistics.mean(r["mean_per_request_tok_s"] for r in rows)
            acc = sum(r["accepted_tokens"] for r in rows)
            drf = sum(r["drafted_tokens"] for r in rows)
            a = (acc / drf * 100) if drf else 0.0
            cells.append(f"{agg:.0f} agg / {per:.0f} per / {a:.0f}%")
        print(f"{c:>3} " + " ".join(f"{x:>26}" for x in cells))
    print()
