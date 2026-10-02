import json
import sys

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "request_done":
            continue
        result = rec.get("result", {})
        t = rec.get("timings_seconds", {})
        rows.append(
            {
                "prompt_tokens": result.get("prompt_tokens"),
                "computed_prefill_tokens": result.get("computed_prefill_tokens"),
                "prefill_s": t.get("prefill"),
                "ttft_s": t.get("ttft"),
                "decode_s": t.get("decode"),
                "total_s": t.get("total"),
            }
        )

print(f"requests logged: {len(rows)} (first is warmup)")
rated = []
for i, r in enumerate(rows):
    p = r["prefill_s"] or 0.0
    c = r["computed_prefill_tokens"] or 0
    rate = (c / p) if p > 0 else 0.0
    label = "warmup" if i == 0 else f"run{i}"
    print(
        f"  {label}: prompt={r['prompt_tokens']} computed={c} "
        f"prefill={p:.4f}s ttft={r['ttft_s']} decode={r['decode_s']} rate={rate:.1f} tok/s"
    )
    if i > 0 and p > 0:
        rated.append(rate)

if rated:
    print(f"  measured mean prefill rate: {sum(rated)/len(rated):.1f} tok/s over {len(rated)} runs")
