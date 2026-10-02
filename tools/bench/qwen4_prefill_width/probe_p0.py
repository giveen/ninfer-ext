import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

with open(bench.FIXTURES["decode"]) as f:
    msgs = json.load(f)
payload = bench.make_payload(msgs, 256)

for mode in ["auto", "4096"]:
    s = bench.Server(mode, "k3", f"p0-{mode}", stats_ms=250)
    slots = s.start.get("expert_cache_slots")
    idle = s.start.get("idle_prefill_chunk")
    print(f"=== mode={mode} idle={idle} slots={slots} ===", flush=True)
    try:
        print("  warmup", flush=True)
        bench.post(payload)
        s.new_done()
        time.sleep(0.5)
        for c in [1, 8]:
            print(f"  wave C={c}", flush=True)
            mark = len(s.throughput_records())
            wall, errs = bench.wave(s, payload, c)
            if errs:
                print("   errors", errs)
            time.sleep(0.6)
            recs = s.throughput_records()[mark:]
            for rec in recs:
                ec = rec["expert_cache"]
                sched = rec.get("scheduler", {})
                print(
                    f"    dt={rec.get('interval_seconds', 0):.2f} "
                    f"hits={ec['interval_hits']} misses={ec['interval_misses']} "
                    f"hit_rate={ec['interval_hit_rate']:.3f} "
                    f"resident={ec['resident_slots']} "
                    f"layer_min={ec['layer_min_resident']} layer_max={ec['layer_max_resident']} "
                    f"decode_ready={sched.get('decode_ready')} running={sched.get('running')}",
                    flush=True,
                )
    finally:
        s.stop()
