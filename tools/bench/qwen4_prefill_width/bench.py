import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = str(HERE.parents[2])
# Scratch/output directory for runs and request logs; set STRATA_AB_DIR to keep them elsewhere.
AB = os.environ.get("STRATA_AB_DIR", "/tmp/strata-ab")
os.makedirs(AB, exist_ok=True)
SERVE = f"{ROOT}/build/apps/ninfer-serve"
ART = f"{ROOT}/models/Qwen3.8-Flash-NVFP4/qwen3_8_flash_next_nvfp4.ninfer"
HOST = "127.0.0.1"
PORT = 18090
BASE = f"http://{HOST}:{PORT}"

FIXTURES = {
    "8k": f"{ROOT}/examples/cli/messages/long_niah_8k.json",
    "64k": f"{ROOT}/examples/cli/messages/long_niah_64k.json",
    "128k": f"{ROOT}/examples/cli/messages/long_niah_128k.json",
    "256k": f"{ROOT}/examples/cli/messages/long_niah_256k.json",
    "decode": f"{ROOT}/examples/cli/messages/long_decode_aime26_15.json",
}


def make_payload(messages, max_tokens):
    return json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()


def post(payload_bytes, timeout=2400):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wait_ready(proc, log_path, seconds=900):
    for _ in range(seconds):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server not ready")


class Server:
    def __init__(self, idle, spec, tag, stats_ms=0, kv=None):
        self.tag = tag
        if kv is None:
            kv = os.environ.get("BENCH_KV", "fp8")
        self.jsonl = f"{AB}/bench-{tag}.jsonl"
        self.log = f"{AB}/bench-{tag}.log"
        for p in (self.jsonl, self.log):
            if os.path.exists(p):
                os.remove(p)
        cmd = [
            SERVE, ART, "--model-id", "qwen3.8-flash-next",
            "--host", HOST, "--port", str(PORT),
            "--max-concurrency", "8", "--max-context", "262144", "--kv-capacity", "262144",
            "--kv-dtype", kv, "--expert-cache", "auto", "--ngram-residency", "stream",
            "--no-prefix-reuse", "--greedy", "--no-thinking",
            "--log-stats-interval-ms", str(stats_ms),
            "--request-log-jsonl", self.jsonl,
        ]
        if idle != "auto":
            cmd += ["--prefill-chunk", "4096", "--idle-prefill-chunk", idle]
        if spec == "k3":
            cmd += ["--spec", "mtp", "--draft-tokens", "3", "--fixed-draft"]
        elif spec == "k7":
            cmd += ["--spec", "mtp", "--draft-tokens", "7", "--fixed-draft"]
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        wait_ready(self.proc, self.log)
        self.load_s = time.time() - t0
        self.start = self.server_start()
        self.done_seen = 0

    def server_start(self):
        with open(self.jsonl) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "server_start":
                    return rec.get("engine", {})
        return {}

    def done_records(self):
        out = []
        try:
            with open(self.jsonl) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("event") == "request_done":
                        out.append(rec)
        except FileNotFoundError:
            pass
        return out

    def new_done(self):
        recs = self.done_records()
        new = recs[self.done_seen:]
        self.done_seen = len(recs)
        return new

    def throughput_records(self):
        out = []
        try:
            with open(self.jsonl) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("event") == "throughput" and "expert_cache" in rec:
                        out.append(rec)
        except FileNotFoundError:
            pass
        return out

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.logf.close()


def summarize_done(rec):
    res = rec.get("result", {})
    t = rec.get("timings_seconds", {})
    return {
        "prompt": res.get("prompt_tokens"),
        "completion": res.get("completion_tokens"),
        "computed_prefill": res.get("computed_prefill_tokens"),
        "prefill_s": t.get("prefill"),
        "ttft_s": t.get("ttft"),
        "decode_s": t.get("decode"),
        "total_s": t.get("total"),
        "spec": rec.get("speculative"),
    }


def run_prefill(server, results):
    fixtures = ["8k", "64k", "128k", "256k"]
    payloads = {}
    for name in fixtures:
        with open(FIXTURES[name]) as f:
            msgs = json.load(f)
        payloads[name] = make_payload(msgs, 1)
    # Warmup primes clocks and the expert cache; not recorded.
    print(f"  [{server.tag}] prefill warmup 8k", flush=True)
    post(payloads["8k"])
    server.new_done()
    for name in fixtures:
        for rep in range(3):
            print(f"  [{server.tag}] prefill {name} rep{rep + 1}", flush=True)
            t0 = time.time()
            post(payloads[name])
            wall = time.time() - t0
            recs = server.new_done()
            if len(recs) != 1:
                raise RuntimeError(f"expected 1 done record, got {len(recs)}")
            s = summarize_done(recs[0])
            s.update({"test": "prefill", "fixture": name, "rep": rep, "wall_s": wall})
            results.append(s)
            print(
                f"    prompt={s['prompt']} prefill={s['prefill_s']:.3f}s "
                f"rate={(s['computed_prefill'] / s['prefill_s']):.1f} tok/s",
                flush=True,
            )


def wave(server, payload, c, timeout=2400):
    errs = []

    def worker():
        try:
            post(payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errs.append(repr(exc))

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    return wall, errs


def run_decode(server, results):
    with open(FIXTURES["decode"]) as f:
        msgs = json.load(f)
    payload = make_payload(msgs, 256)
    print(f"  [{server.tag}] decode warmup", flush=True)
    post(payload)
    server.new_done()
    for c in [1, 2, 4, 8]:
        for rep in range(2):
            print(f"  [{server.tag}] decode C={c} rep{rep + 1}", flush=True)
            wall, errs = wave(server, payload, c)
            if errs:
                raise RuntimeError(f"wave errors: {errs}")
            recs = server.new_done()
            if len(recs) != c:
                raise RuntimeError(f"expected {c} done records, got {len(recs)}")
            rows = [summarize_done(r) for r in recs]
            dec_tokens = sum(max(0, (r["completion"] or 1) - 1) for r in rows)
            per_req = [
                (max(0, (r["completion"] or 1) - 1) / r["decode_s"])
                for r in rows
                if r["decode_s"]
            ]
            accepted = drafted = 0
            for r in rows:
                sp = r["spec"] or {}
                accepted += sp.get("accepted_tokens", 0)
                drafted += sp.get("drafted_tokens", 0)
            results.append(
                {
                    "test": "decode",
                    "c": c,
                    "rep": rep,
                    "wall_s": wall,
                    "decode_tokens": dec_tokens,
                    "aggregate_tok_s": dec_tokens / wall if wall else 0.0,
                    "mean_per_request_tok_s": sum(per_req) / len(per_req) if per_req else 0.0,
                    "prompt": rows[0]["prompt"],
                    "completion": rows[0]["completion"],
                    "prefill_s": rows[0]["prefill_s"],
                    "accepted_tokens": accepted,
                    "drafted_tokens": drafted,
                    "decode_s_mean": sum(r["decode_s"] or 0 for r in rows) / len(rows),
                }
            )
            print(
                f"    C={c} wall={wall:.2f}s aggregate={dec_tokens / wall:.1f} tok/s "
                f"per-req={results[-1]['mean_per_request_tok_s']:.1f} tok/s "
                f"accept={accepted}/{drafted}",
                flush=True,
            )


def main():
    raw = sys.argv[1:] if len(sys.argv) > 1 else ["auto", "4096"]
    modes = []
    prefill_only = set()
    decode_only = set()
    for m in raw:
        name, _, filt = m.partition(":")
        modes.append(name)
        if filt == "prefill":
            prefill_only.add(name)
        elif filt == "decode":
            decode_only.add(name)
    out_path = os.environ.get("BENCH_OUT", f"{AB}/bench-results.json")
    all_results = []
    meta = []
    for idle in modes:
        if idle in prefill_only:
            specs = ["none"]
        elif idle in decode_only:
            specs = ["k3", "k7"]
        else:
            specs = ["none", "k3", "k7"]
        for spec in specs:
            tag = f"{idle}-{spec}"
            print(f"=== server {tag} ===", flush=True)
            server = Server(idle, spec, tag, kv=os.environ.get("BENCH_KV", "fp8"))
            e = server.start
            print(
                f"  kv={e.get('kv_cache')} idle={e.get('idle_prefill_chunk')} "
                f"slots={e.get('expert_cache_slots')} "
                f"cacheGiB={(e.get('expert_cache_bytes', 0) / 2**30):.2f}",
                flush=True,
            )
            meta.append(
                {
                    "tag": tag,
                    "load_s": server.load_s,
                    "engine": server.start,
                }
            )
            try:
                if spec == "none":
                    run_prefill(server, all_results)
                else:
                    run_decode(server, all_results)
            finally:
                server.stop()
            with open(out_path, "w") as f:
                json.dump({"meta": meta, "results": all_results}, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
