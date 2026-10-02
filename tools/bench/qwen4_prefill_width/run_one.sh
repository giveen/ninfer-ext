#!/usr/bin/env bash
# A/B one idle-prefill width for the Strata chunk experiment on Flash-Next NVFP4.
# Usage: run_one.sh <idle_width> [port]
set -u

IDLE="${1:?idle width required}"
PORT="${2:-18080}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
AB="${STRATA_AB_DIR:-/tmp/strata-ab}"
mkdir -p "$AB"
ART="$ROOT/models/Qwen3.8-Flash-NVFP4/qwen3_8_flash_next_nvfp4.ninfer"
PAYLOAD="$HERE/payloads/request.json"

JSONL="$AB/run-${IDLE}.jsonl"
LOG="$AB/serve-${IDLE}.log"
rm -f "$JSONL" "$LOG"

cd "$ROOT" || exit 1

"$ROOT/build/apps/ninfer-serve" "$ART" \
  --model-id qwen3.8-flash-next \
  --host 127.0.0.1 --port "$PORT" \
  --max-concurrency 1 --max-context 65536 --kv-capacity 65536 \
  --kv-dtype fp8 --expert-cache auto --ngram-residency stream --spec mtp \
  --no-prefix-reuse --greedy --no-thinking \
  --prefill-chunk 4096 --idle-prefill-chunk "$IDLE" \
  --log-stats-interval-ms 0 \
  --request-log-jsonl "$JSONL" \
  > "$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER EXITED EARLY (idle=$IDLE)"; tail -40 "$LOG"; exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "SERVER NOT READY within 600s (idle=$IDLE)"; tail -40 "$LOG"
  kill "$SERVER_PID" 2>/dev/null; exit 1
fi
echo "--- serve ready (idle=$IDLE) ---"
grep -iE "expert|prefill|workspace|cache|kv_capacity|memory" "$LOG" | head -40

post() {
  curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$PAYLOAD" >/dev/null
}

# Warmup (not measured): full-load shapes, CUDA graph capture.
echo "warmup..."
post || { echo "warmup failed"; kill "$SERVER_PID"; exit 1; }

# Measured: three fresh cold requests of the same ~32K prompt.
for i in 1 2 3; do
  echo "measured $i..."
  post || { echo "measure $i failed"; kill "$SERVER_PID"; exit 1; }
done

sleep 1
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "--- done idle=$IDLE ---"
