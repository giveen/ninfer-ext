#!/usr/bin/env bash
# A/B one idle-prefill width: long-prompt prefill phase + short-prompt decode phase.
# Usage: run_ab.sh <idle_width> [port]
set -u

IDLE="${1:?idle width required}"
PORT="${2:-18080}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
AB="${STRATA_AB_DIR:-/tmp/strata-ab}"
mkdir -p "$AB"
ART="$ROOT/models/Qwen3.8-Flash-NVFP4/qwen3_8_flash_next_nvfp4.ninfer"

JSONL="$AB/ab-${IDLE}.jsonl"
LOG="$AB/ab-${IDLE}.log"
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
  echo "SERVER NOT READY (idle=$IDLE)"; tail -40 "$LOG"
  kill "$SERVER_PID" 2>/dev/null; exit 1
fi
echo "--- serve ready (idle=$IDLE) ---"
grep -iE "capacity|prefill \|" "$LOG" | head -20

post() {
  curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$1" >/dev/null
}

LONG="$HERE/payloads/request.json"
DECODE="$HERE/payloads/request_decode.json"

echo "prefill warmup..."
post "$LONG" || { echo "prefill warmup failed"; kill "$SERVER_PID"; exit 1; }
for i in 1 2 3; do echo "prefill $i..."; post "$LONG" || { echo "prefill $i failed"; kill "$SERVER_PID"; exit 1; }; done

echo "decode warmup..."
post "$DECODE" || { echo "decode warmup failed"; kill "$SERVER_PID"; exit 1; }
for i in 1 2 3; do echo "decode $i..."; post "$DECODE" || { echo "decode $i failed"; kill "$SERVER_PID"; exit 1; }; done

sleep 1
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "--- done idle=$IDLE ---"
