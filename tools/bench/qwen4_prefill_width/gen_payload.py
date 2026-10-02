import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
src = os.path.join(REPO, "examples/cli/messages/long_niah_8k.json")
out = sys.argv[1]
mult = int(sys.argv[2]) if len(sys.argv) > 2 else 4

with open(src) as f:
    base = json.load(f)

user_content = next(m["content"] for m in base if m["role"] == "user")
system_content = next((m["content"] for m in base if m["role"] == "system"), None)

body = user_content * mult
payload = {
    "model": "qwen3.8-flash-next",
    "messages": (
        ([{"role": "system", "content": system_content}] if system_content else [])
        + [{"role": "user", "content": body}]
    ),
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}

with open(out, "w") as f:
    json.dump(payload, f)

print("content_chars", len(body), "payload_bytes", len(json.dumps(payload)))
