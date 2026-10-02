import json
import sys

# usage: gen_openai.py <messages_fixture.json> <out.json> <max_tokens>
fixture, out, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(fixture) as f:
    messages = json.load(f)
payload = {
    "model": "qwen3.8-flash-next",
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": False,
}
with open(out, "w") as f:
    json.dump(payload, f)
print("wrote", out, "messages", len(messages), "max_tokens", max_tokens)
