import json
import sys

out = sys.argv[1]
payload = {
    "model": "qwen3.8-flash-next",
    "messages": [
        {"role": "user", "content": "Count from 1 to 300, one integer per line, nothing else."}
    ],
    "max_tokens": 256,
    "temperature": 0,
    "stream": False,
}
with open(out, "w") as f:
    json.dump(payload, f)
print("wrote", out)
