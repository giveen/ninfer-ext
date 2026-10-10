"""Writes the Gemma 4 chat-template fixture from HuggingFace's own rendering.

The oracle is `transformers`' `apply_chat_template` over the checkpoint's chat_template.jinja and
tokenizer, so it needs an interpreter whose transformers ships Gemma 4 (the 5.x line), for example:

    /mnt/storage/ninfer/eval/.venv/bin/python tools/verify/gemma4_chat_fixture.py \
        /mnt/storage/models/gemma/full-31b tests/fixtures/gemma4/chat_template.json

Every case records the rendered text and its token ids; ninfer_gemma4_frontend_test requires both to
match exactly.
"""
import hashlib
import json
import sys

from transformers import AutoTokenizer

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}

CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}},
}

CASES = [
    {"name": "single_user", "thinking": False,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "system_and_history_thinking", "thinking": True,
     "messages": [
         {"role": "system", "content": "  You are terse.  "},
         {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello."},
         {"role": "user", "content": "Count to three."},
     ]},
    {"name": "unicode_and_whitespace", "thinking": False,
     "messages": [{"role": "user", "content": "Ünïcödé — 中文 😀\n\n  indented\tline "}]},
    {"name": "tool_declaration", "thinking": False, "tools": [WEATHER],
     "messages": [{"role": "user", "content": "Weather in Paris for two days?"}]},
    {"name": "tool_round_trip", "thinking": False, "tools": [WEATHER],
     "messages": [
         {"role": "user", "content": "Weather in Paris for two days?"},
         {"role": "assistant", "content": "", "tool_calls": [CALL]},
         {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 18}"},
     ]},
    {"name": "tool_round_trip_thinking", "thinking": True, "tools": [WEATHER],
     "messages": [
         {"role": "user", "content": "Weather in Paris for two days?"},
         {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
          "tool_calls": [CALL]},
         {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
         {"role": "assistant", "content": "It is sunny."},
         {"role": "user", "content": "Thanks"},
     ]},
]


def main():
    checkpoint, output = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    with open(f"{checkpoint}/chat_template.jinja", "rb") as source:
        template_digest = hashlib.sha256(source.read()).hexdigest()
    cases = []
    for case in CASES:
        text = tokenizer.apply_chat_template(
            case["messages"], tools=case.get("tools"), tokenize=False,
            add_generation_prompt=True, enable_thinking=case["thinking"])
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({**case, "text": text, "ids": ids})
    import transformers
    fixture = {
        "source": f"{checkpoint}/chat_template.jinja",
        "source_sha256": template_digest,
        "oracle": f"transformers {transformers.__version__}: apply_chat_template(tokenize=False, "
                  "add_generation_prompt=True), then encode(add_special_tokens=False)",
        "cases": cases,
    }
    with open(output, "w", encoding="utf-8") as sink:
        json.dump(fixture, sink, ensure_ascii=False, indent=1)
        sink.write("\n")


if __name__ == "__main__":
    main()
