"""Writes the Nemotron-H frontend fixtures from the artifact's own text resources.

Two oracles, both reading the artifact rather than a checkpoint directory, so a fixture can never
describe bytes the loader would not see:

  * the tokenizer oracle is `tokenizers` (Rust) reading the artifact's `tokenizer.json`, which is the
    pipeline the checkpoint was trained with;
  * the chat-template oracle is `transformers`' `apply_chat_template` over the artifact's own
    `chat_template.jinja`.

The tokenizer cases walk the Nanotron-v2 split branch by branch -- the camelCase letter pair, the
single digit, the punctuation run that swallows trailing CR/LF/slash, newline runs, the trailing-space
rule, combining marks, every letter class the pair distinguishes, and the added tokens the protocol
depends on -- because those are the cases a reimplementation of the pattern gets wrong.

    python3 -m tools.verify.nemotron_h_frontend_fixture [artifact]

Defaults to `/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4.ninfer` and writes
`tests/fixtures/nemotron_h/tokenizer_ids.json` and `tests/fixtures/nemotron_h/chat_cases.json`.
"""
from __future__ import annotations

import hashlib
import json
import sys

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

from tools.artifact.reader import Artifact

ARTIFACT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4.ninfer"
OUT_DIR = "tests/fixtures/nemotron_h"

# One case per shape the split distinguishes. The comment names the branch it exercises.
TOKENIZER_CASES = [
    # The camelCase-aware letter pair: an upper run then a lower run, and the reverse.
    "PascalCase and HTTPServer",
    "iOS XMLHttpRequest JSON2XML",
    "ABCDef abcDEF ABc abC A B a",
    "aBc dEf",
    # The optional one-codepoint prefix, which is how a space attaches to the next word.
    "a b  c   d",
    " leading and trailing ",
    "don't it's they'll we've I'm",
    # A single digit is its own word; letters and digits do not join.
    "1 23 456 7th 3D x5 5x a1b2c3",
    "0.5 -12 1,000 3.14159",
    # The punctuation run, and its tail swallowing CR, LF and slash.
    "!!! ... ??? ,;:",
    "end. Next! Really?",
    "path/to/file a/b/c 1/2 x\\y",
    "a!/\n b...\r\n c//",
    "((nested)) [brackets] {braces}",
    "quote\"and'apostrophe",
    # Newline runs and mixed whitespace.
    "\n",
    "\n\n\n",
    "a\nb",
    "a\n\nb",
    "a\r\nb",
    "a \n b",
    "a \t b",
    "a\t\tb\n \n c",
    "trailing spaces   ",
    "   ",
    "\t",
    # Combining marks, which the letter pair admits and the punctuation class does not exclude.
    "e\u0301 a\u0301b \u0301leading",
    "नमस्ते दुनिया",
    # Every letter class the pair names: Lu, Ll, Lt, Lm, Lo, M.
    "\u01c5 titlecase \u01c4 \u01c6",
    "\u02b0\u02b1 modifier \u02b2",
    "\u03b1\u03b2\u03b3 \u0391\u0392\u0393",
    "\u0430\u0431\u0432 \u0410\u0411\u0412",
    "T\u00fcrk\u00e7e \u0130stanbul",
    "\u4e2d\u6587\u6d4b\u8bd5 \u65e5\u672c\u8a9e",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8",
    # Numbers outside Nd, and letters that are neither cased nor characters of a word.
    "\u0663 \u00bd \u2167",
    "\u3042\u3044\u3046 \u30a2\u30a4\u30a6",
    # Symbols and emoji are not letters, so they take the punctuation run.
    "\U0001f600\U0001f389 emoji \u2764\ufe0f",
    "\u20ac100 $5 \u00a310",
    # Real prose, where every branch meets the others.
    " = Robert Boulter = \n\n Robert Boulter is an English film , television and theatre actor . "
    "He had a guest @-@ starring role on the television series The Bill in 2000 .",
    "The capital of France is Paris, the largest city in the country and one of the most visited "
    "places in the world. Millions of people travel there every year.",
    # The tokens the protocol depends on: they are added tokens, so they are recognized as units.
    "<|im_start|>user\nhi<|im_end|>\n",
    "<think>reasoning</think>answer",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
    "</tool_call>",
    "text <|im_end|> text",
    "<tool_response>\n{\"temp\": 18}\n</tool_response>",
    # A protobuf-ish blob, for the punctuation and digit branches together.
    "{\"a\":1,\"b\":[2,3],\"c\":{\"d\":4}}",
]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city"},
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

CHAT_CASES = [
    {"name": "single_user", "thinking": False,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "single_user_thinking", "thinking": True,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "system_and_history", "thinking": False,
     "messages": [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Name a colour."},
         {"role": "assistant", "content": "Blue."},
         {"role": "user", "content": "And a second one?"},
     ]},
    {"name": "history_with_reasoning", "thinking": True,
     "messages": [
         {"role": "system", "content": "Be brief."},
         {"role": "user", "content": "Name a colour."},
         {"role": "assistant", "content": "Blue.", "reasoning_content": "The sky is blue."},
         {"role": "user", "content": "And a second one?"},
     ]},
    {"name": "unicode_and_whitespace", "thinking": False,
     "messages": [{"role": "user", "content": "\u00dcn\u00efc\u00f6d\u00e9 \u2014 \u4e2d\u6587 "
                                                "\U0001f600\n\n  indented\tline "}]},
    {"name": "tool_declaration", "thinking": False, "tools": [WEATHER],
     "messages": [{"role": "user", "content": "Weather in Paris for two days?"}]},
    {"name": "tool_declaration_thinking", "thinking": True, "tools": [WEATHER],
     "messages": [{"role": "user", "content": "Weather in Paris for two days?"}]},
    {"name": "tool_round_trip", "thinking": False, "tools": [WEATHER],
     "messages": [
         {"role": "user", "content": "Weather in Paris for two days?"},
         {"role": "assistant", "content": "", "tool_calls": [CALL]},
         {"role": "tool", "content": "{\"temp\": 18}"},
     ]},
    {"name": "tool_round_trip_thinking", "thinking": True, "tools": [WEATHER],
     "messages": [
         {"role": "user", "content": "Weather in Paris for two days?"},
         {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
          "tool_calls": [CALL]},
         {"role": "tool", "content": "sunny"},
         {"role": "assistant", "content": "It is sunny."},
         {"role": "user", "content": "Thanks"},
     ]},
]


def read_resources(artifact: Artifact) -> dict[str, bytes]:
    return {
        role: artifact.read_object(f"resource/text/{role}")
        for role in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "chat_template.jinja")
    }


def byte_offsets(text: str, spans) -> list[list[int]]:
    """The reference reports character offsets; the runtime walks bytes, so both sides must be in the
    same space before a boundary can be compared."""
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + len(character.encode("utf-8")))
    return [[prefix[start], prefix[end]] for start, end in spans]


def write_tokenizer_fixture(resources: dict[str, bytes]) -> None:
    tokenizer_json = resources["tokenizer.json"]
    oracle = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    import tokenizers

    cases = [
        {
            "text": text,
            "ids": oracle.encode(text, add_special_tokens=False).ids,
            # The split's own pieces, in the runtime's byte coordinates: the oracle for the family's
            # word-boundary rule on its own, independent of the merge table and vocabulary.
            "pieces": byte_offsets(
                text, (span for _, span in oracle.pre_tokenizer.pre_tokenize_str(text))
            ),
        }
        for text in TOKENIZER_CASES
    ]
    fixture = {
        "source": f"{ARTIFACT}:resource/text/tokenizer.json",
        "source_sha256": hashlib.sha256(tokenizer_json).hexdigest(),
        "source_bytes": len(tokenizer_json),
        "vocab_size": oracle.get_vocab_size(),
        "oracle": f"tokenizers {tokenizers.__version__}: "
                  "Tokenizer.from_str(...).encode(text, add_special_tokens=False)",
        "cases": cases,
    }
    with open(f"{OUT_DIR}/tokenizer_ids.json", "w", encoding="utf-8") as sink:
        json.dump(fixture, sink, ensure_ascii=False, indent=1)
        sink.write("\n")
    print(f"{OUT_DIR}/tokenizer_ids.json: {len(cases)} cases, vocab {oracle.get_vocab_size()}")


def write_chat_fixture(resources: dict[str, bytes]) -> None:
    template = resources["chat_template.jinja"]
    tokenizer = Tokenizer.from_str(resources["tokenizer.json"].decode("utf-8"))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                   chat_template=template.decode("utf-8"),
                                   eos_token="<|im_end|>", bos_token="<s>", unk_token="<unk>")
    cases = []
    for case in CHAT_CASES:
        text = fast.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=case["thinking"])
        cases.append({**case, "text": text, "ids": fast.encode(text, add_special_tokens=False)})
    import transformers

    fixture = {
        "source": f"{ARTIFACT}:resource/text/chat_template.jinja",
        "source_sha256": hashlib.sha256(template).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(resources["tokenizer.json"]).hexdigest(),
        "oracle": f"transformers {transformers.__version__}: "
                  "PreTrainedTokenizerFast(tokenizer_object=..., chat_template=...) "
                  ".apply_chat_template(tokenize=False, add_generation_prompt=True), then "
                  "encode(add_special_tokens=False)",
        "cases": cases,
    }
    with open(f"{OUT_DIR}/chat_cases.json", "w", encoding="utf-8") as sink:
        json.dump(fixture, sink, ensure_ascii=False, indent=1)
        sink.write("\n")
    print(f"{OUT_DIR}/chat_cases.json: {len(cases)} cases")


def main() -> None:
    with Artifact(ARTIFACT) as artifact:
        resources = read_resources(artifact)
    for role, payload in resources.items():
        print(f"{role}: {len(payload)} bytes")
    write_tokenizer_fixture(resources)
    write_chat_fixture(resources)


if __name__ == "__main__":
    main()
