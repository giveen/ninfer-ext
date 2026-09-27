"""Sample disjoint EXL3 calibration/evaluation traces through ``ninfer-serve``.

The trace contains qbench-compatible token rows plus a Safetensors calibration matrix suitable
for ExLlamaV3 reference runs. Model inference always goes through the public NInfer HTTP server;
an opt-in server trace provides exact generated token IDs, while Python renders and tokenizes the
artifact's chat template and supplies deterministic local tool fixtures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import socket
import subprocess
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from jinja2 import Environment
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from tools.artifact.reader import Artifact


@dataclass(frozen=True, slots=True)
class PromptChunk:
    stream_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TraceRow:
    source: str
    input_ids: list[int]
    response_ids: list[int]
    transcript: str
    server_completion_tokens: int | None = None
    finish_reason: str = "unknown"


def select_streams(manifest: dict[str, Any], split: str) -> list[dict[str, str]]:
    """Assign shards 00/01 to calibration and 02/03 to the disjoint evaluation split."""

    if split not in ("calibration", "evaluation"):
        raise ValueError("split must be 'calibration' or 'evaluation'")
    selected = []
    for stream in manifest["streams"]:
        stream_id = stream.get("id")
        if not isinstance(stream_id, str) or "-" not in stream_id:
            raise ValueError(f"invalid corpus stream ID: {stream_id!r}")
        try:
            shard = int(stream_id.rsplit("-", 1)[1])
        except ValueError:
            raise ValueError(f"corpus stream ID has no numeric shard: {stream_id!r}") from None
        if not 0 <= shard <= 3:
            raise ValueError(f"unsupported corpus shard in {stream_id!r}")
        if (shard < 2) == (split == "calibration"):
            selected.append(stream)
    if not selected:
        raise ValueError(f"corpus manifest has no streams for split {split!r}")
    return selected


def load_prompt_chunks(
    manifest_path: Path,
    split: str,
    tokenizer: Tokenizer,
    prompt_tokens: int,
) -> list[PromptChunk]:
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[PromptChunk] = []
    for stream in select_streams(manifest, split):
        path = manifest_path.parent / stream["path"]
        token_ids = tokenizer.encode(path.read_text(encoding="utf-8")).ids
        for begin in range(0, len(token_ids), prompt_tokens):
            selected = token_ids[begin : begin + prompt_tokens]
            if len(selected) < min(128, prompt_tokens):
                continue
            text = tokenizer.decode(selected, skip_special_tokens=True).strip()
            if text:
                chunks.append(PromptChunk(stream["id"], text))
    if not chunks:
        raise ValueError(f"corpus split {split!r} produced no usable prompt chunks")
    return chunks


def template_environment() -> Environment:
    environment = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    environment.globals["raise_exception"] = raise_exception
    # Match NInfer's template engine (and HF): Python's default ", "/": " separators, no ASCII
    # escaping, with the same optional arguments.
    def tojson(value, ensure_ascii=False, sort_keys=False, separators=None, indent=None):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=tuple(separators) if separators is not None else None,
            indent=indent,
        )

    environment.filters["tojson"] = tojson
    return environment


def template_tools(tools: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Canonical tool objects as ninfer-serve passes them to the template (translate.cpp)."""
    if tools is None:
        return None
    canonical = []
    for tool in tools:
        source = tool["function"]
        function = {"name": source["name"], "parameters": source["parameters"], "strict": False}
        if source.get("description"):
            function["description"] = source["description"]
        canonical.append({"type": "function", "function": function})
    return canonical


def render_chat(
    template,
    messages: Sequence[dict[str, Any]],
    *,
    generation_prompt: bool,
    tools: Sequence[dict[str, Any]] | None = None,
) -> str:
    return template.render(
        messages=list(messages),
        tools=template_tools(tools),
        add_generation_prompt=generation_prompt,
        enable_thinking=True,
        reasoning_effort="xhigh",
        preserve_thinking=True,
        preserve_reasoning=True,
        tool_call_format="xml",
    )


def response_token_ids(
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    completed_messages: Sequence[dict[str, Any]],
    template,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[int], str]:
    completed = render_chat(
        template, completed_messages, generation_prompt=False, tools=tools
    )
    if not completed_messages:
        raise ValueError("completed trace turn has no assistant message")
    prompt_text = render_chat(
        template, completed_messages[:-1], generation_prompt=True, tools=tools
    )
    if not completed.startswith(prompt_text):
        raise ValueError(
            "completed chat rendering changed its prompt text; refusing to create a "
            "misaligned trace row"
        )
    response_text = completed[len(prompt_text) :]
    response_ids = tokenizer.encode(response_text, add_special_tokens=False).ids
    return response_ids, completed


def pack_calibration_rows(rows: Sequence[TraceRow], width: int) -> tuple[np.ndarray, np.ndarray]:
    if width <= 0:
        raise ValueError("calibration row width must be positive")
    if not rows:
        raise ValueError("calibration rows must not be empty")
    packed = np.zeros((len(rows), width), dtype=np.int64)
    lengths = np.zeros((len(rows),), dtype=np.int64)
    for index, row in enumerate(rows):
        tokens = row.input_ids + row.response_ids
        if not tokens:
            raise ValueError(f"trace row {index} has no tokens")
        length = min(len(tokens), width)
        packed[index, :length] = tokens[:length]
        lengths[index] = length
    return packed, lengths


def read_artifact_text_resources(path: Path) -> tuple[dict[str, Any], Tokenizer, str]:
    with Artifact(path) as artifact:
        component = artifact.directory.components["text"]
        resources = component["resources"]
        tokenizer_data = artifact.read_object(resources["tokenizer.json"])
        template_data = artifact.read_object(resources["chat_template.jinja"])
        metadata = dict(artifact.directory.metadata)
    tokenizer = Tokenizer.from_str(tokenizer_data.decode("utf-8"))
    template = template_environment().from_string(template_data.decode("utf-8"))
    return metadata, tokenizer, template


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def wait_for_server(process: subprocess.Popen[bytes], host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/v1/models"
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ninfer-serve exited early with status {process.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
                last_error = f"readiness endpoint returned HTTP {response.status}"
        except (OSError, HTTPError, URLError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for ninfer-serve: {last_error}")


def post_chat(
    host: str,
    port: int,
    payload: dict[str, Any],
    timeout: float,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    if tools is not None:
        payload = {**payload, "tools": list(tools), "tool_choice": tool_choice or "auto"}
    elif tool_choice is not None:
        payload = {**payload, "tool_choice": tool_choice}
    request = Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ninfer-serve returned HTTP {error.code}: {detail}") from error
    if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
        raise RuntimeError("ninfer-serve returned an invalid Chat Completions response")
    if not result["choices"] or not isinstance(result["choices"][0], dict):
        raise RuntimeError("ninfer-serve returned no Chat Completions choice")
    choice = result["choices"][0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("ninfer-serve Chat Completions choice has no message")
    message = dict(message)
    message["_trace_finish_reason"] = choice.get("finish_reason", "unknown")
    usage = result.get("usage")
    if isinstance(usage, dict):
        message["_trace_prompt_tokens"] = usage.get("prompt_tokens")
        message["_trace_completion_tokens"] = usage.get("completion_tokens")
    return message


def make_prompt(chunk: PromptChunk) -> str:
    return (
        "Continue the following passage in its original language and style. Add a useful next "
        "section without repeating it.\n\n" + chunk.text
    )


def tool_case(split: str, index: int) -> tuple[str, list[dict[str, Any]]]:
    cities = (
        ("Oslo", "Bergen", "Helsinki", "Stockholm", "Copenhagen")
        if split == "calibration"
        else ("Tallinn", "Reykjavik", "Dublin", "Lisbon", "Warsaw")
    )
    city = cities[index % len(cities)]
    prompt = (
        f"Use the weather lookup tool to check the forecast for {city}, then advise what "
        "clothing a visitor should pack."
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get a current weather summary and forecast for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    return prompt, tools


def fake_tool_result(call: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ValueError("NInfer returned a malformed tool call")
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("NInfer returned a malformed tool call")
    if function["name"] != "get_weather":
        raise ValueError(f"no local trace fixture for tool {function['name']!r}")
    raw_arguments = function.get("arguments", "{}")
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        raise ValueError("NInfer returned malformed tool arguments")
    weather = ("light rain", "cloudy", "sunny", "windy", "showers")
    result = {
        "location": arguments.get("location", "unknown"),
        "forecast": weather[index % len(weather)],
        "high_c": 7 + index % 9,
        "low_c": 1 + index % 5,
    }
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"trace-tool-{index}"
    return {
        "role": "tool",
        "name": function["name"],
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def sample_turn(
    host: str,
    port: int,
    tokenizer: Tokenizer,
    template,
    messages: Sequence[dict[str, Any]],
    *,
    source: str,
    model_id: str,
    seed: int,
    args: argparse.Namespace,
    tools: Sequence[dict[str, Any]] | None,
    tool_choice: str | None = None,
) -> tuple[dict[str, Any], TraceRow]:
    prompt_text = render_chat(
        template, messages, generation_prompt=True, tools=tools
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    max_context = max(4096, args.row_tokens + 512)
    if len(prompt_ids) + args.max_new_tokens > max_context:
        raise ValueError(f"prompt from {source} exceeds the configured server context")
    payload = {
        "model": model_id,
        "messages": list(messages),
        "max_completion_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": seed,
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
        "preserve_thinking": True,
        "stream": False,
    }
    response = post_chat(
        host, port, payload, args.timeout, tools=tools, tool_choice=tool_choice
    )
    assistant = {"role": "assistant"}
    for key in ("content", "reasoning_content", "tool_calls"):
        if key in response:
            assistant[key] = response[key]
    response_ids, transcript = response_token_ids(
        tokenizer, prompt_ids, [*messages, assistant], template, tools=tools
    )
    if not response_ids:
        raise RuntimeError(f"NInfer returned an empty assistant turn for {source}")
    reported_prompt = response.get("_trace_prompt_tokens")
    if isinstance(reported_prompt, int) and reported_prompt != len(prompt_ids):
        raise ValueError(
            f"artifact tokenizer/template produced {len(prompt_ids)} prompt tokens for {source}, "
            f"but ninfer-serve reports {reported_prompt}"
        )
    reported_completion = response.get("_trace_completion_tokens")
    if not isinstance(reported_completion, int):
        reported_completion = None
    return assistant, TraceRow(
        source,
        prompt_ids,
        response_ids,
        transcript,
        reported_completion,
        str(response.get("_trace_finish_reason", "unknown")),
    )


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "trace": Path(str(prefix) + ".trace.json"),
        "calibration": Path(str(prefix) + ".calibration.safetensors"),
        "text": Path(str(prefix) + ".text"),
        "server_log": Path(str(prefix) + ".server.jsonl"),
        "server_output": Path(str(prefix) + ".server.log"),
        "generated_tokens": Path(str(prefix) + ".generated-token-ids.jsonl"),
    }


def read_generated_token_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"ninfer-serve did not produce its token trace: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid token trace JSON on line {line_number}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"token trace line {line_number} is not an object")
        if (
            record.get("schema") != "ninfer_generated_token_trace"
            or record.get("schema_version") != 1
            or not isinstance(record.get("request_id"), int)
            or not isinstance(record.get("prompt_tokens"), int)
            or not isinstance(record.get("completion_tokens"), int)
            or not isinstance(record.get("generated_token_ids"), list)
        ):
            raise RuntimeError(f"token trace line {line_number} has an invalid schema")
        token_ids = record["generated_token_ids"]
        if any(type(token) is not int or token < 0 for token in token_ids):
            raise RuntimeError(f"token trace line {line_number} has invalid token IDs")
        if record["completion_tokens"] != len(token_ids):
            raise RuntimeError(f"token trace line {line_number} has an inconsistent token count")
        records.append(record)
    return records


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--serve", type=Path, default=Path("build/apps/ninfer-serve"))
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--row-tokens", type=int, default=2048)
    parser.add_argument("--prompt-tokens", type=int, default=1400)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tool-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.rows <= 0 or args.row_tokens <= 0 or args.prompt_tokens <= 0:
        parser.error("rows and token limits must be positive")
    if args.max_new_tokens <= 0 or args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("generation and startup timeouts must be positive")
    if not 0.0 <= args.temperature or not 0.0 < args.top_p <= 1.0:
        parser.error("temperature must be nonnegative and top-p must be in (0, 1]")
    if not 0.0 <= args.tool_fraction <= 1.0:
        parser.error("tool-fraction must be in [0, 1]")
    if args.port < 0 or args.port > 65535:
        parser.error("port must be 0 (choose an unused port) or in [1,65535]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.artifact.is_file() or not args.serve.is_file():
        raise FileNotFoundError("artifact and ninfer-serve executable must exist")
    if not args.corpus_manifest.is_file():
        raise FileNotFoundError(args.corpus_manifest)

    paths = output_paths(args.output_prefix)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"trace output already exists: {path}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata, tokenizer, template = read_artifact_text_resources(args.artifact)
    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    streams = select_streams(manifest, args.split)
    chunks = load_prompt_chunks(args.corpus_manifest, args.split, tokenizer, args.prompt_tokens)
    if len(chunks) < args.rows:
        raise ValueError(
            f"split {args.split!r} has {len(chunks)} prompt chunks, fewer than requested "
            f"{args.rows} rows"
        )
    random.Random(args.seed).shuffle(chunks)
    chunks = chunks[: args.rows]

    host = "127.0.0.1"
    port = args.port or free_port()
    model_id = metadata.get("name", "qwen3.8-27b")
    if not isinstance(model_id, str) or not model_id:
        model_id = "qwen3.8-27b"
    server_command = [
        str(args.serve),
        str(args.artifact),
        "--host",
        host,
        "--port",
        str(port),
        "--max-context",
        str(max(4096, args.row_tokens + 512)),
        "--kv-capacity",
        str(max(4096, args.row_tokens + 512)),
        "--max-concurrency",
        "1",
        "--kv-dtype",
        "fp8",
        "--device",
        str(args.device),
        "--no-cuda-graph",
        "--request-log-jsonl",
        str(paths["server_log"]),
        "--generation-token-trace-jsonl",
        str(paths["generated_tokens"]),
    ]

    traces: list[dict[str, Any]] = []
    rows: list[TraceRow] = []
    process: subprocess.Popen[bytes] | None = None
    with paths["server_output"].open("xb") as server_log:
        try:
            process = subprocess.Popen(
                server_command,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            wait_for_server(process, host, port, args.startup_timeout)
            transcripts = []
            tool_rows = round(args.rows * args.tool_fraction)
            tool_call_count = 0
            for index, chunk in enumerate(chunks):
                tools = None
                user_text = make_prompt(chunk)
                if index < tool_rows:
                    tool_prompt, tools = tool_case(args.split, index)
                    user_text += "\n\n" + tool_prompt
                user_message = {"role": "user", "content": user_text}
                messages = [user_message]
                assistant, trace_row = sample_turn(
                    host,
                    port,
                    tokenizer,
                    template,
                    messages,
                    source=chunk.stream_id,
                    model_id=model_id,
                    seed=args.seed + index,
                    args=args,
                    tools=tools,
                )
                messages.append(assistant)
                rows.append(trace_row)
                traces.append(
                    {
                        "conversation": index,
                        "turn": 0,
                        "source_stream": chunk.stream_id,
                        "input_ids": trace_row.input_ids,
                        "response_ids": trace_row.response_ids,
                        "server_completion_tokens": trace_row.server_completion_tokens,
                        "finish_reason": trace_row.finish_reason,
                        "tools": tools is not None,
                    }
                )
                print(
                    f"[{index + 1}/{args.rows}] {chunk.stream_id}: "
                    f"{len(trace_row.input_ids)} prompt + "
                    f"{len(trace_row.response_ids)} sampled tokens",
                    flush=True,
                )
                tool_calls = assistant.get("tool_calls")
                call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
                tool_call_count += call_count
                traces[-1]["tool_call_count"] = call_count
                if tools is not None and isinstance(tool_calls, list) and tool_calls:
                    messages.extend(
                        fake_tool_result(call, index + offset)
                        for offset, call in enumerate(tool_calls)
                    )
                    final_assistant, followup = sample_turn(
                        host,
                        port,
                        tokenizer,
                        template,
                        messages,
                        source=chunk.stream_id,
                        model_id=model_id,
                        seed=args.seed + args.rows + index,
                        args=args,
                        tools=tools,
                        tool_choice="none",
                    )
                    messages.append(final_assistant)
                    rows.append(followup)
                    traces.append(
                        {
                            "conversation": index,
                            "turn": 1,
                            "source_stream": chunk.stream_id,
                            "input_ids": followup.input_ids,
                            "response_ids": followup.response_ids,
                            "server_completion_tokens": followup.server_completion_tokens,
                            "finish_reason": followup.finish_reason,
                            "tool_call_count": 0,
                            "tools": True,
                        }
                    )
                    print(
                        f"  tool follow-up: {len(followup.input_ids)} prompt + "
                        f"{len(followup.response_ids)} sampled tokens",
                        flush=True,
                    )
                transcripts.append(
                    render_chat(template, messages, generation_prompt=False, tools=tools)
                )

            stop_server(process)
            process = None
            token_records = read_generated_token_trace(paths["generated_tokens"])
            if len(token_records) != len(rows):
                raise RuntimeError(
                    f"ninfer-serve emitted {len(token_records)} generated-token records for "
                    f"{len(rows)} successful Chat requests"
                )
            reconstructed_response_tokens = []
            trace_request_ids = set()
            for index, (row, trace_record, qbench_row) in enumerate(
                zip(rows, token_records, traces)
            ):
                exact_tokens = trace_record["generated_token_ids"]
                request_id = trace_record["request_id"]
                if request_id in trace_request_ids:
                    raise RuntimeError(
                        f"duplicate request ID in generated token trace: {request_id}"
                    )
                trace_request_ids.add(request_id)
                if trace_record["prompt_tokens"] != len(row.input_ids):
                    raise RuntimeError(
                        f"token trace row {index} has {trace_record['prompt_tokens']} prompt "
                        f"tokens, expected {len(row.input_ids)}"
                    )
                if (
                    row.server_completion_tokens is not None
                    and row.server_completion_tokens != len(exact_tokens)
                ):
                    raise RuntimeError(
                        f"token trace row {index} disagrees with Chat Completions usage"
                    )
                reconstructed_response_tokens.append(len(row.response_ids))
                exact_row = TraceRow(
                    source=row.source,
                    input_ids=row.input_ids,
                    response_ids=exact_tokens,
                    transcript=row.transcript,
                    server_completion_tokens=len(exact_tokens),
                    finish_reason=row.finish_reason,
                )
                rows[index] = exact_row
                qbench_row["response_ids"] = exact_tokens
                qbench_row["response_tokens_exact"] = True

            packed, lengths = pack_calibration_rows(rows, args.row_tokens)
            exact_completion_tokens = [len(row.response_ids) for row in rows]
            trace_data = {
                "model": model_id,
                "split": args.split,
                "seed": args.seed,
                "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
                "template_vars": {
                    "enable_thinking": True,
                    "reasoning_effort": "xhigh",
                    "preserve_thinking": True,
                },
                "tokenization": {
                    "prompt": "artifact tokenizer over rendered chat template; server count checked",
                    "response": "exact Engine generated_token_ids from the opt-in server trace",
                    "response_text_retokenization_count_mismatches": sum(
                        reconstructed != exact
                        for reconstructed, exact in zip(
                            reconstructed_response_tokens, exact_completion_tokens
                        )
                    ),
                },
                "meta": {
                    "rows": len(rows),
                    "input_tokens": sum(len(row.input_ids) for row in rows),
                    "output_tokens": sum(len(row.response_ids) for row in rows),
                    "server_output_tokens": sum(exact_completion_tokens),
                    "packed_tokens": int(lengths.sum()),
                    "row_tokens": args.row_tokens,
                    "source_streams": sorted({stream["id"] for stream in streams}),
                    "artifact": str(args.artifact.resolve()),
                    "tool_seed_rows": tool_rows,
                    "tool_calls": tool_call_count,
                },
                "rows": traces,
            }
            save_file({"input_ids": packed, "lengths": lengths}, str(paths["calibration"]))
            paths["trace"].write_text(
                json.dumps(trace_data, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            paths["text"].write_text("\n\n".join(transcripts), encoding="utf-8")
            print(
                f"wrote {len(rows)} rows to {paths['trace']} and {paths['calibration']}; "
                f"text stream: {paths['text']}",
                flush=True,
            )
        finally:
            if process is not None and process.poll() is None:
                stop_server(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
