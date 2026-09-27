from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from tools.artifact.schema import ResourceSpec
from tools.artifact.writer import ArtifactWriter
from tools.exl3.sample_traces import (
    fake_tool_result,
    load_prompt_chunks,
    TraceRow,
    pack_calibration_rows,
    render_chat,
    read_artifact_text_resources,
    response_token_ids,
    select_streams,
    template_environment,
    template_tools,
    tool_case,
)


def _manifest() -> dict:
    return {
        "streams": [
            {"id": f"wikitext-{index:02}", "path": f"{index}.txt"}
            for index in range(4)
        ]
    }


def test_calibration_and_evaluation_splits_are_disjoint_and_balanced() -> None:
    calibration = select_streams(_manifest(), "calibration")
    evaluation = select_streams(_manifest(), "evaluation")
    assert [item["id"] for item in calibration] == ["wikitext-00", "wikitext-01"]
    assert [item["id"] for item in evaluation] == ["wikitext-02", "wikitext-03"]
    assert {item["id"] for item in calibration}.isdisjoint(
        {item["id"] for item in evaluation}
    )
    with pytest.raises(ValueError, match="split"):
        select_streams(_manifest(), "other")


def test_pack_calibration_rows_records_true_lengths_and_right_padding() -> None:
    rows = [
        TraceRow("wikitext-00", [1, 2], [3], "first"),
        TraceRow("pg19-00", [4, 5], [6, 7, 8, 9], "second"),
    ]
    packed, lengths = pack_calibration_rows(rows, 4)
    assert packed.tolist() == [[1, 2, 3, 0], [4, 5, 6, 7]]
    assert lengths.tolist() == [3, 4]
    with pytest.raises(ValueError, match="must not be empty"):
        pack_calibration_rows([], 4)


def test_tool_traces_use_disjoint_seed_locations_and_scripted_results() -> None:
    calibration_prompt, calibration_tools = tool_case("calibration", 0)
    evaluation_prompt, evaluation_tools = tool_case("evaluation", 0)
    assert "Oslo" in calibration_prompt and "Tallinn" in evaluation_prompt
    assert calibration_tools == evaluation_tools
    result = fake_tool_result(
        {
            "id": "call-1",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Oslo","days":2}',
            },
        },
        0,
    )
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call-1"
    assert '"location": "Oslo"' in result["content"]


def test_rendered_turn_retokenizes_response_after_rendered_prompt_prefix() -> None:
    template = template_environment().from_string(
        "{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )

    class ByteTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False):
            return SimpleNamespace(ids=list(text.encode("utf-8")))

    messages = [{"role": "user", "content": "question"}]
    prompt = render_chat(template, messages, generation_prompt=True)
    prompt_ids = ByteTokenizer.encode(prompt).ids
    response_ids, transcript = response_token_ids(
        ByteTokenizer,
        prompt_ids,
        [*messages, {"role": "assistant", "content": "answer"}],
        template,
    )
    assert response_ids == list(b"answer;")
    assert transcript.endswith("assistant:answer;")
    with pytest.raises(ValueError, match="changed its prompt text"):
        response_token_ids(ByteTokenizer, [999], messages, template)


def test_trace_sampler_reads_tokenizer_and_template_from_artifact(tmp_path) -> None:
    tokenizer_json = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).to_str()
    chat_template = (
        "{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    resources = {"tokenizer.json": "tokenizer", "chat_template.jinja": "template"}
    output = tmp_path / "trace-model.ninfer"
    with ArtifactWriter(
        output,
        [
            ResourceSpec("tokenizer", len(tokenizer_json.encode())),
            ResourceSpec("template", len(chat_template.encode())),
        ],
        components={"text": {"config": {}, "resources": resources}},
        bindings={},
    ) as writer:
        writer.write_object("tokenizer", tokenizer_json.encode())
        writer.write_object("template", chat_template.encode())

    metadata, tokenizer, template = read_artifact_text_resources(output)
    assert metadata == {}
    assert tokenizer.encode("test", add_special_tokens=False).ids
    assert render_chat(
        template, [{"role": "user", "content": "hello"}], generation_prompt=True
    ) == "user:hello;assistant:"


def test_local_qwen38_artifact_can_render_and_tokenize_trace_rows() -> None:
    artifact_path = Path("models/qwen3_8_27b.ninfer")
    if not artifact_path.is_file():
        pytest.skip("local Qwen3.8 artifact is not installed")
    _, tokenizer, template = read_artifact_text_resources(artifact_path)
    messages = [{"role": "user", "content": "Continue this short test passage."}]
    prompt = render_chat(template, messages, generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    response_ids, transcript = response_token_ids(
        tokenizer,
        prompt_ids,
        [*messages, {"role": "assistant", "content": "The passage continues."}],
        template,
    )
    assert prompt_ids
    assert response_ids
    assert "The passage continues." in transcript
    tool_prompt, tools = tool_case("calibration", 0)
    tool_rendered = render_chat(
        template,
        [{"role": "user", "content": tool_prompt}],
        generation_prompt=True,
        tools=tools,
    )
    assert "get_weather" in tool_rendered


def test_qwen38_corpus_splits_have_enough_disjoint_trace_prompts() -> None:
    artifact_path = Path("models/qwen3_8_27b.ninfer")
    manifest_path = Path("eval/corpora/perplexity-1m/manifest.json")
    if not artifact_path.is_file():
        pytest.skip("local Qwen3.8 artifact is not installed")
    _, tokenizer, _ = read_artifact_text_resources(artifact_path)
    calibration = load_prompt_chunks(manifest_path, "calibration", tokenizer, 1400)
    evaluation = load_prompt_chunks(manifest_path, "evaluation", tokenizer, 1400)
    assert len(calibration) >= 250
    assert len(evaluation) >= 250
    assert {row.stream_id for row in calibration}.isdisjoint(
        {row.stream_id for row in evaluation}
    )


def test_template_tojson_matches_ninfer_default_separators() -> None:
    # ninfer-serve's template engine (like HF) renders tojson with ", "/": " and no ASCII escaping.
    template = template_environment().from_string(
        "{{ v|tojson }}|{{ v|tojson(separators=(',', ':'), sort_keys=true) }}"
    )
    assert template.render(v={"z": "北京", "a": [1, 2]}) == (
        '{"z": "北京", "a": [1, 2]}|{"a":[1,2],"z":"北京"}'
    )


def test_template_tools_match_the_server_canonical_tool_object() -> None:
    _, tools = tool_case("evaluation", 0)
    [canonical] = template_tools(tools)
    function = canonical["function"]
    assert canonical["type"] == "function"
    assert list(function) == ["name", "parameters", "strict", "description"]
    assert function["strict"] is False
    assert template_tools(None) is None
