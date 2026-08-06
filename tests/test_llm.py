"""T-1.6 verification: the Ollama client, against a mocked transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.brain.llm import OllamaClient, ToolCall, _parse_arguments
from jarvis.config import JarvisConfig, load_config
from jarvis.util.errors import LlmError


def _ndjson(*objects: dict[str, Any]) -> str:
    return "\n".join(json.dumps(obj) for obj in objects) + "\n"


def _client(cfg: JarvisConfig, handler: Any) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient(cfg, httpx.Client(transport=transport))


def _stream_handler(body: str, status: int = 200) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


class TestStreaming:
    def test_content_deltas_arrive_in_order(self, cfg: JarvisConfig) -> None:
        body = _ndjson(
            {"message": {"content": "The "}, "done": False},
            {"message": {"content": "answer "}, "done": False},
            {"message": {"content": "is 4."}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 9},
        )
        with _client(cfg, _stream_handler(body)) as client:
            chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

        assert "".join(chunk.content for chunk in chunks) == "The answer is 4."
        assert chunks[-1].done is True
        assert chunks[-1].metrics == {"eval_count": 9}

    def test_blank_lines_are_skipped(self, cfg: JarvisConfig) -> None:
        body = '{"message":{"content":"a"},"done":false}\n\n{"message":{},"done":true}\n'
        with _client(cfg, _stream_handler(body)) as client:
            assert "".join(c.content for c in client.chat_stream([])) == "a"

    def test_malformed_stream_lines_are_skipped(self, cfg: JarvisConfig) -> None:
        body = '{"message":{"content":"a"},"done":false}\nNOT JSON\n{"message":{},"done":true}\n'
        with _client(cfg, _stream_handler(body)) as client:
            assert "".join(c.content for c in client.chat_stream([])) == "a"

    def test_non_streaming_chat_assembles_the_content(self, cfg: JarvisConfig) -> None:
        body = _ndjson(
            {"message": {"content": "Half "}, "done": False},
            {"message": {"content": "four."}, "done": True},
        )
        with _client(cfg, _stream_handler(body)) as client:
            assert client.chat([]).content == "Half four."


class TestOptions:
    def test_configured_options_are_sent(self, tmp_path: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml",
            llm={"temperature": 0.25, "num_ctx": 4096, "keep_alive": "42m", "model": "qwen3:8b"},
        )
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, text=_ndjson({"message": {}, "done": True}))

        with _client(config, handler) as client:
            list(client.chat_stream([{"role": "user", "content": "x"}]))

        body = seen[0]
        assert body["model"] == "qwen3:8b"
        assert body["options"]["temperature"] == 0.25
        assert body["options"]["num_ctx"] == 4096
        assert body["keep_alive"] == "42m"
        assert body["stream"] is True

    def test_thinking_is_off_by_default(self, cfg: JarvisConfig) -> None:
        """§3: thinking mode costs time to first token."""
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, text=_ndjson({"message": {}, "done": True}))

        with _client(cfg, handler) as client:
            list(client.chat_stream([]))
        assert seen[0]["think"] is False

    def test_tools_are_included_when_given(self, cfg: JarvisConfig) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, text=_ndjson({"message": {}, "done": True}))

        schema = [{"type": "function", "function": {"name": "sys.cpu", "parameters": {}}}]
        with _client(cfg, handler) as client:
            list(client.chat_stream([], tools=schema))
        assert seen[0]["tools"] == schema

    def test_no_tools_key_when_none_offered(self, cfg: JarvisConfig) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, text=_ndjson({"message": {}, "done": True}))

        with _client(cfg, handler) as client:
            list(client.chat_stream([]))
        assert "tools" not in seen[0]


class TestToolCallParsing:
    def test_dict_arguments(self, cfg: JarvisConfig) -> None:
        body = _ndjson(
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "sys.cpu", "arguments": {"interval_s": 0.5}}}
                    ]
                },
                "done": True,
            }
        )
        with _client(cfg, _stream_handler(body)) as client:
            calls = [c for chunk in client.chat_stream([]) for c in chunk.tool_calls]
        assert calls == [ToolCall(name="sys.cpu", arguments={"interval_s": 0.5})]

    def test_json_string_arguments(self, cfg: JarvisConfig) -> None:
        """Several models emit a JSON string rather than an object."""
        body = _ndjson(
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "sys.cpu", "arguments": '{"interval_s": 0.5}'}}
                    ]
                },
                "done": True,
            }
        )
        with _client(cfg, _stream_handler(body)) as client:
            calls = [c for chunk in client.chat_stream([]) for c in chunk.tool_calls]
        assert calls[0].arguments == {"interval_s": 0.5}

    def test_empty_arguments(self, cfg: JarvisConfig) -> None:
        body = _ndjson(
            {"message": {"tool_calls": [{"function": {"name": "sys.gpu"}}]}, "done": True}
        )
        with _client(cfg, _stream_handler(body)) as client:
            calls = [c for chunk in client.chat_stream([]) for c in chunk.tool_calls]
        assert calls[0].arguments == {}

    def test_malformed_tool_json_is_retried_then_surfaced(self, tmp_path: Path) -> None:
        """T-1.6: retry up to the configured count, then verbalise the failure."""
        config = load_config(tmp_path / "absent.yaml", llm={"tool_json_retries": 2})
        bad = {
            "message": {
                "tool_calls": [{"function": {"name": "sys.cpu", "arguments": "{not json"}}]
            },
            "done": False,
        }
        body = _ndjson(bad, bad, bad, bad)

        with _client(config, _stream_handler(body)) as client, pytest.raises(LlmError) as excinfo:
            list(client.chat_stream([]))
        assert "malformed" in str(excinfo.value)
        assert excinfo.value.speakable == "I could not work out how to do that."

    def test_a_recovered_tool_call_is_returned(self, tmp_path: Path) -> None:
        """A retry that succeeds must produce the call, not an error."""
        config = load_config(tmp_path / "absent.yaml", llm={"tool_json_retries": 2})
        body = _ndjson(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "sys.cpu", "arguments": "{oops"}}]
                },
                "done": False,
            },
            {
                "message": {
                    "tool_calls": [{"function": {"name": "sys.cpu", "arguments": {"x": 1}}}]
                },
                "done": True,
            },
        )
        with _client(config, _stream_handler(body)) as client:
            calls = [c for chunk in client.chat_stream([]) for c in chunk.tool_calls]
        assert calls[0].arguments == {"x": 1}

    @pytest.mark.parametrize(
        ("raw", "expected", "bad"),
        [
            (None, {}, False),
            ({}, {}, False),
            ({"a": 1}, {"a": 1}, False),
            ('{"a": 1}', {"a": 1}, False),
            ("", {}, False),
            ("not json", {}, True),
            ("[1,2]", {}, True),
            (42, {}, True),
        ],
    )
    def test_argument_normalisation(self, raw: Any, expected: dict, bad: bool) -> None:
        assert _parse_arguments(raw) == (expected, bad)


class TestFailureModes:
    def test_connection_refused_says_ollama_is_not_running(self, cfg: JarvisConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with _client(cfg, handler) as client, pytest.raises(LlmError) as excinfo:
            list(client.chat_stream([]))
        assert "Ollama is not running" in excinfo.value.speakable

    def test_timeout_has_its_own_message(self, cfg: JarvisConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        with _client(cfg, handler) as client, pytest.raises(LlmError) as excinfo:
            list(client.chat_stream([]))
        assert "too long" in excinfo.value.speakable

    def test_model_not_found_names_ollama_pull(self, cfg: JarvisConfig) -> None:
        with _client(cfg, _stream_handler("", 404)) as client, pytest.raises(LlmError) as excinfo:
            list(client.chat_stream([]))
        assert "ollama pull" in excinfo.value.speakable

    def test_server_error_is_distinct(self, cfg: JarvisConfig) -> None:
        with _client(cfg, _stream_handler("", 500)) as client, pytest.raises(LlmError) as excinfo:
            list(client.chat_stream([]))
        assert "error" in excinfo.value.speakable.lower()

    def test_three_failure_modes_have_three_messages(self, cfg: JarvisConfig) -> None:
        """Collapsing these into one message makes troubleshooting impossible."""
        messages = set()
        for handler in (
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("x", request=r)),
            lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("x", request=r)),
            lambda _r: httpx.Response(404),
        ):
            with _client(cfg, handler) as client:
                try:
                    list(client.chat_stream([]))
                except LlmError as exc:
                    messages.add(exc.speakable)
        assert len(messages) == 3


class TestDiscovery:
    def test_is_available_true(self, cfg: JarvisConfig) -> None:
        with _client(cfg, lambda _r: httpx.Response(200, json={"models": []})) as client:
            assert client.is_available() is True

    def test_is_available_never_raises(self, cfg: JarvisConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        with _client(cfg, handler) as client:
            assert client.is_available() is False

    def test_list_models(self, cfg: JarvisConfig) -> None:
        payload = {"models": [{"name": "qwen3:8b"}, {"name": "moondream:latest"}]}
        with _client(cfg, lambda _r: httpx.Response(200, json=payload)) as client:
            assert client.list_models() == ["qwen3:8b", "moondream:latest"]

    def test_has_model_tolerates_the_latest_suffix(self, cfg: JarvisConfig) -> None:
        payload = {"models": [{"name": "moondream:latest"}]}
        with _client(cfg, lambda _r: httpx.Response(200, json=payload)) as client:
            assert client.has_model("moondream") is True
            assert client.has_model("moondream:latest") is True
            assert client.has_model("qwen3:8b") is False

    def test_ensure_model_passes_when_present(self, cfg: JarvisConfig) -> None:
        payload = {"models": [{"name": cfg.llm_model()}]}
        with _client(cfg, lambda _r: httpx.Response(200, json=payload)) as client:
            client.ensure_model()

    def test_ensure_model_raises_with_the_pull_command(self, cfg: JarvisConfig) -> None:
        with (
            _client(cfg, lambda _r: httpx.Response(200, json={"models": []})) as client,
            pytest.raises(LlmError) as excinfo,
        ):
            client.ensure_model()
        assert "ollama pull" in excinfo.value.speakable


class TestVision:
    def test_image_is_sent_in_the_images_array(self, cfg: JarvisConfig) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "A terminal window."}})

        with _client(cfg, handler) as client:
            answer = client.vision("what is this", "BASE64DATA")

        assert answer == "A terminal window."
        assert seen[0]["messages"][0]["images"] == ["BASE64DATA"]
        assert seen[0]["stream"] is False

    def test_vision_uses_the_configured_model(self, cfg: JarvisConfig) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "x"}})

        with _client(cfg, handler) as client:
            client.vision("q", "img")
        assert seen[0]["model"] == cfg.tools.vision_model
