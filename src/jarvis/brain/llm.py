"""Ollama client: streaming chat, native tool calls, and vision.

T-1.6. The latency budget in §3 allows 400 ms to first token, which rules out
buffering: :meth:`OllamaClient.chat_stream` yields each content delta as the
NDJSON line arrives.

Failure modes are distinguished on purpose. "Ollama is not running", "the model
timed out", and "that model is not installed" need three different spoken
answers, and collapsing them into one generic error makes the assistant useless
to troubleshoot.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from jarvis.config import JarvisConfig
from jarvis.util.errors import LlmError

__all__ = [
    "REASONING_TAGS",
    "ChatChunk",
    "ChatResponse",
    "OllamaClient",
    "ReasoningFilter",
    "ToolCall",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One piece of a streaming response.

    ``content`` is the speakable channel and has already had reasoning removed.
    ``reasoning`` carries what was removed, for the log only. Nothing downstream
    should ever speak it.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    metrics: dict[str, Any] | None = None
    reasoning: str = ""


#: Correction appended before re-asking for a tool call that would not parse.
#: Concrete about what was wrong and what is wanted, because a small model given
#: "that was invalid" tends to apologise in prose instead of calling the tool.
_TOOL_JSON_NUDGE = (
    "Your last tool call could not be used: its arguments were not valid JSON. "
    "Call the tool again now. Put the arguments in a single valid JSON object "
    "and write nothing else."
)

#: Reasoning delimiters that models emit inline in the content channel. Qwen3
#: and DeepSeek-R1 both use the first pair.
REASONING_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
)


def _pending_prefix(text: str, tags: Sequence[str]) -> int:
    """Length of the trailing run that might yet turn into one of ``tags``.

    A stream can split ``</think>`` across two lines. Emitting ``</thi``
    because it is not yet a whole tag would put it in the user's ears, so the
    longest suffix that is a proper prefix of some tag is held back until the
    next delta decides what it is.
    """
    longest = max((len(tag) for tag in tags), default=0)
    for size in range(min(longest - 1, len(text)), 0, -1):
        if any(tag.startswith(text[-size:]) for tag in tags):
            return size
    return 0


class ReasoningFilter:
    """Removes inline reasoning blocks from a streamed content channel.

    Ollama's ``think`` parameter is supposed to make this unnecessary, either by
    suppressing reasoning or by moving it to its own field. It does not always:
    an older Ollama ignores the parameter, and a model whose template does not
    honour it emits ``<think>`` inline regardless. When that happens the content
    channel goes straight to the sentence chunker, and the assistant reads its
    own deliberations aloud. Reported from the field as "it felt like it was
    reading instructions".

    Filtering here rather than at the speech end is deliberate. Reasoning is not
    part of the reply, so it should not reach the transcript, the memory, the
    HUD, or the time-to-first-token measurement either. Marking first token on a
    ``<think>`` would report a fast turn that the user experienced as a long
    silence.
    """

    def __init__(self, tags: Sequence[tuple[str, str]] = REASONING_TAGS) -> None:
        self._openings = tuple(opening for opening, _ in tags)
        self._closing_for = dict(tags)
        self._buffer = ""
        self._closing: str | None = None
        self._dropped: list[str] = []

    @property
    def inside_block(self) -> bool:
        """Whether a reasoning block is currently open."""
        return self._closing is not None

    def take_reasoning(self) -> str:
        """Return and clear whatever has been filtered out so far."""
        text, self._dropped = "".join(self._dropped), []
        return text

    def feed(self, delta: str) -> str:
        """Add a delta and return the part of it that is safe to speak."""
        if not delta:
            return ""
        self._buffer += delta
        speakable: list[str] = []

        while True:
            if self._closing is None:
                found = self._find_opening()
                if found is None:
                    held = _pending_prefix(self._buffer, self._openings)
                    cut = len(self._buffer) - held
                    speakable.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    break
                index, opening = found
                speakable.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(opening) :]
                self._closing = self._closing_for[opening]
                continue

            index = self._buffer.find(self._closing)
            if index >= 0:
                self._dropped.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(self._closing) :]
                self._closing = None
                continue

            held = _pending_prefix(self._buffer, (self._closing,))
            cut = len(self._buffer) - held
            self._dropped.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            break

        return "".join(speakable)

    def flush(self) -> str:
        """Finish the stream and return anything still held back.

        A block left open at the end of a stream is dropped rather than spoken.
        Text after an unterminated ``<think>`` is reasoning the model never
        finished, and reading it out is the exact failure this class exists to
        prevent.
        """
        remaining, self._buffer = self._buffer, ""
        if self._closing is not None:
            self._dropped.append(remaining)
            _log.warning(
                "the model left a reasoning block unterminated, dropping it",
                extra={"context": {"characters": len(remaining)}},
            )
            self._closing = None
            return ""
        return remaining

    def _find_opening(self) -> tuple[int, str] | None:
        """The earliest opening tag in the buffer, if any."""
        best: tuple[int, str] | None = None
        for opening in self._openings:
            index = self._buffer.find(opening)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, opening)
        return best


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A complete response, assembled from the stream."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """HTTP client for a local Ollama instance.

    Args:
        config: Supplies the base URL, model, options, and timeouts.
        client: An httpx client. Tests inject one with a mock transport.
    """

    def __init__(self, config: JarvisConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.llm.base_url,
            timeout=httpx.Timeout(
                config.llm.request_timeout_s, connect=config.llm.connect_timeout_s
            ),
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client, if this instance owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _url(self, path: str) -> str:
        """Absolute URL for a path, so an injected client needs no base_url."""
        return f"{self._config.llm.base_url}{path}"

    # -- discovery ---------------------------------------------------------

    def is_available(self) -> bool:
        """True when Ollama answers. Never raises."""
        try:
            response = self._client.get(self._url("/api/tags"), timeout=5.0)
        except (httpx.HTTPError, OSError):
            return False
        return response.status_code == 200

    def list_models(self) -> list[str]:
        """Names of the models Ollama has pulled.

        Raises:
            LlmError: Ollama could not be reached.
        """
        try:
            response = self._client.get(self._url("/api/tags"), timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as exc:
            raise _not_running(exc) from exc
        except httpx.TimeoutException as exc:
            raise _timed_out(exc) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LlmError(f"could not list models: {exc}") from exc

        return [
            str(entry.get("name", ""))
            for entry in payload.get("models", [])
            if isinstance(entry, dict) and entry.get("name")
        ]

    def has_model(self, name: str) -> bool:
        """Whether ``name`` is pulled, tolerating the implicit ``:latest`` tag."""
        wanted = name if ":" in name else f"{name}:latest"
        available = set(self.list_models())
        return name in available or wanted in available

    def ensure_model(self, name: str | None = None) -> None:
        """Raise a speakable error when the required model is missing.

        Raises:
            LlmError: The model is not pulled, or Ollama is unreachable.
        """
        model = name or self._config.llm_model()
        if not self.has_model(model):
            raise LlmError(
                f"model {model!r} is not installed in Ollama",
                speakable=f"The {model} model is not installed. Run ollama pull {model}.",
                context={"model": model},
            )

    # -- chat --------------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        model: str | None,
        stream: bool,
        think: bool | None,
    ) -> dict[str, Any]:
        """Build the /api/chat request body."""
        llm = self._config.llm
        body: dict[str, Any] = {
            "model": model or self._config.llm_model(),
            "messages": list(messages),
            "stream": stream,
            "keep_alive": llm.keep_alive,
            "options": {
                "temperature": llm.temperature,
                "top_p": llm.top_p,
                "num_ctx": llm.num_ctx,
                "num_predict": llm.num_predict,
            },
        }
        # Qwen3 thinking costs time to first token, so it is opt in (§3).
        body["think"] = llm.think if think is None else think
        if tools:
            body["tools"] = list(tools)
        return body

    def chat_stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        model: str | None = None,
        think: bool | None = None,
    ) -> Iterator[ChatChunk]:
        """Stream a chat completion, yielding deltas as they arrive.

        Args:
            messages: Chat history, in Ollama's message shape.
            tools: JSON schemas from the tool registry.
            model: Overrides the configured model.
            think: Overrides the configured thinking mode.

        Yields:
            One :class:`ChatChunk` per NDJSON line, ending with ``done=True``.

        Raises:
            LlmError: Ollama is unreachable, timed out, or rejected the request.
        """
        attempts = max(int(self._config.llm.tool_json_retries), 0) + 1
        conversation: list[dict[str, Any]] = list(messages)
        said: list[str] = []

        for attempt in range(attempts):
            # Only the first attempt speaks. A corrective round exists to get a
            # parseable tool call, and the user has already heard whatever the
            # model said the first time round.
            speaking = attempt == 0
            malformed = False
            finished = False
            reasoning = ReasoningFilter()

            for chunk, bad in self._stream_once(
                conversation, tools=tools, model=model, think=think, attempt=attempt
            ):
                malformed = malformed or bad

                spoken = reasoning.feed(chunk.content)
                if chunk.done:
                    spoken += reasoning.flush()
                dropped = reasoning.take_reasoning()
                if dropped:
                    _log.debug(
                        "dropped inline reasoning from the spoken channel",
                        extra={"context": {"characters": len(dropped)}},
                    )
                if speaking and spoken:
                    said.append(spoken)

                # Once anything in this message failed to parse, the terminal
                # marker is held back: a tool call was lost, so this is not the
                # end of the turn even though Ollama says it is. Yielding done
                # here is what made a bad call a silent no-op, because the
                # orchestrator took the empty answer as the whole reply.
                out = replace(
                    chunk,
                    content=spoken if speaking else "",
                    done=chunk.done and not malformed,
                    reasoning=chunk.reasoning + dropped,
                )
                if out.content or out.tool_calls or out.done or out.reasoning:
                    yield out
                if chunk.done:
                    finished = True
                    break

            if not finished:
                raise LlmError(
                    "the model's response ended without completing",
                    speakable="My language model stopped halfway through.",
                )
            if not malformed:
                return

            if attempt + 1 < attempts:
                _log.warning(
                    "the model emitted malformed tool JSON, asking it again",
                    extra={"context": {"attempt": attempt + 1, "of": attempts}},
                )
                # Give it back what it said plus a correction, so the retry has
                # the context of its own broken attempt rather than starting
                # blind and producing a different answer.
                conversation = [
                    *messages,
                    {"role": "assistant", "content": "".join(said)},
                    {"role": "system", "content": _TOOL_JSON_NUDGE},
                ]

        raise LlmError(
            "the model kept emitting malformed tool call JSON",
            speakable="I could not work out how to do that.",
        )

    def _stream_once(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        model: str | None,
        think: bool | None,
        attempt: int,
    ) -> Iterator[tuple[ChatChunk, bool]]:
        """One request to /api/chat, yielding ``(chunk, had_malformed_call)``.

        The malformed flag is per message, and the chunk beside it still carries
        everything that did parse: the content, the metrics, and any sibling
        tool call that was well formed. Dropping the whole message, which is
        what the old code did, threw away a valid call because another call in
        the same message was broken.
        """
        body = self._payload(messages, tools=tools, model=model, stream=True, think=think)
        _log.debug(
            "chat request",
            extra={
                "context": {
                    "model": body["model"],
                    "messages": len(messages),
                    "tools": len(tools or []),
                    "attempt": attempt,
                }
            },
        )
        try:
            with self._client.stream("POST", self._url("/api/chat"), json=body) as response:
                self._raise_for_status(response, body["model"])
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        _log.warning("skipping a malformed stream line")
                        continue
                    yield self._parse_chunk(payload)
        except httpx.ConnectError as exc:
            raise _not_running(exc) from exc
        except httpx.TimeoutException as exc:
            raise _timed_out(exc) from exc
        except httpx.HTTPError as exc:
            raise LlmError(
                f"the chat request failed: {exc}",
                speakable="My language model is not responding.",
            ) from exc


    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Non-streaming convenience wrapper over :meth:`chat_stream`."""
        content: list[str] = []
        calls: list[ToolCall] = []
        metrics: dict[str, Any] = {}
        for chunk in self.chat_stream(messages, tools=tools, model=model):
            content.append(chunk.content)
            calls.extend(chunk.tool_calls)
            if chunk.metrics:
                metrics = chunk.metrics
        return ChatResponse(content="".join(content), tool_calls=calls, metrics=metrics)

    def vision(self, prompt: str, image_b64: str, *, model: str | None = None) -> str:
        """Ask a vision model about an image.

        Args:
            prompt: The question.
            image_b64: Base64-encoded image bytes.
            model: Overrides the configured vision model.

        Returns:
            The model's answer.

        Raises:
            LlmError: The request failed or the model is not installed.
        """
        body = {
            "model": model or self._config.tools.vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "keep_alive": self._config.llm.keep_alive,
            "options": {"temperature": 0.2, "num_predict": 300},
        }
        try:
            response = self._client.post(self._url("/api/chat"), json=body)
            self._raise_for_status(response, str(body["model"]))
            payload = response.json()
        except httpx.ConnectError as exc:
            raise _not_running(exc) from exc
        except httpx.TimeoutException as exc:
            raise _timed_out(exc) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LlmError(
                f"the vision request failed: {exc}",
                speakable="I could not look at the screen.",
            ) from exc

        message = payload.get("message") or {}
        return str(message.get("content", "")).strip()

    # -- parsing -----------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        """Turn an HTTP error into a specific, speakable LlmError."""
        if response.status_code < 400:
            return
        if response.status_code == 404:
            raise LlmError(
                f"Ollama does not have the model {model!r}",
                speakable=f"The {model} model is not installed. Run ollama pull {model}.",
                context={"model": model},
            )
        raise LlmError(
            f"Ollama returned status {response.status_code}",
            speakable="My language model returned an error.",
            context={"status": response.status_code},
        )

    def _parse_chunk(self, payload: dict[str, Any]) -> tuple[ChatChunk, bool]:
        """Turn one NDJSON object into a chunk.

        Returns:
            ``(chunk, tool_json_was_malformed)``.
        """
        message = payload.get("message") or {}
        content = str(message.get("content") or "")
        # Newer Ollama moves reasoning into its own field. Kept for the log,
        # never spoken, and deliberately not merged into content.
        thinking = str(message.get("thinking") or "")
        done = bool(payload.get("done", False))

        calls: list[ToolCall] = []
        malformed = False
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                malformed = True
                continue
            function = raw.get("function") or {}
            name = str(function.get("name") or "")
            if not name:
                malformed = True
                continue
            arguments, bad = _parse_arguments(function.get("arguments"))
            if bad:
                malformed = True
                continue
            calls.append(ToolCall(name=name, arguments=arguments, id=str(raw.get("id") or "")))

        metrics: dict[str, Any] | None = None
        if done:
            metrics = {
                key: payload[key]
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if key in payload
            }

        chunk = ChatChunk(
            content=content,
            tool_calls=calls,
            done=done,
            metrics=metrics,
            reasoning=thinking,
        )
        return chunk, malformed


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Normalise a tool call's arguments.

    Ollama returns a dict, but several models emit a JSON string instead, and a
    few emit a string that is not valid JSON at all. Returns
    ``(arguments, malformed)``.
    """
    if raw is None:
        return {}, False
    if isinstance(raw, dict):
        return raw, False
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}, False
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}, True
        if isinstance(parsed, dict):
            return parsed, False
        return {}, True
    return {}, True


def _not_running(exc: BaseException) -> LlmError:
    """The error for a refused connection, which means Ollama is down."""
    return LlmError(
        f"could not connect to Ollama: {exc}",
        speakable="Ollama is not running, so I cannot think at the moment.",
    )


def _timed_out(exc: BaseException) -> LlmError:
    """The error for a request that took too long."""
    return LlmError(
        f"the Ollama request timed out: {exc}",
        speakable="My language model took too long to answer.",
    )
