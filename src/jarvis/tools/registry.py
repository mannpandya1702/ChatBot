"""Tool registry and the ``@tool`` decorator.

CLAUDE.md §5: every tool is a pure function taking a pydantic input model and
returning a pydantic output model, registered through a decorator that generates
the JSON schema Ollama receives. Schema quality drives tool-calling reliability
more than model size does, so the generated schema is fully dereferenced: no
``$ref``, no ``$defs``, nothing a small model has to chase.

§6 defence in depth: :meth:`ToolRegistry.dispatch` refuses to run a mutating tool
unless the caller passes ``confirmed=True``. The orchestrator is supposed to
route those through ``tools/gate.py`` first, and this is the backstop for when it
does not.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.util.errors import (
    JarvisError,
    PlatformUnsupportedError,
    SafetyViolationError,
    as_speakable,
    redact,
)
from jarvis.util.platform import is_windows

__all__ = [
    "ToolCategory",
    "ToolInput",
    "ToolOutput",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "registry",
    "tool",
]

_log = logging.getLogger(__name__)


class ToolCategory(StrEnum):
    """Grouping used for prompt organisation and for enabling whole families."""

    SYSTEM = "system"
    APPS = "apps"
    MEDIA = "media"
    FILES = "files"
    VISION = "vision"
    WEB = "web"
    REMINDERS = "reminders"
    SHELL = "shell"
    UTILITY = "utility"


class ToolInput(BaseModel):
    """Base class for tool inputs. Rejects unknown arguments outright."""

    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    """Base class for tool outputs.

    Keep these small and phrasable (§5). Cap list fields at five items.
    """

    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    """What :meth:`ToolRegistry.dispatch` always returns. Never raises."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    speakable: str | None = None
    duration_ms: float = 0.0

    def for_llm(self) -> dict[str, Any]:
        """Compact payload handed back to the model as the tool message."""
        if self.ok:
            return self.data
        return {"error": self.error, "speakable": self.speakable}


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything the registry and the LLM need to know about one tool."""

    name: str
    description: str
    category: ToolCategory
    read_only: bool
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    # Typed loosely on purpose: the decorator validates the annotation, but a
    # caller can still register a function that lies about its return type.
    func: Callable[..., Any]
    requires_windows: bool = False
    #: Consulted before the tool is offered to the model. Receives the config.
    is_enabled: Callable[[Any], bool] | None = field(default=None, compare=False)

    @property
    def requires_confirmation(self) -> bool:
        """Mutating tools always need an explicit confirmation (§6)."""
        return not self.read_only

    def json_schema(self) -> dict[str, Any]:
        """Ollama-compatible function schema, fully dereferenced."""
        params = _flatten_schema(self.input_model.model_json_schema())
        params.pop("title", None)
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


def _flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref`` pointers and strip ``$defs``.

    Small models chase references badly. Inlining them costs a few tokens and
    removes a whole class of malformed tool calls.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                if key in seen or key not in defs:
                    # Recursive model, or a dangling pointer. Degrade to "object".
                    return {"type": "object"}
                merged = resolve(deepcopy(defs[key]), seen | {key})
                extra = {k: v for k, v in node.items() if k != "$ref"}
                if isinstance(merged, dict):
                    merged.update(resolve(extra, seen))
                return merged
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        return node

    flattened = resolve(deepcopy(schema), frozenset())
    return flattened if isinstance(flattened, dict) else {"type": "object"}


class ToolRegistry:
    """Holds every registered tool and dispatches calls safely."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # -- registration ------------------------------------------------------

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        """Add ``spec``. Duplicate names are a programming error unless replacing."""
        if spec.name in self._tools and not replace:
            msg = f"tool {spec.name!r} is already registered"
            raise ValueError(msg)
        self._tools[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        """Remove a tool. Used by tests."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Drop every registration. Used by tests."""
        self._tools.clear()

    # -- lookup ------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(sorted(self._tools.values(), key=lambda s: s.name))

    def names(self) -> list[str]:
        """Sorted tool names."""
        return sorted(self._tools)

    def select(
        self,
        *,
        config: Any = None,  # JarvisConfig, kept loose to avoid a cycle
        category: ToolCategory | None = None,
        read_only: bool | None = None,
        include_unavailable: bool = False,
    ) -> list[ToolSpec]:
        """Tools matching the filters, in a stable order.

        Windows-only tools are dropped on other platforms, and tools whose
        ``is_enabled`` predicate returns False for this config are dropped too,
        so the model is never shown something it cannot successfully call.
        """
        out: list[ToolSpec] = []
        for spec in self:
            if category is not None and spec.category is not category:
                continue
            if read_only is not None and spec.read_only is not read_only:
                continue
            if not include_unavailable:
                if spec.requires_windows and not is_windows():
                    continue
                disabled = (
                    spec.is_enabled is not None
                    and config is not None
                    and not spec.is_enabled(config)
                )
                if disabled:
                    continue
            out.append(spec)
        return out

    def ollama_tools(self, **filters: Any) -> list[dict[str, Any]]:
        """The ``tools`` array passed to Ollama's chat endpoint."""
        return [spec.json_schema() for spec in self.select(**filters)]

    # -- dispatch ----------------------------------------------------------

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> ToolResult:
        """Run a tool. Always returns a :class:`ToolResult`, never raises.

        Args:
            name: Registered tool name.
            arguments: Raw arguments from the model, validated against the input
                model before the function sees them.
            confirmed: Must be True for a mutating tool. §6 backstop.
        """
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                tool=name,
                ok=False,
                error=f"unknown tool: {name}",
                speakable="I do not have a tool for that.",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        if spec.requires_confirmation and not confirmed:
            _log.warning(
                "refused unconfirmed mutating tool call",
                extra={"context": {"tool": name}},
            )
            return _fail(
                spec,
                SafetyViolationError(
                    f"tool {name!r} changes system state and was called without confirmation",
                    speakable="I need you to confirm that first.",
                ),
                started,
            )

        if spec.requires_windows and not is_windows():
            return _fail(spec, PlatformUnsupportedError(spec.name), started)

        try:
            payload = spec.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            _log.warning(
                "tool arguments failed validation",
                extra={"context": {"tool": name, "errors": exc.errors(include_url=False)}},
            )
            return ToolResult(
                tool=name,
                ok=False,
                error=f"invalid arguments: {_summarise_validation(exc)}",
                speakable="I did not understand the details of that request.",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            result = spec.func(payload)
        except JarvisError as exc:
            _log.exception("tool failed", extra={"context": {"tool": name, **exc.context}})
            return _fail(spec, exc, started)
        except Exception as exc:  # noqa: BLE001 - §5, a tool must never kill the loop
            _log.exception("tool raised", extra={"context": {"tool": name}})
            return _fail(spec, exc, started)

        if not isinstance(result, BaseModel):
            _log.error(
                "tool returned a non-model value",
                extra={"context": {"tool": name, "type": type(result).__name__}},
            )
            return ToolResult(
                tool=name,
                ok=False,
                error=f"tool returned {type(result).__name__}, expected a pydantic model",
                speakable="I could not read that sensor.",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        return ToolResult(
            tool=name,
            ok=True,
            data=result.model_dump(mode="json"),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


def _fail(spec: ToolSpec, exc: BaseException, started: float) -> ToolResult:
    """Build the failure result for a tool that raised.

    The message is redacted because this result travels: it goes to the model
    as the tool message and is persisted into conversation memory, so a stray
    home directory path in an OSError would be replayed on every later turn.
    The unredacted traceback is already in the log, which stays on the machine.
    """
    return ToolResult(
        tool=spec.name,
        ok=False,
        error=redact(f"{type(exc).__name__}: {exc}"),
        speakable=as_speakable(exc),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _summarise_validation(exc: ValidationError) -> str:
    """One-line summary of a validation failure, suitable for the model to read."""
    parts = []
    for err in exc.errors(include_url=False)[:3]:
        location = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


#: Process-wide registry. Tool modules register into this on import.
registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.SYSTEM,
    read_only: bool = True,
    requires_windows: bool = False,
    is_enabled: Callable[[Any], bool] | None = None,
    target: ToolRegistry | None = None,
) -> Callable[[Callable[[InputT], OutputT]], Callable[[InputT], OutputT]]:
    """Register a tool function.

    The decorated function must take exactly one argument annotated with a
    pydantic model and must be annotated with a pydantic model return type::

        @tool(name="sys.cpu", description="Current CPU utilisation in percent.")
        def cpu(params: CpuInput) -> CpuOutput:
            ...

    Args:
        name: Dotted tool name the model calls, for example ``sys.cpu``.
        description: Precise, unit-bearing description (§5).
        category: Grouping.
        read_only: False marks the tool as mutating, which forces confirmation.
        requires_windows: Skip registration exposure on non-Windows hosts.
        is_enabled: Predicate over the config deciding whether to offer this tool.
        target: Registry to register into. Defaults to the process-wide one.

    Returns:
        The original function, unchanged, so it stays directly callable and
        directly testable.
    """

    def decorator(func: Callable[[InputT], OutputT]) -> Callable[[InputT], OutputT]:
        input_model, output_model = _models_from_signature(func)
        spec = ToolSpec(
            name=name,
            description=description.strip(),
            category=category,
            read_only=read_only,
            input_model=input_model,
            output_model=output_model,
            func=func,
            requires_windows=requires_windows,
            is_enabled=is_enabled,
        )
        # Explicit None check: ToolRegistry defines __len__, so an empty
        # registry is falsy and `target or registry` would silently fall
        # back to the process-wide one.
        (registry if target is None else target).register(spec)
        func.__jarvis_tool__ = spec  # type: ignore[attr-defined]
        return func

    return decorator


def _models_from_signature(func: Callable[..., Any]) -> tuple[type[BaseModel], type[BaseModel]]:
    """Extract the input and output pydantic models from a tool's annotations."""
    signature = inspect.signature(func)
    params = [p for p in signature.parameters.values() if p.kind is not p.KEYWORD_ONLY]
    if len(params) != 1:
        msg = (
            f"tool {func.__qualname__} must take exactly one positional argument, "
            f"got {len(params)}"
        )
        raise TypeError(msg)

    try:
        hints = get_type_hints(func)
    except Exception as exc:  # unresolvable forward reference
        msg = f"tool {func.__qualname__} has annotations that cannot be resolved: {exc}"
        raise TypeError(msg) from exc

    input_model = hints.get(params[0].name)
    output_model = hints.get("return")

    if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        msg = f"tool {func.__qualname__} argument must be annotated with a pydantic model"
        raise TypeError(msg)
    if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        msg = f"tool {func.__qualname__} return type must be annotated with a pydantic model"
        raise TypeError(msg)
    return input_model, output_model
