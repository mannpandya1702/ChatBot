"""T-2.1 verification: the tool registry, schema generation, and dispatch safety."""

from __future__ import annotations

import json

import pytest
from pydantic import Field

from jarvis.tools.registry import (
    ToolCategory,
    ToolInput,
    ToolOutput,
    ToolRegistry,
    ToolResult,
    tool,
)
from jarvis.util.errors import ToolExecutionError


class EchoInput(ToolInput):
    text: str = Field(description="Text to echo back.")
    times: int = Field(default=1, ge=1, le=5, description="Repeat count.")


class EchoOutput(ToolOutput):
    echoed: str
    length_chars: int


class EmptyInput(ToolInput):
    pass


class NestedChild(ToolInput):
    value: int = Field(description="A nested integer.")


class NestedInput(ToolInput):
    child: NestedChild
    label: str = "x"


@pytest.fixture
def reg() -> ToolRegistry:
    return ToolRegistry()


def _register_echo(reg: ToolRegistry, **kwargs: object) -> None:
    @tool(
        name=kwargs.pop("name", "test.echo"),  # type: ignore[arg-type]
        description="Echo text back. Returns the text and its length in characters.",
        target=reg,
        **kwargs,  # type: ignore[arg-type]
    )
    def echo(params: EchoInput) -> EchoOutput:
        text = params.text * params.times
        return EchoOutput(echoed=text, length_chars=len(text))


class TestRegistration:
    def test_registers_and_looks_up(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert "test.echo" in reg
        assert len(reg) == 1
        spec = reg.get("test.echo")
        assert spec is not None
        assert spec.input_model is EchoInput
        assert spec.output_model is EchoOutput

    def test_default_is_read_only(self, reg: ToolRegistry) -> None:
        """§6: read-only by default. A tool must opt in to mutating."""
        _register_echo(reg)
        spec = reg.get("test.echo")
        assert spec is not None
        assert spec.read_only is True
        assert spec.requires_confirmation is False

    def test_mutating_tools_require_confirmation(self, reg: ToolRegistry) -> None:
        _register_echo(reg, read_only=False)
        spec = reg.get("test.echo")
        assert spec is not None
        assert spec.requires_confirmation is True

    def test_duplicate_name_is_rejected(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        with pytest.raises(ValueError, match="already registered"):
            _register_echo(reg)

    def test_decorator_returns_the_original_function(self, reg: ToolRegistry) -> None:
        @tool(name="t.direct", description="Direct call.", target=reg)
        def direct(params: EchoInput) -> EchoOutput:
            return EchoOutput(echoed=params.text, length_chars=len(params.text))

        assert direct(EchoInput(text="hi")).echoed == "hi"

    def test_rejects_unannotated_argument(self, reg: ToolRegistry) -> None:
        with pytest.raises(TypeError, match="pydantic model"):

            @tool(name="t.bad", description="Bad.", target=reg)
            def bad(params: str) -> EchoOutput:  # type: ignore[type-var]
                raise NotImplementedError

    def test_rejects_unannotated_return(self, reg: ToolRegistry) -> None:
        with pytest.raises(TypeError, match="return type"):

            @tool(name="t.bad2", description="Bad.", target=reg)
            def bad2(params: EchoInput) -> dict:  # type: ignore[type-var]
                raise NotImplementedError

    def test_rejects_multiple_arguments(self, reg: ToolRegistry) -> None:
        with pytest.raises(TypeError, match="exactly one"):

            @tool(name="t.bad3", description="Bad.", target=reg)
            def bad3(a: EchoInput, b: EchoInput) -> EchoOutput:  # type: ignore[type-var]
                raise NotImplementedError


class TestSchema:
    def test_shape_matches_the_ollama_contract(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        spec = reg.get("test.echo")
        assert spec is not None
        schema = spec.json_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test.echo"
        assert "Echo text back" in schema["function"]["description"]
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert set(params["properties"]) == {"text", "times"}
        assert params["required"] == ["text"]

    def test_descriptions_survive(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        spec = reg.get("test.echo")
        assert spec is not None
        props = spec.json_schema()["function"]["parameters"]["properties"]
        assert props["text"]["description"] == "Text to echo back."

    def test_nested_models_are_inlined(self, reg: ToolRegistry) -> None:
        """Small models chase $ref badly, so the schema must be dereferenced."""

        @tool(name="t.nested", description="Nested.", target=reg)
        def nested(params: NestedInput) -> EchoOutput:
            raise NotImplementedError

        spec = reg.get("t.nested")
        assert spec is not None
        schema = spec.json_schema()
        text = json.dumps(schema)
        assert "$ref" not in text
        assert "$defs" not in text
        child = schema["function"]["parameters"]["properties"]["child"]
        assert child["properties"]["value"]["type"] == "integer"

    def test_empty_input_still_produces_an_object(self, reg: ToolRegistry) -> None:
        @tool(name="t.empty", description="No args.", target=reg)
        def empty(params: EmptyInput) -> EchoOutput:
            raise NotImplementedError

        spec = reg.get("t.empty")
        assert spec is not None
        params = spec.json_schema()["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}

    def test_ollama_tools_returns_a_list(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        tools = reg.ollama_tools()
        assert len(tools) == 1
        json.dumps(tools)


class TestDispatch:
    def test_happy_path(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        result = reg.dispatch("test.echo", {"text": "ab", "times": 2})
        assert result.ok is True
        assert result.data == {"echoed": "abab", "length_chars": 4}
        assert result.duration_ms >= 0

    def test_defaults_are_applied(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert reg.dispatch("test.echo", {"text": "z"}).data["echoed"] == "z"

    def test_unknown_tool_is_soft_failure(self, reg: ToolRegistry) -> None:
        result = reg.dispatch("nope.missing", {})
        assert result.ok is False
        assert "unknown tool" in (result.error or "")
        assert result.speakable == "I do not have a tool for that."

    def test_bad_arguments_are_soft_failure(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        result = reg.dispatch("test.echo", {"text": "x", "times": 99})
        assert result.ok is False
        assert "invalid arguments" in (result.error or "")

    def test_missing_required_argument(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert reg.dispatch("test.echo", {}).ok is False

    def test_unknown_argument_is_rejected(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert reg.dispatch("test.echo", {"text": "x", "bogus": 1}).ok is False

    def test_raising_tool_never_propagates(self, reg: ToolRegistry) -> None:
        """§5: a tool exception must never crash the loop."""

        @tool(name="t.boom", description="Explodes.", target=reg)
        def boom(params: EmptyInput) -> EchoOutput:
            raise RuntimeError("internal path C:/Users/mann/secret")

        result = reg.dispatch("t.boom", {})
        assert result.ok is False
        assert result.speakable == "Something went wrong on my end."
        assert "secret" not in (result.speakable or "")

    def test_jarvis_error_speakable_is_surfaced(self, reg: ToolRegistry) -> None:
        @tool(name="t.sensor", description="Sensor.", target=reg)
        def sensor(params: EmptyInput) -> EchoOutput:
            raise ToolExecutionError("t.sensor", "NVML down")

        result = reg.dispatch("t.sensor", {})
        assert result.ok is False
        assert result.speakable == "I could not read that sensor."

    def test_non_model_return_is_caught(self, reg: ToolRegistry) -> None:
        @tool(name="t.liar", description="Lies about its return type.", target=reg)
        def liar(params: EmptyInput) -> EchoOutput:
            return "not a model"  # type: ignore[return-value]

        result = reg.dispatch("t.liar", {})
        assert result.ok is False
        assert "expected a pydantic model" in (result.error or "")

    def test_result_for_llm_is_compact(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert reg.dispatch("test.echo", {"text": "q"}).for_llm() == {
            "echoed": "q",
            "length_chars": 1,
        }

    def test_failure_for_llm_carries_the_error(self, reg: ToolRegistry) -> None:
        payload = reg.dispatch("nope", {}).for_llm()
        assert "error" in payload
        assert "speakable" in payload


class TestConfirmationBackstop:
    """§6 defence in depth: the registry itself refuses unconfirmed mutations."""

    def test_mutating_tool_without_confirmation_is_refused(self, reg: ToolRegistry) -> None:
        calls: list[str] = []

        @tool(name="t.mutate", description="Mutates.", read_only=False, target=reg)
        def mutate(params: EmptyInput) -> EchoOutput:
            calls.append("ran")
            return EchoOutput(echoed="done", length_chars=4)

        result = reg.dispatch("t.mutate", {})
        assert result.ok is False
        assert calls == [], "the function must not have executed"
        assert result.speakable == "I need you to confirm that first."

    def test_mutating_tool_with_confirmation_runs(self, reg: ToolRegistry) -> None:
        @tool(name="t.mutate2", description="Mutates.", read_only=False, target=reg)
        def mutate2(params: EmptyInput) -> EchoOutput:
            return EchoOutput(echoed="done", length_chars=4)

        result = reg.dispatch("t.mutate2", {}, confirmed=True)
        assert result.ok is True

    def test_read_only_tool_ignores_the_flag(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        assert reg.dispatch("test.echo", {"text": "a"}, confirmed=False).ok is True


class TestSelection:
    def test_filters_by_category(self, reg: ToolRegistry) -> None:
        _register_echo(reg, name="a.one", category=ToolCategory.SYSTEM)
        _register_echo(reg, name="b.two", category=ToolCategory.FILES)
        assert [s.name for s in reg.select(category=ToolCategory.FILES)] == ["b.two"]

    def test_filters_by_read_only(self, reg: ToolRegistry) -> None:
        _register_echo(reg, name="a.ro")
        _register_echo(reg, name="b.rw", read_only=False)
        assert [s.name for s in reg.select(read_only=True)] == ["a.ro"]

    def test_disabled_tools_are_hidden(self, reg: ToolRegistry) -> None:
        _register_echo(reg, name="a.off", is_enabled=lambda _cfg: False)
        _register_echo(reg, name="b.on", is_enabled=lambda _cfg: True)
        names = [s.name for s in reg.select(config=object())]
        assert names == ["b.on"]

    def test_no_config_means_no_filtering(self, reg: ToolRegistry) -> None:
        _register_echo(reg, name="a.off", is_enabled=lambda _cfg: False)
        assert len(reg.select()) == 1

    def test_iteration_is_sorted(self, reg: ToolRegistry) -> None:
        _register_echo(reg, name="z.last")
        _register_echo(reg, name="a.first")
        assert [s.name for s in reg] == ["a.first", "z.last"]

    def test_clear_empties_the_registry(self, reg: ToolRegistry) -> None:
        _register_echo(reg)
        reg.clear()
        assert len(reg) == 0


class TestToolResultModel:
    def test_round_trips_through_json(self) -> None:
        result = ToolResult(tool="x", ok=True, data={"a": 1}, duration_ms=1.5)
        assert ToolResult.model_validate_json(result.model_dump_json()) == result
