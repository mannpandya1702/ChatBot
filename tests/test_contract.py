"""Mechanical enforcement of the CLAUDE.md §0 constraints.

§0 says violating any of these is a build failure. A constraint that is only
checked by reading the code is a constraint that will eventually be broken, so
each one that can be checked automatically is checked here.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "jarvis"


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _tracked(pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return [line for line in result.stdout.splitlines() if line.strip()]


class TestFullyLocal:
    """§0.1: no cloud anything.

    Network access is permitted only for package installs, model downloads, and
    explicitly user-enabled tools. Installs and downloads happen in the
    PowerShell scripts, never at runtime, so the runtime tree should contain no
    remote address at all.
    """

    #: The only hosts the runtime may name. SearXNG and Home Assistant are
    #: user-supplied at runtime and are never hardcoded.
    ALLOWED = re.compile(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?/?$")

    def test_no_remote_url_in_the_runtime_tree(self) -> None:
        offenders: list[str] = []
        url = re.compile(r"https?://[^\s\"'<>)\]}]+")

        for path in _source_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in url.finditer(line):
                    candidate = match.group().rstrip(".,;")
                    if self.ALLOWED.match(candidate):
                        continue
                    # A doc link in a comment is not a request.
                    stripped = line.lstrip()
                    if stripped.startswith("#") or stripped.startswith(("'''", '"""')):
                        continue
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {candidate}")

        assert not offenders, "§0.1 forbids cloud access:\n" + "\n".join(offenders)

    def test_no_telemetry_or_analytics_imports(self) -> None:
        banned = {"sentry_sdk", "posthog", "analytics", "mixpanel", "segment", "datadog"}
        offenders: list[str] = []
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in banned:
                            offenders.append(f"{path.name}: {alias.name}")
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".")[0] in banned
                ):
                    offenders.append(f"{path.name}: {node.module}")
        assert offenders == []

    def test_no_api_key_configuration_exists(self) -> None:
        """§0.1 and §0.2: no API keys, because there is nothing to key into."""
        from jarvis.config import JarvisConfig

        def walk(model: type, prefix: str = "") -> list[str]:
            found = []
            for name, field in model.model_fields.items():
                full = f"{prefix}{name}"
                if "api_key" in name or name.endswith("_key") or "apikey" in name:
                    found.append(full)
                annotation = field.annotation
                if hasattr(annotation, "model_fields"):
                    found.extend(walk(annotation, f"{full}."))
            return found

        # home_assistant_token is a user-supplied local credential, which §9
        # explicitly permits, and it is the only secret in the tree.
        assert walk(JarvisConfig) == []


class TestZeroCost:
    """§0.2: no paid services, no trials, no accounts."""

    def test_no_rejected_component_is_present(self) -> None:
        """§1 names these explicitly. None may be introduced."""
        banned = [
            "langchain", "llamaindex", "llama_index", "crewai", "electron",
            "coqui", "TTS.api", "porcupine", "pvporcupine", "edge_tts", "edge-tts",
        ]
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        for name in banned:
            assert name not in text, f"{name} is explicitly rejected by §1"

    def test_no_rejected_component_is_imported(self) -> None:
        banned = {
            "langchain", "llama_index", "crewai", "TTS", "pvporcupine", "edge_tts",
        }
        offenders: list[str] = []
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in banned:
                        offenders.append(f"{path.name}: {name}")
        assert offenders == []


class TestNoSecretsInTheRepo:
    """§0.6: no tokens, no absolute user paths."""

    def test_config_yaml_is_not_tracked(self) -> None:
        assert _tracked("config/config.yaml") == []

    def test_only_the_example_config_is_tracked(self) -> None:
        assert _tracked("config/*") == ["config/config.example.yaml"]

    def test_no_model_artifacts_are_tracked(self) -> None:
        assert _tracked("models/*") == []

    def test_no_logs_are_tracked(self) -> None:
        assert _tracked("logs/*") == []

    def test_no_absolute_user_path_in_the_runtime_tree(self) -> None:
        patterns = [
            re.compile(r"C:\\+Users\\+[A-Za-z]"),
            re.compile(r"/home/[a-z][a-z0-9_-]*/"),
            re.compile(r"/Users/[a-z][a-z0-9_-]*/"),
        ]
        offenders: list[str] = []
        for path in _source_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for pattern in patterns:
                    if pattern.search(line):
                        offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:70]}")
        assert not offenders, "§0.6 forbids hardcoded user paths:\n" + "\n".join(offenders)

    def test_no_credential_looking_literal(self) -> None:
        suspicious = re.compile(
            r"(?i)(api[_-]?key|secret|password|token)\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']"
        )
        offenders: list[str] = []
        for path in _source_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if suspicious.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        assert offenders == []


class TestReadOnlyByDefault:
    """§0.5 and §6: read-only unless explicitly allowlisted."""

    def test_every_mutating_tool_is_on_the_allowlist(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        config = load_defaults()
        main_module._register_tools(config)

        mutating = {spec.name for spec in registry if not spec.read_only}
        allowed = set(config.gate.mutating_allowlist)
        assert mutating <= allowed, f"not allowlisted: {sorted(mutating - allowed)}"

    def test_the_allowlist_is_exactly_the_six_named_tools(self) -> None:
        from jarvis.config import load_defaults

        assert sorted(load_defaults().gate.mutating_allowlist) == [
            "apps.focus",
            "apps.launch",
            "media.control",
            "reminders.create",
            "reminders.delete",
            "shell.run",
        ]

    def test_the_majority_of_tools_are_read_only(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        main_module._register_tools(load_defaults())
        read_only = [spec for spec in registry if spec.read_only]
        assert len(read_only) > len(list(registry)) / 2

    def test_no_tool_auto_executes_an_llm_string(self) -> None:
        """§0.5: no LLM-generated command is ever auto-executed.

        Enforced structurally: nothing outside the shell tool may pass a value
        to subprocess with shell=True, and the shell tool passes fixed argv.
        """
        offenders: list[str] = []
        for path in _source_files():
            text = path.read_text(encoding="utf-8")
            if "shell=True" in text:
                offenders.append(str(path.relative_to(ROOT)))
        assert offenders == []


class TestSpeakableOutputs:
    """§5: tool output is spoken, so lists are capped at 5."""

    def test_every_tool_description_is_substantial(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        main_module._register_tools(load_defaults())
        for spec in registry:
            assert len(spec.description) > 60, f"{spec.name} has a thin description"

    def test_tool_descriptions_carry_units_or_say_why_not(self) -> None:
        """Schema quality drives tool-calling reliability more than model size."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        main_module._register_tools(load_defaults())
        unit_words = (
            "percent", "megabyte", "gigabyte", "megahertz", "celsius", "watt",
            "second", "minute", "hour", "revolutions", "megabits", "count",
        )
        missing = [
            spec.name
            for spec in registry
            if spec.category.value == "system"
            and not any(word in spec.description.lower() for word in unit_words)
        ]
        assert missing == [], f"system tools with no units in their description: {missing}"

    def test_every_list_field_documents_its_cap(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        main_module._register_tools(load_defaults())
        offenders: list[str] = []
        for spec in registry:
            for name, field in spec.output_model.model_fields.items():
                annotation = str(field.annotation)
                if not annotation.startswith("list["):
                    continue
                description = (field.description or "").lower()
                if not any(token in description for token in ("5", "3", "up to", "at most")):
                    offenders.append(f"{spec.name}.{name}")
        assert offenders == [], f"list fields with no documented cap: {offenders}"
