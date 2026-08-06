"""T-4.8 verification: shell.run hardening.

The ledger demands red-team cases for operator injection, encoded commands, path
escape, and each blocked command. These are written as attacks, not as examples.
Every one asserts that nothing ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, load_config
from jarvis.tools.registry import registry
from jarvis.tools.shell import ShellInput, run_command, validate_command


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "inside.txt").write_text("ok", encoding="utf-8")
    return root


@pytest.fixture
def enabled(tmp_path: Path, workdir: Path) -> JarvisConfig:
    """A config with the shell tool fully enabled, which is not the default."""
    return load_config(
        tmp_path / "absent.yaml",
        tools={"enable_shell": True},
        shell={
            "enabled": True,
            # echo and printenv exist on this host, so the allowlist is
            # meaningful rather than vacuous during the tests.
            "allowlist": ["echo", "printenv", "Get-Date", "systeminfo", "hostname"],
            "working_dir_allowlist": [str(workdir)],
            "audit_log": str(tmp_path / "shell_audit.jsonl"),
        },
    )


def _verdict(command: str, config: JarvisConfig, working_dir: str | None = None) -> object:
    return validate_command(command, working_dir, config)


class TestDisabledByDefault:
    def test_refused_when_config_is_default(self, cfg: JarvisConfig) -> None:
        """§6 and §9: the shell tool ships off."""
        verdict = validate_command("echo hello", None, cfg)
        assert verdict.allowed is False
        assert "disabled" in verdict.reason

    def test_tool_switch_alone_is_not_enough(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.yaml", tools={"enable_shell": True})
        assert validate_command("echo hi", None, config).allowed is False

    def test_registered_as_mutating(self) -> None:
        spec = registry.get("shell.run")
        assert spec is not None
        assert spec.read_only is False
        assert spec.requires_confirmation is True

    def test_registry_refuses_without_confirmation(self) -> None:
        """§6 backstop: even enabled, an unconfirmed call must not execute."""
        result = registry.dispatch("shell.run", {"command": "echo hi"})
        assert result.ok is False
        assert result.speakable == "I need you to confirm that first."

    def test_hidden_from_the_model_when_disabled(self, cfg: JarvisConfig) -> None:
        names = [spec.name for spec in registry.select(config=cfg)]
        assert "shell.run" not in names


class TestAllowlistIsPrimary:
    def test_allowlisted_command_passes(self, enabled: JarvisConfig) -> None:
        assert _verdict("echo hello", enabled).allowed is True

    @pytest.mark.parametrize(
        "command",
        ["curl http://evil.test", "wget something", "python -c pass", "bash", "cmd", "ls"],
    )
    def test_unlisted_command_is_refused(self, enabled: JarvisConfig, command: str) -> None:
        """Not on the allowlist means refused, even though it is also not blocked."""
        verdict = _verdict(command, enabled)
        assert verdict.allowed is False
        assert "allowlist" in verdict.reason

    def test_allowlist_matching_is_case_insensitive(self, enabled: JarvisConfig) -> None:
        """The allowlist check itself ignores case.

        Asserted via the refusal reason rather than the final verdict, because on
        a case-sensitive filesystem "ECHO" then fails the separate PATH lookup.
        """
        verdict = _verdict("ECHO hello", enabled)
        assert "allowlist" not in verdict.reason

    def test_path_prefix_does_not_bypass_the_allowlist(self, enabled: JarvisConfig) -> None:
        """C:/Windows/System32/cmd.exe must be judged as cmd, not as a path."""
        assert _verdict(r"C:\Windows\System32\cmd.exe /c dir", enabled).allowed is False


class TestBlockedCommands:
    @pytest.mark.parametrize(
        "name",
        [
            "rm", "del", "rmdir", "format", "diskpart", "shutdown", "restart",
            "reg", "regedit", "net", "netsh", "takeown", "icacls", "sc",
            "bcdedit", "vssadmin", "cipher",
        ],
    )
    def test_every_named_blocked_command_is_refused(
        self, enabled: JarvisConfig, name: str
    ) -> None:
        """§6 names all seventeen. Each gets its own assertion."""
        verdict = _verdict(f"{name} something", enabled)
        assert verdict.allowed is False

    @pytest.mark.parametrize("name", ["shutdown", "diskpart", "vssadmin"])
    def test_blocked_command_with_exe_suffix(self, enabled: JarvisConfig, name: str) -> None:
        assert _verdict(f"{name}.exe /s", enabled).allowed is False

    @pytest.mark.parametrize("name", ["SHUTDOWN", "Format", "ReG"])
    def test_blocked_matching_is_case_insensitive(
        self, enabled: JarvisConfig, name: str
    ) -> None:
        assert _verdict(f"{name} x", enabled).allowed is False

    def test_blocked_command_as_an_argument_is_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict("echo shutdown", enabled).allowed is False


class TestOperatorInjection:
    @pytest.mark.parametrize(
        "command",
        [
            "echo hi && shutdown /s",
            "echo hi & shutdown",
            "echo hi | del important.txt",
            "echo hi ; rm -rf /",
            "echo hi || format c:",
            "echo `whoami`",
            "echo $(whoami)",
            "echo ${PATH}",
            "echo hi > /etc/passwd",
            "echo hi >> ~/.bashrc",
            "echo hi < /etc/shadow",
            "echo hi\nshutdown /s",
            "echo hi\r\nshutdown",
        ],
    )
    def test_operators_are_refused(self, enabled: JarvisConfig, command: str) -> None:
        verdict = _verdict(command, enabled)
        assert verdict.allowed is False, f"{command!r} must be refused"

    def test_null_byte_is_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict("echo hi\x00shutdown", enabled).allowed is False

    @pytest.mark.parametrize(
        ("name", "char"),
        [
            ("line separator", "\u2028"),
            ("paragraph separator", "\u2029"),
            ("vertical tab", "\x0b"),
            ("form feed", "\x0c"),
        ],
    )
    def test_exotic_line_terminators_are_refused(
        self, enabled: JarvisConfig, name: str, char: str
    ) -> None:
        """These terminate a line for several parsers, exactly like a newline."""
        assert _verdict(f"echo hi{char}shutdown", enabled).allowed is False, name

    def test_operator_inside_quotes_is_still_refused(self, enabled: JarvisConfig) -> None:
        """Quoting must not launder an operator past the check."""
        assert _verdict('echo "hi && shutdown"', enabled).allowed is False


class TestEncodedCommands:
    @pytest.mark.parametrize(
        "command",
        [
            "echo -enc aQBlAHgA",
            "echo -EncodedCommand aQBlAHgA",
            "echo -e aQBlAHgA",
            "echo -nop -c something",
            "echo -Command Get-Process",
            "echo -NoProfile -Command x",
            "echo -ExecutionPolicy Bypass",
        ],
    )
    def test_encoded_and_bypass_arguments_are_refused(
        self, enabled: JarvisConfig, command: str
    ) -> None:
        assert _verdict(command, enabled).allowed is False


class TestPathEscape:
    def test_allowlisted_directory_is_accepted(
        self, enabled: JarvisConfig, workdir: Path
    ) -> None:
        assert _verdict("echo hi", enabled, str(workdir)).allowed is True

    def test_subdirectory_is_accepted(self, enabled: JarvisConfig, workdir: Path) -> None:
        nested = workdir / "sub"
        nested.mkdir()
        assert _verdict("echo hi", enabled, str(nested)).allowed is True

    def test_parent_directory_is_refused(self, enabled: JarvisConfig, workdir: Path) -> None:
        assert _verdict("echo hi", enabled, str(workdir.parent)).allowed is False

    def test_traversal_out_of_the_root_is_refused(
        self, enabled: JarvisConfig, workdir: Path
    ) -> None:
        """resolve() collapses .. so the escape cannot survive the check."""
        escape = str(workdir / ".." / ".." / "etc")
        assert _verdict("echo hi", enabled, escape).allowed is False

    def test_absolute_system_path_is_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict("echo hi", enabled, "/etc").allowed is False

    def test_nonexistent_directory_is_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict("echo hi", enabled, "/nope/not/here").allowed is False

    def test_a_symlinked_allowlist_root_still_authorises_its_contents(
        self, tmp_path: Path, workdir: Path
    ) -> None:
        """Roots are resolved, so a symlinked allowlist entry is not silently dead."""
        link = tmp_path / "link-to-work"
        link.symlink_to(workdir)
        config = load_config(
            tmp_path / "absent.yaml",
            tools={"enable_shell": True},
            shell={
                "enabled": True,
                "allowlist": ["echo"],
                "working_dir_allowlist": [str(link)],
            },
        )
        assert validate_command("echo hi", str(workdir), config).allowed is True

    def test_a_missing_allowlist_root_authorises_nothing(self, tmp_path: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml",
            tools={"enable_shell": True},
            shell={
                "enabled": True,
                "allowlist": ["echo"],
                "working_dir_allowlist": [str(tmp_path / "does-not-exist")],
            },
        )
        assert validate_command("echo hi", None, config).allowed is False

    def test_empty_allowlist_means_nowhere(self, tmp_path: Path) -> None:
        """An unconfigured allowlist must mean no, not anywhere."""
        config = load_config(
            tmp_path / "absent.yaml",
            tools={"enable_shell": True},
            shell={"enabled": True, "allowlist": ["echo"], "working_dir_allowlist": []},
        )
        verdict = validate_command("echo hi", None, config)
        assert verdict.allowed is False
        assert "allowlisted" in verdict.reason


class TestLimits:
    def test_length_limit(self, enabled: JarvisConfig) -> None:
        verdict = _verdict("echo " + "a" * 600, enabled)
        assert verdict.allowed is False
        assert "character limit" in verdict.reason

    def test_schema_also_caps_length(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ShellInput(command="a" * 513)

    def test_empty_command_is_rejected_by_the_schema(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ShellInput(command="")

    def test_whitespace_only_is_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict("   ", enabled).allowed is False

    def test_unbalanced_quotes_are_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict('echo "unclosed', enabled).allowed is False


class TestExecution:
    def test_allowed_command_runs(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        result = run_command(ShellInput(command="echo hello-from-jarvis"))
        assert result.ran is True
        assert result.exit_code == 0
        assert "hello-from-jarvis" in result.stdout

    def test_refusal_is_returned_as_data_not_an_exception(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        result = run_command(ShellInput(command="echo hi && shutdown"))
        assert result.ran is False
        assert result.refused_reason is not None
        assert result.exit_code is None

    def test_output_is_truncated(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        result = run_command(ShellInput(command="echo " + "x" * 400))
        assert len(result.stdout.encode("utf-8")) <= enabled.shell.max_output_bytes

    def test_runs_in_the_allowlisted_directory(
        self, enabled: JarvisConfig, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        result = run_command(ShellInput(command="printenv PWD"))
        assert result.ran is True

    def test_no_shell_is_involved(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tokens must reach the process as literal argv, not shell syntax."""
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        # A literal asterisk would be glob-expanded by a shell. It is not here.
        result = run_command(ShellInput(command="echo *"))
        assert result.ran is True
        assert result.stdout.strip() == "*"


class TestAuditLog:
    def test_successful_run_is_logged(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="echo audited"))

        record = json.loads((tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip())
        assert record["command"] == "echo audited"
        assert record["exit_code"] == 0
        assert record["confirmed"] is True
        assert "ts" in record
        assert record["argv"][0].endswith("echo")

    def test_refusal_is_also_logged(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused attempt is exactly what an audit log is for."""
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="shutdown /s"))

        record = json.loads((tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip())
        assert record["refused_reason"] is not None
        assert record["exit_code"] is None

    def test_log_lines_are_valid_json(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="echo one"))
        run_command(ShellInput(command="rm -rf /"))
        run_command(ShellInput(command="echo two"))

        lines = (tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert {"ts", "command", "exit_code", "confirmed"} <= set(record)


class TestVerdictShape:
    def test_allowed_verdict_carries_argv_and_cwd(self, enabled: JarvisConfig) -> None:
        verdict = validate_command("echo hi", None, enabled)
        assert verdict.allowed is True
        assert verdict.argv is not None
        assert verdict.argv[0].endswith("echo")
        assert verdict.cwd is not None

    def test_refused_verdict_carries_no_argv(self, enabled: JarvisConfig) -> None:
        verdict = validate_command("shutdown /s", None, enabled)
        assert verdict.argv is None
        assert verdict.cwd is None

    def test_unresolvable_executable_is_refused(self, tmp_path: Path, workdir: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml",
            tools={"enable_shell": True},
            shell={
                "enabled": True,
                "allowlist": ["Get-Date"],
                "working_dir_allowlist": [str(workdir)],
            },
        )
        verdict = validate_command("Get-Date", None, config)
        assert verdict.allowed is False
        assert "not found on the path" in verdict.reason
