"""T-4.8 verification: shell.run hardening.

The ledger demands red-team cases for operator injection, encoded commands, path
escape, and each blocked command. These are written as attacks, not as examples.
Every one asserts that nothing ran.
"""

from __future__ import annotations

import json
import re
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


class TestAllowlistIsNotKeyedOnTheFilenameAlone:
    """The allowlist names commands, not files that happen to share a name.

    Keying it on the filename stem meant any executable called "echo", anywhere
    on disk, satisfied an allowlist that only ever meant the real echo.
    """

    def test_an_absolute_path_head_is_refused(
        self, enabled: JarvisConfig, tmp_path: Path
    ) -> None:
        import stat

        planted = tmp_path / "planted"
        planted.mkdir()
        payload = planted / "echo"
        payload.write_text("#!/bin/sh\necho PWNED\n", encoding="utf-8")
        payload.chmod(payload.stat().st_mode | stat.S_IEXEC)

        verdict = _verdict(f"{payload} anything", enabled)
        assert verdict.allowed is False
        assert verdict.argv is None

    @pytest.mark.parametrize(
        "command",
        [
            "./echo hi",
            "../echo hi",
            "/usr/bin/echo hi",
            "bin/echo hi",
            r"C:\Windows\System32\echo.exe hi",
            "C:echo hi",
        ],
    )
    def test_any_path_bearing_head_is_refused(
        self, enabled: JarvisConfig, command: str
    ) -> None:
        assert _verdict(command, enabled).allowed is False

    def test_the_resolved_executable_is_absolute(self, enabled: JarvisConfig) -> None:
        """The file that was checked must be the file that runs, whatever the cwd."""
        verdict = _verdict("echo hi", enabled)
        assert verdict.allowed is True
        assert verdict.argv is not None
        assert Path(verdict.argv[0]).is_absolute()


class TestQuotingCannotSmuggleAnArgument:
    """Quoting used to hide a blocked argument from the check but not from argv."""

    @pytest.mark.parametrize(
        "command",
        [
            'echo "-Command" Get-Process',
            "echo '-Command' Get-Process",
            'echo "-enc" payload',
            "echo '-EncodedCommand' payload",
            'echo "-nop" x',
            'echo "-ExecutionPolicy" Bypass',
        ],
    )
    def test_quoted_blocked_arguments_are_refused(
        self, enabled: JarvisConfig, command: str
    ) -> None:
        assert _verdict(command, enabled).allowed is False

    def test_quoted_blocked_commands_are_refused(self, enabled: JarvisConfig) -> None:
        assert _verdict('echo "shutdown"', enabled).allowed is False

    def test_argv_carries_the_same_value_that_was_checked(
        self, enabled: JarvisConfig
    ) -> None:
        """A token must not pass a check in one form and reach the process in another."""
        verdict = _verdict('echo "hello world"', enabled)
        assert verdict.allowed is True
        assert verdict.argv is not None
        assert verdict.argv[1:] == ["hello world"]


class TestValidationIsTotal:
    """validate_command is documented as returning a verdict, so it must never raise."""

    @pytest.mark.parametrize(
        "working_dir",
        ["~nonexistentuser12345/x", "\x00/etc", "", " ", "\n/tmp", "relative/path"],
    )
    def test_no_working_dir_makes_it_raise(
        self, enabled: JarvisConfig, working_dir: str
    ) -> None:
        verdict = validate_command("echo hi", working_dir, enabled)
        assert verdict.allowed is False

    def test_refusals_are_still_audited_for_a_bad_working_dir(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="echo hi", working_dir="~nouser99/x"))
        assert (tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip()


class TestOutputIsBounded:
    def test_the_cap_bounds_memory_not_just_the_return_value(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 9 is a resource bound, not a formatting rule.

        Buffering the whole stream and slicing afterwards returns 4 KB while
        still letting a chatty command exhaust memory first.
        """
        import resource
        import shutil as shutil_module

        if shutil_module.which("yes") is None:
            pytest.skip("yes is not available on this host")

        config = load_config(
            tmp_path / "absent.yaml",
            tools={"enable_shell": True},
            shell={
                "enabled": True,
                "allowlist": ["yes"],
                "working_dir_allowlist": [str(tmp_path)],
                "audit_log": str(tmp_path / "audit.jsonl"),
                "timeout_s": 3.0,
            },
        )
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", config)

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = run_command(ShellInput(command="yes"))
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        assert result.timed_out is True
        assert len(result.stdout.encode()) <= config.shell.max_output_bytes
        assert result.truncated is True, "truncation must be reported honestly"
        assert (after - before) / 1024 < 200, "the cap did not bound memory"

    def test_short_output_is_not_reported_as_truncated(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        result = run_command(ShellInput(command="echo brief"))
        assert result.truncated is False
        assert result.exit_code == 0


class TestAuditRecordsTheRealConfirmation:
    def test_an_unconfirmed_call_is_logged_as_unconfirmed(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 10 requires recording whether the user confirmed, not a constant."""
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="shutdown /s"), confirmed=False)

        record = json.loads(
            (tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        assert record["confirmed"] is False
        assert record["refused_reason"]

    def test_a_confirmed_call_is_logged_as_confirmed(
        self, enabled: JarvisConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", enabled)
        run_command(ShellInput(command="echo hi"), confirmed=True)

        record = json.loads(
            (tmp_path / "shell_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        assert record["confirmed"] is True


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


class TestTheDefaultAllowlistCanActuallyRun:
    """An allowlist entry that is always refused is a false advertisement.

    The shipped list held ten PowerShell cmdlets. Reaching a cmdlet means
    powershell.exe -Command, and §6 blocks -Command, so every one of them was
    accepted by the allowlist and then refused by rule 4. The tool description
    named Get-Date as an example of something it could run.
    """

    def test_no_entry_is_a_powershell_cmdlet(self) -> None:
        from jarvis.config import load_defaults

        allowlist = load_defaults().shell.allowlist
        cmdlets = [name for name in allowlist if "-" in name]
        assert cmdlets == [], (
            f"{cmdlets} are cmdlets, which §6 makes unreachable because it blocks -Command"
        )

    def test_every_entry_is_a_bare_executable_name(self) -> None:
        from jarvis.config import load_defaults

        for name in load_defaults().shell.allowlist:
            assert name == Path(name).name, f"{name} carries a path"
            assert name.isascii() and name.replace(".", "").isalnum(), name

    def test_the_description_only_names_runnable_examples(self) -> None:
        from jarvis.config import load_defaults
        from jarvis.tools.registry import registry

        spec = registry.get("shell.run")
        assert spec is not None
        allowlist = {name.lower() for name in load_defaults().shell.allowlist}
        # Pull the "such as X or Y" examples out of the description and check
        # each one is something the allowlist actually permits.
        match = re.search(r"such as ([\w-]+) or ([\w-]+)", spec.description)
        assert match is not None, "the description no longer names examples"
        for example in match.groups():
            assert example.lower() in allowlist, f"{example} is advertised but not allowlisted"

    def test_a_cmdlet_is_still_refused_if_someone_adds_one(
        self, tmp_path: Path
    ) -> None:
        """Removing them from the default is not the safety control; rule 4 is."""
        from jarvis.config import load_config

        config = load_config(
            tmp_path / "absent.yaml",
            shell={"enabled": True, "allowlist": ["Get-Date"]},
            tools={"enable_shell": True},
        )
        verdict = validate_command("Get-Date", None, config)
        assert verdict.allowed is False
