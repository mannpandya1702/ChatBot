"""T-4.2 verification: apps.launch, apps.focus, apps.list_windows."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig
from jarvis.tools import apps as apps_module
from jarvis.tools.apps import (
    AppFocusInput,
    AppLaunchInput,
    WindowListInput,
    focus_window,
    launch_app,
    list_windows,
    resolve_application,
)
from jarvis.tools.registry import registry
from jarvis.util.errors import SafetyViolationError


class TestResolution:
    def test_known_app_resolves(self) -> None:
        assert resolve_application("notepad") == "notepad.exe"
        assert resolve_application("calculator") == "calc.exe"

    def test_resolution_is_case_insensitive(self) -> None:
        assert resolve_application("NotePad") == "notepad.exe"

    def test_multiword_names(self) -> None:
        assert resolve_application("visual studio code") == "code.exe"
        assert resolve_application("task manager") == "taskmgr.exe"

    def test_unknown_name_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setattr(apps_module, "_find_start_menu_shortcut", lambda _n: None)
        assert resolve_application("nonexistent-app-xyz") is None

    def test_empty_name(self) -> None:
        assert resolve_application("   ") is None


class TestLaunchSafety:
    """apps.launch must not become a second, unhardened shell.run."""

    @pytest.mark.parametrize(
        "name",
        [
            "cmd",
            "powershell",
            "pwsh",
            "wscript",
            "cscript",
            "mshta",
            "rundll32",
            "regedit",
            "wmic",
            "certutil",
        ],
    )
    def test_shells_and_script_hosts_are_refused(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setattr(apps_module, "_find_start_menu_shortcut", lambda _n: None)
        with pytest.raises(SafetyViolationError):
            resolve_application(f"{name}.exe")

    @pytest.mark.parametrize(
        "name",
        [
            r"C:\Windows\System32\cmd.exe",
            "/usr/bin/bash",
            "notepad & shutdown",
            "notepad | del",
            "notepad; rm -rf /",
            'notepad "arg"',
            "payload.bat",
            "script.ps1",
            "evil.vbs",
            "thing.cmd",
        ],
    )
    def test_paths_and_command_lines_are_refused(self, name: str) -> None:
        with pytest.raises(SafetyViolationError):
            resolve_application(name)

    def test_refusal_speakable_is_plain(self) -> None:
        with pytest.raises(SafetyViolationError) as excinfo:
            resolve_application("C:/Windows/cmd.exe")
        assert "only open applications by name" in excinfo.value.speakable


class TestOffWindows:
    def test_launch_reports_platform(self) -> None:
        result = launch_app(AppLaunchInput(name="notepad"))
        assert result.launched is False
        assert "Windows" in result.detail

    def test_focus_reports_platform(self) -> None:
        result = focus_window(AppFocusInput(name="chrome"))
        assert result.focused is False
        assert "Windows" in result.detail

    def test_list_windows_is_empty(self) -> None:
        assert list_windows(WindowListInput()).count == 0


class TestRegistration:
    def test_launch_and_focus_are_mutating(self) -> None:
        """§6 lists apps.launch and apps.focus on the allowlist."""
        for name in ("apps.launch", "apps.focus"):
            spec = registry.get(name)
            assert spec is not None
            assert spec.read_only is False
            assert spec.requires_confirmation is True

    def test_list_windows_is_read_only(self) -> None:
        spec = registry.get("apps.list_windows")
        assert spec is not None
        assert spec.read_only is True

    def test_all_require_windows(self) -> None:
        for name in ("apps.launch", "apps.focus", "apps.list_windows"):
            spec = registry.get(name)
            assert spec is not None
            assert spec.requires_windows is True

    def test_hidden_from_the_model_off_windows(self, cfg: JarvisConfig) -> None:
        """A tool the model cannot successfully call must not be offered."""
        names = [spec.name for spec in registry.select(config=cfg)]
        assert "apps.launch" not in names

    def test_unconfirmed_launch_is_refused(self) -> None:
        result = registry.dispatch("apps.launch", {"name": "notepad"})
        assert result.ok is False

    def test_allowlist_matches_the_contract(self, cfg: JarvisConfig) -> None:
        assert "apps.launch" in cfg.gate.mutating_allowlist
        assert "apps.focus" in cfg.gate.mutating_allowlist
        assert "apps.list_windows" not in cfg.gate.mutating_allowlist

    def test_schema_caps_the_name_length(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppLaunchInput(name="x" * 200)


class TestSourceSafety:
    def test_no_shell_true_anywhere(self) -> None:
        source = Path(apps_module.__file__).read_text(encoding="utf-8")
        assert "shell=True" not in source
