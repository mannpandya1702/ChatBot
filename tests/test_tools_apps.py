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


class TestStartMenuCannotSmuggleAShell:
    """The Start Menu reaches shells under display names, not executable names.

    Checking only the executable name let "Windows PowerShell.lnk" through,
    which handed the model a shell with none of the §6 hardening on it.
    """

    @staticmethod
    def _start_menu(tmp_path: Path, *names: str) -> Path:
        programs = tmp_path / "Microsoft/Windows/Start Menu/Programs"
        programs.mkdir(parents=True, exist_ok=True)
        for name in names:
            (programs / name).write_bytes(b"")
        return tmp_path

    @pytest.mark.parametrize(
        ("shortcut", "asked_for"),
        [
            ("Windows PowerShell.lnk", "powershell"),
            ("Windows PowerShell ISE.lnk", "powershell ise"),
            ("Windows PowerShell (x86).lnk", "powershell (x86)"),
            ("Command Prompt.lnk", "command prompt"),
            ("Windows Terminal.lnk", "windows terminal"),
            ("Developer Command Prompt.lnk", "developer command prompt"),
            ("Registry Editor.lnk", "registry editor"),
            ("Python 3.12.lnk", "python 3.12"),
        ],
    )
    def test_a_shell_shortcut_is_refused(
        self,
        shortcut: str,
        asked_for: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = self._start_menu(tmp_path, shortcut)
        monkeypatch.setattr(apps_module, "is_windows", lambda: True)
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setenv("APPDATA", str(root))
        monkeypatch.setenv("PROGRAMDATA", str(root))

        with pytest.raises(SafetyViolationError):
            resolve_application(asked_for)

    @pytest.mark.parametrize(
        ("shortcut", "asked_for"),
        [
            ("Adobe Photoshop.lnk", "photoshop"),
            ("PowerToys.lnk", "powertoys"),
            ("Slack.lnk", "slack"),
            ("Blender.lnk", "blender"),
        ],
    )
    def test_an_ordinary_application_still_resolves(
        self,
        shortcut: str,
        asked_for: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The word check must not swallow names that merely look similar."""
        root = self._start_menu(tmp_path, shortcut)
        monkeypatch.setattr(apps_module, "is_windows", lambda: True)
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setenv("APPDATA", str(root))
        monkeypatch.setenv("PROGRAMDATA", str(root))

        resolved = resolve_application(asked_for)
        assert resolved is not None
        assert Path(resolved).name == shortcut

    def test_a_shortcut_pointing_at_a_shell_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shortcut's name is a display name, so the target is checked too."""
        root = self._start_menu(tmp_path, "Innocent Helper.lnk")
        monkeypatch.setattr(apps_module, "is_windows", lambda: True)
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setenv("APPDATA", str(root))
        monkeypatch.setenv("PROGRAMDATA", str(root))
        # The name passes; only the target gives it away.
        assert apps_module._forbidden_reason("Innocent Helper.lnk") is None
        monkeypatch.setattr(
            apps_module,
            "_shortcut_target",
            lambda _p: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        )

        with pytest.raises(SafetyViolationError):
            resolve_application("innocent helper")

    def test_an_unreadable_shortcut_target_still_falls_back_to_the_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing to read the target may never turn a refusal into a launch."""
        root = self._start_menu(tmp_path, "Command Prompt.lnk")
        monkeypatch.setattr(apps_module, "is_windows", lambda: True)
        monkeypatch.setattr(apps_module.shutil, "which", lambda _n: None)
        monkeypatch.setenv("APPDATA", str(root))
        monkeypatch.setenv("PROGRAMDATA", str(root))
        monkeypatch.setattr(apps_module, "_shortcut_target", lambda _p: None)

        with pytest.raises(SafetyViolationError):
            resolve_application("command prompt")

    def test_the_tool_description_matches_the_behaviour(self) -> None:
        """§5: the description drives tool choice, so it must be true."""
        spec = next(s for s in registry if s.name == "apps.launch")
        assert "shell" in spec.description.lower()


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
