"""T-4.4 verification: files.search."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import JarvisConfig, load_config
from jarvis.tools import files as files_module
from jarvis.tools.files import (
    FileSearchInput,
    find_everything_cli,
    search_files,
)
from jarvis.tools.registry import registry


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small filesystem with predictable names."""
    root = tmp_path / "docs"
    (root / "taxes").mkdir(parents=True)
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".git").mkdir()

    (root / "invoice-2024.pdf").write_text("a", encoding="utf-8")
    (root / "invoice-2025.pdf").write_text("bb", encoding="utf-8")
    (root / "notes.txt").write_text("ccc", encoding="utf-8")
    (root / "taxes" / "tax-return.pdf").write_text("dddd", encoding="utf-8")
    (root / "node_modules" / "pkg" / "invoice.js").write_text("e", encoding="utf-8")
    (root / ".git" / "invoice-config").write_text("f", encoding="utf-8")
    return root


@pytest.fixture
def no_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the scandir fallback path."""
    monkeypatch.setattr(files_module, "find_everything_cli", lambda _config=None: None)


class TestEverythingDiscovery:
    def test_absent_returns_none(self, monkeypatch: pytest.MonkeyPatch, cfg: JarvisConfig) -> None:
        monkeypatch.setattr(files_module.shutil, "which", lambda _name: None)
        monkeypatch.setattr(Path, "is_file", lambda _self: False)
        assert find_everything_cli(cfg) is None

    def test_configured_path_wins(self, tmp_path: Path) -> None:
        exe = tmp_path / "es.exe"
        exe.write_text("", encoding="utf-8")
        config = load_config(
            tmp_path / "absent.yaml", tools={"everything_cli_path": str(exe)}
        )
        assert find_everything_cli(config) == exe

    def test_missing_configured_path_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = load_config(
            tmp_path / "absent.yaml", tools={"everything_cli_path": str(tmp_path / "nope.exe")}
        )
        monkeypatch.setattr(files_module.shutil, "which", lambda _name: None)
        assert find_everything_cli(config) is None


class TestScandirSearch:
    def test_finds_matches(self, tree: Path, no_everything: None) -> None:
        result = search_files(FileSearchInput(query="invoice", search_root=str(tree)))
        assert result.backend == "scandir"
        names = [hit.name for hit in result.results]
        assert "invoice-2024.pdf" in names
        assert "invoice-2025.pdf" in names

    def test_matching_is_case_insensitive(self, tree: Path, no_everything: None) -> None:
        assert search_files(FileSearchInput(query="INVOICE", search_root=str(tree))).results

    def test_searches_subdirectories(self, tree: Path, no_everything: None) -> None:
        result = search_files(FileSearchInput(query="tax-return", search_root=str(tree)))
        assert [hit.name for hit in result.results] == ["tax-return.pdf"]

    def test_skips_noise_directories(self, tree: Path, no_everything: None) -> None:
        """node_modules and .git are slow and never what a person means."""
        result = search_files(FileSearchInput(query="invoice", search_root=str(tree)))
        paths = [hit.path for hit in result.results]
        assert not any("node_modules" in path for path in paths)
        assert not any(".git" in path for path in paths)

    def test_extension_filter(self, tree: Path, no_everything: None) -> None:
        result = search_files(
            FileSearchInput(query="invoice", search_root=str(tree), extension="pdf")
        )
        assert all(hit.name.endswith(".pdf") for hit in result.results)

    def test_extension_filter_tolerates_a_leading_dot(
        self, tree: Path, no_everything: None
    ) -> None:
        result = search_files(
            FileSearchInput(query="invoice", search_root=str(tree), extension=".pdf")
        )
        assert result.results

    def test_results_capped_at_five(self, tmp_path: Path, no_everything: None) -> None:
        """§5 caps spoken list results at five items."""
        root = tmp_path / "many"
        root.mkdir()
        for index in range(20):
            (root / f"report-{index}.txt").write_text("x", encoding="utf-8")
        result = search_files(FileSearchInput(query="report", search_root=str(root)))
        assert len(result.results) == 5
        assert result.truncated is True
        assert result.total_found == 20

    def test_no_matches_is_not_an_error(self, tree: Path, no_everything: None) -> None:
        result = search_files(FileSearchInput(query="nothing-like-this", search_root=str(tree)))
        assert result.results == []
        assert result.total_found == 0

    def test_missing_root_returns_empty(self, no_everything: None) -> None:
        result = search_files(
            FileSearchInput(query="anything", search_root="/nope/does/not/exist")
        )
        assert result.results == []

    def test_hits_carry_size_and_time(self, tree: Path, no_everything: None) -> None:
        hit = search_files(FileSearchInput(query="notes", search_root=str(tree))).results[0]
        assert hit.size_mb >= 0
        assert hit.modified
        assert hit.is_directory is False

    def test_newest_first(self, tmp_path: Path, no_everything: None) -> None:
        import os
        import time

        root = tmp_path / "ordered"
        root.mkdir()
        for index in range(3):
            path = root / f"doc-{index}.txt"
            path.write_text("x", encoding="utf-8")
            os.utime(path, (time.time() - index * 100, time.time() - index * 100))
        result = search_files(FileSearchInput(query="doc", search_root=str(root)))
        assert [hit.name for hit in result.results] == ["doc-0.txt", "doc-1.txt", "doc-2.txt"]


class TestEverythingSearch:
    def test_uses_everything_when_present(
        self, tmp_path: Path, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe = tmp_path / "es.exe"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(files_module, "find_everything_cli", lambda _config=None: exe)

        target = tree / "invoice-2024.pdf"

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(argv, 0, f"{target}\n", "")

        monkeypatch.setattr(files_module.subprocess, "run", fake_run)
        result = search_files(FileSearchInput(query="invoice"))
        assert result.backend == "everything"
        assert result.results[0].name == "invoice-2024.pdf"

    def test_everything_failure_returns_empty_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe = tmp_path / "es.exe"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(files_module, "find_everything_cli", lambda _config=None: exe)

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            raise OSError("es.exe crashed")

        monkeypatch.setattr(files_module.subprocess, "run", fake_run)
        assert search_files(FileSearchInput(query="invoice")).results == []

    def test_everything_argv_has_no_shell_metacharacters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The query reaches es.exe as argv, never as a shell string."""
        exe = tmp_path / "es.exe"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(files_module, "find_everything_cli", lambda _config=None: exe)
        seen: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            seen.append(argv)
            assert kwargs.get("shell") in (None, False)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(files_module.subprocess, "run", fake_run)
        search_files(FileSearchInput(query="a && shutdown"))
        assert seen[0][-1] == "a && shutdown"


class TestSafety:
    def test_module_never_writes_or_deletes(self) -> None:
        """T-4.4: read-only. No open for write, no unlink, no rename."""
        source = Path(files_module.__file__).read_text(encoding="utf-8")
        for forbidden in (".unlink(", ".rmdir(", ".rename(", "shutil.rmtree", "shutil.move"):
            assert forbidden not in source, f"{forbidden} is a mutating call"


class TestRegistration:
    def test_registered_read_only(self) -> None:
        spec = registry.get("files.search")
        assert spec is not None
        assert spec.read_only is True

    def test_dispatch(self, tree: Path, no_everything: None) -> None:
        result = registry.dispatch(
            "files.search", {"query": "notes", "search_root": str(tree)}
        )
        assert result.ok is True
        assert result.data["results"]
