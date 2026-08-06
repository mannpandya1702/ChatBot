"""T-2.4 verification: sys.disk."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from jarvis.tools import sys_disk
from jarvis.tools.registry import registry
from jarvis.tools.sys_disk import DiskInput, disk_status, smart_health
from jarvis.util.errors import ToolExecutionError


class TestVolumeClassification:
    @pytest.mark.parametrize(
        ("percent", "free_gb", "expected"),
        [
            (10, 500, "healthy"),
            (74, 200, "healthy"),
            (80, 100, "filling"),
            (91, 50, "low"),
            (96, 30, "critical"),
            # Absolute floor beats percentage: a nearly empty small disk is still critical.
            (50, 3, "critical"),
            (50, 10, "low"),
            # And a huge disk at 80 percent still has plenty left.
            (80, 800, "filling"),
        ],
    )
    def test_thresholds(self, percent: float, free_gb: float, expected: str) -> None:
        assert sys_disk._describe_volume(percent, free_gb) == expected


class TestDiskStatus:
    def test_returns_volumes(self) -> None:
        result = disk_status(DiskInput())
        assert result.volumes
        assert result.total_free_gb >= 0

    def test_volumes_capped_at_five(self) -> None:
        assert len(disk_status(DiskInput()).volumes) <= 5

    def test_volumes_sorted_fullest_first(self) -> None:
        """The drive at risk should be the first thing mentioned aloud."""
        volumes = disk_status(DiskInput()).volumes
        assert volumes == sorted(volumes, key=lambda v: v.percent_used, reverse=True)

    def test_usage_fields_are_consistent(self) -> None:
        for volume in disk_status(DiskInput()).volumes:
            assert volume.total_gb >= volume.used_gb
            assert 0 <= volume.percent_used <= 100


class TestTotalsCoverEveryVolume:
    """§5 forbids inventing metrics, and an understated total is an invented one.

    The spoken list is capped at 5, but the totals are not: a machine with more
    volumes than that was being told it had less free space than it really did.
    """

    @staticmethod
    def _fake_volumes(count: int) -> Any:
        from jarvis.tools.sys_disk import VolumeUsage

        def _make() -> list[VolumeUsage]:
            return [
                VolumeUsage(
                    mount=f"V{index}:",
                    filesystem="NTFS",
                    total_gb=100.0,
                    used_gb=float(index),
                    free_gb=10.0,
                    # Descending, so _volumes' fullest-first order is preserved.
                    percent_used=float(90 - index),
                    status="healthy",
                )
                for index in range(count)
            ]

        return _make

    def test_free_space_sums_every_volume_not_just_the_spoken_five(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys_disk, "_volumes", self._fake_volumes(8))

        result = disk_status(DiskInput(include_throughput=False, include_smart=False))

        assert len(result.volumes) == 5, "the spoken list is still capped"
        assert result.volume_count == 8
        assert result.total_free_gb == pytest.approx(80.0), "8 volumes at 10 GB each"
        assert result.total_capacity_gb == pytest.approx(800.0)

    def test_the_totals_match_the_list_when_it_is_not_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys_disk, "_volumes", self._fake_volumes(3))

        result = disk_status(DiskInput(include_throughput=False, include_smart=False))

        assert result.volume_count == 3
        assert len(result.volumes) == 3
        assert result.total_free_gb == pytest.approx(
            sum(v.free_gb for v in result.volumes)
        )

    def test_no_volumes_reports_zero_rather_than_failing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys_disk, "_volumes", lambda: [])

        result = disk_status(DiskInput(include_throughput=False, include_smart=False))

        assert result.volume_count == 0
        assert result.total_free_gb == 0.0

    def test_the_real_totals_are_at_least_the_listed_ones(self) -> None:
        result = disk_status(DiskInput(include_throughput=False, include_smart=False))
        assert result.total_free_gb >= sum(v.free_gb for v in result.volumes) - 0.01
        assert result.volume_count >= len(result.volumes)

    def test_throughput_is_opt_in(self) -> None:
        result = disk_status(DiskInput())
        assert result.read_mb_per_s is None
        assert result.write_mb_per_s is None

    def test_throughput_when_requested(self) -> None:
        result = disk_status(DiskInput(include_throughput=True))
        # Counters can be unavailable in a container, in which case None is correct.
        assert result.read_mb_per_s is None or result.read_mb_per_s >= 0

    def test_smart_is_opt_in(self) -> None:
        assert disk_status(DiskInput()).smart_status is None

    def test_unreadable_partition_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = sys_disk.psutil.disk_usage

        def selective(path: str) -> Any:
            if path == "/nonexistent-mount":
                raise PermissionError("denied")
            return real(path)

        class Part:
            device = "/dev/fake"
            mountpoint = "/nonexistent-mount"
            fstype = "ext4"
            opts = "rw"

        real_parts = sys_disk.psutil.disk_partitions

        monkeypatch.setattr(sys_disk.psutil, "disk_usage", selective)
        monkeypatch.setattr(
            sys_disk.psutil, "disk_partitions", lambda all=False: [Part(), *real_parts(all)]
        )
        assert disk_status(DiskInput()).volumes is not None

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(all: bool = False) -> Any:
            raise RuntimeError("no partitions")

        monkeypatch.setattr(sys_disk.psutil, "disk_partitions", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            disk_status(DiskInput())
        assert excinfo.value.speakable == "I could not read the drive information."


class TestSmartHealth:
    def test_missing_smartctl_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """T-2.4: degrade gracefully when smartctl is not present."""
        monkeypatch.setattr(sys_disk.shutil, "which", lambda _name: None)
        assert "not installed" in smart_health()

    def test_healthy_drive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys_disk.shutil, "which", lambda _name: "/usr/sbin/smartctl")

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            stdout = "/dev/sda -d sat\n" if "--scan" in argv else "SMART overall-health: PASSED\n"
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        monkeypatch.setattr(sys_disk.subprocess, "run", fake_run)
        assert smart_health() == "healthy"

    def test_failing_drive_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys_disk.shutil, "which", lambda _name: "/usr/sbin/smartctl")

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            stdout = "/dev/sdb -d sat\n" if "--scan" in argv else "SMART overall-health: FAILED!\n"
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        monkeypatch.setattr(sys_disk.subprocess, "run", fake_run)
        result = smart_health()
        assert "failing" in result
        assert "/dev/sdb" in result

    def test_no_drives_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys_disk.shutil, "which", lambda _name: "/usr/sbin/smartctl")
        monkeypatch.setattr(
            sys_disk.subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""),
        )
        assert "no drives" in smart_health()

    def test_smartctl_timeout_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys_disk.shutil, "which", lambda _name: "/usr/sbin/smartctl")

        def timeout(argv: list[str], **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(argv, 10)

        monkeypatch.setattr(sys_disk.subprocess, "run", timeout)
        assert "unavailable" in smart_health()


class TestSafety:
    def test_never_calls_win32_product(self) -> None:
        """T-2.4 forbids Win32_Product; querying it triggers MSI reconfiguration."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(sys_disk.__file__).read_text(encoding="utf-8"))
        # The module docstring names Win32_Product to explain why it is banned.
        # Exclude docstring nodes by identity, then any remaining string literal
        # mentioning it would be an actual query.
        docstring_ids = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstring_ids.add(id(first.value))
        offenders = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "Win32_Product" in node.value
            and id(node) not in docstring_ids
        ]
        assert offenders == [], "Win32_Product must never be queried"


class TestRegistration:
    def test_registered_and_read_only(self) -> None:
        spec = registry.get("sys.disk")
        assert spec is not None
        assert spec.read_only is True

    def test_dispatch_and_serialise(self) -> None:
        result = registry.dispatch("sys.disk", {})
        assert result.ok is True
        json.dumps(result.data)


class TestSnapshot:
    def test_returns_hud_fields(self) -> None:
        assert set(sys_disk.snapshot()) == {"disk_percent", "disk_free_gb"}

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(all: bool = False) -> Any:
            raise RuntimeError("gone")

        monkeypatch.setattr(sys_disk.psutil, "disk_partitions", boom)
        assert sys_disk.snapshot()["disk_percent"] == 0.0
