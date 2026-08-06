"""T-3.2, T-3.3, T-3.4 verification: the Tauri HUD shell and frontend.

A full ``npm run tauri build`` needs a Windows host with the Rust and WebView2
toolchains, so that stays a manual check (recorded in PROGRESS.md). What can be
verified here is everything short of the compile: the frontend parses, the
window is configured the way §7 T-3.2 requires, the security policy does not
let the HUD talk to anything but the local core, and the orb declares the
particle count and per-state palette T-3.3 asks for.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "jarvis" / "ui" / "app"
JS_FILES = ["src/orb.js", "src/panels.js", "src/main.js", "vite.config.js"]


def _node() -> str:
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not installed")
    return exe


class TestFilesExist:
    @pytest.mark.parametrize(
        "relative",
        [
            "package.json",
            "vite.config.js",
            "src/index.html",
            "src/style.css",
            "src/main.js",
            "src/orb.js",
            "src/panels.js",
            "src-tauri/tauri.conf.json",
            "src-tauri/Cargo.toml",
            "src-tauri/build.rs",
            "src-tauri/src/main.rs",
        ],
    )
    def test_present_and_non_empty(self, relative: str) -> None:
        path = APP / relative
        assert path.is_file(), f"{relative} is missing"
        assert path.stat().st_size > 0


class TestJavaScriptParses:
    @pytest.mark.parametrize("relative", JS_FILES)
    def test_syntax(self, relative: str) -> None:
        """A syntax error would only surface at build time otherwise."""
        result = subprocess.run(
            [_node(), "--check", str(APP / relative)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{relative} failed to parse:\n{result.stderr}"


class TestTauriWindow:
    @pytest.fixture
    def conf(self) -> dict:
        return json.loads((APP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))

    def test_window_flags_match_the_contract(self, conf: dict) -> None:
        """T-3.2: frameless, transparent, always on top, no taskbar entry."""
        window = conf["app"]["windows"][0]
        assert window["decorations"] is False
        assert window["transparent"] is True
        assert window["alwaysOnTop"] is True
        assert window["skipTaskbar"] is True

    def test_tray_is_configured(self, conf: dict) -> None:
        assert "trayIcon" in conf["app"]

    def test_tray_menu_has_show_hide_quit(self) -> None:
        source = (APP / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
        for item in ('"show"', '"hide"', '"quit"'):
            assert item in source

    def test_click_through_is_the_default(self) -> None:
        """The desktop underneath must stay usable."""
        source = (APP / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
        assert "set_ignore_cursor_events(true)" in source

    def test_click_through_can_be_toggled_off(self) -> None:
        """Without a toggle the drag handle and transcript would be unreachable."""
        source = (APP / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
        assert "set_click_through" in source

    def test_csp_only_allows_the_local_core(self, conf: dict) -> None:
        """§0.1: the HUD must not be able to reach anything off this machine."""
        csp = conf["app"]["security"]["csp"]
        assert "connect-src" in csp
        assert "127.0.0.1:8765" in csp or "localhost:8765" in csp
        assert "https://" not in csp
        assert "*" not in csp.split("connect-src")[1].split(";")[0]

    def test_frontend_dist_points_at_the_build_output(self, conf: dict) -> None:
        """T-3.2: the shipped HUD has to be the bundled one, not the sources.

        orb.js imports 'three' by bare specifier, which no browser can resolve.
        Pointing frontendDist at src/ shipped that import verbatim, so the built
        HUD loaded a blank window.
        """
        dist = conf["build"]["frontendDist"]
        assert dist == "../dist", f"frontendDist is {dist!r}, which skips bundling"

        vite = (APP / "vite.config.js").read_text(encoding="utf-8")
        assert "outDir: '../dist'" in vite, "vite and tauri disagree about the output directory"

    def test_the_bundler_runs_before_the_build(self, conf: dict) -> None:
        """Without this, frontendDist points at a directory nothing populated."""
        assert conf["build"]["beforeBuildCommand"] == "npm run build"

    def test_dev_mode_serves_through_vite(self, conf: dict) -> None:
        assert conf["build"]["beforeDevCommand"] == "npm run dev"
        assert conf["build"]["devUrl"] == "http://localhost:5173"

    def test_the_dev_url_matches_the_vite_port(self, conf: dict) -> None:
        vite = (APP / "vite.config.js").read_text(encoding="utf-8")
        port = conf["build"]["devUrl"].rsplit(":", 1)[1]
        assert f"port: {port}" in vite

    def test_the_sources_still_use_a_bare_specifier(self) -> None:
        """Guards the reason the bundler is mandatory rather than optional.

        If this ever stops being true the import was inlined by hand, and the
        frontendDist assertion above should be revisited rather than silently
        left pointing at a build step nothing needs.
        """
        source = (APP / "src" / "orb.js").read_text(encoding="utf-8")
        assert "from 'three'" in source


class TestOrb:
    @pytest.fixture
    def source(self) -> str:
        return (APP / "src" / "orb.js").read_text(encoding="utf-8")

    def test_particle_count_is_about_two_thousand(self, source: str) -> None:
        """T-3.3 asks for roughly 2000 points."""
        assert "const PARTICLE_COUNT = 2000;" in source

    def test_uses_buffer_geometry_and_points_material(self, source: str) -> None:
        assert "BufferGeometry" in source
        assert "PointsMaterial" in source

    def test_additive_blending(self, source: str) -> None:
        assert "AdditiveBlending" in source

    def test_per_particle_sine_and_cosine_drift(self, source: str) -> None:
        assert "Math.sin(this.time * speed + phase)" in source
        assert "Math.cos(this.time * speed * 0.7 + phase)" in source

    def test_every_state_has_a_palette(self, source: str) -> None:
        """T-3.3 names a distinct palette for each of the five states."""
        for state in ("idle", "listening", "thinking", "speaking", "error"):
            assert f"{state}:" in source

    def test_displacement_is_driven_by_audio_level(self, source: str) -> None:
        assert "this.level" in source
        assert "displacement" in source

    def test_no_allocation_in_the_update_loop(self, source: str) -> None:
        """Allocating per frame at 60 fps would cause visible GC stutter."""
        body = source.split("update(deltaSeconds)")[1].split("\n  }")[0]
        assert "new Float32Array" not in body
        assert "new THREE." not in body


class TestPanels:
    @pytest.fixture
    def source(self) -> str:
        return (APP / "src" / "panels.js").read_text(encoding="utf-8")

    def test_sparklines_exist(self, source: str) -> None:
        assert "export class Sparkline" in source

    def test_missing_metric_is_a_gap_not_a_zero(self, source: str) -> None:
        """A machine with no NVIDIA GPU must not show a flat zero GPU line."""
        assert "value === null || value === undefined ? null" in source

    def test_history_is_bounded(self, source: str) -> None:
        assert "while (this.values.length > HISTORY_LENGTH) this.values.shift();" in source


class TestReconnect:
    @pytest.fixture
    def source(self) -> str:
        return (APP / "src" / "main.js").read_text(encoding="utf-8")

    def test_reconnects_on_drop(self, source: str) -> None:
        """T-3.5 requires reconnect-on-drop."""
        assert "scheduleReconnect" in source
        assert "onclose" in source

    def test_backoff_is_capped(self, source: str) -> None:
        """A core that stays down must not be hammered."""
        assert "RECONNECT_MAX_MS" in source
        assert "Math.min(this.retryDelay * 2, RECONNECT_MAX_MS)" in source

    def test_backoff_resets_only_on_a_real_connection(self, source: str) -> None:
        opened = source.split("socket.onopen")[1].split("};")[0]
        assert "RECONNECT_MIN_MS" in opened

    def test_malformed_frames_are_ignored(self, source: str) -> None:
        assert "JSON.parse(event.data)" in source
        assert "catch" in source

    def test_disconnected_state_is_shown(self, source: str) -> None:
        """A frozen HUD showing stale data would be worse than an honest one."""
        assert "setConnected" in source


class TestBundleIcons:
    """T-3.2 / M-8: `npm run tauri build` fails outright without these.

    They are checked structurally rather than just for existence, because a
    malformed .ico fails deep inside the Windows bundler with an error that does
    not name the file.
    """

    ICONS = APP / "src-tauri" / "icons"

    def test_both_icons_are_present(self) -> None:
        for name in ("icon.png", "icon.ico"):
            path = self.ICONS / name
            assert path.is_file(), f"{name} is missing, the bundle cannot be built"
            assert path.stat().st_size > 0

    def test_tauri_config_points_at_them(self) -> None:
        conf = json.loads((APP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
        for relative in conf["bundle"]["icon"]:
            assert (APP / "src-tauri" / relative).is_file(), f"{relative} is referenced but absent"

    def test_the_tray_icon_exists_too(self) -> None:
        conf = json.loads((APP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
        relative = conf["app"]["trayIcon"]["iconPath"]
        assert (APP / "src-tauri" / relative).is_file()

    def test_the_png_is_a_square_rgba_image(self) -> None:
        import struct

        data = (self.ICONS / "icon.png").read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        assert data[12:16] == b"IHDR"
        width, height = struct.unpack(">II", data[16:24])
        depth, colour_type = data[24], data[25]
        assert width == height >= 256, "Tauri wants a large square source icon"
        assert depth == 8
        assert colour_type == 6, "must be RGBA, the HUD icon is transparent"

    def test_the_ico_is_structurally_valid(self) -> None:
        """Every directory entry must point at bytes that are actually there."""
        import struct

        data = (self.ICONS / "icon.ico").read_bytes()
        reserved, kind, count = struct.unpack("<HHH", data[:6])
        assert reserved == 0
        assert kind == 1, "type 1 is an icon, type 2 is a cursor"
        assert count > 0

        seen = []
        for index in range(count):
            offset = 6 + 16 * index
            entry = struct.unpack("<BBBBHHII", data[offset : offset + 16])
            width, _height, _colours, _pad, _planes, bpp, size, position = entry
            assert size > 0
            assert position + size <= len(data), "an entry runs past the end of the file"
            assert bpp == 32
            seen.append(width or 256)

        assert 16 in seen, "no 16 pixel entry, the title bar icon would be a scaled blur"
        assert 256 in seen, "no 256 pixel entry, large icon view would be a scaled blur"
        assert len(seen) == len(set(seen)), "duplicate sizes in the directory"

    def test_regenerating_is_deterministic(self, tmp_path: Path) -> None:
        """The committed files must match what the script produces today.

        Otherwise the script drifts from the artefacts and nobody notices until
        someone regenerates and gets a different icon.
        """
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_icons.py"), "--out", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        for name in ("icon.png", "icon.ico"):
            assert (tmp_path / name).read_bytes() == (self.ICONS / name).read_bytes(), (
                f"{name} differs from what scripts/make_icons.py produces"
            )
