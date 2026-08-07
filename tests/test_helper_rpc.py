"""T-2.8 verification: the elevated helper's RPC boundary.

The helper runs elevated, so this is the highest-value attack surface in the
system. These tests are written as probes against §6: can anything other than
the three named, parameterless sensor reads get through?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import JarvisConfig
from jarvis.helper import __main__ as helper_main
from jarvis.helper.rpc import (
    ALLOWED_METHODS,
    ErrorCode,
    HelperClient,
    HelperServer,
    RpcError,
    pipe_path,
)
from jarvis.util.errors import HelperUnavailableError, JarvisError


def _request(method: str, **extra: Any) -> str:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    payload.update(extra)
    return json.dumps(payload)


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def server(calls: list[str]) -> HelperServer:
    def make(name: str) -> Any:
        def handler() -> dict[str, Any]:
            calls.append(name)
            return {"ok": True, "method": name}

        return handler

    return HelperServer({name: make(name) for name in ALLOWED_METHODS})


class TestMethodSurface:
    def test_the_surface_is_exactly_three_methods(self) -> None:
        """§6 names exactly get_thermals, get_fans, get_pending_updates."""
        assert set(ALLOWED_METHODS) == {"get_thermals", "get_fans", "get_pending_updates"}

    def test_each_permitted_method_works(self, server: HelperServer, calls: list[str]) -> None:
        for method in sorted(ALLOWED_METHODS):
            reply = json.loads(server.handle_message(_request(method)))
            assert reply["result"]["method"] == method
        assert sorted(calls) == sorted(ALLOWED_METHODS)

    @pytest.mark.parametrize(
        "method",
        [
            "install_updates",
            "run",
            "exec",
            "shutdown",
            "get_thermals_extra",
            "GET_THERMALS",
            "__init__",
            "eval",
        ],
    )
    def test_unknown_methods_are_refused(
        self, server: HelperServer, calls: list[str], method: str
    ) -> None:
        reply = json.loads(server.handle_message(_request(method)))
        assert reply["error"]["code"] == ErrorCode.METHOD_NOT_FOUND
        assert calls == []

    def test_a_handler_outside_the_allowlist_is_rejected_at_construction(self) -> None:
        """A future edit that adds a fourth method must fail loudly at startup."""
        with pytest.raises(ValueError, match="may not expose"):
            HelperServer({"get_thermals": dict, "install_update": dict})

    def test_registered_handler_not_in_allowlist_never_dispatches(self) -> None:
        """Even if construction were bypassed, dispatch checks the allowlist."""
        server = HelperServer({"get_thermals": lambda: {"ok": True}})
        reply = json.loads(server.handle_message(_request("get_fans")))
        assert reply["error"]["code"] == ErrorCode.METHOD_NOT_FOUND


class TestParameterRejection:
    """§6: the helper never takes an LLM-authored string."""

    @pytest.mark.parametrize(
        "params",
        [
            {"command": "shutdown /s"},
            ["rm", "-rf"],
            {"path": "C:/Windows/System32"},
            "arbitrary string",
            {"sensor": "cpu"},
        ],
    )
    def test_any_params_field_is_refused(
        self, server: HelperServer, calls: list[str], params: Any
    ) -> None:
        reply = json.loads(server.handle_message(_request("get_thermals", params=params)))
        assert reply["error"]["code"] == ErrorCode.INVALID_PARAMS
        assert calls == [], "the handler must not run when params were supplied"

    def test_empty_params_are_tolerated(self, server: HelperServer, calls: list[str]) -> None:
        """A well-behaved client may send an empty params object."""
        for empty in (None, {}, []):
            reply = json.loads(server.handle_message(_request("get_fans", params=empty)))
            assert "result" in reply
        assert len(calls) == 3


class TestMalformedInput:
    def test_invalid_json_is_an_error_not_a_crash(self, server: HelperServer) -> None:
        reply = json.loads(server.handle_message("{not json"))
        assert reply["error"]["code"] == ErrorCode.PARSE_ERROR

    def test_non_object_payload(self, server: HelperServer) -> None:
        reply = json.loads(server.handle_message("[1, 2, 3]"))
        assert reply["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_missing_jsonrpc_version(self, server: HelperServer) -> None:
        reply = json.loads(server.handle_message(json.dumps({"id": 1, "method": "get_fans"})))
        assert reply["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_wrong_jsonrpc_version(self, server: HelperServer) -> None:
        raw = json.dumps({"jsonrpc": "1.0", "id": 1, "method": "get_fans"})
        assert json.loads(server.handle_message(raw))["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_non_string_method(self, server: HelperServer) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": {"evil": True}})
        assert json.loads(server.handle_message(raw))["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_oversized_message_is_refused(self, server: HelperServer) -> None:
        """A pipe peer should never send more than a few hundred bytes."""
        raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "get_fans", "pad": "x" * 10_000})
        assert json.loads(server.handle_message(raw))["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_invalid_utf8_bytes(self, server: HelperServer) -> None:
        reply = json.loads(server.handle_message(b"\xff\xfe\x00bad"))
        assert reply["error"]["code"] == ErrorCode.PARSE_ERROR

    def test_bytes_input_works(self, server: HelperServer) -> None:
        reply = json.loads(server.handle_message(_request("get_fans").encode("utf-8")))
        assert reply["result"]["ok"] is True

    def test_handle_message_never_raises(self, server: HelperServer) -> None:
        for raw in ["", "null", "0", '"str"', "{}", "[]", "\x00"]:
            assert json.loads(server.handle_message(raw))


class TestHandlerFailure:
    def test_raising_handler_becomes_an_error_response(self) -> None:
        """An elevated process must not die on a sensor failure."""

        def explode() -> dict[str, Any]:
            raise RuntimeError("sensor bus wedged")

        server = HelperServer({"get_thermals": explode})
        reply = json.loads(server.handle_message(_request("get_thermals")))
        assert reply["error"]["code"] == ErrorCode.SENSOR_UNAVAILABLE
        assert "sensor bus wedged" in reply["error"]["message"]

    def test_rpc_error_preserves_its_code(self) -> None:
        def explode() -> dict[str, Any]:
            raise RpcError(ErrorCode.INTERNAL_ERROR, "not elevated")

        server = HelperServer({"get_fans": explode})
        reply = json.loads(server.handle_message(_request("get_fans")))
        assert reply["error"]["code"] == ErrorCode.INTERNAL_ERROR


class TestPipePath:
    def test_shape(self) -> None:
        assert pipe_path("jarvis-helper") == r"\\.\pipe\jarvis-helper"


class TestClient:
    def _client(self, cfg: JarvisConfig, server: HelperServer) -> HelperClient:
        return HelperClient(cfg, transport=server.handle_message)

    def test_round_trip(self, cfg: JarvisConfig, server: HelperServer) -> None:
        assert self._client(cfg, server).get_thermals()["ok"] is True

    def test_rejects_a_method_outside_the_allowlist(
        self, cfg: JarvisConfig, server: HelperServer
    ) -> None:
        with pytest.raises(ValueError, match="not a helper method"):
            self._client(cfg, server).call("install_updates")

    def test_results_are_cached(self, cfg: JarvisConfig, calls: list[str]) -> None:
        server = HelperServer({"get_thermals": lambda: (calls.append("hit"), {"v": 1})[1]})
        client = HelperClient(cfg, transport=server.handle_message)
        client.get_thermals()
        client.get_thermals()
        client.get_thermals()
        assert len(calls) == 1, "repeated questions must not hammer the sensors"

    def test_cache_expires(self, cfg: JarvisConfig, calls: list[str]) -> None:
        server = HelperServer({"get_thermals": lambda: (calls.append("hit"), {"v": 1})[1]})
        now = [0.0]
        client = HelperClient(cfg, transport=server.handle_message, clock=lambda: now[0])
        client.get_thermals()
        now[0] = cfg.helper.cache_ttl_s + 1
        client.get_thermals()
        assert len(calls) == 2

    def test_clear_cache(self, cfg: JarvisConfig, calls: list[str]) -> None:
        server = HelperServer({"get_thermals": lambda: (calls.append("hit"), {"v": 1})[1]})
        client = HelperClient(cfg, transport=server.handle_message)
        client.get_thermals()
        client.clear_cache()
        client.get_thermals()
        assert len(calls) == 2

    def test_error_response_becomes_helper_unavailable(
        self, cfg: JarvisConfig, server: HelperServer
    ) -> None:
        client = HelperClient(cfg, transport=lambda _r: server.handle_message("{bad"))
        with pytest.raises(HelperUnavailableError):
            client.get_fans()

    def test_malformed_response_becomes_helper_unavailable(self, cfg: JarvisConfig) -> None:
        client = HelperClient(cfg, transport=lambda _r: "not json at all")
        with pytest.raises(HelperUnavailableError, match="malformed JSON"):
            client.get_fans()

    def test_missing_result_object(self, cfg: JarvisConfig) -> None:
        client = HelperClient(
            cfg, transport=lambda _r: json.dumps({"jsonrpc": "2.0", "id": 1, "result": "nope"})
        )
        with pytest.raises(HelperUnavailableError, match="no result object"):
            client.get_fans()

    def test_is_available_never_raises(self, cfg: JarvisConfig) -> None:
        def dead(_request: str) -> str:
            raise HelperUnavailableError("pipe is gone")

        assert HelperClient(cfg, transport=dead).is_available() is False

    def test_is_available_true_when_answering(
        self, cfg: JarvisConfig, server: HelperServer
    ) -> None:
        assert self._client(cfg, server).is_available() is True

    def test_request_carries_no_params(self, cfg: JarvisConfig) -> None:
        """The client must never invent a params field."""
        seen: list[dict[str, Any]] = []

        def capture(request: str) -> str:
            seen.append(json.loads(request))
            return json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})

        HelperClient(cfg, transport=capture).get_thermals()
        assert "params" not in seen[0]


class TestHelperEntrypoint:
    def test_build_handlers_matches_the_allowlist(self) -> None:
        class FakeMonitor:
            def get_thermals(self) -> dict[str, Any]:
                return {}

            def get_fans(self) -> dict[str, Any]:
                return {}

        handlers = helper_main.build_handlers(FakeMonitor())  # type: ignore[arg-type]
        assert set(handlers) == ALLOWED_METHODS

    def test_is_elevated_is_false_off_windows(self) -> None:
        assert helper_main.is_elevated() is False

    def test_check_flag_reports_the_method_surface(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = helper_main.main(["--check"])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["methods"]) == ALLOWED_METHODS
        assert code == 1  # not elevated on this host

    def test_no_install_capability_exists_anywhere(self) -> None:
        """T-2.9: the helper reports updates and never installs them."""
        from pathlib import Path

        import jarvis.helper.lhm as lhm_module

        source = Path(lhm_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("Install-WindowsUpdate", ".Install(", "AcceptEula", "Download()"):
            assert forbidden not in source, f"{forbidden} would install updates"


class TestVendoredSensorLibrary:
    """M-9: the DLL the elevated helper loads, and the script that fetches it.

    The three have to agree on one path, one version, and one hash, or the
    helper reports every sensor unavailable and nothing says why.
    """

    ROOT = Path(__file__).resolve().parents[1]
    DLL = ROOT / "vendor" / "LibreHardwareMonitorLib.dll"

    def test_the_dll_is_present(self) -> None:
        assert self.DLL.is_file(), (
            "vendor/LibreHardwareMonitorLib.dll is missing; "
            "run scripts/fetch_vendor.ps1 on the Windows host"
        )

    def test_it_is_a_windows_binary(self) -> None:
        """A stray text file or an HTML error page would fail silently later."""
        head = self.DLL.read_bytes()[:2]
        assert head == b"MZ", f"not a PE image, starts with {head!r}"

    def test_the_helper_looks_where_the_file_actually_is(self) -> None:
        """The loader path is a string literal, so nothing else checks it."""
        source = (self.ROOT / "src" / "jarvis" / "helper" / "lhm.py").read_text(
            encoding="utf-8"
        )
        assert '"vendor" / "LibreHardwareMonitorLib.dll"' in source

    def test_the_recorded_hash_matches_the_committed_file(self) -> None:
        import hashlib
        import re

        digest = hashlib.sha256(self.DLL.read_bytes()).hexdigest()
        readme = (self.ROOT / "vendor" / "README.md").read_text(encoding="utf-8")
        assert digest in readme.lower(), "vendor/README.md records a different hash"

        script = (self.ROOT / "scripts" / "fetch_vendor.ps1").read_text(encoding="utf-8")
        pinned = re.findall(r"'([0-9A-Fa-f]{64})'", script)
        assert digest.upper() in [p.upper() for p in pinned], (
            "scripts/fetch_vendor.ps1 pins a hash that is not the committed file"
        )

    def test_the_fetch_script_verifies_what_it_downloads(self) -> None:
        """An unverified binary that an elevated process loads is not acceptable."""
        script = (self.ROOT / "scripts" / "fetch_vendor.ps1").read_text(encoding="utf-8")
        assert "Get-FileHash" in script
        assert "Remove-Item $Target" in script, "a mismatched download must not be left on disk"

    def test_the_licence_is_recorded(self) -> None:
        """§1 permits MPL-2.0 specifically, so the file has to say so."""
        readme = (self.ROOT / "vendor" / "README.md").read_text(encoding="utf-8")
        assert "MPL-2.0" in readme


class TestTheElevatedHelperVerifiesWhatItLoads:
    """clr.AddReference executes whatever it is pointed at, as Administrator.

    scripts/fetch_vendor.ps1 checks the library's hash when it downloads it,
    which is the wrong moment: the file then sits on disk until the next time
    the helper starts. Combined with a project root that resolved into %TEMP%
    under PyInstaller, that made the path the entire trust boundary. Verifying
    the bytes at load time is what demotes the path to a convenience.
    """

    def test_the_vendored_library_is_accepted(self) -> None:
        from jarvis.helper.lhm import verify_assembly
        from jarvis.util.platform import project_root

        dll = project_root() / "vendor" / "LibreHardwareMonitorLib.dll"
        if not dll.is_file():
            pytest.skip("the vendored DLL is not present in this checkout")
        verify_assembly(dll)

    def test_a_substituted_library_is_refused(self, tmp_path: Path) -> None:
        from jarvis.helper.lhm import verify_assembly

        planted = tmp_path / "LibreHardwareMonitorLib.dll"
        planted.write_bytes(b"ATTACKER SUPPLIED PAYLOAD")

        with pytest.raises(JarvisError, match="does not match the vendored"):
            verify_assembly(planted)

    def test_the_refusal_is_speakable(self, tmp_path: Path) -> None:
        from jarvis.helper.lhm import verify_assembly

        planted = tmp_path / "x.dll"
        planted.write_bytes(b"nope")
        with pytest.raises(JarvisError) as excinfo:
            verify_assembly(planted)
        assert "altered" in (excinfo.value.speakable or "")

    def test_an_explicitly_configured_path_is_allowed_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Choosing another build by hand is a decision, not an attack."""
        import logging

        from jarvis.helper.lhm import verify_assembly

        other = tmp_path / "other.dll"
        other.write_bytes(b"a different build")
        with caplog.at_level(logging.WARNING):
            verify_assembly(other, explicit=True)
        assert any("unverified" in record.message for record in caplog.records)
