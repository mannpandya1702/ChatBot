"""Elevated sidecar entrypoint.

Run as ``jarvis-helper.exe`` with a UAC manifest, or during development as
``uv run python -m jarvis.helper``. This is the only JARVIS process that runs
elevated (§6), and it does exactly three things: read temperatures, read fan
speeds, and list pending Windows updates.

It accepts no command line input that reaches a sensor call, and no arbitrary
commands over the pipe. See ``rpc.py`` for the input validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from jarvis.helper.lhm import LibreHardwareMonitor, get_pending_updates
from jarvis.helper.rpc import ALLOWED_METHODS, HelperServer
from jarvis.util.errors import JarvisError
from jarvis.util.logging import setup_logging

_log = logging.getLogger(__name__)


def build_handlers(monitor: LibreHardwareMonitor | None = None) -> dict[str, Any]:
    """Wire the three permitted methods to their implementations.

    Args:
        monitor: Injected for tests. A real one is created when omitted.

    Returns:
        A mapping accepted by :class:`HelperServer`.
    """
    hardware = monitor if monitor is not None else LibreHardwareMonitor()
    handlers = {
        "get_thermals": hardware.get_thermals,
        "get_fans": hardware.get_fans,
        "get_pending_updates": get_pending_updates,
    }
    # Belt and braces: the server rejects anything outside ALLOWED_METHODS, but
    # a mismatch here would be a bug worth catching at startup rather than at
    # request time, inside an elevated process.
    assert set(handlers) == set(ALLOWED_METHODS)
    return handlers


def is_elevated() -> bool:
    """True when this process has administrator rights.

    Returns False rather than raising on a non-Windows host, where the concept
    does not apply.
    """
    from jarvis.util.platform import is_windows

    if not is_windows():
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - treat an unanswerable question as "no"
        return False


def main(argv: list[str] | None = None) -> int:
    """Serve the helper RPC, or answer one method and exit with ``--once``.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="jarvis-helper",
        description="Elevated sensor sidecar for JARVIS. Reads temperatures, fans, and "
        "pending Windows updates. Performs no other action.",
    )
    parser.add_argument("--pipe-name", default="jarvis-helper", help="Named pipe to listen on.")
    parser.add_argument(
        "--once",
        choices=sorted(ALLOWED_METHODS),
        help="Run one method, print the JSON result, and exit. Used for diagnostics.",
    )
    parser.add_argument("--check", action="store_true", help="Report elevation status and exit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level, console=True, json_file="helper.jsonl", force=True)

    if args.check:
        elevated = is_elevated()
        print(json.dumps({"elevated": elevated, "methods": sorted(ALLOWED_METHODS)}))  # noqa: T201
        return 0 if elevated else 1

    if not is_elevated():
        _log.warning(
            "the helper is not running elevated; temperature and fan sensors will be "
            "unavailable. Start jarvis-helper.exe and approve the prompt."
        )

    monitor = LibreHardwareMonitor()
    handlers = build_handlers(monitor)

    if args.once:
        try:
            result = handlers[args.once]()
        except JarvisError as exc:
            print(json.dumps({"error": exc.message, "speakable": exc.speakable}))  # noqa: T201
            return 1
        finally:
            monitor.close()
        print(json.dumps(result, default=str))  # noqa: T201
        return 0

    server = HelperServer(handlers)
    try:
        server.serve_forever(args.pipe_name)
    except KeyboardInterrupt:
        _log.info("helper stopped by the operator")
    except JarvisError as exc:
        _log.error("helper could not start: %s", exc.message)
        return 1
    finally:
        server.stop()
        monitor.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
