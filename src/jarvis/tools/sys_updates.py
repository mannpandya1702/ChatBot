"""Pending Windows updates, read through the elevated helper.

CLAUDE.md T-2.9 is explicit: report the count, names, and total size. **Never
install anything.** There is no install path in this module, no install method on
the helper, and no mutating tool for updates on the §6 allowlist.
"""

from __future__ import annotations

import logging

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.tools.sys_thermal import _helper
from jarvis.util.errors import HelperUnavailableError

__all__ = ["UpdatesInput", "UpdatesOutput", "updates_status"]

_log = logging.getLogger(__name__)


class UpdatesInput(ToolInput):
    """Arguments for the Windows Update query."""


class UpdatesOutput(ToolOutput):
    """Pending updates, or an explanation of why the query failed."""

    available: bool = Field(
        description="False when the elevated helper is not running or the query failed."
    )
    reason: str | None = Field(default=None, description="Why the query is unavailable.")
    count: int = Field(default=0, description="How many updates are pending.")
    names: list[str] = Field(
        default_factory=list, description="Up to 5 update titles, since the answer is spoken."
    )
    total_size_mb: float | None = Field(
        default=None, description="Combined download size in megabytes, when reported."
    )
    source: str | None = Field(
        default=None,
        description="Which backend answered, PSWindowsUpdate or Microsoft.Update.Session.",
    )


@tool(
    name="sys.updates",
    description=(
        "How many Windows updates are pending, their names, and their total download size in "
        "megabytes. Use this for questions like do I have any updates, is Windows up to date, "
        "or what updates are waiting. This only reports; it never installs anything. Requires "
        "the elevated helper; if it is not running the result says so. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def updates_status(params: UpdatesInput) -> UpdatesOutput:
    """Query pending Windows updates via the helper.

    Args:
        params: No arguments.

    Returns:
        Update count, names, and size, or ``available=False`` with a reason.
    """
    try:
        payload = _helper().get_pending_updates()
    except HelperUnavailableError as exc:
        _log.info("update query unavailable: %s", exc.message)
        return UpdatesOutput(available=False, reason=exc.speakable)

    if not payload.get("available", False):
        return UpdatesOutput(
            available=False,
            reason=str(payload.get("reason", "the update query did not return a result")),
        )

    names = [str(name) for name in payload.get("names", [])][:5]
    return UpdatesOutput(
        available=True,
        count=int(payload.get("count", 0)),
        names=names,
        total_size_mb=payload.get("total_size_mb"),
        source=payload.get("source"),
    )
