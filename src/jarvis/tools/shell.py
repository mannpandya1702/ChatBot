"""Gated shell execution.

CLAUDE.md §6 lists ten mandatory hardening rules for this tool. Every one is
implemented here, and each is labelled with the rule it enforces so a reviewer
can check them off:

1.  Allowlist first. A command whose head is not on the allowlist is refused.
    The denylist is a secondary layer, not the primary control.
2.  Blocked commands: rm, del, rmdir, format, diskpart, shutdown, restart, reg,
    regedit, net, netsh, takeown, icacls, sc, bcdedit, vssadmin, cipher.
3.  Blocked shell operators: & && | ; backtick $( and redirection.
4.  Blocked PowerShell args: -enc, -EncodedCommand, -Command, -e, -nop.
5.  Fixed argv. **The command string from the model is never passed to a shell.**
    It is tokenised with shlex and handed to subprocess as a list, with
    shell=False, so there is no shell to inject into in the first place.
6.  Max command length 512 characters.
7.  Working directory restricted to a configured allowlist. An empty allowlist
    means no working directory may be chosen at all.
8.  Hard 30 second timeout.
9.  Output truncated to 4 KB.
10. Every invocation logged to logs/shell_audit.jsonl with timestamp, command,
    exit code, and whether the user confirmed.

Disabled by default. Both ``tools.enable_shell`` and ``shell.enabled`` must be
true, and the call still has to pass the §6 confirmation gate before the
registry will run it.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from jarvis.config import JarvisConfig, get_config
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import SafetyViolationError

__all__ = [
    "ShellInput",
    "ShellOutput",
    "ShellVerdict",
    "run_command",
    "validate_command",
]

_log = logging.getLogger(__name__)

#: Characters that can never appear in a command, whatever the allowlist says.
#: Newlines matter as much as pipes: a newline in an argument becomes a second
#: statement the moment anything re-parses the string.
_ALWAYS_FORBIDDEN_CHARS = frozenset("&|;`\n\r\x00<>\u2028\u2029\x0b\x0c")


@dataclass(frozen=True, slots=True)
class ShellVerdict:
    """The result of validating a command before it runs."""

    allowed: bool
    reason: str
    #: Fully resolved argv, only when allowed.
    argv: list[str] | None = None
    #: Resolved working directory, only when allowed.
    cwd: Path | None = None


class ShellInput(ToolInput):
    """Arguments for the shell tool."""

    command: str = Field(
        min_length=1,
        max_length=512,
        description=(
            "The command to run. Must start with an allowlisted command name. Shell operators, "
            "pipes, redirection, and command substitution are refused. This runs only after "
            "the user confirms out loud."
        ),
    )
    working_dir: str | None = Field(
        default=None,
        description=(
            "Directory to run in. Must be inside the configured allowlist. Leave unset to use "
            "the first allowlisted directory."
        ),
    )


class ShellOutput(ToolOutput):
    """The outcome of a shell invocation."""

    ran: bool = Field(description="Whether the command actually executed.")
    refused_reason: str | None = Field(
        default=None, description="Why the command was refused, when it was."
    )
    exit_code: int | None = Field(default=None, description="Process exit code.")
    stdout: str = Field(default="", description="Standard output, truncated to 4 kilobytes.")
    stderr: str = Field(default="", description="Standard error, truncated to 4 kilobytes.")
    truncated: bool = Field(default=False, description="Whether output was cut short.")
    duration_s: float = Field(default=0.0, description="How long the command took, in seconds.")
    timed_out: bool = Field(default=False, description="Whether the hard timeout fired.")


def _head(command: str) -> str:
    """The command name, without path or extension, lowercased."""
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return ""
    if not tokens:
        return ""
    return Path(tokens[0].strip('"').strip("'")).stem.lower()


def validate_command(
    command: str,
    working_dir: str | None,
    config: JarvisConfig,
) -> ShellVerdict:
    """Apply every §6 rule to a candidate command.

    Pure and side-effect free, so the hardening can be tested exhaustively
    without ever running a process.

    Args:
        command: The raw command string from the model.
        working_dir: Requested working directory, or None.
        config: Supplies the allowlist, denylist, and limits.

    Returns:
        A verdict. ``allowed`` is only ever True when every rule passed.
    """
    shell_cfg = config.shell

    # Rule: disabled by default, and both switches must agree.
    if not (shell_cfg.enabled and config.tools.enable_shell):
        return ShellVerdict(False, "the shell tool is disabled in configuration")

    text = command.strip()
    if not text:
        return ShellVerdict(False, "the command is empty")

    # Rule 6: max command length.
    if len(text) > shell_cfg.max_command_length:
        return ShellVerdict(
            False,
            f"the command is longer than the {shell_cfg.max_command_length} character limit",
        )

    # Rule 3: blocked operators. Checked against the raw text before tokenising,
    # because tokenising can hide an operator inside a quoted argument.
    for char in text:
        if char in _ALWAYS_FORBIDDEN_CHARS:
            return ShellVerdict(False, f"the command contains the forbidden character {char!r}")
    lowered = text.lower()
    for operator in shell_cfg.blocked_operators:
        if operator and operator in text:
            return ShellVerdict(False, f"the command contains the forbidden operator {operator!r}")
    if "$(" in text or "${" in text or "%(" in text:
        return ShellVerdict(False, "the command contains command substitution")

    # Tokenise. Failure here means unbalanced quotes, which is refused rather
    # than guessed at.
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError as exc:
        return ShellVerdict(False, f"the command could not be parsed: {exc}")
    if not tokens:
        return ShellVerdict(False, "the command is empty")

    head = _head(text)
    if not head:
        return ShellVerdict(False, "the command name could not be determined")

    # Rule 2: denylist, as a secondary layer.
    blocked = {name.lower() for name in shell_cfg.blocked_commands}
    if head in blocked:
        return ShellVerdict(False, f"{head} is on the blocked command list")
    # A blocked name anywhere in the argument list is also refused, which stops
    # "powershell Get-Process; shutdown" style constructions even if the
    # operator check were somehow bypassed.
    for token in tokens[1:]:
        if Path(token.strip("\"'")).stem.lower() in blocked:
            return ShellVerdict(False, f"the command references the blocked command {token!r}")

    # Rule 4: blocked PowerShell arguments.
    for token in tokens[1:]:
        if token.lower().lstrip("/-") in {
            arg.lstrip("-") for arg in shell_cfg.blocked_powershell_args
        }:
            return ShellVerdict(False, f"the argument {token!r} is not permitted")
    if "-e" in lowered.split() or "-enc" in lowered.split():
        return ShellVerdict(False, "encoded command arguments are not permitted")

    # Rule 1: the allowlist is the primary control, applied last so its refusal
    # is the default outcome for anything unrecognised.
    allowed = {name.lower() for name in shell_cfg.allowlist}
    if head not in allowed:
        return ShellVerdict(
            False,
            f"{head} is not on the allowlist of permitted commands",
        )

    # Rule 7: working directory allowlist. An empty allowlist means no directory
    # may be selected, so the tool cannot be pointed anywhere at all.
    roots: list[Path] = []
    for entry in shell_cfg.working_dir_allowlist:
        candidate_root = config.resolve_path(Path(entry))
        try:
            roots.append(candidate_root.resolve(strict=True))
        except (OSError, RuntimeError):
            # An allowlisted directory that does not exist cannot authorise
            # anything, so it is dropped rather than compared against.
            _log.warning(
                "an allowlisted working directory does not exist",
                extra={"context": {"path": str(candidate_root)}},
            )
    if not roots:
        return ShellVerdict(
            False,
            "no working directory is allowlisted, so the shell tool cannot run anywhere",
        )

    if working_dir is None:
        chosen = roots[0]
    else:
        candidate = Path(working_dir).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return ShellVerdict(False, "the requested working directory does not exist")
        # resolve() collapses .. so a traversal out of an allowlisted root
        # cannot survive this comparison.
        if not any(candidate == root or root in candidate.parents for root in roots):
            return ShellVerdict(False, "the requested working directory is not allowlisted")
        chosen = candidate

    if not chosen.is_dir():
        return ShellVerdict(False, "the working directory does not exist")

    # Rule 5: resolve the executable ourselves and build a fixed argv. No shell
    # is involved at any point.
    executable = shutil.which(tokens[0].strip("\"'"))
    if executable is None:
        return ShellVerdict(False, f"{tokens[0]} was not found on the path")

    argv = [executable, *[token.strip('"') for token in tokens[1:]]]
    return ShellVerdict(True, "allowed", argv=argv, cwd=chosen)


def _audit(
    config: JarvisConfig,
    *,
    command: str,
    argv: list[str] | None,
    exit_code: int | None,
    refused: str | None,
    confirmed: bool,
    duration_s: float,
) -> None:
    """Rule 10: append the invocation to the audit log. Never raises."""
    record = {
        "ts": time.time(),
        "command": command,
        "argv": argv,
        "exit_code": exit_code,
        "refused_reason": refused,
        "confirmed": confirmed,
        "duration_s": round(duration_s, 3),
    }
    path = config.resolve_path(config.shell.audit_log)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        _log.exception("could not write the shell audit log")


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Rule 9: cut output to the configured limit."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


@tool(
    name="shell.run",
    description=(
        "Run a single allowlisted read-only command and return its output. "
        "Only commands on the configured allowlist are permitted, such as Get-Date or "
        "systeminfo. Pipes, redirection, chaining, and command substitution are refused. "
        "This tool changes system state in principle, so it always requires the user to "
        "confirm out loud first, and it is disabled unless explicitly turned on in the "
        "configuration. Output is truncated to 4 kilobytes."
    ),
    category=ToolCategory.SHELL,
    read_only=False,
    is_enabled=lambda config: config.tools.enable_shell and config.shell.enabled,
)
def run_command(params: ShellInput) -> ShellOutput:
    """Validate and run a shell command.

    The registry already refuses to call this without ``confirmed=True``, and the
    gate refuses to grant confirmation unless the tool is enabled. This function
    re-validates anyway, because a safety control that depends on its caller is
    not a control.

    Args:
        params: The command and optional working directory.

    Returns:
        The outcome, including a refusal reason when the command was rejected.

    Raises:
        SafetyViolationError: Never raised for a refusal, which is returned as
            data. Reserved for a configuration state that should be impossible.
    """
    config = get_config()
    started = time.perf_counter()

    verdict = validate_command(params.command, params.working_dir, config)
    if not verdict.allowed:
        _log.warning(
            "refused shell command",
            extra={"context": {"reason": verdict.reason, "command": params.command[:120]}},
        )
        _audit(
            config,
            command=params.command,
            argv=None,
            exit_code=None,
            refused=verdict.reason,
            confirmed=True,
            duration_s=time.perf_counter() - started,
        )
        return ShellOutput(ran=False, refused_reason=verdict.reason)

    if verdict.argv is None or verdict.cwd is None:  # pragma: no cover - invariant
        raise SafetyViolationError("an allowed verdict must carry argv and cwd")

    timed_out = False
    exit_code: int | None = None
    stdout = ""
    stderr = ""
    try:
        # Rule 5: fixed argv, shell=False. There is no shell to inject into.
        completed = subprocess.run(
            verdict.argv,
            cwd=str(verdict.cwd),
            capture_output=True,
            text=True,
            timeout=config.shell.timeout_s,  # Rule 8
            check=False,
            shell=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = f"the command exceeded the {config.shell.timeout_s} second limit"
    except (OSError, subprocess.SubprocessError) as exc:
        stderr = f"the command could not be started: {exc}"

    limit = config.shell.max_output_bytes
    stdout, cut_out = _truncate(stdout, limit)
    stderr, cut_err = _truncate(stderr, limit)
    duration = time.perf_counter() - started

    _audit(
        config,
        command=params.command,
        argv=verdict.argv,
        exit_code=exit_code,
        refused=None,
        confirmed=True,
        duration_s=duration,
    )

    return ShellOutput(
        ran=True,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        truncated=cut_out or cut_err,
        duration_s=round(duration, 3),
        timed_out=timed_out,
    )
