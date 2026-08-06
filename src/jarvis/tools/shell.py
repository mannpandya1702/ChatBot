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

import contextlib
import json
import logging
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


#: Characters that mean the head names a location rather than a command.
_PATH_MARKERS = ("/", "\\", ":")


def _normalise(token: str) -> str:
    """Strip the quoting shlex leaves behind in non-posix mode.

    Every check and the final argv are built from this one function, so a token
    can never look benign to a check and dangerous to the process.
    """
    text = token.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


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
    for operator in shell_cfg.blocked_operators:
        if operator and operator in text:
            return ShellVerdict(False, f"the command contains the forbidden operator {operator!r}")
    if "$(" in text or "${" in text or "%(" in text:
        return ShellVerdict(False, "the command contains command substitution")

    # Tokenise. Failure here means unbalanced quotes, which is refused rather
    # than guessed at.
    try:
        raw_tokens = shlex.split(text, posix=False)
    except ValueError as exc:
        return ShellVerdict(False, f"the command could not be parsed: {exc}")
    if not raw_tokens:
        return ShellVerdict(False, "the command is empty")

    # Normalise ONCE. Every check below and the argv built at the end use these
    # same values, so a token cannot pass a check in one form and reach the
    # process in another. Quoting "-Command" used to do exactly that.
    tokens = [_normalise(token) for token in raw_tokens]
    if not tokens[0]:
        return ShellVerdict(False, "the command name could not be determined")

    # Rule 1a: the allowlist names commands, not locations. A head carrying a
    # path separator or a drive letter is refused outright, because otherwise
    # /anywhere/echo passes an allowlist that only ever meant the real echo.
    head_token = tokens[0]
    if any(marker in head_token for marker in _PATH_MARKERS):
        return ShellVerdict(
            False,
            "the command must be a bare command name, not a path",
        )

    head = Path(head_token).stem.lower()
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
        if Path(token).stem.lower() in blocked:
            return ShellVerdict(False, f"the command references the blocked command {token!r}")

    # Rule 4: blocked PowerShell arguments, checked on the normalised token so
    # that quoting cannot smuggle one through.
    blocked_args = {arg.lstrip("/-").lower() for arg in shell_cfg.blocked_powershell_args}
    blocked_args.update({"e", "ec", "enc", "encodedcommand", "command"})
    for token in tokens[1:]:
        if token.lstrip("/-").lower() in blocked_args:
            return ShellVerdict(False, f"the argument {token!r} is not permitted")

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
        # Every failure mode here must become a verdict, not an exception: this
        # function is documented as pure and total, and run_command relies on
        # that to guarantee an audit record for every attempt.
        if any(char in working_dir for char in _ALWAYS_FORBIDDEN_CHARS):
            return ShellVerdict(False, "the working directory contains a forbidden character")
        try:
            candidate = Path(working_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return ShellVerdict(False, "the requested working directory does not exist")
        # resolve() collapses .. so a traversal out of an allowlisted root
        # cannot survive this comparison.
        if not any(candidate == root or root in candidate.parents for root in roots):
            return ShellVerdict(False, "the requested working directory is not allowlisted")
        chosen = candidate

    if not chosen.is_dir():
        return ShellVerdict(False, "the working directory does not exist")

    # Rule 5: resolve the executable ourselves and build a fixed argv. No shell
    # is involved at any point. The resolved path is made absolute so the file
    # that was checked is the file that runs, regardless of the child's cwd.
    executable = shutil.which(head_token)
    if executable is None:
        return ShellVerdict(False, f"{head_token} was not found on the path")
    try:
        resolved = str(Path(executable).resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return ShellVerdict(False, f"{head_token} could not be resolved to a real file")

    argv = [resolved, *tokens[1:]]
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


class _CappedReader:
    """Reads a pipe on its own thread, keeping at most ``limit`` bytes.

    Rule 9 caps output at 4 KB. Buffering the whole stream and slicing
    afterwards would honour the letter of that and none of its point, since a
    chatty command could exhaust memory before the slice happened.

    Draining rather than closing early matters: a child that gets EPIPE on its
    first write dies with a broken pipe, which would misreport a working
    command as a failed one.
    """

    def __init__(self, pipe: Any, limit: int) -> None:
        self._pipe = pipe
        self._limit = limit
        self._chunks: list[bytes] = []
        self.total = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def overflowed(self) -> bool:
        """Whether the stream produced more than the cap allowed."""
        return self.total > self._limit

    @property
    def data(self) -> bytes:
        """The retained bytes, never more than the cap."""
        return b"".join(self._chunks)

    def add(self, extra: bytes) -> None:
        """Append a locally generated note, such as the timeout message."""
        self._chunks.append(extra)

    def _run(self) -> None:
        try:
            while True:
                block = self._pipe.read(4096)
                if not block:
                    return
                if self.total < self._limit:
                    self._chunks.append(block[: self._limit - self.total])
                self.total += len(block)
        except (OSError, ValueError):
            return


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
def run_command(params: ShellInput, *, confirmed: bool = True) -> ShellOutput:
    """Validate and run a shell command.

    The registry already refuses to call this without ``confirmed=True``, and the
    gate refuses to grant confirmation unless the tool is enabled. This function
    re-validates anyway, because a safety control that depends on its caller is
    not a control.

    Args:
        params: The command and optional working directory.
        confirmed: Whether the user actually confirmed this invocation. Rule 10
            requires the audit log to record that, so it is threaded through
            rather than assumed. The registry refuses an unconfirmed call before
            this function is reached; the flag is for the record, not the check.

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
            confirmed=confirmed,
            duration_s=time.perf_counter() - started,
        )
        return ShellOutput(ran=False, refused_reason=verdict.reason)

    if verdict.argv is None or verdict.cwd is None:  # pragma: no cover - invariant
        raise SafetyViolationError("an allowed verdict must carry argv and cwd")

    limit = config.shell.max_output_bytes
    timed_out = False
    exit_code: int | None = None
    out_reader: _CappedReader | None = None
    err_reader: _CappedReader | None = None
    start_error = b""

    try:
        # Rule 5: fixed argv, shell=False. There is no shell to inject into.
        # Popen rather than run() so the readers can stop at the cap; see
        # _read_capped. Binary mode so the cap is measured in the bytes the
        # limit is stated in.
        process = subprocess.Popen(
            verdict.argv,
            cwd=str(verdict.cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        start_error = f"the command could not be started: {exc}".encode()
    else:
        # Two readers, because reading one pipe to exhaustion while the child
        # fills the other would deadlock.
        out_reader = _CappedReader(process.stdout, limit)
        err_reader = _CappedReader(process.stderr, limit)
        out_reader.thread.start()
        err_reader.thread.start()

        try:
            exit_code = process.wait(timeout=config.shell.timeout_s)  # Rule 8
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                exit_code = None
            err_reader.add(
                f"the command exceeded the {config.shell.timeout_s} second limit".encode()
            )
        finally:
            out_reader.thread.join(timeout=5)
            err_reader.thread.join(timeout=5)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    out_bytes = out_reader.data if out_reader is not None else b""
    err_bytes = err_reader.data if err_reader is not None else start_error
    overflowed = (out_reader is not None and out_reader.overflowed) or (
        err_reader is not None and err_reader.overflowed
    )

    stdout, cut_out = _truncate(out_bytes.decode("utf-8", errors="replace"), limit)
    stderr, cut_err = _truncate(err_bytes.decode("utf-8", errors="replace"), limit)
    duration = time.perf_counter() - started

    _audit(
        config,
        command=params.command,
        argv=verdict.argv,
        exit_code=exit_code,
        refused=None,
        confirmed=confirmed,
        duration_s=duration,
    )

    return ShellOutput(
        ran=True,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        truncated=overflowed or cut_out or cut_err,
        duration_s=round(duration, 3),
        timed_out=timed_out,
    )
