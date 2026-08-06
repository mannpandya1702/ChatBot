"""Confirmation gate for mutating tools.

CLAUDE.md §6, read twice as instructed. The rules this file exists to enforce:

* Read-only is the default. A tool may change system state only if it is on the
  configured mutating allowlist.
* Every mutating call produces a :class:`ConfirmationRequired` result. The
  orchestrator speaks the intended action back and waits for an explicit
  affirmative within the configured window.
* **Timeout equals denial.** So does silence, so does an ambiguous answer. There
  is no "obviously safe" bypass and no default-yes anywhere in this module.
* Every decision is appended to an audit log.

The gate is deliberately paranoid about the classification step. "no, don't do
it" contains the affirmative phrase "do it", so negatives are matched first and
matching is done on whole words, never on substrings.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from jarvis.config import JarvisConfig
from jarvis.tools.registry import ToolRegistry, ToolResult
from jarvis.tools.registry import registry as default_registry

__all__ = [
    "ConfirmationGate",
    "ConfirmationOutcome",
    "ConfirmationRequest",
    "ConfirmationRequired",
    "GateDecision",
    "GateResult",
    "classify_response",
    "describe_action",
]

_log = logging.getLogger(__name__)


class GateDecision(StrEnum):
    """What the gate decided about a call."""

    #: Read-only tool, run it.
    ALLOWED = "allowed"
    #: Mutating and allowlisted. Speak the action back and wait for a yes.
    CONFIRMATION_REQUIRED = "confirmation_required"
    #: Refused outright. Never becomes allowed, no matter what the user says.
    DENIED = "denied"


class ConfirmationOutcome(StrEnum):
    """How the user's reply was interpreted."""

    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    #: Anything we are not sure about. Treated exactly like a negative.
    UNCLEAR = "unclear"
    #: The window elapsed. Treated exactly like a negative (§6).
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class ConfirmationRequired:
    """Handed back to the orchestrator when a call needs a spoken yes."""

    token: str
    tool: str
    arguments: dict[str, Any]
    #: The sentence to speak back to the user.
    prompt: str
    #: Wall-clock deadline. Past this the answer is a denial.
    expires_at: float

    @property
    def seconds_remaining(self) -> float:
        """Seconds left before this request expires. Never negative."""
        return max(0.0, self.expires_at - time.time())

    @property
    def expired(self) -> bool:
        """True once the confirmation window has closed."""
        return time.time() >= self.expires_at


@dataclass(frozen=True, slots=True)
class GateResult:
    """The gate's verdict on one call."""

    decision: GateDecision
    tool: str
    reason: str
    confirmation: ConfirmationRequired | None = None

    @property
    def allowed(self) -> bool:
        """True only when the call may execute right now."""
        return self.decision is GateDecision.ALLOWED


@dataclass(slots=True)
class ConfirmationRequest:
    """Internal pending-request record."""

    token: str
    tool: str
    arguments: dict[str, Any]
    prompt: str
    created_at: float
    expires_at: float
    resolved: bool = False
    outcome: ConfirmationOutcome | None = None
    transcript: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _phrase_pattern(phrases: list[str]) -> re.Pattern[str]:
    """Compile phrases into a whole-word alternation.

    Word boundaries matter: without them "no" matches inside "now" and "know",
    and the gate would read an enthusiastic "now do it" as a refusal.
    """
    ordered = sorted((p.strip().lower() for p in phrases if p.strip()), key=len, reverse=True)
    if not ordered:
        return re.compile(r"(?!x)x")  # matches nothing
    alternation = "|".join(re.escape(p) for p in ordered)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


def classify_response(
    utterance: str,
    config: JarvisConfig,
) -> ConfirmationOutcome:
    """Interpret the user's spoken reply to a confirmation prompt.

    Negatives are checked first on purpose. "no, do not do it" contains the
    affirmative phrase "do it", and the safe reading of a reply containing both
    is refusal.

    Args:
        utterance: What speech recognition produced.
        config: Supplies the affirmative and negative phrase lists.

    Returns:
        The outcome. Anything not clearly affirmative is not a yes.
    """
    text = (utterance or "").strip().lower()
    if not text:
        return ConfirmationOutcome.UNCLEAR

    # Normalise punctuation and contractions so "don't" matches "do not".
    text = text.replace("'", "").replace("\u2019", "")  # ASCII and typographic
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\b(cant|cannot)\b", "can not", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    negatives = list(config.gate.negatives)
    # Explicit refusal forms that are not single words.
    negatives.extend(["do not", "dont", "not now", "forget it", "hold off", "wait"])

    if _phrase_pattern(negatives).search(text):
        return ConfirmationOutcome.NEGATIVE
    if _phrase_pattern(list(config.gate.affirmatives)).search(text):
        return ConfirmationOutcome.AFFIRMATIVE
    return ConfirmationOutcome.UNCLEAR


def describe_action(tool: str, arguments: dict[str, Any], config: JarvisConfig) -> str:
    """Build the sentence spoken back before a mutating call runs.

    Kept plain and specific. The user has to be able to tell from one spoken
    sentence exactly what is about to happen.
    """
    address = config.persona.user_address_form.strip()
    suffix = f", {address}" if address else ""

    detail = ", ".join(f"{key} {value}" for key, value in sorted(arguments.items()) if value != "")
    match tool:
        case "apps.launch":
            what = f"launch {arguments.get('name', 'that application')}"
        case "apps.focus":
            what = f"bring {arguments.get('name', 'that window')} to the front"
        case "media.control":
            what = f"send the {arguments.get('action', 'media')} command"
        case "reminders.create":
            what = f"set a reminder for {arguments.get('when', 'that time')}"
        case "reminders.delete":
            what = f"delete reminder {arguments.get('reminder_id', 'that one')}"
        case "shell.run":
            what = f"run the command {arguments.get('command', '')}"
        case _:
            what = f"run {tool}" + (f" with {detail}" if detail else "")
    return f"You want me to {what}{suffix}. Shall I go ahead?"


class ConfirmationGate:
    """Enforces the §6 confirmation contract in front of the tool registry."""

    def __init__(
        self,
        config: JarvisConfig,
        *,
        registry: ToolRegistry | None = None,
        audit_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._registry = default_registry if registry is None else registry
        self._clock = clock
        self._lock = threading.RLock()
        self._pending: dict[str, ConfirmationRequest] = {}
        # Single-use execution grants, issued only by an affirmative resolve().
        # dispatch() consumes one to run a mutating tool; nothing else creates them.
        self._grants: dict[str, dict[str, Any]] = {}
        self._audit_path = (
            audit_path
            if audit_path is not None
            else config.resolve_path(config.gate.audit_log)
        )

    # -- decision ----------------------------------------------------------

    def check(self, tool: str, arguments: dict[str, Any] | None = None) -> GateResult:
        """Decide whether ``tool`` may run now, needs a yes, or is refused."""
        arguments = arguments or {}
        spec = self._registry.get(tool)

        if spec is None:
            return self._deny(tool, arguments, "unknown tool")

        if spec.read_only:
            # Read-only tools never touch the allowlist and never prompt.
            return GateResult(GateDecision.ALLOWED, tool, "read-only tool")

        # Mutating from here down.
        if tool not in self._config.gate.mutating_allowlist:
            return self._deny(
                tool,
                arguments,
                "mutating tool is not on the configured allowlist",
            )

        if tool == "shell.run" and not (
            self._config.shell.enabled and self._config.tools.enable_shell
        ):
            return self._deny(tool, arguments, "shell tool is disabled in configuration")

        request = self._open(tool, arguments)
        return GateResult(
            GateDecision.CONFIRMATION_REQUIRED,
            tool,
            "mutating tool requires spoken confirmation",
            confirmation=ConfirmationRequired(
                token=request.token,
                tool=request.tool,
                arguments=dict(request.arguments),
                prompt=request.prompt,
                expires_at=request.expires_at,
            ),
        )

    def _deny(self, tool: str, arguments: dict[str, Any], reason: str) -> GateResult:
        """Record and return a refusal."""
        _log.warning("gate refused call", extra={"context": {"tool": tool, "reason": reason}})
        self._audit(
            event="denied",
            tool=tool,
            arguments=arguments,
            reason=reason,
            confirmed=False,
        )
        return GateResult(GateDecision.DENIED, tool, reason)

    def _open(self, tool: str, arguments: dict[str, Any]) -> ConfirmationRequest:
        """Create a pending confirmation and start its clock."""
        now = self._clock()
        request = ConfirmationRequest(
            token=uuid.uuid4().hex[:16],
            tool=tool,
            arguments=dict(arguments),
            prompt=describe_action(tool, arguments, self._config),
            created_at=now,
            expires_at=now + self._config.gate.confirmation_timeout_s,
        )
        with self._lock:
            self._expire_locked(now)
            self._pending[request.token] = request
        self._audit(
            event="confirmation_requested",
            tool=tool,
            arguments=arguments,
            reason="mutating tool",
            confirmed=False,
            token=request.token,
        )
        return request

    # -- resolution --------------------------------------------------------

    def resolve(self, token: str, utterance: str) -> ConfirmationOutcome:
        """Apply the user's spoken reply to a pending confirmation.

        An unknown, already-resolved, or expired token yields ``TIMEOUT``, which
        is a denial. There is no path here that turns a stale token into a yes.
        """
        now = self._clock()
        with self._lock:
            request = self._pending.get(token)
            if request is None or request.resolved:
                self._audit(
                    event="confirmation_unknown_token",
                    tool=request.tool if request else "unknown",
                    arguments={},
                    reason="token not pending",
                    confirmed=False,
                    token=token,
                )
                return ConfirmationOutcome.TIMEOUT
            if now >= request.expires_at:
                request.resolved = True
                request.outcome = ConfirmationOutcome.TIMEOUT
                self._pending.pop(token, None)
                self._audit(
                    event="confirmation_timeout",
                    tool=request.tool,
                    arguments=request.arguments,
                    reason="window elapsed before a reply",
                    confirmed=False,
                    token=token,
                )
                return ConfirmationOutcome.TIMEOUT

            outcome = classify_response(utterance, self._config)
            request.resolved = True
            request.outcome = outcome
            request.transcript = utterance
            self._pending.pop(token, None)
            if outcome is ConfirmationOutcome.AFFIRMATIVE:
                # The grant is short lived and single use. It exists so the
                # orchestrator can carry the yes back to dispatch() without the
                # gate having to trust a boolean it was handed.
                self._grants[token] = {
                    "tool": request.tool,
                    "arguments": dict(request.arguments),
                    "expires_at": now + _GRANT_TTL_S,
                }

        self._audit(
            event="confirmation_resolved",
            tool=request.tool,
            arguments=request.arguments,
            reason=f"user said: {utterance!r}",
            confirmed=outcome is ConfirmationOutcome.AFFIRMATIVE,
            token=token,
            outcome=str(outcome),
        )
        return outcome

    def expire(self, token: str) -> ConfirmationOutcome:
        """Explicitly time a request out. Used when the listen window closes."""
        with self._lock:
            request = self._pending.pop(token, None)
            if request is not None:
                request.resolved = True
                request.outcome = ConfirmationOutcome.TIMEOUT
        if request is not None:
            self._audit(
                event="confirmation_timeout",
                tool=request.tool,
                arguments=request.arguments,
                reason="listen window closed",
                confirmed=False,
                token=token,
            )
        return ConfirmationOutcome.TIMEOUT

    def _expire_locked(self, now: float) -> None:
        """Drop pending requests whose window has closed. Caller holds the lock."""
        stale = [tok for tok, req in self._pending.items() if now >= req.expires_at]
        for token in stale:
            self._pending.pop(token, None)

    @property
    def pending_count(self) -> int:
        """How many confirmations are currently awaiting a reply."""
        with self._lock:
            self._expire_locked(self._clock())
            return len(self._pending)

    def is_pending(self, token: str) -> bool:
        """True when ``token`` is still awaiting a reply and has not expired."""
        with self._lock:
            request = self._pending.get(token)
            if request is None or request.resolved:
                return False
            return self._clock() < request.expires_at

    # -- execution ---------------------------------------------------------

    def dispatch(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmation_token: str | None = None,
    ) -> ToolResult | ConfirmationRequired:
        """Run a tool through the gate.

        Returns a :class:`ConfirmationRequired` when the caller must go and ask
        the user first, otherwise the tool's :class:`ToolResult`.

        Passing ``confirmation_token`` asserts that this exact token was already
        resolved affirmatively. The gate verifies that claim against its own
        record rather than trusting it, so a stale or forged token cannot
        smuggle a mutating call through.
        """
        arguments = arguments or {}

        if confirmation_token is not None:
            granted = self._consume_grant(confirmation_token, tool, arguments)
            if not granted:
                return ToolResult(
                    tool=tool,
                    ok=False,
                    error="confirmation token is not valid for this call",
                    speakable="I could not confirm that, so I have not done it.",
                )
            self._audit(
                event="executed",
                tool=tool,
                arguments=arguments,
                reason="confirmed by user",
                confirmed=True,
                token=confirmation_token,
            )
            return self._registry.dispatch(tool, arguments, confirmed=True)

        result = self.check(tool, arguments)
        if result.decision is GateDecision.CONFIRMATION_REQUIRED:
            assert result.confirmation is not None
            return result.confirmation
        if result.decision is GateDecision.DENIED:
            return ToolResult(
                tool=tool,
                ok=False,
                error=f"refused: {result.reason}",
                speakable="I am not permitted to do that.",
            )
        return self._registry.dispatch(tool, arguments, confirmed=False)

    def _consume_grant(self, token: str, tool: str, arguments: dict[str, Any]) -> bool:
        """Validate a grant token against the recorded confirmation."""
        with self._lock:
            grant = self._grants.pop(token, None)
        if grant is None:
            _log.warning(
                "rejected unknown confirmation token",
                extra={"context": {"tool": tool, "token": token}},
            )
            return False
        if grant["tool"] != tool or grant["arguments"] != arguments:
            _log.warning(
                "confirmation token does not match the call it was issued for",
                extra={"context": {"tool": tool, "granted_for": grant["tool"]}},
            )
            return False
        if self._clock() > float(grant["expires_at"]):
            _log.warning("confirmation grant expired", extra={"context": {"tool": tool}})
            return False
        return True

    # -- audit -------------------------------------------------------------

    def _audit(
        self,
        *,
        event: str,
        tool: str,
        arguments: dict[str, Any],
        reason: str,
        confirmed: bool,
        token: str | None = None,
        outcome: str | None = None,
    ) -> None:
        """Append one line to the gate audit log. Never raises."""
        record: dict[str, Any] = {
            "ts": self._clock(),
            "event": event,
            "tool": tool,
            "arguments": _redact(arguments),
            "reason": reason,
            "confirmed": confirmed,
        }
        if token is not None:
            record["token"] = token
        if outcome is not None:
            record["outcome"] = outcome
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            _log.exception("could not write the gate audit log")


#: How long an affirmative answer stays usable. Long enough for the
#: orchestrator to hand it to dispatch(), far too short to be replayed later.
_GRANT_TTL_S = 30.0

_SENSITIVE = ("token", "password", "secret", "key", "credential")


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Blank out anything that looks like a credential before it hits the log."""
    return {
        key: ("<redacted>" if any(marker in key.lower() for marker in _SENSITIVE) else value)
        for key, value in arguments.items()
    }
