"""Rolling conversation memory with a running summary and SQLite persistence.

T-1.7. Two things this has to get right:

* Compaction must never lose the thread. The most recent ``keep_recent``
  messages survive verbatim, and the new summary always incorporates the old
  one, so the conversation does not develop amnesia at the seams.
* A summariser failure must not destroy data. When the LLM cannot be reached,
  the window is allowed to grow past its cap rather than dropping messages that
  were never summarised.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.config import JarvisConfig

__all__ = [
    "ConversationMemory",
    "MemoryMessage",
    "estimate_tokens",
]

_log = logging.getLogger(__name__)

#: Tool results can be enormous. §5 says outputs stay phrasable, so what gets
#: stored is capped well below what a model would choke on.
_MAX_TOOL_CHARS = 2_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    tokens      INTEGER NOT NULL,
    tool_name   TEXT,
    tool_calls  TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
"""


#: Columns added after the first release, as ``name -> ALTER TABLE type``.
#: ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists,
#: so a database written before a column was added keeps the old shape and the
#: insert fails on every turn. Anyone who has run JARVIS once has such a file.
_MIGRATIONS: dict[str, str] = {"tool_calls": "TEXT"}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any column the running code expects and the file does not have."""
    try:
        existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(messages)")}
    except sqlite3.Error:
        _log.exception("could not inspect the conversation table")
        return
    if not existing:
        return
    for column, kind in _MIGRATIONS.items():
        if column in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {column} {kind}")
            conn.commit()
            _log.info("migrated the conversation store", extra={"context": {"column": column}})
        except sqlite3.Error:
            _log.exception("could not add a column", extra={"context": {"column": column}})


def _load_tool_calls(row: sqlite3.Row) -> list[dict[str, Any]]:
    """Decode a stored tool_calls column, tolerating old rows and bad JSON."""
    try:
        raw = row["tool_calls"]
    except (IndexError, KeyError):
        return []
    if not raw:
        return []
    try:
        decoded = json.loads(str(raw))
    except (json.JSONDecodeError, ValueError):
        _log.warning("discarding an unreadable stored tool call")
        return []
    return decoded if isinstance(decoded, list) else []


def estimate_tokens(text: str) -> int:
    """Approximate token count for a string.

    Roughly four characters per token, with a floor of one for any non-empty
    string. This is deliberately an estimate: pulling in a real tokenizer would
    add a heavy dependency to answer a question whose only consumer is a
    "should I compact yet" threshold, where being off by ten percent changes
    nothing.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class MemoryMessage:
    """One stored message.

    ``tool_calls`` is set on assistant messages that asked for a tool, and
    ``tool_name`` on the tool messages that answered. Both halves have to be
    kept: a tool result on its own is an anonymous blob of JSON, and the only
    thing that says which call produced it is the assistant message before it.
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)

    def to_chat(self) -> dict[str, Any]:
        """Render in the shape Ollama's chat endpoint expects.

        The key for a tool result's name is ``tool_name``. Ollama's Message
        struct declares exactly Role, Content, Thinking, Images, ToolCalls,
        ToolName and ToolCallID, and Go's JSON decoder discards anything else
        without complaint, so a message sent under ``name`` arrives nameless and
        nothing says so.
        """
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool" and self.tool_name:
            message["tool_name"] = self.tool_name
        if self.role == "assistant" and self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class ConversationMemory:
    """A rolling message window plus a running summary.

    Args:
        config: Supplies the caps, the session id, and the database path.
        summariser: Called with the messages being compacted, returns the new
            summary text. Injected so tests never need an LLM. In production the
            orchestrator passes a function that calls Ollama.
        db_path: Overrides the configured path. Used by tests.
    """

    def __init__(
        self,
        config: JarvisConfig,
        *,
        summariser: Callable[[list[MemoryMessage], str], str] | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._config = config
        self._summariser = summariser
        self._lock = threading.RLock()
        self._messages: list[MemoryMessage] = []
        self._summary = ""
        self._session_id = config.memory.session_id
        self._conn: sqlite3.Connection | None = None
        self._defer_depth = 0
        self._compaction_pending = False

        if config.memory.enabled:
            path = db_path or config.resolve_path(config.memory.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
                _migrate(self._conn)
            self.load()

    # -- properties --------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Which conversation this memory belongs to."""
        return self._session_id

    @property
    def summary(self) -> str:
        """The running summary of everything compacted so far."""
        with self._lock:
            return self._summary

    @property
    def message_count(self) -> int:
        """How many messages are in the live window."""
        with self._lock:
            return len(self._messages)

    @property
    def total_tokens(self) -> int:
        """Estimated tokens in the window plus the summary."""
        with self._lock:
            return sum(m.tokens for m in self._messages) + estimate_tokens(self._summary)

    # -- adding ------------------------------------------------------------

    def add_user(self, text: str) -> MemoryMessage:
        """Record something the user said."""
        return self._add(MemoryMessage(role="user", content=text))

    def add_assistant(
        self, text: str, tool_calls: Sequence[dict[str, Any]] | None = None
    ) -> MemoryMessage:
        """Record something the assistant said, and any tools it asked for.

        Args:
            text: What the assistant said, which may be empty when it went
                straight to a tool without speaking first.
            tool_calls: The calls it requested, in Ollama's wire shape. These
                have to be stored: without the assistant message that requested
                them, the tool results that follow are anonymous JSON with
                nothing tying them to a question.
        """
        return self._add(
            MemoryMessage(role="assistant", content=text, tool_calls=list(tool_calls or []))
        )

    def add_tool(self, name: str, payload: Any) -> MemoryMessage:
        """Record a tool result.

        Long results are truncated, with the truncation marked, so a verbose
        tool cannot push the whole conversation out of the window.
        """
        text = payload if isinstance(payload, str) else _compact_json(payload)
        if len(text) > _MAX_TOOL_CHARS:
            text = text[:_MAX_TOOL_CHARS] + " ... (truncated)"
        return self._add(MemoryMessage(role="tool", content=text, tool_name=name))

    def _add(self, message: MemoryMessage) -> MemoryMessage:
        """Append, persist, and compact if needed."""
        with self._lock:
            self._messages.append(message)
            self._persist(message)
            deferred = self._defer_depth > 0
        if deferred:
            self._compaction_pending = True
        else:
            self.maybe_compact()
        return message

    @contextmanager
    def deferred_compaction(self) -> Iterator[None]:
        """Hold compaction until the end of the block.

        Compaction runs a full, non-streaming LLM generation. Triggered from
        :meth:`_add` it lands wherever a message happens to be recorded, which
        includes between a tool result and the answering round: the user has
        finished speaking, a tool has already run, and the turn stops dead for
        several seconds to summarise old messages nobody asked about.

        Wrapping a turn in this moves it to the end, where the reply is already
        queued and playing, so it overlaps with speech instead of preceding it.
        The work still happens on the same thread and the caps still hold; only
        the moment changes.

        Reentrant, and compacts at most once on the way out however many
        messages were added inside.
        """
        with self._lock:
            self._defer_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._defer_depth -= 1
                run = self._defer_depth == 0 and self._compaction_pending
                if run:
                    self._compaction_pending = False
            if run:
                self.maybe_compact()

    # -- reading -----------------------------------------------------------

    def messages(self, system_prompt: str) -> list[dict[str, Any]]:
        """The exact message list handed to Ollama.

        Order is: the system prompt, then the running summary as a second system
        message when one exists, then the retained window.
        """
        with self._lock:
            out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            if self._summary:
                out.append(
                    {
                        "role": "system",
                        "content": f"Earlier in this conversation:\n{self._summary}",
                    }
                )
            out.extend(message.to_chat() for message in self._messages)
            return out

    def window(self) -> list[MemoryMessage]:
        """A copy of the live message window."""
        with self._lock:
            return list(self._messages)

    # -- compaction --------------------------------------------------------

    def needs_compaction(self) -> bool:
        """Whether the window has outgrown either configured cap."""
        with self._lock:
            return (
                self.total_tokens > self._config.memory.max_tokens
                or len(self._messages) > self._config.memory.max_messages
            )

    def maybe_compact(self) -> bool:
        """Compact when a cap has been passed. Returns whether it did."""
        if not self.needs_compaction():
            return False
        return self.compact()

    def compact(self) -> bool:
        """Fold the oldest messages into the running summary.

        Returns:
            True when the window actually shrank.
        """
        keep = self._config.memory.keep_recent
        with self._lock:
            if len(self._messages) <= keep:
                return False
            older = self._messages[:-keep]
            recent = self._messages[-keep:]
            previous = self._summary

        if self._summariser is None:
            # No summariser wired up. Dropping the messages would lose the
            # conversation silently, so keep a factual note instead.
            merged = _fallback_summary(previous, older)
        else:
            try:
                merged = self._summariser(older, previous)
            except Exception:  # noqa: BLE001 - never lose data over a failed call
                _log.exception(
                    "the summariser failed, leaving the window uncompacted",
                    extra={"context": {"session": self._session_id, "pending": len(older)}},
                )
                return False

        merged = (merged or "").strip()
        if not merged:
            merged = _fallback_summary(previous, older)

        with self._lock:
            self._summary = merged
            self._messages = recent
            self._persist_summary(merged)
            self._prune_persisted(keep)

        _log.info(
            "memory compacted",
            extra={
                "context": {
                    "session": self._session_id,
                    "folded": len(older),
                    "kept": len(recent),
                    "summary_tokens": estimate_tokens(merged),
                }
            },
        )
        return True

    # -- persistence -------------------------------------------------------

    def _persist(self, message: MemoryMessage) -> None:
        """Write one message. Caller holds the lock."""
        if self._conn is None:
            return
        try:
            cursor = self._conn.execute(
                "INSERT INTO messages"
                " (session_id, role, content, timestamp, tokens, tool_name, tool_calls)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self._session_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    message.tokens,
                    message.tool_name,
                    json.dumps(message.tool_calls) if message.tool_calls else None,
                ),
            )
            self._conn.commit()
            message.id = int(cursor.lastrowid or 0)
        except sqlite3.Error:
            _log.exception("could not persist a message")

    def _persist_summary(self, summary: str) -> None:
        """Write the running summary. Caller holds the lock."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO summaries (session_id, summary, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET summary = ?, updated_at = ?",
                (self._session_id, summary, time.time(), summary, time.time()),
            )
            self._conn.commit()
        except sqlite3.Error:
            _log.exception("could not persist the summary")

    def _prune_persisted(self, keep: int) -> None:
        """Drop rows that have been folded into the summary. Caller holds the lock."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
                "  SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ")",
                (self._session_id, self._session_id, keep),
            )
            self._conn.commit()
        except sqlite3.Error:
            _log.exception("could not prune persisted messages")

    def load(self) -> None:
        """Restore this session from SQLite, so context survives a restart."""
        if self._conn is None:
            return
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (self._session_id,),
                ).fetchall()
                summary_row = self._conn.execute(
                    "SELECT summary FROM summaries WHERE session_id = ?",
                    (self._session_id,),
                ).fetchone()
            except sqlite3.Error:
                _log.exception("could not load the conversation")
                return

            self._messages = [
                MemoryMessage(
                    role=str(row["role"]),
                    content=str(row["content"]),
                    timestamp=float(row["timestamp"]),
                    tokens=int(row["tokens"]),
                    tool_name=row["tool_name"],
                    tool_calls=_load_tool_calls(row),
                    id=int(row["id"]),
                )
                for row in rows
            ]
            self._summary = str(summary_row["summary"]) if summary_row else ""

    def clear(self) -> None:
        """Wipe this session, in memory and on disk."""
        with self._lock:
            self._messages = []
            self._summary = ""
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (self._session_id,)
                )
                self._conn.execute(
                    "DELETE FROM summaries WHERE session_id = ?", (self._session_id,)
                )
                self._conn.commit()
            except sqlite3.Error:
                _log.exception("could not clear the conversation")

    def close(self) -> None:
        """Close the database handle."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> ConversationMemory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _compact_json(payload: Any) -> str:
    """Render a tool payload compactly, without ever raising."""
    import json

    try:
        return json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(payload)


def _fallback_summary(previous: str, folded: list[MemoryMessage]) -> str:
    """A factual note used when no summariser is available.

    Keeps the user's own turns, which carry the intent, and drops tool payloads,
    which go stale. Better than losing the thread entirely.
    """
    lines = [previous.strip()] if previous.strip() else []
    for message in folded:
        if message.role == "user":
            text = message.content.strip().replace("\n", " ")
            if text:
                lines.append(f"The user asked: {text[:160]}")
    if not lines:
        return previous.strip()
    # Keep the note bounded, otherwise it becomes the thing that needs compacting.
    return "\n".join(lines[-12:])
