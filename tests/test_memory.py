"""T-1.7 verification: rolling memory, compaction, and persistence."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import pytest

from jarvis.brain.memory import ConversationMemory, MemoryMessage, estimate_tokens
from jarvis.config import JarvisConfig, load_config


@pytest.fixture
def config(tmp_path: Path) -> JarvisConfig:
    return load_config(
        tmp_path / "absent.yaml", memory={"db_path": str(tmp_path / "memory.db")}
    )


def _joining_summariser(messages: list[MemoryMessage], previous: str) -> str:
    """Deterministic summariser that proves the previous summary is folded in."""
    parts = [previous] if previous else []
    parts.extend(f"{m.role}:{m.content[:20]}" for m in messages)
    return " | ".join(parts)


class TestTokenEstimate:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_non_empty_is_at_least_one(self) -> None:
        assert estimate_tokens("a") >= 1

    def test_is_monotonic(self) -> None:
        assert estimate_tokens("a" * 100) > estimate_tokens("a" * 10)

    def test_roughly_four_characters_per_token(self) -> None:
        assert 20 <= estimate_tokens("x" * 100) <= 30


class TestBasics:
    def test_add_and_read_back(self, config: JarvisConfig) -> None:
        with ConversationMemory(config) as memory:
            memory.add_user("hello")
            memory.add_assistant("good evening")
            assert memory.message_count == 2

    def test_messages_puts_the_system_prompt_first(self, config: JarvisConfig) -> None:
        with ConversationMemory(config) as memory:
            memory.add_user("hi")
            messages = memory.messages("SYSTEM")
        assert messages[0] == {"role": "system", "content": "SYSTEM"}
        assert messages[-1]["content"] == "hi"

    def test_ordering_is_preserved(self, config: JarvisConfig) -> None:
        with ConversationMemory(config) as memory:
            for index in range(5):
                memory.add_user(f"m{index}")
            contents = [m["content"] for m in memory.messages("S")[1:]]
        assert contents == ["m0", "m1", "m2", "m3", "m4"]

    def test_tool_messages_carry_their_name(self, config: JarvisConfig) -> None:
        """Under ``tool_name``, which is the key Ollama actually reads.

        This asserted ``name`` until the round trip was checked against Ollama's
        Message struct. That key is not in the struct, so Go dropped it and the
        result reached the model anonymous. The test passed the whole time,
        because it only ever asked the code to agree with itself.
        """
        with ConversationMemory(config) as memory:
            memory.add_tool("sys.cpu", {"cpu_percent": 43.2})
            message = memory.messages("S")[-1]
        assert message["role"] == "tool"
        assert message["tool_name"] == "sys.cpu"
        assert "43.2" in message["content"]

    def test_long_tool_output_is_truncated(self, config: JarvisConfig) -> None:
        """§5: a verbose tool must not push the conversation out of the window."""
        with ConversationMemory(config) as memory:
            message = memory.add_tool("files.search", {"data": "x" * 50_000})
        assert len(message.content) < 3_000
        assert "truncated" in message.content

    def test_unserialisable_tool_payload_does_not_raise(self, config: JarvisConfig) -> None:
        with ConversationMemory(config) as memory:
            memory.add_tool("weird", {"obj": object()})

    def test_total_tokens_tracks_content(self, config: JarvisConfig) -> None:
        with ConversationMemory(config) as memory:
            before = memory.total_tokens
            memory.add_user("a fairly long message that should raise the count")
            assert memory.total_tokens > before


class TestCompaction:
    def _config(self, tmp_path: Path, **memory: object) -> JarvisConfig:
        base = {"db_path": str(tmp_path / "memory.db"), "keep_recent": 4}
        base.update(memory)
        return load_config(tmp_path / "absent.yaml", memory=base)

    def test_triggers_on_the_message_cap(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_messages=6)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            for index in range(12):
                memory.add_user(f"message {index}")
            assert memory.message_count <= 6
            assert memory.summary

    def test_triggers_on_the_token_cap(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_tokens=300, max_messages=500)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            for index in range(40):
                memory.add_user(f"message {index} " + "padding " * 10)
            assert memory.summary

    def test_keeps_the_most_recent_verbatim(self, tmp_path: Path) -> None:
        """The conversation must not develop amnesia at the seam.

        The window slides: after the final compaction kept keep_recent, further
        messages arrived, so the exact length varies. What must hold is that the
        newest messages are all present, verbatim, in order, and unsummarised.
        """
        config = self._config(tmp_path, max_messages=6, keep_recent=4)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            for index in range(12):
                memory.add_user(f"message {index}")
            contents = [m.content for m in memory.window()]

        assert len(contents) >= 4
        assert contents[-1] == "message 11"
        # Contiguous and ascending, so nothing was dropped out of the middle.
        indices = [int(text.split()[-1]) for text in contents]
        assert indices == list(range(indices[0], indices[0] + len(indices)))

    def test_the_previous_summary_is_folded_in(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_messages=6, keep_recent=2)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            for index in range(8):
                memory.add_user(f"first-batch-{index}")
            first_summary = memory.summary
            assert first_summary

            for index in range(8):
                memory.add_user(f"second-batch-{index}")
            second_summary = memory.summary

        assert first_summary in second_summary, "the earlier summary was lost"

    def test_the_summary_appears_in_the_message_list(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_messages=6)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            for index in range(12):
                memory.add_user(f"message {index}")
            messages = memory.messages("SYSTEM")

        assert messages[0]["content"] == "SYSTEM"
        assert messages[1]["role"] == "system"
        assert "Earlier in this conversation" in messages[1]["content"]

    def test_a_failing_summariser_does_not_lose_data(self, tmp_path: Path) -> None:
        """Better an oversized window than a conversation with holes in it."""

        def broken(_messages: list[MemoryMessage], _previous: str) -> str:
            raise RuntimeError("ollama is down")

        config = self._config(tmp_path, max_messages=6)
        with ConversationMemory(config, summariser=broken) as memory:
            for index in range(12):
                memory.add_user(f"message {index}")
            contents = [m.content for m in memory.window()]

        assert len(contents) == 12, "messages were dropped without being summarised"
        assert contents[0] == "message 0"

    def test_an_empty_summary_falls_back(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_messages=6)
        with ConversationMemory(config, summariser=lambda _m, _p: "  ") as memory:
            for index in range(12):
                memory.add_user(f"question {index}")
            assert memory.summary, "an empty summary must not erase the history"

    def test_no_summariser_still_keeps_the_thread(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, max_messages=6)
        with ConversationMemory(config) as memory:
            for index in range(12):
                memory.add_user(f"question {index}")
            assert "question" in memory.summary

    def test_compact_is_a_no_op_on_a_short_window(self, config: JarvisConfig) -> None:
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            memory.add_user("one")
            assert memory.compact() is False


class TestPersistence:
    def test_round_trips_across_instances(self, config: JarvisConfig) -> None:
        """T-1.7: context must survive a restart."""
        first = ConversationMemory(config)
        first.add_user("remember this")
        first.add_assistant("noted")
        first.close()

        second = ConversationMemory(config)
        try:
            contents = [m.content for m in second.window()]
            assert contents == ["remember this", "noted"]
        finally:
            second.close()

    def test_the_summary_survives_too(self, tmp_path: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml",
            memory={"db_path": str(tmp_path / "m.db"), "max_messages": 6, "keep_recent": 2},
        )
        first = ConversationMemory(config, summariser=_joining_summariser)
        for index in range(12):
            first.add_user(f"message {index}")
        expected = first.summary
        first.close()

        second = ConversationMemory(config)
        try:
            assert second.summary == expected
        finally:
            second.close()

    def test_sessions_are_isolated(self, tmp_path: Path) -> None:
        path = str(tmp_path / "shared.db")
        alpha = ConversationMemory(
            load_config(tmp_path / "a.yaml", memory={"db_path": path, "session_id": "alpha"})
        )
        beta = ConversationMemory(
            load_config(tmp_path / "b.yaml", memory={"db_path": path, "session_id": "beta"})
        )
        try:
            alpha.add_user("alpha only")
            beta.add_user("beta only")
            assert [m.content for m in alpha.window()] == ["alpha only"]
            assert [m.content for m in beta.window()] == ["beta only"]
        finally:
            alpha.close()
            beta.close()

    def test_clear_wipes_both_layers(self, config: JarvisConfig) -> None:
        first = ConversationMemory(config)
        first.add_user("temporary")
        first.clear()
        first.close()

        second = ConversationMemory(config)
        try:
            assert second.window() == []
        finally:
            second.close()

    def test_disabled_memory_writes_no_file(self, tmp_path: Path) -> None:
        db = tmp_path / "should-not-exist.db"
        config = load_config(
            tmp_path / "absent.yaml", memory={"enabled": False, "db_path": str(db)}
        )
        with ConversationMemory(config) as memory:
            memory.add_user("in memory only")
            assert memory.message_count == 1
        assert not db.exists()


class TestThreadSafety:
    def test_concurrent_adds_do_not_corrupt_the_window(self, tmp_path: Path) -> None:
        """The orchestrator is threaded, so memory must be too.

        The caps are raised above the write count so this measures locking
        rather than compaction, which has its own tests.
        """
        config = load_config(
            tmp_path / "absent.yaml",
            memory={
                "db_path": str(tmp_path / "memory.db"),
                "max_messages": 500,
                "max_tokens": 100_000,
            },
        )
        with ConversationMemory(config) as memory:

            def worker(index: int) -> None:
                for step in range(20):
                    memory.add_user(f"worker {index} step {step}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert memory.message_count == 80
            # Every write is present exactly once.
            contents = {m.content for m in memory.window()}
            assert len(contents) == 80


class TestTheToolCallRoundTrip:
    """A tool result has to arrive attached to the call that asked for it.

    Ollama's Message struct is Role, Content, Thinking, Images, ToolCalls,
    ToolName, ToolCallID. Go's JSON decoder drops unknown keys silently, so a
    name sent under the wrong key arrives nameless with nothing to say so. And
    Qwen3's template renders a tool result with no identity of its own: the only
    thing binding a result to a question is the tool_calls block on the
    assistant message before it. Send neither and the model sees anonymous JSON
    appear after the user's question, which is what "it felt like it was reading
    instructions" looks like from the other side.
    """

    def test_a_tool_result_names_itself_under_the_key_ollama_reads(
        self, cfg: JarvisConfig
    ) -> None:
        message = MemoryMessage(role="tool", content="{}", tool_name="sys.cpu")
        assert message.to_chat()["tool_name"] == "sys.cpu"

    def test_it_does_not_use_the_key_ollama_discards(self, cfg: JarvisConfig) -> None:
        """`name` is not in Ollama's Message struct, so it never arrives."""
        assert "name" not in MemoryMessage(role="tool", content="{}", tool_name="sys.cpu").to_chat()

    def test_an_assistant_message_carries_its_tool_calls(self, cfg: JarvisConfig) -> None:
        calls = [{"type": "function", "function": {"name": "sys.cpu", "arguments": {}}}]
        chat = MemoryMessage(role="assistant", content="", tool_calls=calls).to_chat()
        assert chat["tool_calls"] == calls

    def test_a_plain_assistant_message_carries_no_empty_block(self, cfg: JarvisConfig) -> None:
        assert "tool_calls" not in MemoryMessage(role="assistant", content="Light.").to_chat()

    def test_a_user_message_never_carries_tool_calls(self, cfg: JarvisConfig) -> None:
        calls = [{"type": "function", "function": {"name": "x", "arguments": {}}}]
        message = MemoryMessage(role="user", content="hi", tool_calls=calls)
        assert "tool_calls" not in message.to_chat()

    def test_add_assistant_stores_the_calls(self, cfg: JarvisConfig, tmp_path: Path) -> None:
        memory = ConversationMemory(cfg, db_path=tmp_path / "m.db")
        calls = [{"type": "function", "function": {"name": "sys.gpu", "arguments": {}}}]
        memory.add_assistant("", tool_calls=calls)
        assert memory.messages("sys")[-1]["tool_calls"] == calls

    def test_the_calls_survive_a_restart(self, cfg: JarvisConfig, tmp_path: Path) -> None:
        """They are part of the conversation, so they have to persist with it."""
        path = tmp_path / "m.db"
        call = {"name": "sys.disk", "arguments": {"unit": "gb"}}
        calls = [{"type": "function", "function": call}]
        first = ConversationMemory(cfg, db_path=path)
        first.add_user("how much space")
        first.add_assistant("", tool_calls=calls)
        first.add_tool("sys.disk", {"free_gb": 41.2})
        first.close()

        reopened = ConversationMemory(cfg, db_path=path)
        restored = reopened.messages("sys")
        assert [m.get("tool_calls") for m in restored if m["role"] == "assistant"] == [calls]
        assert [m.get("tool_name") for m in restored if m["role"] == "tool"] == ["sys.disk"]

    def test_a_database_from_before_the_column_existed_still_opens(
        self, cfg: JarvisConfig, tmp_path: Path
    ) -> None:
        """Anyone who has run JARVIS once has such a file.

        CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
        so without a migration every insert would fail on a live install.
        """
        import sqlite3

        path = tmp_path / "old.db"
        old = sqlite3.connect(str(path))
        old.executescript(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, timestamp REAL NOT NULL,
                tokens INTEGER NOT NULL, tool_name TEXT
            );
            CREATE TABLE summaries (
                session_id TEXT PRIMARY KEY, summary TEXT NOT NULL, updated_at REAL NOT NULL
            );
            """
        )
        old.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, tokens, tool_name)"
            " VALUES (?, 'user', 'hello', 1.0, 2, NULL)",
            (cfg.memory.session_id,),
        )
        old.commit()
        old.close()

        memory = ConversationMemory(cfg, db_path=path)
        assert [m["content"] for m in memory.messages("sys") if m["role"] == "user"] == ["hello"]
        calls = [{"function": {"name": "sys.cpu", "arguments": {}}}]
        memory.add_assistant("", tool_calls=calls)
        memory.close()

        # Reopened rather than read back from the live window. Persistence
        # failures are swallowed and logged so a full disk cannot kill the turn
        # loop, which means an unmigrated insert looks fine in memory and loses
        # the conversation on the next restart. Only disk proves the migration.
        reopened = ConversationMemory(cfg, db_path=path)
        stored = reopened.messages("sys")
        assert [m["content"] for m in stored if m["role"] == "user"] == ["hello"]
        assert [m.get("tool_calls") for m in stored if m["role"] == "assistant"] == [calls]

    def test_unreadable_stored_calls_do_not_break_a_restart(
        self, cfg: JarvisConfig, tmp_path: Path
    ) -> None:
        import sqlite3

        path = tmp_path / "m.db"
        memory = ConversationMemory(cfg, db_path=path)
        memory.add_assistant("hm", tool_calls=[{"function": {"name": "x", "arguments": {}}}])
        memory.close()

        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE messages SET tool_calls = 'not json'")
        conn.commit()
        conn.close()

        reopened = ConversationMemory(cfg, db_path=path)
        assert reopened.message_count == 1
        assert not reopened.messages("sys")[-1].get("tool_calls")


class TestCompactionStaysOffTheTurnPath:
    """Compaction is a full, non-streaming LLM generation.

    Triggered from _add it landed wherever a message happened to be recorded,
    which includes between a tool result and the answering round: the user has
    stopped speaking, a tool has already run, and the turn stops dead for
    several seconds to summarise old messages nobody asked about.
    """

    def _config(self, tmp_path: Path) -> JarvisConfig:
        return load_config(
            tmp_path / "absent.yaml",
            memory={"db_path": str(tmp_path / "m.db"), "max_messages": 4, "keep_recent": 2},
        )

    def test_nothing_compacts_inside_the_block(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def summariser(messages: list[MemoryMessage], previous: str) -> str:
            calls.append(len(messages))
            return "summary"

        with ConversationMemory(self._config(tmp_path), summariser=summariser) as memory:
            with memory.deferred_compaction():
                for index in range(10):
                    memory.add_user(f"message {index}")
                assert not calls, "compaction ran mid-turn"
            assert calls, "compaction never ran at all"

    def test_it_compacts_once_however_many_were_added(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def summariser(messages: list[MemoryMessage], previous: str) -> str:
            calls.append(len(messages))
            return "summary"

        with ConversationMemory(self._config(tmp_path), summariser=summariser) as memory:
            with memory.deferred_compaction():
                for index in range(20):
                    memory.add_user(f"message {index}")
            assert len(calls) == 1

    def test_the_caps_still_hold_afterwards(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        with ConversationMemory(config, summariser=_joining_summariser) as memory:
            with memory.deferred_compaction():
                for index in range(20):
                    memory.add_user(f"message {index}")
            assert memory.message_count <= config.memory.max_messages
            assert memory.summary

    def test_it_is_reentrant(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def summariser(messages: list[MemoryMessage], previous: str) -> str:
            calls.append(len(messages))
            return "summary"

        with ConversationMemory(self._config(tmp_path), summariser=summariser) as memory:
            with memory.deferred_compaction():
                for index in range(10):
                    memory.add_user(f"outer {index}")
                with memory.deferred_compaction():
                    memory.add_user("inner")
                assert not calls, "the inner block compacted"
            assert len(calls) == 1

    def test_an_exception_still_releases_the_hold(self, tmp_path: Path) -> None:
        with ConversationMemory(self._config(tmp_path), summariser=_joining_summariser) as memory:
            with contextlib.suppress(RuntimeError), memory.deferred_compaction():
                memory.add_user("one")
                raise RuntimeError("turn failed")
            for index in range(10):
                memory.add_user(f"after {index}")
            assert memory.summary, "compaction stayed held after a failed turn"

    def test_without_the_block_it_compacts_inline(self, tmp_path: Path) -> None:
        """Every other caller keeps the old behaviour."""
        with ConversationMemory(self._config(tmp_path), summariser=_joining_summariser) as memory:
            for index in range(10):
                memory.add_user(f"message {index}")
            assert memory.summary
