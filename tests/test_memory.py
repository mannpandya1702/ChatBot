"""T-1.7 verification: rolling memory, compaction, and persistence."""

from __future__ import annotations

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
        with ConversationMemory(config) as memory:
            memory.add_tool("sys.cpu", {"cpu_percent": 43.2})
            message = memory.messages("S")[-1]
        assert message["role"] == "tool"
        assert message["name"] == "sys.cpu"
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
