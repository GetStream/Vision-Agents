"""Tests for TranscriptStore."""

import pytest

from vision_agents.core.agents.transcript import TranscriptStore


class TestTranscriptStore:
    def test_empty_agent_user_id_raises(self):
        with pytest.raises(ValueError, match="agent_user_id"):
            TranscriptStore(agent_user_id="")

    def test_update_user_transcript_replacement_flow(self):
        store = TranscriptStore(agent_user_id="agent-1")

        result = store.update_user_transcript(
            participant_id="p1", user_id="user-1", text="Hello", mode="replacement"
        )
        assert result is not None
        assert result.text == "Hello"
        assert result.user_id == "user-1"
        assert result.mode == "replacement"

        result2 = store.update_user_transcript(
            participant_id="p1", user_id="user-1", text="Hello world", mode="final"
        )
        assert result2 is not None
        assert result2.text == "Hello world"
        assert result2.message_id == result.message_id
        assert result2.mode == "final"

    def test_update_user_transcript_preserves_message_id(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="A", mode="replacement"
        )
        r2 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="AB", mode="replacement"
        )
        assert r1.message_id == r2.message_id

    def test_update_user_transcript_different_participants(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Hi", mode="replacement"
        )
        r2 = store.update_user_transcript(
            participant_id="p2", user_id="u2", text="Hey", mode="replacement"
        )
        assert r1.message_id != r2.message_id
        assert r1.user_id == "u1"
        assert r2.user_id == "u2"

    def test_update_user_transcript_empty_text_skipped(self):
        store = TranscriptStore(agent_user_id="agent-1")
        result = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="", mode="final"
        )
        assert result is None

    def test_update_agent_transcript_delta_flow(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_agent_transcript(text="I'm ", mode="delta")
        assert r1 is not None
        assert r1.text == "I'm "
        assert r1.user_id == "agent-1"
        assert r1.mode == "delta"

        r2 = store.update_agent_transcript(text="doing well", mode="delta")
        assert r2.text == "doing well"
        assert r2.message_id == r1.message_id

        r3 = store.update_agent_transcript(text="", mode="final")
        assert r3.text == "I'm doing well"
        assert r3.mode == "final"

    def test_update_agent_transcript_empty_text_skipped(self):
        store = TranscriptStore(agent_user_id="agent-1")
        result = store.update_agent_transcript(text="", mode="delta")
        assert result is None

    def test_pending_users_returns_and_clears(self):
        store = TranscriptStore(agent_user_id="agent-1")

        store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Hello", mode="delta"
        )
        store.update_user_transcript(
            participant_id="p2", user_id="u2", text="World", mode="delta"
        )

        pending = store.flush_users_transcripts()
        assert len(pending) == 2
        assert all(p.mode == "final" for p in pending)

        assert store.flush_users_transcripts() == []

    def test_pending_agent_returns_and_clears(self):
        store = TranscriptStore(agent_user_id="agent-1")

        store.update_agent_transcript(text="Response", mode="delta")

        pending = store.flush_agent_transcript()
        assert pending is not None
        assert pending.text == "Response"
        assert pending.mode == "final"

        assert store.flush_agent_transcript() is None

    def test_pending_agent_returns_none_when_empty(self):
        store = TranscriptStore(agent_user_id="agent-1")
        assert store.flush_agent_transcript() is None

    def test_final_clears_user_entry(self):
        store = TranscriptStore(agent_user_id="agent-1")

        store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Done", mode="final"
        )

        assert store.flush_users_transcripts() == []

    def test_final_clears_agent_entry(self):
        store = TranscriptStore(agent_user_id="agent-1")

        store.update_agent_transcript(text="Done", mode="final")

        assert store.flush_agent_transcript() is None

    def test_update_user_transcript_delta_returns_raw_text(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Hello ", mode="delta"
        )
        assert r1.text == "Hello "

        r2 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="world", mode="delta"
        )
        assert r2.text == "world"
        assert r2.message_id == r1.message_id

    def test_update_agent_transcript_replacement_flow(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_agent_transcript(text="Thinking", mode="replacement")
        assert r1.text == "Thinking"
        assert r1.mode == "replacement"

        r2 = store.update_agent_transcript(text="Thinking about it", mode="replacement")
        assert r2.text == "Thinking about it"
        assert r2.message_id == r1.message_id

        r3 = store.update_agent_transcript(text="Thinking about it.", mode="final")
        assert r3.text == "Thinking about it."
        assert r3.mode == "final"

    def test_final_assigns_new_message_id_for_next_entry(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_agent_transcript(text="First", mode="final")
        r2 = store.update_agent_transcript(text="Second", mode="final")
        assert r1.message_id != r2.message_id

    def test_user_final_assigns_new_message_id_for_next_entry(self):
        store = TranscriptStore(agent_user_id="agent-1")

        r1 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="First", mode="final"
        )
        r2 = store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Second", mode="final"
        )
        assert r1.message_id != r2.message_id

    def test_get_buffer_user(self):
        store = TranscriptStore(agent_user_id="agent-1")

        assert store.get_buffer(participant_id="p1", user_id="u1") is None

        store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Hello", mode="delta"
        )
        buffer = store.get_buffer(participant_id="p1", user_id="u1")
        assert buffer is not None
        assert buffer.text == "Hello"

    def test_get_buffer_agent(self):
        store = TranscriptStore(agent_user_id="agent-1")

        assert store.get_buffer(participant_id="p-agent", user_id="agent-1") is None

        store.update_agent_transcript(text="Hi ", mode="delta")
        buffer = store.get_buffer(participant_id="p-agent", user_id="agent-1")
        assert buffer is not None
        assert buffer.text == "Hi "

    def test_flush_users_skips_empty_buffers(self):
        store = TranscriptStore(agent_user_id="agent-1")

        store.update_user_transcript(
            participant_id="p1", user_id="u1", text="Hello", mode="delta"
        )
        store.update_user_transcript(
            participant_id="p2", user_id="u2", text="", mode="replacement"
        )

        pending = store.flush_users_transcripts()
        assert len(pending) == 1
        assert pending[0].user_id == "u1"

    def test_update_user_transcript_invalid_mode_raises(self):
        store = TranscriptStore(agent_user_id="agent-1")
        with pytest.raises(ValueError, match="Invalid transcript mode"):
            store.update_user_transcript(
                participant_id="p1", user_id="u1", text="hi", mode="invalid"
            )

    def test_update_agent_transcript_invalid_mode_raises(self):
        store = TranscriptStore(agent_user_id="agent-1")
        with pytest.raises(ValueError, match="Invalid transcript mode"):
            store.update_agent_transcript(text="hi", mode="invalid")
