"""Tests for TranscriptBuffer."""

import pytest

from vision_agents.core.agents.transcript import TranscriptBuffer


@pytest.fixture
def buffer():
    return TranscriptBuffer()


class TestTranscriptBuffer:
    def test_single_final_event(self, buffer):
        buffer.update("hello", mode="final")
        assert len(buffer) == 1
        assert buffer.text == "hello"
        assert buffer.segments == ["hello"]

    def test_multiple_final_events_create_separate_segments(self, buffer):
        buffer.update("hello", mode="final")
        buffer.update("world", mode="final")
        assert buffer.segments == ["hello", "world"]
        assert buffer.text == "hello world"

    def test_replacement_events_update_last_segment(self, buffer):
        buffer.update("I", mode="replacement")
        assert buffer.segments == ["I"]

        buffer.update("I am", mode="replacement")
        assert buffer.segments == ["I am"]

        buffer.update("I am walking", mode="replacement")
        assert buffer.segments == ["I am walking"]

        assert len(buffer) == 1
        assert buffer.text == "I am walking"

    def test_replacement_with_corrections(self, buffer):
        buffer.update("What is the fact", mode="replacement")
        assert buffer.segments == ["What is the fact"]

        buffer.update("What is the fastest human ability", mode="replacement")
        assert buffer.segments == ["What is the fastest human ability"]

        buffer.update("What is the fastest human alive?", mode="replacement")
        assert buffer.segments == ["What is the fastest human alive?"]

        assert len(buffer) == 1

    def test_final_event_finalizes_replacement(self, buffer):
        buffer.update("I", mode="replacement")
        buffer.update("I am", mode="replacement")
        buffer.update("I am walking to the store", mode="final")

        assert buffer.segments == ["I am walking to the store"]
        assert len(buffer) == 1

    def test_new_replacement_after_final_starts_new_segment(self, buffer):
        buffer.update("Hello", mode="replacement")
        buffer.update("Hello there", mode="final")
        assert buffer.segments == ["Hello there"]

        buffer.update("How", mode="replacement")
        buffer.update("How are you", mode="replacement")
        assert buffer.segments == ["Hello there", "How are you"]

        buffer.update("How are you doing?", mode="final")
        assert buffer.segments == ["Hello there", "How are you doing?"]

    def test_multiple_utterances(self, buffer):
        buffer.update("I", mode="replacement")
        buffer.update("I am", mode="replacement")
        buffer.update("I am walking", mode="final")

        buffer.update("To", mode="replacement")
        buffer.update("To the", mode="replacement")
        buffer.update("To the store", mode="final")

        assert buffer.segments == ["I am walking", "To the store"]
        assert buffer.text == "I am walking To the store"

    def test_reset_clears_buffer(self, buffer):
        buffer.update("hello", mode="replacement")
        buffer.update("hello world", mode="final")
        buffer.update("goodbye", mode="final")
        assert len(buffer) == 2

        buffer.reset()
        assert len(buffer) == 0
        assert buffer.text == ""
        assert buffer.segments == []

    def test_reset_clears_pending_state(self, buffer):
        buffer.update("hello", mode="replacement")
        buffer.reset()

        buffer.update("world", mode="final")
        assert buffer.segments == ["world"]

    def test_duplicate_replacement_ignored(self, buffer):
        buffer.update("I am walking", mode="replacement")
        buffer.update("I am walking", mode="replacement")
        assert buffer.segments == ["I am walking"]

    def test_final_adds_segment_when_no_pending(self, buffer):
        buffer.update("Hello world", mode="final")
        buffer.update("Goodbye world", mode="final")
        assert buffer.segments == ["Hello world", "Goodbye world"]

    def test_duplicate_final_event_ignored(self, buffer):
        buffer.update("What is the fastest animal?", mode="replacement")
        buffer.update("What is the fastest animal?", mode="final")
        buffer.update("What is the fastest animal?", mode="final")
        assert buffer.segments == ["What is the fastest animal?"]
        assert len(buffer) == 1

    def test_duplicate_final_without_replacement_ignored(self, buffer):
        buffer.update("Hello", mode="final")
        buffer.update("Hello", mode="final")
        assert buffer.segments == ["Hello"]

    def test_replacement_after_final_with_same_text_ignored(self, buffer):
        buffer.update("Tell me about Deepgram.", mode="replacement")
        buffer.update("Tell me about Deepgram.", mode="final")
        buffer.update("Tell me about Deepgram.", mode="replacement")
        buffer.update("Tell me about Deepgram.", mode="final")
        assert buffer.segments == ["Tell me about Deepgram."]
        assert len(buffer) == 1

    def test_empty_text_ignored(self, buffer):
        buffer.update("", mode="final")
        buffer.update("   ", mode="replacement")
        buffer.update("", mode="delta")
        assert len(buffer) == 0

    def test_whitespace_only_delta_preserved(self, buffer):
        buffer.update("Hello", mode="delta")
        buffer.update(" ", mode="delta")
        buffer.update("world", mode="delta")
        assert buffer.text == "Hello world"

    def test_bool_false_when_empty(self, buffer):
        assert not buffer

    def test_bool_true_when_has_content(self, buffer):
        buffer.update("hello", mode="final")
        assert buffer

    def test_has_pending_false_when_empty(self, buffer):
        assert not buffer.has_pending

    def test_has_pending_true_after_replacement(self, buffer):
        buffer.update("hello", mode="replacement")
        assert buffer.has_pending

    def test_has_pending_false_after_final(self, buffer):
        buffer.update("hello", mode="replacement")
        buffer.update("hello world", mode="final")
        assert not buffer.has_pending

    def test_delta_appends_to_current_segment(self, buffer):
        buffer.update("I ", mode="delta")
        assert buffer.segments == ["I "]

        buffer.update("am ", mode="delta")
        assert buffer.segments == ["I am "]

        buffer.update("walking", mode="delta")
        assert buffer.segments == ["I am walking"]

        assert len(buffer) == 1

    def test_delta_then_final_with_text(self, buffer):
        buffer.update("Hello ", mode="delta")
        buffer.update("world", mode="delta")
        buffer.update("Hello world!", mode="final")

        assert buffer.segments == ["Hello world!"]
        assert not buffer.has_pending

    def test_delta_then_empty_final(self, buffer):
        """Empty final just finalizes without changing text."""
        buffer.update("Hello ", mode="delta")
        buffer.update("world", mode="delta")
        buffer.update("", mode="final")

        assert buffer.segments == ["Hello world"]
        assert not buffer.has_pending

    def test_delta_multiple_segments(self, buffer):
        buffer.update("First ", mode="delta")
        buffer.update("segment", mode="delta")
        buffer.update("", mode="final")

        buffer.update("Second ", mode="delta")
        buffer.update("segment", mode="delta")
        buffer.update("", mode="final")

        assert buffer.segments == ["First segment", "Second segment"]

    def test_empty_final_without_pending_is_noop(self, buffer):
        buffer.update("", mode="final")
        assert len(buffer) == 0
        assert not buffer.has_pending

    def test_mixed_delta_and_replacement(self, buffer):
        """Delta and replacement can be used in separate segments."""
        buffer.update("delta ", mode="delta")
        buffer.update("text", mode="delta")
        buffer.update("", mode="final")

        buffer.update("replacement start", mode="replacement")
        buffer.update("replacement done", mode="final")

        assert buffer.segments == ["delta text", "replacement done"]
