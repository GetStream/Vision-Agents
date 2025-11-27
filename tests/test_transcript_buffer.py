"""Tests for TranscriptBuffer."""

import pytest

from tests.base_test import BaseTest
from vision_agents.core.agents.transcript_buffer import TranscriptBuffer
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent


class TestTranscriptBuffer(BaseTest):
    @pytest.fixture
    def buffer(self):
        return TranscriptBuffer()

    def test_single_final_event(self, buffer):
        buffer.update("hello")
        assert len(buffer) == 1
        assert buffer.text == "hello"
        assert buffer.segments == ["hello"]

    def test_multiple_final_events_create_separate_segments(self, buffer):
        """Each final event (string) creates a new segment."""
        buffer.update("hello")
        buffer.update("world")
        assert buffer.segments == ["hello", "world"]
        assert buffer.text == "hello world"

    def test_partial_events_update_last_segment(self, buffer):
        """Partial events update the current working segment."""
        buffer.update(STTPartialTranscriptEvent(text="I"))
        assert buffer.segments == ["I"]

        buffer.update(STTPartialTranscriptEvent(text="I am"))
        assert buffer.segments == ["I am"]

        buffer.update(STTPartialTranscriptEvent(text="I am walking"))
        assert buffer.segments == ["I am walking"]

        assert len(buffer) == 1
        assert buffer.text == "I am walking"

    def test_partial_with_corrections(self, buffer):
        """Partial events can be corrections, not just extensions."""
        buffer.update(STTPartialTranscriptEvent(text="What is the fact"))
        assert buffer.segments == ["What is the fact"]

        buffer.update(
            STTPartialTranscriptEvent(text="What is the fastest human ability")
        )
        assert buffer.segments == ["What is the fastest human ability"]

        buffer.update(
            STTPartialTranscriptEvent(text="What is the fastest human alive?")
        )
        assert buffer.segments == ["What is the fastest human alive?"]

        assert len(buffer) == 1

    def test_final_event_finalizes_partial(self, buffer):
        """Final event replaces the partial and finalizes it."""
        buffer.update(STTPartialTranscriptEvent(text="I"))
        buffer.update(STTPartialTranscriptEvent(text="I am"))
        buffer.update(STTTranscriptEvent(text="I am walking to the store"))

        assert buffer.segments == ["I am walking to the store"]
        assert len(buffer) == 1

    def test_new_partial_after_final_starts_new_segment(self, buffer):
        """After a final event, new partials start a fresh segment."""
        buffer.update(STTPartialTranscriptEvent(text="Hello"))
        buffer.update(STTTranscriptEvent(text="Hello there"))
        assert buffer.segments == ["Hello there"]

        buffer.update(STTPartialTranscriptEvent(text="How"))
        buffer.update(STTPartialTranscriptEvent(text="How are you"))
        assert buffer.segments == ["Hello there", "How are you"]

        buffer.update(STTTranscriptEvent(text="How are you doing?"))
        assert buffer.segments == ["Hello there", "How are you doing?"]

    def test_multiple_utterances(self, buffer):
        """Complete flow with multiple utterances."""
        # First utterance
        buffer.update(STTPartialTranscriptEvent(text="I"))
        buffer.update(STTPartialTranscriptEvent(text="I am"))
        buffer.update(STTTranscriptEvent(text="I am walking"))

        # Second utterance
        buffer.update(STTPartialTranscriptEvent(text="To"))
        buffer.update(STTPartialTranscriptEvent(text="To the"))
        buffer.update(STTTranscriptEvent(text="To the store"))

        assert buffer.segments == ["I am walking", "To the store"]
        assert buffer.text == "I am walking To the store"

    def test_reset_clears_buffer(self, buffer):
        buffer.update(STTPartialTranscriptEvent(text="hello"))
        buffer.update(STTTranscriptEvent(text="hello world"))
        buffer.update("goodbye")
        assert len(buffer) == 2

        buffer.reset()
        assert len(buffer) == 0
        assert buffer.text == ""
        assert buffer.segments == []

    def test_reset_clears_pending_state(self, buffer):
        """Reset clears the pending partial state."""
        buffer.update(STTPartialTranscriptEvent(text="hello"))
        buffer.reset()

        # After reset, a new final should create segment 1, not update
        buffer.update("world")
        assert buffer.segments == ["world"]

    def test_duplicate_partial_ignored(self, buffer):
        """Duplicate partial events are ignored."""
        buffer.update(STTPartialTranscriptEvent(text="I am walking"))
        buffer.update(STTPartialTranscriptEvent(text="I am walking"))
        assert buffer.segments == ["I am walking"]

    def test_stt_transcript_event_adds_segment(self, buffer):
        """STTTranscriptEvent adds a segment directly when no partial pending."""
        buffer.update(STTTranscriptEvent(text="Hello world"))
        buffer.update(STTTranscriptEvent(text="Goodbye world"))
        assert buffer.segments == ["Hello world", "Goodbye world"]

    def test_duplicate_final_event_ignored(self, buffer):
        """Duplicate final events are ignored."""
        buffer.update(STTPartialTranscriptEvent(text="What is the fastest animal?"))
        buffer.update(STTTranscriptEvent(text="What is the fastest animal?"))
        buffer.update(STTTranscriptEvent(text="What is the fastest animal?"))
        assert buffer.segments == ["What is the fastest animal?"]
        assert len(buffer) == 1

    def test_duplicate_final_without_partial_ignored(self, buffer):
        """Duplicate final events without preceding partial are ignored."""
        buffer.update(STTTranscriptEvent(text="Hello"))
        buffer.update(STTTranscriptEvent(text="Hello"))
        assert buffer.segments == ["Hello"]

    def test_partial_after_final_with_same_text_ignored(self, buffer):
        """Partial after final with same text should not create duplicate."""
        buffer.update(STTPartialTranscriptEvent(text="Tell me about Deepgram."))
        buffer.update(STTTranscriptEvent(text="Tell me about Deepgram."))
        buffer.update(STTPartialTranscriptEvent(text="Tell me about Deepgram."))
        buffer.update(STTTranscriptEvent(text="Tell me about Deepgram."))
        assert buffer.segments == ["Tell me about Deepgram."]
        assert len(buffer) == 1

    def test_string_adds_segment(self, buffer):
        """Plain strings are treated as final events."""
        buffer.update("Hello")
        buffer.update("World")
        assert buffer.segments == ["Hello", "World"]

    def test_empty_text_ignored(self, buffer):
        """Empty or whitespace-only text is ignored."""
        buffer.update("")
        buffer.update("   ")
        buffer.update(STTPartialTranscriptEvent(text=""))
        buffer.update(STTPartialTranscriptEvent(text="  "))
        assert len(buffer) == 0

    def test_bool_false_when_empty(self, buffer):
        assert not buffer

    def test_bool_true_when_has_content(self, buffer):
        buffer.update("hello")
        assert buffer
