"""Tests for TranscriptBuffer."""

import pytest

from tests.base_test import BaseTest
from vision_agents.core.agents.transcript_buffer import TranscriptBuffer


class TestTranscriptBuffer(BaseTest):
    @pytest.fixture
    def buffer(self):
        return TranscriptBuffer()

    def test_single_event(self, buffer):
        buffer.update("hello")
        assert len(buffer) == 1
        assert buffer.text == "hello"
        assert buffer.segments == ["hello"]

    def test_extending_text_updates_last_segment(self, buffer):
        """Test that extending text updates the existing segment."""
        buffer.update("I")
        assert buffer.segments == ["I"]

        buffer.update("I am")
        assert buffer.segments == ["I am"]

        buffer.update("I am walking")
        assert buffer.segments == ["I am walking"]

        assert len(buffer) == 1
        assert buffer.text == "I am walking"

    def test_non_overlapping_text_creates_new_segment(self, buffer):
        buffer.update("I")
        buffer.update("I am walking")
        assert buffer.segments == ["I am walking"]

        buffer.update("To")
        buffer.update("To the grocery store")
        assert buffer.segments == ["I am walking", "To the grocery store"]
        assert len(buffer) == 2

    def test_reset_clears_buffer(self, buffer):
        buffer.update("hello world")
        buffer.update("goodbye")
        assert len(buffer) == 2

        buffer.reset()
        assert len(buffer) == 0
        assert buffer.text == ""
        assert buffer.segments == []

    def test_text_property_joins_segments(self, buffer):
        buffer.update("Hello")
        buffer.update("How are you")
        assert buffer.text == "Hello How are you"

    def test_duplicate_stale_event_ignored(self, buffer):
        """Test that a stale event with shorter text is ignored."""
        buffer.update("I am walking")
        buffer.update("I am walking")
        assert buffer.segments == ["I am walking"]
