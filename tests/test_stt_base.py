"""
Test the base STT class consistency improvements.
"""

import pytest
from unittest.mock import Mock
from getstream.plugins.common import STT
from getstream.video.rtc.track_util import PcmData
import numpy as np


class MockSTT(STT):
    """Mock STT implementation for testing base class functionality."""

    def __init__(self):
        super().__init__()
        self.process_audio_impl_called = False
        self.process_audio_impl_result = None

    async def _process_audio_impl(self, pcm_data, user_metadata=None):
        self.process_audio_impl_called = True
        return self.process_audio_impl_result

    async def close(self):
        self._is_closed = True


@pytest.fixture
def mock_stt():
    return MockSTT()


@pytest.fixture
def valid_pcm_data():
    """Create valid PCM data for testing."""
    samples = np.random.randint(-1000, 1000, size=1000, dtype=np.int16)
    return PcmData(samples=samples, sample_rate=16000, format="s16")


def test_validate_pcm_data_valid(mock_stt, valid_pcm_data):
    """Test that valid PCM data passes validation."""
    assert mock_stt._validate_pcm_data(valid_pcm_data) is True


def test_validate_pcm_data_none(mock_stt):
    """Test that None PCM data fails validation."""
    assert mock_stt._validate_pcm_data(None) is False


def test_validate_pcm_data_no_samples(mock_stt):
    """Test that PCM data without samples fails validation."""
    pcm_data = Mock()
    pcm_data.samples = None
    pcm_data.sample_rate = 16000
    assert mock_stt._validate_pcm_data(pcm_data) is False


def test_validate_pcm_data_invalid_sample_rate(mock_stt):
    """Test that PCM data with invalid sample rate fails validation."""
    pcm_data = Mock()
    pcm_data.samples = np.array([1, 2, 3])
    pcm_data.sample_rate = 0
    assert mock_stt._validate_pcm_data(pcm_data) is False


def test_validate_pcm_data_empty_samples(mock_stt):
    """Test that PCM data with empty samples fails validation."""
    pcm_data = Mock()
    pcm_data.samples = np.array([])
    pcm_data.sample_rate = 16000
    assert mock_stt._validate_pcm_data(pcm_data) is False


@pytest.mark.asyncio
async def test_emit_transcript_event(mock_stt):
    """Test that transcript events are emitted correctly."""
    # Set up event listener
    transcript_events = []

    def on_transcript(event):
        transcript_events.append(event)

    mock_stt.on("transcript", on_transcript)

    # Emit a transcript event
    text = "Hello world"
    user_metadata = {"user_id": "123"}
    metadata = {"confidence": 0.95, "processing_time_ms": 100}

    mock_stt._emit_transcript_event(text, user_metadata, metadata)

    # Verify event was emitted
    assert len(transcript_events) == 1
    event = transcript_events[0]
    assert event.text == text
    assert event.user_metadata == user_metadata
    assert event.confidence == metadata["confidence"]
    assert event.processing_time_ms == metadata["processing_time_ms"]


@pytest.mark.asyncio
async def test_emit_partial_transcript_event(mock_stt):
    """Test that partial transcript events are emitted correctly."""
    # Set up event listener
    partial_events = []

    def on_partial_transcript(event):
        partial_events.append(event)

    mock_stt.on("partial_transcript", on_partial_transcript)

    # Emit a partial transcript event
    text = "Hello"
    user_metadata = {"user_id": "123"}
    metadata = {"confidence": 0.8}

    mock_stt._emit_partial_transcript_event(text, user_metadata, metadata)

    # Verify event was emitted
    assert len(partial_events) == 1
    event = partial_events[0]
    assert event.text == text
    assert event.user_metadata == user_metadata
    assert event.confidence == metadata["confidence"]


@pytest.mark.asyncio
async def test_emit_error_event(mock_stt):
    """Test that error events are emitted correctly."""
    # Set up event listener
    error_events = []

    def on_error(event):
        error_events.append(event)

    mock_stt.on("error", on_error)

    # Emit an error event
    test_error = Exception("Test error")
    mock_stt._emit_error_event(test_error, "test context")

    # Verify event was emitted
    assert len(error_events) == 1
    event = error_events[0]
    assert event.error == test_error
    assert event.context == "test context"


@pytest.mark.asyncio
async def test_process_audio_with_invalid_data(mock_stt):
    """Test that process_audio handles invalid data gracefully."""
    # Try to process None data
    await mock_stt.process_audio(None)

    # Verify that _process_audio_impl was not called
    assert mock_stt.process_audio_impl_called is False


@pytest.mark.asyncio
async def test_process_audio_with_valid_data(mock_stt, valid_pcm_data):
    """Test that process_audio processes valid data correctly."""
    # Set up mock result
    mock_stt.process_audio_impl_result = [(True, "Hello world", {"confidence": 0.95})]

    # Set up event listener
    transcript_events = []

    def on_transcript(event):
        transcript_events.append(event)

    mock_stt.on("transcript", on_transcript)

    # Process audio
    user_metadata = {"user_id": "123"}
    await mock_stt.process_audio(valid_pcm_data, user_metadata)

    # Verify that _process_audio_impl was called
    assert mock_stt.process_audio_impl_called is True

    # Verify that transcript event was emitted
    assert len(transcript_events) == 1
    event = transcript_events[0]
    assert event.text == "Hello world"
    assert event.user_metadata == user_metadata
    assert event.confidence == 0.95
    assert event.processing_time_ms is not None  # Should be added by base class


@pytest.mark.asyncio
async def test_process_audio_when_closed(mock_stt, valid_pcm_data):
    """Test that process_audio ignores requests when STT is closed."""
    # Close the STT
    await mock_stt.close()

    # Try to process audio
    await mock_stt.process_audio(valid_pcm_data)

    # Verify that _process_audio_impl was not called
    assert mock_stt.process_audio_impl_called is False


@pytest.mark.asyncio
async def test_process_audio_handles_exceptions(mock_stt, valid_pcm_data):
    """Test that process_audio handles exceptions from _process_audio_impl."""

    # Set up mock to raise an exception
    class MockSTTWithException(MockSTT):
        async def _process_audio_impl(self, pcm_data, user_metadata=None):
            raise Exception("Test exception")

    mock_stt_with_exception = MockSTTWithException()

    # Set up error event listener
    error_events = []

    def on_error(error):
        error_events.append(error)

    mock_stt_with_exception.on("error", on_error)

    # Process audio (should not raise exception)
    await mock_stt_with_exception.process_audio(valid_pcm_data)

    # Verify that error event was emitted
    assert len(error_events) == 1
    event = error_events[0]
    assert str(event.error) == "Test exception"
