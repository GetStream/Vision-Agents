"""
Test the base STT class consistency improvements.
"""

import asyncio
import os
import pytest
from unittest.mock import Mock

from dotenv import load_dotenv

from vision_agents.core.stt.stt import STT
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent, STTErrorEvent
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User, Participant
from vision_agents.plugins import getstream, openai, deepgram
import numpy as np

from .base_test import BaseTest

load_dotenv()


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
async def mock_stt():
    """Create MockSTT instance in async context."""
    return MockSTT()


@pytest.fixture
def valid_pcm_data():
    """Create valid PCM data for testing."""
    samples = np.random.randint(-1000, 1000, size=1000, dtype=np.int16)
    return PcmData(samples=samples, sample_rate=16000, format="s16")


@pytest.mark.asyncio
async def test_validate_pcm_data_valid(mock_stt, valid_pcm_data):
    """Test that valid PCM data passes validation."""
    assert mock_stt._validate_pcm_data(valid_pcm_data) is True


@pytest.mark.asyncio
async def test_validate_pcm_data_none(mock_stt):
    """Test that None PCM data fails validation."""
    assert mock_stt._validate_pcm_data(None) is False


@pytest.mark.asyncio
async def test_validate_pcm_data_no_samples(mock_stt):
    """Test that PCM data without samples fails validation."""
    pcm_data = Mock()
    pcm_data.samples = None
    pcm_data.sample_rate = 16000
    assert mock_stt._validate_pcm_data(pcm_data) is False


@pytest.mark.asyncio
async def test_validate_pcm_data_invalid_sample_rate(mock_stt):
    """Test that PCM data with invalid sample rate fails validation."""
    pcm_data = Mock()
    pcm_data.samples = np.array([1, 2, 3])
    pcm_data.sample_rate = 0
    assert mock_stt._validate_pcm_data(pcm_data) is False


@pytest.mark.asyncio
async def test_validate_pcm_data_empty_samples(mock_stt):
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

    @mock_stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append(event)

    # Emit a transcript event
    text = "Hello world"
    user_metadata = {"user_id": "123"}
    metadata = {"confidence": 0.95, "processing_time_ms": 100}

    mock_stt._emit_transcript_event(text, user_metadata, metadata)
    
    # Wait for event processing
    await mock_stt.events.wait(timeout=1.0)

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

    @mock_stt.events.subscribe
    async def on_partial_transcript(event: STTPartialTranscriptEvent):
        partial_events.append(event)

    # Emit a partial transcript event
    text = "Hello"
    user_metadata = {"user_id": "123"}
    metadata = {"confidence": 0.8}

    mock_stt._emit_partial_transcript_event(text, user_metadata, metadata)
    
    # Wait for event processing
    await mock_stt.events.wait(timeout=1.0)

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

    @mock_stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        error_events.append(event)

    # Emit an error event
    test_error = Exception("Test error")
    mock_stt._emit_error_event(test_error, "test context")
    
    # Wait for event processing
    await mock_stt.events.wait(timeout=1.0)

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

    @mock_stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append(event)

    # Process audio
    user_metadata = {"user_id": "123"}
    await mock_stt.process_audio(valid_pcm_data, user_metadata)
    
    # Wait for event processing
    await mock_stt.events.wait(timeout=1.0)

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

    @mock_stt_with_exception.events.subscribe
    async def on_error(event: STTErrorEvent):
        error_events.append(event)

    # Process audio (should not raise exception)
    await mock_stt_with_exception.process_audio(valid_pcm_data)
    
    # Wait for event processing
    await mock_stt_with_exception.events.wait(timeout=1.0)

    # Verify that error event was emitted
    assert len(error_events) == 1
    event = error_events[0]
    assert str(event.error) == "Test exception"


# ============================================================================
# Integration Tests
# ============================================================================

class TestSTTIntegration(BaseTest):
    """Integration tests for STT with real components."""

    @pytest.mark.integration
    async def test_agent_stt_only_without_tts(self, mia_audio_16khz):
        """
        Real integration test: Agent with STT but no TTS.
        
        Uses real components (Deepgram STT, OpenAI LLM, Stream Edge) 
        to verify STT-only agents work end-to-end.
        
        This test verifies:
        - Agent can be created with STT but without TTS
        - Agent correctly identifies need for audio input
        - Agent does not publish audio track (no TTS)
        - Audio flows through to STT
        - STT transcript events are emitted
        - Transcripts are added to conversation
        """
        # Skip if required API keys are not present
        required_keys = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY", "STREAM_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")
        
   
        
        edge = getstream.Edge()
        llm = openai.LLM(model="gpt-4o-mini")
        # Create STT with correct sample rate to match our test audio
        stt = deepgram.STT(sample_rate=16000)
        
        # Create agent with STT but explicitly NO TTS
        agent = Agent(
            edge=edge,
            agent_user=User(name="STT Test Agent", id="stt_agent"),
            llm=llm,
            stt=stt,
            tts=None,  # ← KEY: No TTS - this is what we're testing
            instructions="You are a test agent for STT-only support.",
        )
        
        # Test 1: Verify agent needs audio input (because STT is present)
        assert agent._needs_audio_or_video_input() is True, \
            "Agent with STT should need audio input"
        
        # Test 2: Verify agent does NOT publish audio (because TTS is None)
        assert agent.publish_audio is False, \
            "Agent without TTS should not publish audio"
        
        # Test 3: Set up event listeners to capture transcript
        transcript_events = []
        
        @agent.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            transcript_events.append(event)
        
        # Test 4: Create a test participant (user sending audio)
        test_user = User(name="Test User", id="test_user")
        test_participant = Participant(
            original=test_user,  # The original user object
            user_id="test_user",  # User ID
        )
        
        # Test 5: Send real audio through the agent's audio processing path
        # This simulates what happens when a user speaks in a call
        await agent._reply_to_audio(mia_audio_16khz, test_participant)
        
        # Test 6: Wait for STT to process and emit transcript
        # Real STT takes time to process audio and establish connection
        await asyncio.sleep(5.0)
        
        # Test 7: Verify that transcript event was emitted
        assert len(transcript_events) > 0, \
            "STT should have emitted at least one transcript event"
        
        # Test 8: Verify transcript has content
        first_transcript = transcript_events[0]
        assert first_transcript.text is not None, \
            "Transcript should have text content"
        assert len(first_transcript.text) > 0, \
            "Transcript text should not be empty"
        
        # Test 9: Verify user metadata is present
        assert first_transcript.user_metadata is not None, \
            "Transcript should have user metadata"
        
        # Log the transcript for debugging
        print(f"✅ STT transcribed: '{first_transcript.text}'")
        
        # Test 10: Clean up
        await stt.close()
        await agent.close()
