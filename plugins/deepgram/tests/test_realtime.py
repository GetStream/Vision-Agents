import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from getstream.plugins import DeepgramSTT
from getstream.video.rtc.track_util import PcmData


# Mock the Deepgram client for testing real-time transcript emission
class MockDeepgramConnection:
    def __init__(self):
        self.event_handlers = {}
        self.sent_data = []
        self.finished = False

    def on(self, event, handler):
        """Register event handlers"""
        self.event_handlers[event] = handler
        return handler

    def send(self, data):
        """Mock send data"""
        self.sent_data.append(data)
        return True

    def finish(self):
        """Close the connection"""
        self.finished = True

    def start(self, options):
        """Start the connection"""
        pass

    def emit_transcript(self, text, is_final=True):
        """Helper to emit a transcript event"""
        from deepgram import LiveTranscriptionEvents

        if LiveTranscriptionEvents.Transcript in self.event_handlers:
            # Create a mock result
            transcript_data = {
                "is_final": is_final,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": text,
                            "confidence": 0.95,
                            "words": [
                                {
                                    "word": word,
                                    "start": i * 0.5,
                                    "end": (i + 1) * 0.5,
                                    "confidence": 0.95,
                                }
                                for i, word in enumerate(text.split())
                            ],
                        }
                    ]
                },
                "channel_index": 0,
            }

            # Convert to JSON string for the handler
            transcript_json = json.dumps(transcript_data)

            # Call the handler with the connection and the result
            self.event_handlers[LiveTranscriptionEvents.Transcript](
                self, result=transcript_json
            )

    def emit_error(self, error_message):
        """Helper to emit an error event"""
        from deepgram import LiveTranscriptionEvents

        if LiveTranscriptionEvents.Error in self.event_handlers:
            # Call the error handler
            self.event_handlers[LiveTranscriptionEvents.Error](
                self, error=error_message
            )


class MockDeepgramClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.listen = MagicMock()
        self.listen.websocket = MagicMock()
        self.listen.websocket.v = MagicMock(return_value=MockDeepgramConnection())


@pytest.mark.asyncio
@patch("getstream.plugins.deepgram.stt.stt.DeepgramClient", MockDeepgramClient)
async def test_real_time_transcript_emission():
    """
    Test that transcripts are emitted in real-time without needing a second audio chunk.

    This test verifies that:
    1. A transcript can be emitted immediately after receiving it from the server
    2. Events are properly emitted to listeners through the hybrid approach
    3. Both collection and immediate emission work correctly
    """
    # Create the Deepgram STT instance
    stt = DeepgramSTT(api_key="test-api-key")

    # Collection for events
    transcript_events = []
    partial_transcript_events = []
    error_events = []

    # Register event handlers
    @stt.on("transcript")
    def on_transcript(event):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    @stt.on("partial_transcript")
    def on_partial_transcript(event):
        partial_transcript_events.append((event.text, event.user_metadata, {"is_final": False}))

    @stt.on("error")
    def on_error(event):
        error_events.append(event.error)

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Immediately trigger a transcript from the server
    stt.dg_connection.emit_transcript("hello world", is_final=True)

    # With the new hybrid approach, events should be emitted immediately
    # Wait a very small amount to allow synchronous execution to complete
    await asyncio.sleep(0.01)

    # Check that we received the transcript event
    assert len(transcript_events) == 1, "Expected 1 transcript event"
    assert transcript_events[0][0] == "hello world", "Incorrect transcript text"
    assert transcript_events[0][2]["is_final"], "Transcript should be marked as final"

    # No errors should have occurred
    assert len(error_events) == 0, f"Unexpected errors: {error_events}"

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("getstream.plugins.deepgram.stt.stt.DeepgramClient", MockDeepgramClient)
async def test_real_time_partial_transcript_emission():
    """
    Test that partial transcripts are emitted in real-time.
    """
    # Create the Deepgram STT instance
    stt = DeepgramSTT(api_key="test-api-key")

    # Collection for events
    transcript_events = []
    partial_transcript_events = []

    # Register event handlers
    @stt.on("transcript")
    def on_transcript(event):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    @stt.on("partial_transcript")
    def on_partial_transcript(event):
        partial_transcript_events.append((event.text, event.user_metadata, {"is_final": False}))

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Emit a partial transcript
    stt.dg_connection.emit_transcript("typing in prog", is_final=False)

    # With immediate emission, we don't need to wait long
    await asyncio.sleep(0.01)

    # Emit another partial transcript
    stt.dg_connection.emit_transcript("typing in progress", is_final=False)

    # Wait again
    await asyncio.sleep(0.01)

    # Emit the final transcript
    stt.dg_connection.emit_transcript("typing in progress complete", is_final=True)

    # Wait a small amount of time for processing
    await asyncio.sleep(0.01)

    # Check that we received the partial transcript events
    assert len(partial_transcript_events) == 2, "Expected 2 partial transcript events"
    assert (
        partial_transcript_events[0][0] == "typing in prog"
    ), "Incorrect partial transcript text"
    assert (
        partial_transcript_events[1][0] == "typing in progress"
    ), "Incorrect partial transcript text"

    # Check that we received the final transcript event
    assert len(transcript_events) == 1, "Expected 1 final transcript event"
    assert (
        transcript_events[0][0] == "typing in progress complete"
    ), "Incorrect final transcript text"

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("getstream.plugins.deepgram.stt.stt.DeepgramClient", MockDeepgramClient)
async def test_real_time_error_emission():
    """
    Test that errors are emitted in real-time.
    """
    # Create the Deepgram STT instance
    stt = DeepgramSTT(api_key="test-api-key")

    # Collection for events
    error_events = []

    # Register event handler
    @stt.on("error")
    def on_error(error):
        error_events.append(error)

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Trigger an error by emitting it directly from the mock connection
    stt.dg_connection.emit_error("Test error message")

    # With immediate emission, error should be available quickly
    await asyncio.sleep(0.01)

    # Check that we received the error event
    assert len(error_events) == 1, "Expected 1 error event"
    assert "Test error message" in str(error_events[0]), "Incorrect error message"

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("getstream.plugins.deepgram.stt.stt.DeepgramClient", MockDeepgramClient)
async def test_close_cleanup():
    """
    Test that the STT service is properly closed and cleaned up.
    """
    # Create the Deepgram STT instance
    stt = DeepgramSTT(api_key="test-api-key")

    # Verify the service is running
    assert not stt._is_closed, "Service should be running after initialization"
    assert stt._running, "Running flag should be True"

    # Close the STT service
    await stt.close()

    # Verify the service has been stopped
    assert stt._is_closed, "Service should be closed"
    assert not stt._running, "Running flag should be False"
    assert stt.keep_alive_task is None, "Keep-alive task should be None after close"

    # Try to emit a transcript after closing (should not crash)
    transcript_events = []

    @stt.on("transcript")
    def on_transcript(event):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    # Process audio after close should be ignored
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    result = await stt._process_audio_impl(pcm_data)

    # Should return None since service is closed
    assert result is None, "Should return None when closed"

    # No events should have been received
    assert len(transcript_events) == 0, "Should not receive events after close"


@pytest.mark.asyncio
@patch("getstream.plugins.deepgram.stt.stt.DeepgramClient", MockDeepgramClient)
async def test_asynchronous_mode_behavior():
    """
    Test that Deepgram operates in asynchronous mode:
    1. Events are emitted immediately when they arrive
    2. _process_audio_impl always returns None (no result collection)
    """
    # Create the Deepgram STT instance
    stt = DeepgramSTT(api_key="test-api-key")

    # Collection for events
    transcript_events = []

    # Register event handler
    @stt.on("transcript")
    def on_transcript(event):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    # Send some audio data
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Trigger a transcript
    stt.dg_connection.emit_transcript("test message", is_final=True)

    # Event should be emitted immediately
    await asyncio.sleep(0.01)
    assert len(transcript_events) == 1, "Event should be emitted immediately"

    # _process_audio_impl should always return None in asynchronous mode
    results = await stt._process_audio_impl(pcm_data, {"user_id": "test"})

    # Should always return None for asynchronous mode
    assert results is None, "Asynchronous mode should always return None"

    # Cleanup
    await stt.close()
