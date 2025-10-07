from unittest.mock import patch, MagicMock
import asyncio

import numpy as np
import pytest

from vision_agents.plugins import kokoro
from vision_agents.core.tts.events import TTSAudioEvent
from getstream.video.rtc.audio_track import AudioStreamTrack


############################
# Test utilities & fixtures
############################


class MockAudioTrack(AudioStreamTrack):
    def __init__(self, framerate: int = 24_000):
        self.framerate = framerate
        self.written_data: list[bytes] = []

    async def write(self, data: bytes):
        self.written_data.append(data)
        return True


class _MockKPipeline:  # noqa: D401
    """Very small stub that mimics ``kokoro.KPipeline`` callable behaviour."""

    def __init__(self, *_, **__):
        pass

    def __call__(self, text, *, voice, speed, split_pattern):  # noqa: D401
        # Produce two mini 20 ms chunks of silence at 24 kHz
        blank = np.zeros(480, dtype=np.float32)  # 480 samples @ 24 kHz = 20 ms
        for _ in range(2):
            yield text, voice, blank


############################
# Unit-tests
############################


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_tts_initialization():
    tts = kokoro.TTS()
    assert tts is not None


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_synthesize_returns_iterator():
    tts = kokoro.TTS()
    stream = await tts.stream_audio("Hello")

    # Should be an async iterator (list of bytes)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_send_writes_and_emits():
    tts = kokoro.TTS()
    track = MockAudioTrack()
    tts.set_output_track(track)

    received = []

    @tts.events.subscribe
    async def _on_audio(event: TTSAudioEvent):
        # Extract the audio data from the event
        if hasattr(event, "audio_data") and event.audio_data is not None:
            received.append(event.audio_data)
        else:
            received.append(b"")

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    await tts.send("Hello world")

    # Allow events to be processed
    await asyncio.sleep(0.01)

    assert len(track.written_data) == 2
    assert track.written_data == received


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_invalid_framerate():
    tts = kokoro.TTS()
    bad_track = MockAudioTrack(framerate=16_000)

    with pytest.raises(TypeError):
        tts.set_output_track(bad_track)


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_send_without_track():
    tts = kokoro.TTS()
    with pytest.raises(ValueError):
        await tts.send("Hi")


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_tts_with_custom_client():
    """Test that Kokoro TTS can be initialized with a custom client."""
    # Create a custom mock client
    custom_client = _MockKPipeline()

    # Initialize TTS with the custom client
    tts = kokoro.TTS(client=custom_client)

    # Verify that the custom client is used
    assert tts.client is custom_client


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_tts_stop_method():
    """Test that the stop method properly flushes the audio track."""
    tts = kokoro.TTS()

    # Create a mock audio track with flush method
    track = MockAudioTrack()
    track.flush = MagicMock(return_value=asyncio.Future())
    track.flush.return_value.set_result(None)

    tts.set_output_track(track)

    # Call stop method
    await tts.stop_audio()

    # Verify that flush was called on the track
    track.flush.assert_called_once()


@pytest.mark.asyncio
@patch("vision_agents.plugins.kokoro.tts.KPipeline", _MockKPipeline)
async def test_kokoro_tts_stop_method_handles_exceptions():
    """Test that the stop method handles flush exceptions gracefully."""
    tts = kokoro.TTS()

    # Create a mock audio track with flush method that raises an exception
    track = MockAudioTrack()
    track.flush = MagicMock(side_effect=Exception("Flush error"))

    tts.set_output_track(track)

    # Call stop method - should not raise an exception
    await tts.stop_audio()

    # Verify that flush was called on the track
    track.flush.assert_called_once()
