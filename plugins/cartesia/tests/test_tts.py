import os
import asyncio
from unittest.mock import patch, MagicMock

import pytest

from getstream.plugins import CartesiaTTS
from getstream.video.rtc.audio_track import AudioStreamTrack


############################
# Test utilities & fixtures
############################


# A simple async iterator yielding a predefined list of byte chunks
class _AsyncBytesIterator:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration


# Mock implementation of the Cartesia SDK
class MockAsyncCartesia:
    """Light-weight stub mimicking the public surface used by CartesiaTTS."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.tts = MagicMock()

        # Pre-generate two fake PCM byte chunks (2000 samples each)
        mock_audio = [b"\x00\x00" * 1000, b"\x00\x00" * 1000]

        self.tts.bytes = MagicMock(
            side_effect=lambda *_, **__: _AsyncBytesIterator(mock_audio.copy())
        )


# Re-usable audio track stub
class MockAudioTrack(AudioStreamTrack):
    def __init__(self, framerate: int = 16000):
        self.framerate = framerate
        self.written_data = []

    async def write(self, data: bytes):
        self.written_data.append(data)
        return True


############################
# Unit tests
############################


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
async def test_cartesia_tts_initialization():
    """CartesiaTTS should instantiate and store the provided api_key."""
    tts = CartesiaTTS(api_key="test-api-key")
    assert tts is not None
    assert tts.client.api_key == "test-api-key"


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
@patch.dict(os.environ, {"CARTESIA_API_KEY": "env-var-api-key"})
async def test_cartesia_tts_initialization_with_env_var():
    """When no api_key arg is supplied CartesiaTTS should read CARTESIA_API_KEY."""
    tts = CartesiaTTS()  # no explicit key
    assert tts.client.api_key == "env-var-api-key"


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
async def test_cartesia_synthesize_returns_async_iterator():
    """synthesize() should yield an async iterator of PCM byte chunks."""
    tts = CartesiaTTS(api_key="test")
    stream = await tts.stream_audio("Hello")

    # Must be async iterable
    assert hasattr(stream, "__aiter__")

    collected = []
    async for chunk in stream:
        collected.append(chunk)

    assert len(collected) == 2
    assert all(isinstance(c, (bytes, bytearray)) for c in collected)


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
async def test_cartesia_send_writes_to_track_and_emits_event():
    tts = CartesiaTTS(api_key="test")
    track = MockAudioTrack()
    tts.set_output_track(track)

    received = []

    @tts.on("audio")
    def _on_audio(event):
        received.append(event.audio_data)

    await tts.send("Hello world")

    # Data should be forwarded to track
    assert len(track.written_data) == 2
    assert track.written_data == received


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
async def test_cartesia_invalid_framerate_raises():
    tts = CartesiaTTS(api_key="test")
    bad_track = MockAudioTrack(framerate=44100)

    with pytest.raises(TypeError, match="framerate 44100"):
        tts.set_output_track(bad_track)


@pytest.mark.asyncio
@patch("getstream.plugins.cartesia.tts.tts.AsyncCartesia", MockAsyncCartesia)
async def test_cartesia_send_without_track_raises():
    tts = CartesiaTTS(api_key="test")

    with pytest.raises(ValueError, match="No output track set"):
        await tts.send("Hello, world!")


############################
# Optional integration test
############################


@pytest.mark.asyncio
async def test_cartesia_with_real_api():
    """Integration test against Cartesia cloud – skipped if CARTESIA_API_KEY unset."""
    # Skip if the Cartesia SDK is not available
    try:
        import cartesia  # noqa: F401
    except ImportError:
        pytest.skip("cartesia package not installed – skipping live API test.")

    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        pytest.skip("CARTESIA_API_KEY env var not set – skipping live API test.")

    tts = CartesiaTTS(api_key=api_key)
    track = MockAudioTrack()
    tts.set_output_track(track)

    # Wait until we either receive audio or hit a timeout
    audio_received = asyncio.Event()

    @tts.on("audio")
    def _on_audio(event):
        audio_received.set()

    try:
        await asyncio.wait_for(tts.send("Hello from Cartesia!"), timeout=30)
    except asyncio.TimeoutError:
        pytest.fail("Timed out waiting for Cartesia audio response")

    assert len(track.written_data) > 0, "No audio data received from Cartesia"
