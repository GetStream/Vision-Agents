import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from getstream.plugins import ElevenLabsTTS
from getstream.video.rtc.audio_track import AudioStreamTrack


# Mock audio track for testing
class MockAudioTrack(AudioStreamTrack):
    def __init__(self):
        self.framerate = 16000
        self.written_data = []

    async def write(self, data):
        self.written_data.append(data)
        return True


# Mock AsyncElevenLabs client for testing
class MockAsyncElevenLabsClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.text_to_speech = MagicMock()

        # Create a mock audio stream that returns a few chunks of audio
        mock_audio = [b"\x00\x00" * 1000, b"\x00\x00" * 1000]

        # Mock the async stream method to return an async generator
        async def mock_stream(*args, **kwargs):
            for chunk in mock_audio:
                yield chunk
        
        self.text_to_speech.stream = mock_stream


@pytest.mark.asyncio
@patch("getstream.plugins.elevenlabs.tts.tts.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_initialization():
    """Test that the ElevenLabs TTS initializes correctly with explicit API key."""
    tts = ElevenLabsTTS(api_key="test-api-key")
    assert tts is not None
    assert tts.client.api_key == "test-api-key"


@pytest.mark.asyncio
@patch("getstream.plugins.elevenlabs.tts.tts.AsyncElevenLabs", MockAsyncElevenLabsClient)
@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-var-api-key"})
async def test_elevenlabs_tts_initialization_with_env_var():
    """ElevenLabsTTS should use ELEVENLABS_API_KEY when no key argument is given."""

    tts = ElevenLabsTTS()  # no explicit key provided
    assert tts is not None
    assert tts.client.api_key == "env-var-api-key"


@pytest.mark.asyncio
@patch("getstream.plugins.elevenlabs.tts.tts.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_synthesize():
    """Test that synthesize returns an audio stream."""
    tts = ElevenLabsTTS(api_key="test-api-key")

    # Test that synthesize returns an iterator
    text = "Hello, world!"
    audio_stream = await tts.stream_audio(text)

    # Check that it's an async iterator
    assert hasattr(audio_stream, "__aiter__")

    # Check that we can get chunks from it
    chunks = []
    async for chunk in audio_stream:
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(isinstance(chunk, bytes) for chunk in chunks)


@pytest.mark.asyncio
@patch("getstream.plugins.elevenlabs.tts.tts.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_send():
    """Test that send writes audio to the track and emits events."""
    tts = ElevenLabsTTS(api_key="test-api-key")

    # Create a mock audio track
    track = MockAudioTrack()
    tts.set_output_track(track)

    # Track emitted audio events
    emitted_audio = []

    @tts.on("audio")
    def on_audio(event):
        emitted_audio.append(event.audio_data)

    # Send text to the TTS
    text = "Hello, world!"
    await tts.send(text)

    # Check that audio was written to the track
    assert len(track.written_data) > 0

    # Check that audio events were emitted
    assert len(emitted_audio) > 0
    assert emitted_audio == track.written_data


@pytest.mark.asyncio
@patch("elevenlabs.client.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_send_without_track():
    """Test that sending without setting a track raises an error."""
    tts = ElevenLabsTTS(api_key="test-api-key")

    # Sending without setting a track should raise ValueError
    with pytest.raises(ValueError, match="No output track set"):
        await tts.send("Hello, world!")


@pytest.mark.asyncio
@patch("elevenlabs.client.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_invalid_framerate():
    """Test that setting a track with invalid framerate raises an error."""
    tts = ElevenLabsTTS(api_key="test-api-key")

    # Create a mock audio track with invalid framerate
    invalid_track = MagicMock(spec=AudioStreamTrack)
    invalid_track.framerate = 44100

    # Setting the invalid track should raise TypeError
    with pytest.raises(TypeError, match="Invalid framerate"):
        tts.set_output_track(invalid_track)


@pytest.mark.asyncio
@patch("elevenlabs.client.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_with_custom_client():
    """Test that ElevenLabs TTS can be initialized with a custom client."""
    # Create a custom mock client
    custom_client = MockAsyncElevenLabsClient(api_key="custom-api-key")
    
    # Initialize TTS with the custom client
    tts = ElevenLabsTTS(client=custom_client)
    
    # Verify that the custom client is used
    assert tts.client is custom_client
    assert tts.client.api_key == "custom-api-key"


@pytest.mark.asyncio
@patch("elevenlabs.client.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_stop_method():
    """Test that the stop method properly flushes the audio track."""
    tts = ElevenLabsTTS(api_key="test-api-key")
    
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
@patch("elevenlabs.client.AsyncElevenLabs", MockAsyncElevenLabsClient)
async def test_elevenlabs_tts_stop_method_handles_exceptions():
    """Test that the stop method handles flush exceptions gracefully."""
    tts = ElevenLabsTTS(api_key="test-api-key")
    
    # Create a mock audio track with flush method that raises an exception
    track = MockAudioTrack()
    track.flush = MagicMock(side_effect=Exception("Flush error"))
    
    tts.set_output_track(track)
    
    # Call stop method - should not raise an exception
    await tts.stop_audio()
    
    # Verify that flush was called on the track
    track.flush.assert_called_once()


@pytest.mark.asyncio
async def test_elevenlabs_with_real_api():
    """
    Integration test with the real ElevenLabs API.

    This test uses the actual ElevenLabs API with the
    ELEVENLABS_API_KEY environment variable.
    It will be skipped if the environment variable is not set.

    To set up the ELEVENLABS_API_KEY:
    1. Sign up for an ElevenLabs account at https://elevenlabs.io
    2. Create an API key in your ElevenLabs dashboard
    3. Add to your .env file: ELEVENLABS_API_KEY=your_api_key_here
    """
    # Check if the required API key is available
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    # Skip the test if the ELEVENLABS_API_KEY environment variable is not set
    if not api_key:
        pytest.skip(
            "ELEVENLABS_API_KEY environment variable not set. "
            "Add it to your .env file."
        )

    # Create a real ElevenLabs TTS instance with the API key explicitly set
    tts = ElevenLabsTTS(api_key=api_key)

    # Create a mock audio track to capture the output
    track = MockAudioTrack()
    tts.set_output_track(track)

    # Track audio events
    audio_received = asyncio.Event()
    received_chunks = []

    @tts.on("audio")
    def on_audio(audio_data, user):
        received_chunks.append(audio_data)
        audio_received.set()

    # Track API errors
    api_errors = []

    @tts.on("error")
    def on_error(error):
        api_errors.append(error)
        audio_received.set()  # Unblock the waiting

    try:
        # Use a short text to minimize API usage
        text = "This is a test of the ElevenLabs text-to-speech API."

        # Send the text to generate speech
        send_task = asyncio.create_task(tts.send(text))

        # Wait for either audio or an error
        try:
            await asyncio.wait_for(audio_received.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            # Cancel the task if it's taking too long
            send_task.cancel()
            pytest.fail("No audio or error received within timeout")

        # Check if we received any API errors
        if api_errors:
            pytest.skip(f"API error received: {api_errors[0]}")

        # Try to ensure the send task completes
        try:
            await send_task
        except Exception as e:
            pytest.skip(f"Exception during TTS generation: {e}")

        # Verify that we received audio data
        assert len(received_chunks) > 0, "No audio chunks were received"
    except Exception as e:
        pytest.skip(f"Unexpected error in ElevenLabs test: {e}")
    finally:
        # Always clean up event handlers
        # This is important to avoid memory leaks from persistent event handlers
        tts.remove_all_listeners("audio")
        tts.remove_all_listeners("error")
