import asyncio
import os
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1ControlMessage,
    ListenV1ResultsEvent,
)
from deepgram.extensions.types.sockets.listen_v1_results_event import (
    ListenV1ModelInfo,
    ListenV1ResultsMetadata,
)
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.stt.events import (
    STTErrorEvent,
    STTPartialTranscriptEvent,
    STTTranscriptEvent,
)
from vision_agents.plugins import deepgram

from plugins.plugin_test_utils import get_audio_asset, get_json_metadata


class MockDeepgramV5Connection:
    def __init__(self, **options):
        self.event_handlers = {}
        self.sent_media = []
        self.sent_control = []
        self.finished = False
        self.closed = False
        self.options = options

    def on(self, event, handler):
        """Register event handlers"""
        self.event_handlers[event] = handler
        return handler

    async def send_control(self, data):
        """Mock send control messages"""
        self.sent_control.append(data)
        return True

    async def send_media(self, data):
        """Mock send media data"""
        self.sent_media.append(data)
        return True

    async def start_listening(self):
        """
        Start listening for the events.
        Note that this method blocks itself and the actual events must be sent using "emit_***()" methods.
        """
        while not self.closed:
            await asyncio.sleep(0.1)

    def close(self):
        """Close the connection"""
        self.closed = True

    async def emit_result(self, text: str, is_final=True):
        """Helper to emit a transcript event"""

        if EventType.MESSAGE in self.event_handlers:
            # Create a mock result

            mock_result = ListenV1ResultsEvent(
                **{
                    "type": "Results",
                    "channel_index": [0],
                    "channel": {
                        "alternatives": [
                            {"transcript": text, "confidence": 0.9, "words": []}
                        ]
                    },
                    "is_final": is_final,
                    "duration": 0.0,
                    "start": 0.0,
                    "metadata": ListenV1ResultsMetadata(
                        request_id=str(uuid.uuid4()),
                        model_uuid=str(uuid.uuid4()),
                        model_info=ListenV1ModelInfo(
                            name="test", version="test", arch="test"
                        ),
                    ),
                }
            )
            await self.event_handlers[EventType.MESSAGE](mock_result)

    async def emit_message(self, message: dict):
        """
        Helper to emit any message event.
        """
        if EventType.MESSAGE in self.event_handlers:
            # Create a mock result
            await self.event_handlers[EventType.MESSAGE](message)

    async def emit_error(self, error_message):
        """Helper to emit an error event"""

        if EventType.ERROR in self.event_handlers:
            error_obj = Exception(error_message)
            await self.event_handlers[EventType.ERROR](error_obj)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockAsyncDeepgramClient:
    def __init__(self, api_key=None, config=None):
        self.api_key = api_key
        self.config = config
        self.listen = MagicMock()
        self.listen.v1 = MagicMock()
        self.listen.v1.connect = MockDeepgramV5Connection


@pytest.fixture
def mia_mp3_path():
    """Return the path to the mia.mp3 test file."""
    return get_audio_asset("mia.mp3")


@pytest.fixture
def mia_json_path():
    """Return the path to the mia.json metadata file."""
    return get_audio_asset("mia.json")


@pytest.fixture
def mia_metadata():
    """Load the mia.json metadata."""
    return get_json_metadata("mia.json")


@pytest.fixture
def audio_data(mia_mp3_path):
    """Load and prepare the audio data for testing."""
    import numpy as np
    import torch
    import torchaudio
    from scipy import signal

    # Load the mp3 file
    waveform, original_sample_rate = torchaudio.load(mia_mp3_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to numpy array
    data = waveform.numpy().squeeze()

    # Resample to 16kHz if needed (Deepgram's preferred rate)
    target_sample_rate = 48000
    if original_sample_rate != target_sample_rate:
        number_of_samples = round(
            len(data) * float(target_sample_rate) / original_sample_rate
        )
        data = signal.resample(data, number_of_samples)

    # Normalize and convert to int16
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / max(abs(data.max()), abs(data.min()))

    # Convert to int16 PCM
    pcm_samples = (data * 32767.0).astype(np.int16)

    # Return PCM data with the resampled rate
    return PcmData(samples=pcm_samples, sample_rate=target_sample_rate, format="s16")


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_stt_initialization():
    """Test that the Deepgram STT initializes correctly with explicit API key."""
    stt = deepgram.STT(api_key="test-api-key")
    assert stt is not None
    assert stt.deepgram.api_key == "test-api-key"
    await stt.close()


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
@patch.dict(os.environ, {"DEEPGRAM_API_KEY": "env-var-api-key"})
async def test_deepgram_stt_initialization_with_env_var():
    """Test that the Deepgram STT initializes correctly when DEEPGRAM_API_KEY is set."""

    # Initialize without providing an API key – implementation should fall back to env var
    stt = deepgram.STT()
    assert stt is not None
    assert stt.deepgram.api_key == "env-var-api-key"


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_stt_transcript_events(mia_metadata):
    """Test that the Deepgram STT emits transcript events correctly."""
    stt = deepgram.STT()

    # Track events
    transcripts = []

    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_meta: dict[str, bool] = {"is_final": True}
        transcripts.append((event.text, event.user_metadata, transcript_meta))

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Emit a transcript using the mock connection
    await stt.dg_connection.emit_result("This is a final transcript", is_final=True)

    # Process some audio to ensure the connection is active
    audio_data = PcmData(samples=b"\x00\x00" * 1000, sample_rate=48000, format="s16")
    await stt.process_audio(audio_data)

    # Give the async event handlers time to process
    await asyncio.sleep(0.05)

    # Cleanup
    await stt.close()

    # Check that the events were received
    assert len(transcripts) > 0
    assert "This is a final transcript" in transcripts[0][0]


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_process_audio(audio_data, mia_metadata):
    """Test that the Deepgram STT can process audio data."""
    stt = deepgram.STT(api_key="test-api-key")

    # Track the audio data that was sent
    sent_audio_bytes = []

    asyncio.create_task(stt.start())
    await stt.started()

    # Create a custom send method to track sent data
    async def mock_send(data):
        sent_audio_bytes.append(data)
        return True

    # Replace the send method on the connection to track sent data
    stt.dg_connection.send_media = mock_send

    # Process audio - note we're using the implementation method
    await stt._process_audio_impl(audio_data, None)

    # Cleanup
    await stt.close()

    # Check that audio was sent
    assert len(sent_audio_bytes) > 0, "No audio data was sent to Deepgram"


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_end_to_end(audio_data, mia_metadata):
    """Test the entire processing pipeline for Deepgram STT."""
    stt = deepgram.STT(api_key="test-api-key")

    # Track events
    transcripts = []
    errors = []

    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_meta: dict[str, bool] = {"is_final": True}
        transcripts.append((event.text, event.user_metadata, transcript_meta))

    @stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        errors.append(event.error)

    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Emit a transcript using the mock connection
    await stt.dg_connection.emit_result("This is the final result")

    # Process the audio
    await stt.process_audio(audio_data)

    # Give the async event handlers time to process
    await asyncio.sleep(0.05)

    # Check that we received the expected events
    assert len(errors) == 0
    assert len(transcripts) > 0
    assert "This is the final result" in transcripts[0][0]

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_keep_alive_mechanism():
    """Test that the keep-alive mechanism works."""
    # Create a Deepgram STT instance with a short keep-alive interval
    stt = deepgram.STT(api_key="test-api-key", keep_alive_interval=0.1)

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    connection = stt.dg_connection

    # Wait long enough for at least one keep-alive message to be sent
    await asyncio.sleep(0.2)

    # Cleanup
    await stt.close()

    # The audio silence must be sent
    assert len(connection.sent_media) > 0, "No audio silence messages sent after wait"
    # The keep-alive messages must be sent too
    assert len(connection.sent_control) > 0, "No keep-alive messages sent after wait"


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_keep_alive_after_audio():
    """Test that keep-alive messages are sent after audio is processed."""
    # Create a Deepgram STT instance with a short keep-alive interval
    stt = deepgram.STT(api_key="test-api-key", keep_alive_interval=0.1)

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()
    connection = stt.dg_connection

    # Create some empty audio data
    audio_data = PcmData(samples=b"\x00\x00" * 1000, sample_rate=48000, format="s16")

    # Process the audio - this should set the last_activity_time
    await stt.process_audio(audio_data)

    # Wait longer than the keep-alive interval
    await asyncio.sleep(0.2)

    # Cleanup
    await stt.close()

    # We should see that keep-alive messages have been sent
    assert len(connection.sent_media) > 0, (
        "No keep-alive messages sent after audio processing"
    )
    assert len(connection.sent_control) > 0, (
        "No keep-alive messages sent after audio processing"
    )


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_start_idempotent():
    """Test that the start() method is idempotent."""
    # Create a Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()
    # Store the connection object
    conn1 = stt.dg_connection
    assert conn1

    # Try to connect for the second time
    asyncio.create_task(stt.start())
    await stt.started()
    # Ensure that the connection object remains intact
    assert conn1 is stt.dg_connection


@pytest.mark.asyncio
@patch(
    "vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient
)
async def test_deepgram_close_message():
    """Test that the finish message is sent when the connection is closed."""
    # Create a Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    connection = stt.dg_connection

    # Mock the original send_control method to catch the "close" message
    close_message = None

    async def mock_send_control(message: ListenV1ControlMessage):
        nonlocal close_message
        close_message = message

    connection.send_control = mock_send_control

    # Close the STT service
    await stt.close()

    # The "CloseStream" control message must be sent in the end
    assert close_message is not None, (
        "The connection close package was not sent on close"
    )
    assert close_message.type == "CloseStream"

@pytest.mark.asyncio
@patch("vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient)
async def test_real_time_transcript_emission():
    """
    Test that transcripts are emitted in real-time without needing a second audio chunk.

    This test verifies that:
    1. A transcript can be emitted immediately after receiving it from the server
    2. Events are properly emitted to listeners through the hybrid approach
    3. Both collection and immediate emission work correctly
    """
    # Create the Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Collection for events
    transcript_events = []
    partial_transcript_events = []
    error_events = []

    # Register event handlers
    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    @stt.events.subscribe
    async def on_partial_transcript(event: STTPartialTranscriptEvent):
        partial_transcript_events.append(
            (event.text, event.user_metadata, {"is_final": False})
        )

    @stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        error_events.append(event.error)

    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Immediately trigger a transcript from the server
    await stt.dg_connection.emit_result("hello world", is_final=True)

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
@patch("vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient)
async def test_real_time_partial_transcript_emission():
    """
    Test that partial transcripts are emitted in real-time.
    """
    # Create the Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Collection for events
    transcript_events = []
    partial_transcript_events = []

    # Register event handlers
    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    @stt.events.subscribe
    async def on_partial_transcript(event: STTPartialTranscriptEvent):
        partial_transcript_events.append(
            (event.text, event.user_metadata, {"is_final": False})
        )

    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Emit a partial transcript
    await stt.dg_connection.emit_result("typing in prog", is_final=False)

    # With immediate emission, we don't need to wait long
    await asyncio.sleep(0.01)

    # Emit another partial transcript
    await stt.dg_connection.emit_result("typing in progress", is_final=False)

    # Wait again
    await asyncio.sleep(0.01)

    # Emit the final transcript
    await stt.dg_connection.emit_result("typing in progress complete", is_final=True)

    # Wait a small amount of time for processing
    await asyncio.sleep(0.01)

    # Check that we received the partial transcript events
    assert len(partial_transcript_events) == 2, "Expected 2 partial transcript events"
    assert partial_transcript_events[0][0] == "typing in prog", (
        "Incorrect partial transcript text"
    )
    assert partial_transcript_events[1][0] == "typing in progress", (
        "Incorrect partial transcript text"
    )

    # Check that we received the final transcript event
    assert len(transcript_events) == 1, "Expected 1 final transcript event"
    assert transcript_events[0][0] == "typing in progress complete", (
        "Incorrect final transcript text"
    )

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient)
async def test_real_time_error_emission():
    """
    Test that errors are emitted in real-time.
    """
    # Create the Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Collection for events
    error_events = []

    # Register event handler
    @stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        error_events.append(event)

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Send some audio data to ensure the connection is active
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Trigger an error by emitting it directly from the mock connection
    await stt.dg_connection.emit_error("Test error message")

    # With immediate emission, error should be available quickly
    await asyncio.sleep(0.01)

    # Check that we received the error event
    assert len(error_events) == 1, "Expected 1 error event"
    assert "Test error message" in str(error_events[0]), "Incorrect error message"

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient)
async def test_close_cleanup():
    """
    Test that the STT service is properly closed and cleaned up.
    """
    # Create the Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Verify the service is running
    assert not stt._is_closed, "Service should be running after initialization"

    # Close the STT service
    await stt.close()

    # Verify the service has been stopped
    assert stt._is_closed, "Service should be closed"

    # Try to emit a transcript after closing (should not crash)
    transcript_events = []

    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Process audio after close should be ignored
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    result = await stt._process_audio_impl(pcm_data)

    # Should return None since service is closed
    assert result is None, "Should return None when closed"

    # No events should have been received
    assert len(transcript_events) == 0, "Should not receive events after close"


@pytest.mark.asyncio
@patch("vision_agents.plugins.deepgram.stt.AsyncDeepgramClient", MockAsyncDeepgramClient)
async def test_asynchronous_mode_behavior():
    """
    Test that Deepgram operates in asynchronous mode:
    1. Events are emitted immediately when they arrive
    2. _process_audio_impl always returns None (no result collection)
    """
    # Create the Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    asyncio.create_task(stt.start())
    await stt.started()

    # Collection for events
    transcript_events = []

    # Register event handler
    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcript_events.append((event.text, event.user_metadata, {"is_final": True}))

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Send some audio data
    pcm_data = PcmData(samples=b"\x00\x00" * 800, sample_rate=48000, format="s16")
    await stt.process_audio(pcm_data)

    # Trigger a transcript
    await stt.dg_connection.emit_result("test message", is_final=True)

    # Event should be emitted immediately
    await asyncio.sleep(0.01)
    assert len(transcript_events) == 1, "Event should be emitted immediately"

    # _process_audio_impl should always return None in asynchronous mode
    results = await stt._process_audio_impl(pcm_data, {"user_id": "test"})

    # Should always return None for asynchronous mode
    assert results is None, "Asynchronous mode should always return None"

    # Cleanup
    await stt.close()



@pytest.mark.integration
@pytest.mark.asyncio
async def test_deepgram_with_real_api_keep_alive():
    """
    Test Deepgram STT with real API and keep-alive functionality.

    This test uses a real Deepgram API connection to test keep-alive behavior.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set")

    # Use the mia.mp3 audio asset
    mia_audio_path = get_audio_asset("mia.mp3")

    try:
        # Load the audio file
        import soundfile as sf

        audio_data, original_sample_rate = sf.read(mia_audio_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 48kHz if needed (Deepgram's default)
        target_sample_rate = 48000
        if original_sample_rate != target_sample_rate:
            from getstream.audio.utils import resample_audio

            audio_data = resample_audio(
                audio_data, original_sample_rate, target_sample_rate
            )

        # Convert to int16 PCM
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

        pcm_samples = (audio_data * 32767.0).astype(np.int16)

        # Create PCM data
        pcm_data = PcmData(
            samples=pcm_samples, sample_rate=target_sample_rate, format="s16"
        )

    except Exception as e:
        pytest.skip(f"Could not load test audio: {e}")

    stt = deepgram.STT(api_key=api_key, keep_alive_interval=5.0)

    # Track events
    transcripts = []
    partial_transcripts = []
    errors = []

    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        transcripts.append((event.text, event.user_metadata, {"is_final": True}))

    @stt.events.subscribe
    async def on_partial_transcript(event: STTPartialTranscriptEvent):
        partial_transcripts.append(
            (event.text, event.user_metadata, {"is_final": False})
        )

    @stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        errors.append(event.error)

    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)

    timeout = 6.0
    try:
        print(f"Waiting for keep-alive timeout ({timeout} seconds)...")

        # Wait longer than keep-alive interval to test the mechanism
        await asyncio.sleep(timeout)

        # Process audio to trigger keep-alive
        await stt.process_audio(pcm_data)

        # Wait for processing
        await asyncio.sleep(2.0)

        print("STT closed successfully")

    finally:
        await stt.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deepgram_real_integration():
    """
    Integration test with the real Deepgram API using the mia.mp3 test file.

    This test processes the mia.mp3 audio file and validates the transcription results
    against expected content. This is crucial for ensuring the Deepgram plugin
    actually works with real API calls.

    This test will be skipped if DEEPGRAM_API_KEY is not set.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set - cannot run real integration test")

    # Load test audio and metadata
    try:
        mia_audio_path = get_audio_asset("mia.mp3")
        mia_metadata = get_json_metadata("mia.json")
    except Exception as e:
        pytest.skip(f"Could not load test assets: {e}")

    # Load and prepare audio data
    try:
        import soundfile as sf

        audio_data, original_sample_rate = sf.read(mia_audio_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 48kHz (Deepgram's preferred rate)
        target_sample_rate = 48000
        if original_sample_rate != target_sample_rate:
            from getstream.audio.utils import resample_audio

            audio_data = resample_audio(
                audio_data, original_sample_rate, target_sample_rate
            )

        # Convert to int16 PCM
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

        pcm_samples = (audio_data * 32767.0).astype(np.int16)

        print(
            f"Testing with mia.mp3: {len(pcm_samples)} samples at {target_sample_rate}Hz"
        )
        print(f"Audio duration: {len(pcm_samples) / target_sample_rate:.2f} seconds")
        print(f"Audio range: {pcm_samples.min()} to {pcm_samples.max()}")

    except Exception as e:
        pytest.skip(f"Could not process audio file: {e}")

    # Extract expected text from mia.json metadata
    expected_segments = mia_metadata.get("segments", [])
    expected_full_text = " ".join(
        [segment["text"] for segment in expected_segments]
    ).strip()
    expected_words = expected_full_text.lower().split()

    print(f"Expected transcript: {expected_full_text}")
    print(f"Expected word count: {len(expected_words)}")

    stt = deepgram.STT(
        api_key=api_key, sample_rate=target_sample_rate, interim_results=True
    )

    # Track events
    transcripts = []
    partial_transcripts = []
    errors = []

    @stt.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        # Extract words from the text for metadata
        words = event.text.lower().split() if event.text else []
        transcripts.append(
            (
                event.text,
                event.user_metadata,
                {"is_final": True, "confidence": 0.9, "words": words},
            )
        )
        print(f"Final transcript: {event.text}")

    @stt.events.subscribe
    async def on_partial_transcript(event: STTPartialTranscriptEvent):
        partial_transcripts.append(
            (event.text, event.user_metadata, {"is_final": False})
        )
        print(f"Partial transcript: {event.text}")

    @stt.events.subscribe
    async def on_error(event: STTErrorEvent):
        errors.append(event.error)
        print(f"Error: {event.error}")

    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)

    try:
        # Process the audio in chunks to simulate real-time streaming
        chunk_size = 9600  # 0.2 second chunks at 48kHz
        total_samples = len(pcm_samples)

        print(f"Processing audio in chunks of {chunk_size} samples...")

        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk_samples = pcm_samples[i:end_idx]

            chunk_data = PcmData(
                samples=chunk_samples, sample_rate=target_sample_rate, format="s16"
            )

            await stt.process_audio(chunk_data)
            await asyncio.sleep(0.1)  # Small delay between chunks

        # Wait for processing to complete
        print("Waiting for final transcripts...")
        await asyncio.sleep(3.0)

        # Check results
        print(f"Transcripts received: {len(transcripts)}")
        print(f"Partial transcripts received: {len(partial_transcripts)}")
        print(f"Errors received: {len(errors)}")

        if transcripts:
            for i, (text, user, metadata) in enumerate(transcripts):
                print(f"Final transcript {i + 1}: {text}")
                print(f"Metadata: {metadata}")

        if partial_transcripts:
            print(f"Total partial transcripts: {len(partial_transcripts)}")

        if errors:
            for i, error in enumerate(errors):
                print(f"Error {i + 1}: {error}")

        # Validation
        assert len(errors) == 0, f"Received errors: {errors}"

        # We should get at least some results (either final or partial transcripts)
        total_results = len(transcripts) + len(partial_transcripts)
        assert total_results > 0, "No transcripts or partial transcripts received"

        # If we got final transcripts, validate them
        if transcripts:
            # Combine all transcript text
            combined_text = " ".join([t[0] for t in transcripts]).strip()
            actual_words = combined_text.lower().split()

            print(f"Combined final transcript: {combined_text}")
            print(f"Actual word count: {len(actual_words)}")

            # Basic validation
            text, user, metadata = transcripts[0]
            assert isinstance(text, str)
            assert len(text.strip()) > 0
            assert "confidence" in metadata
            assert "is_final" in metadata
            assert metadata["is_final"] is True

            # Content validation - check for key words from the expected transcript
            key_words = [
                "mia",
                "village",
                "map",
                "treasure",
                "cat",
                "whiskers",
                "quiet",
            ]
            found_key_words = [
                word for word in key_words if word in combined_text.lower()
            ]

            print(f"Key words found: {found_key_words}")

            # We should find at least some key words from the story
            assert len(found_key_words) >= 2, (
                f"Expected to find at least 2 key words from {key_words}, but only found {found_key_words}"
            )

            # Check that we got a reasonable amount of text
            assert len(actual_words) >= 10, (
                f"Expected at least 10 words, but got {len(actual_words)}: {combined_text}"
            )

            # Verify metadata structure
            assert "confidence" in metadata
            assert isinstance(metadata["confidence"], (int, float))
            assert "words" in metadata
            assert isinstance(metadata["words"], list)

        # Validate partial transcripts if we have them
        if partial_transcripts:
            # Check that partial transcripts have correct metadata
            text, user, partial_metadata = partial_transcripts[0]
            assert isinstance(text, str)
            assert "is_final" in partial_metadata
            assert partial_metadata["is_final"] is False

        print("Integration test completed successfully!")
        print(f"Final transcripts: {[t[0] for t in transcripts]}")

    finally:
        await stt.close()
