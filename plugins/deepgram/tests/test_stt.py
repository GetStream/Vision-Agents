import json
import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock
import os

from vision_agents.plugins import deepgram
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent, STTErrorEvent
from getstream.video.rtc.track_util import PcmData
from plugins.plugin_test_utils import get_audio_asset, get_json_metadata


# Mock the Deepgram client to avoid real API calls during tests
class MockDeepgramConnection:
    def __init__(self):
        self.event_handlers = {}
        self.sent_data = []
        self.sent_text_messages = []
        self.finished = False
        self.closed = False

    def on(self, event, handler):
        """Register event handlers"""
        self.event_handlers[event] = handler
        return handler

    def send(self, data):
        """Mock send data"""
        self.sent_data.append(data)
        return True

    def send_binary(self, data):
        """Mock send binary data - required by the actual implementation"""
        self.sent_data.append(data)
        return True

    def send_text(self, text_message):
        """Mock send text message (for keep-alive)"""
        self.sent_text_messages.append(text_message)
        return True

    def keep_alive(self):
        """Mock keep_alive method for SDKs that support it"""
        self.sent_text_messages.append(json.dumps({"type": "KeepAlive"}))
        return True

    def finish(self):
        """Close the connection"""
        self.finished = True

    def close(self):
        """Alternative close method"""
        self.closed = True

    def start(self, options):
        """Start the connection"""
        pass

    def emit_transcript(self, text, is_final=True):
        """Helper to emit a transcript event"""
        from deepgram import LiveTranscriptionEvents

        if LiveTranscriptionEvents.Transcript in self.event_handlers:
            # Create a mock result
            mock_result = {
                "channel": {"alternatives": [{"transcript": text, "confidence": 0.9}]},
                "is_final": is_final,
            }
            self.event_handlers[LiveTranscriptionEvents.Transcript](self, mock_result)

    def emit_error(self, error_message):
        """Helper to emit an error event"""
        from deepgram import LiveTranscriptionEvents

        if LiveTranscriptionEvents.Error in self.event_handlers:
            error_obj = Exception(error_message)
            self.event_handlers[LiveTranscriptionEvents.Error](self, error_obj)


class MockDeepgramClient:
    def __init__(self, api_key=None, config=None):
        self.api_key = api_key
        self.config = config
        self.listen = MagicMock()
        self.listen.websocket = MagicMock()
        self.listen.websocket.v = MagicMock(return_value=MockDeepgramConnection())


# Enhanced mock for testing keep-alive functionality
class MockDeepgramConnectionWithKeepAlive:
    def __init__(self):
        self.event_handlers = {}
        self.sent_data = []
        self.sent_text_messages = []
        self.finished = False
        self.closed = False

    def on(self, event, handler):
        """Register event handlers"""
        self.event_handlers[event] = handler
        return handler

    def send(self, data):
        """Mock send audio data"""
        self.sent_data.append(data)
        return True

    def send_binary(self, data):
        """Mock send binary data - required by the actual implementation"""
        self.sent_data.append(data)
        return True

    def send_text(self, text_message):
        """Mock send text message (for keep-alive)"""
        self.sent_text_messages.append(text_message)
        return True

    def keep_alive(self):
        """Mock keep_alive method for SDKs that support it"""
        self.sent_text_messages.append(json.dumps({"type": "KeepAlive"}))
        return True

    def finish(self):
        """Close the connection"""
        self.finished = True

    def close(self):
        """Alternative close method"""
        self.closed = True

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


class MockDeepgramClientWithKeepAlive:
    def __init__(self, api_key=None, config=None):
        self.api_key = api_key
        self.config = config
        self.listen = MagicMock()
        self.listen.websocket = MagicMock()
        self.listen.websocket.v = MagicMock(
            return_value=MockDeepgramConnectionWithKeepAlive()
        )


# Skip reason for keep_alive tests
KEEP_ALIVE_NOT_IMPLEMENTED = (
    "keep_alive functionality not yet implemented in Deepgram STT. "
    "The 'keep_alive_interval' parameter and 'send_keep_alive()' method "
    "do not exist in the current implementation. This is tracked as a feature request."
)


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
    import torchaudio
    import torch
    import numpy as np
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
@patch("stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClient)
async def test_deepgram_stt_initialization():
    """Test that the Deepgram STT initializes correctly with explicit API key."""
    stt = deepgram.STT(api_key="test-api-key")
    assert stt is not None
    assert stt.deepgram.api_key == "test-api-key"
    await stt.close()


@pytest.mark.asyncio
@patch("stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClient)
@patch.dict(os.environ, {"DEEPGRAM_API_KEY": "env-var-api-key"})
async def test_deepgram_stt_initialization_with_env_var():
    """Test that the Deepgram STT initializes correctly when DEEPGRAM_API_KEY is set."""

    # Initialize without providing an API key – implementation should fall back to env var
    stt = deepgram.STT()
    assert stt is not None
    assert stt.deepgram.api_key == "env-var-api-key"


@pytest.mark.asyncio
@patch("stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClient)
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
    stt._setup_connection()

    # Emit a transcript using the mock connection
    stt.dg_connection.emit_transcript("This is a final transcript")

    # Process some audio to ensure the connection is active
    audio_data = PcmData(samples=b"\x00\x00" * 1000, sample_rate=48000, format="s16")
    await stt.process_audio(audio_data)

    # Give the async event handlers time to process
    await asyncio.sleep(0.05)

    # Check that the events were received
    assert len(transcripts) > 0
    assert "This is a final transcript" in transcripts[0][0]

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClient)
async def test_deepgram_process_audio(audio_data, mia_metadata):
    """Test that the Deepgram STT can process audio data."""
    stt = deepgram.STT(api_key="test-api-key")

    # Track the audio data that was sent
    sent_audio_bytes = []

    # Create a custom send method to track sent data
    original_send = stt.dg_connection.send

    def mock_send(data):
        sent_audio_bytes.append(data)
        return True

    # Replace the send method on the connection to track sent data
    stt.dg_connection.send = mock_send

    # Process audio - note we're using the implementation method
    await stt._process_audio_impl(audio_data, None)

    # Restore the original send method
    stt.dg_connection.send = original_send

    # Check that audio was sent
    assert len(sent_audio_bytes) > 0, "No audio data was sent to Deepgram"

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch("stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClient)
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
    stt._setup_connection()

    # Emit a transcript using the mock connection
    stt.dg_connection.emit_transcript("This is the final result")

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


@pytest.mark.skip(reason=KEEP_ALIVE_NOT_IMPLEMENTED)
@pytest.mark.asyncio
@patch(
    "stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClientWithKeepAlive
)
async def test_deepgram_keep_alive_mechanism():
    """Test that the keep-alive mechanism works."""
    # Create a Deepgram STT instance with a short keep-alive interval
    stt = deepgram.STT(api_key="test-api-key", keep_alive_interval=0.1)

    # Set up the connection with the mocked client
    stt._setup_connection()
    connection = stt.dg_connection

    # Wait long enough for at least one keep-alive message to be sent
    await asyncio.sleep(0.2)

    # We should see that keep-alive messages have been sent
    assert len(connection.sent_text_messages) > 0, (
        "No keep-alive messages sent after wait"
    )

    # Cleanup
    await stt.close()


@pytest.mark.skip(reason=KEEP_ALIVE_NOT_IMPLEMENTED)
@pytest.mark.asyncio
@patch(
    "stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClientWithKeepAlive
)
async def test_deepgram_keep_alive_after_audio():
    """Test that keep-alive messages are sent after audio is processed."""
    # Create a Deepgram STT instance with a short keep-alive interval
    stt = deepgram.STT(api_key="test-api-key", keep_alive_interval=0.1)

    # Set up the connection with the mocked client
    stt._setup_connection()
    connection = stt.dg_connection

    # Create some empty audio data
    audio_data = PcmData(samples=b"\x00\x00" * 1000, sample_rate=48000, format="s16")

    # Process the audio - this should set the last_activity_time
    await stt.process_audio(audio_data)

    # Wait longer than the keep-alive interval
    await asyncio.sleep(0.2)

    # We should see that keep-alive messages have been sent
    assert len(connection.sent_text_messages) > 0, (
        "No keep-alive messages sent after audio processing"
    )

    # Cleanup
    await stt.close()


@pytest.mark.skip(reason=KEEP_ALIVE_NOT_IMPLEMENTED)
@pytest.mark.asyncio
@patch(
    "stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClientWithKeepAlive
)
async def test_deepgram_keep_alive_direct():
    """Test that we can directly send keep-alive messages."""
    # Create a Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    stt._setup_connection()
    connection = stt.dg_connection

    # Send a keep-alive message directly
    success = await stt.send_keep_alive()

    # It should succeed
    assert success, "Failed to send keep-alive message"

    # The connection should have received the message
    assert len(connection.sent_text_messages) > 0, (
        "No keep-alive messages were recorded"
    )

    # If the connection has a keep_alive method, then the send_text method should not be used
    if hasattr(connection, "keep_alive"):
        assert connection.sent_text_messages[0] == '{"type": "KeepAlive"}'

    # Cleanup
    await stt.close()


@pytest.mark.asyncio
@patch(
    "stream_agents.plugins.deepgram.stt.DeepgramClient", MockDeepgramClientWithKeepAlive
)
async def test_deepgram_close_message():
    """Test that the finish message is sent when the connection is closed."""
    # Create a Deepgram STT instance
    stt = deepgram.STT(api_key="test-api-key")

    # Set up the connection with the mocked client
    stt._setup_connection()
    connection = stt.dg_connection

    # Track the original finish method and mock it
    original_finish = connection.finish
    finish_called = False

    def mock_finish():
        nonlocal finish_called
        finish_called = True
        original_finish()

    connection.finish = mock_finish

    # Also track the send_text method
    original_send_text = None
    if hasattr(connection, "send_text"):
        original_send_text = connection.send_text
        send_text_called = False

        def mock_send_text(message):
            nonlocal send_text_called
            send_text_called = True
            return original_send_text(message)

        connection.send_text = mock_send_text

    # Close the STT service
    await stt.close()

    # The finish method should have been called
    assert finish_called, "Connection finish method was not called on close"

    # The connection should have been closed
    assert connection.finished, "Connection not marked as finished after close"


@pytest.mark.skip(reason=KEEP_ALIVE_NOT_IMPLEMENTED)
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

    try:
        print("Waiting for keep-alive timeout (3 seconds)...")

        # Wait longer than keep-alive interval to test the mechanism
        await asyncio.sleep(6.0)

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
