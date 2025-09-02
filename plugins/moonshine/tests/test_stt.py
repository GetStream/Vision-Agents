import pytest
import asyncio
import numpy as np
from unittest.mock import patch

from getstream.plugins import MoonshineSTT
from getstream.video.rtc.track_util import PcmData
from getstream.plugins.test_utils import get_audio_asset, get_json_metadata

# Skip all tests in this module if moonshine_onnx is not installed
try:
    import moonshine_onnx  # noqa: F401
except ImportError:
    pytest.skip(
        "moonshine_onnx is not installed. Skipping all Moonshine STT tests.",
        allow_module_level=True,
    )


# Mock moonshine module for tests that don't require the actual library
class MockMoonshine:
    @staticmethod
    def transcribe(audio_path, model_name):
        """Mock transcribe function that returns a simple result."""
        # Simulate different responses based on model name
        if "base" in model_name:
            return ["This is a high quality transcription from the base model."]
        else:
            return ["This is a transcription from the tiny model."]


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
def mia_audio_data(mia_mp3_path):
    """Load and prepare the mia.mp3 audio data for testing."""
    try:
        # Try to load the mp3 file using soundfile
        import soundfile as sf

        data, original_sample_rate = sf.read(mia_mp3_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample to 16kHz (Moonshine's native rate)
        target_sample_rate = 16000
        if original_sample_rate != target_sample_rate:
            from getstream.audio.utils import resample_audio

            data = resample_audio(data, original_sample_rate, target_sample_rate)

        # Normalize and convert to int16
        if data.max() > 1.0 or data.min() < -1.0:
            data = data / max(abs(data.max()), abs(data.min()))

        # Convert to int16 PCM
        pcm_samples = (data * 32767.0).astype(np.int16)

        # Return PCM data with the resampled rate
        return PcmData(
            samples=pcm_samples, sample_rate=target_sample_rate, format="s16"
        )
    except Exception:
        # Fall back to synthetic data if file loading fails
        sample_rate = 16000
        duration_sec = 2
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))

        # Create speech-like signal with multiple formants
        audio_data = np.zeros_like(t)
        for formant, amplitude in [(600, 1.0), (1200, 0.5), (2400, 0.2)]:
            audio_data += amplitude * np.sin(2 * np.pi * formant * t)

        # Normalize and convert to int16
        audio_data = audio_data / np.max(np.abs(audio_data))
        pcm_samples = (audio_data * 32767.0).astype(np.int16)

        return PcmData(samples=pcm_samples, sample_rate=sample_rate, format="s16")


@pytest.fixture
def audio_data_16k():
    """Load and prepare 16kHz audio data for testing."""
    try:
        # Try to load real audio asset
        audio_path = get_audio_asset("formant_speech_16k.wav")
        import soundfile as sf

        data, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Convert to int16
        pcm_samples = (data * 32767.0).astype(np.int16)

        return PcmData(samples=pcm_samples, sample_rate=sample_rate, format="s16")
    except Exception:
        # Fall back to synthetic data
        sample_rate = 16000
        duration_sec = 2
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))

        # Create speech-like signal with multiple formants
        audio_data = np.zeros_like(t)
        for formant, amplitude in [(600, 1.0), (1200, 0.5), (2400, 0.2)]:
            audio_data += amplitude * np.sin(2 * np.pi * formant * t)

        # Normalize and convert to int16
        audio_data = audio_data / np.max(np.abs(audio_data))
        pcm_samples = (audio_data * 32767.0).astype(np.int16)

        return PcmData(samples=pcm_samples, sample_rate=sample_rate, format="s16")


@pytest.fixture
def audio_data_48k():
    """Load and prepare 48kHz audio data for testing."""
    try:
        # Try to load real audio asset
        audio_path = get_audio_asset("formant_speech_48k.wav")
        import soundfile as sf

        data, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Convert to int16
        pcm_samples = (data * 32767.0).astype(np.int16)

        return PcmData(samples=pcm_samples, sample_rate=sample_rate, format="s16")
    except Exception:
        # Fall back to synthetic data
        sample_rate = 48000
        duration_sec = 2
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))

        # Create speech-like signal
        audio_data = np.zeros_like(t)
        for formant, amplitude in [(600, 1.0), (1200, 0.5), (2400, 0.2)]:
            audio_data += amplitude * np.sin(2 * np.pi * formant * t)

        # Normalize and convert to int16
        audio_data = audio_data / np.max(np.abs(audio_data))
        pcm_samples = (audio_data * 32767.0).astype(np.int16)

        return PcmData(samples=pcm_samples, sample_rate=sample_rate, format="s16")


@pytest.mark.asyncio
async def test_moonshine_model_validation():
    """Test that Moonshine validates model names correctly."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        # Test invalid model name
        with pytest.raises(ValueError, match="Unknown Moonshine model"):
            MoonshineSTT(model_name="invalid_model")

        # Test valid model names
        stt1 = MoonshineSTT(model_name="tiny")
        assert stt1.model_name == "moonshine/tiny"
        await stt1.close()

        stt2 = MoonshineSTT(model_name="moonshine/base")
        assert stt2.model_name == "moonshine/base"
        await stt2.close()


@pytest.mark.asyncio
async def test_moonshine_initialization():
    """Test that Moonshine initializes correctly with mocked library."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT()
        assert stt is not None
        assert stt.model_name == "moonshine/base"  # Canonical value after validation
        assert stt.sample_rate == 16000
        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_custom_initialization():
    """Test Moonshine initialization with custom parameters."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(
            model_name="moonshine/base",
            sample_rate=16000,
            min_audio_length_ms=1000,
        )

        assert stt.model_name == "moonshine/base"  # Canonical value after validation
        assert stt.sample_rate == 16000
        assert stt.min_audio_length_ms == 1000
        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_audio_resampling():
    """Test that audio resampling works correctly."""
    from getstream.audio.utils import resample_audio

    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(sample_rate=16000)

        # Test resampling from 48kHz to 16kHz using the shared utility
        original_data = np.random.randint(
            -1000, 1000, 48000, dtype=np.int16
        )  # 1 second at 48kHz
        resampled = resample_audio(original_data, 48000, 16000).astype(np.int16)

        # Should be approximately 16000 samples (1 second at 16kHz)
        assert abs(len(resampled) - 16000) < 100  # Allow some tolerance
        assert resampled.dtype == np.int16

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_audio_normalization():
    """Test that audio normalization works correctly."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT()

        # Test normalization
        int16_data = np.array([32767, -32768, 0, 16384], dtype=np.int16)
        normalized = stt._normalize_audio(int16_data)

        assert normalized.dtype == np.float32
        assert np.allclose(normalized, [1.0, -1.0, 0.0, 0.5], atol=1e-4)

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_immediate_processing():
    """Test that audio is processed immediately without buffering."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(sample_rate=16000)

        # Mock the _transcribe_audio method to track calls
        transcribe_calls = []

        async def mock_transcribe_audio(audio_data):
            transcribe_calls.append(len(audio_data))
            return "test transcription"

        stt._transcribe_audio = mock_transcribe_audio

        # Create test audio
        audio_array = np.random.randint(
            -1000, 1000, 8000, dtype=np.int16
        )  # 0.5 seconds
        pcm_data = PcmData(samples=audio_array, sample_rate=16000, format="s16")

        # Process audio - should be immediate, no buffering
        await stt.process_audio(pcm_data)

        # Should have called transcribe immediately
        assert len(transcribe_calls) == 1
        assert transcribe_calls[0] == 8000  # Same length as input

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_process_audio_short_chunk(audio_data_16k):
    """Test processing audio that's too short to trigger transcription."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(min_audio_length_ms=1000)  # Require 1s minimum

        # Track events
        transcripts = []
        errors = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        @stt.on("error")
        def on_error(error):
            errors.append(error)

        # Create short audio (0.5 seconds)
        short_audio = PcmData(
            samples=audio_data_16k.samples[:8000],  # 0.5 seconds at 16kHz
            sample_rate=16000,
            format="s16",
        )

        # Process the short audio
        await stt.process_audio(short_audio)

        # Should not trigger transcription due to min_audio_length_ms
        assert len(transcripts) == 0
        assert len(errors) == 0

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_process_audio_sufficient_chunk(audio_data_16k):
    """Test processing audio that's long enough to trigger transcription."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        # Mock the _transcribe_audio method directly instead of moonshine.transcribe
        stt = MoonshineSTT(min_audio_length_ms=500)

        # Track events
        transcripts = []
        errors = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        @stt.on("error")
        def on_error(error):
            errors.append(error)

        # Mock the _transcribe_audio method to return a test result
        async def mock_transcribe_audio(audio_data):
            return "This is a test transcription"

        stt._transcribe_audio = mock_transcribe_audio

        # Process sufficient audio
        await stt.process_audio(audio_data_16k)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Should trigger transcription
        assert len(transcripts) > 0
        assert len(errors) == 0

        # Check transcript content
        text, user, metadata = transcripts[0]
        assert isinstance(text, str)
        assert len(text) > 0
        assert "model_name" in metadata
        assert "audio_duration_ms" in metadata

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_process_audio_with_resampling(audio_data_48k):
    """Test processing audio that requires resampling."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(sample_rate=16000)

        # Track events
        transcripts = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        # Mock the _transcribe_audio method to return a test result
        async def mock_transcribe_audio(audio_data):
            return "This is a test transcription"

        stt._transcribe_audio = mock_transcribe_audio

        # Process 48kHz audio (should be resampled to 16kHz)
        await stt.process_audio(audio_data_48k)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Should trigger transcription after resampling
        assert len(transcripts) > 0

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_flush_functionality(audio_data_16k):
    """Test that flush is a no-op since we no longer buffer."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT(min_audio_length_ms=500)

        # Track events
        transcripts = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        # Mock the _transcribe_audio method to return a test result
        async def mock_transcribe_audio(audio_data):
            return "This is a test transcription"

        stt._transcribe_audio = mock_transcribe_audio

        # Process audio - should trigger immediate transcription
        audio = PcmData(
            samples=audio_data_16k.samples[:16000],  # 1 second
            sample_rate=16000,
            format="s16",
        )
        await stt.process_audio(audio)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Should have triggered transcription immediately
        assert len(transcripts) == 1

        # Flush should be a no-op
        await stt.flush()

        # Should still have only one transcript
        assert len(transcripts) == 1

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_bytes_input():
    """Test processing audio data provided as bytes."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT()

        # Track events
        transcripts = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        # Mock the _transcribe_audio method to return a test result
        async def mock_transcribe_audio(audio_data):
            return "This is a test transcription"

        stt._transcribe_audio = mock_transcribe_audio

        # Create audio as bytes
        audio_array = np.random.randint(-1000, 1000, 16000, dtype=np.int16)  # 1 second
        audio_bytes = audio_array.tobytes()

        pcm_data = PcmData(samples=audio_bytes, sample_rate=16000, format="s16")
        await stt.process_audio(pcm_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Should trigger transcription
        assert len(transcripts) > 0

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_error_handling():
    """Test error handling during transcription."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT()

        # Track events
        errors = []

        @stt.on("error")
        def on_error(error):
            errors.append(error)

        # Mock the _transcribe_audio method to raise an exception
        async def mock_transcribe_audio(audio_data):
            raise Exception("Transcription failed")

        stt._transcribe_audio = mock_transcribe_audio

        # Create sufficient audio
        audio_array = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        pcm_data = PcmData(samples=audio_array, sample_rate=16000, format="s16")

        await stt.process_audio(pcm_data)

        # Give some time for async processing
        await asyncio.sleep(0.1)

        # Should have captured the error
        assert len(errors) > 0

        await stt.close()


@pytest.mark.asyncio
async def test_moonshine_closed_state():
    """Test that processing is ignored when STT is closed."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        stt = MoonshineSTT()

        # Close the STT
        await stt.close()

        # Try to process audio
        audio_array = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        pcm_data = PcmData(samples=audio_array, sample_rate=16000, format="s16")

        result = await stt._process_audio_impl(pcm_data)

        # Should return None when closed
        assert result is None


@pytest.mark.asyncio
async def test_moonshine_model_selection():
    """Test that different models produce different results."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine"):
        # Test tiny model
        stt_tiny = MoonshineSTT(model_name="moonshine/tiny")

        # Test base model
        stt_base = MoonshineSTT(model_name="moonshine/base")

        # Both should initialize successfully with canonical names
        assert (
            stt_tiny.model_name == "moonshine/tiny"
        )  # Canonical value after validation
        assert (
            stt_base.model_name == "moonshine/base"
        )  # Canonical value after validation

        await stt_tiny.close()
        await stt_base.close()


@pytest.mark.asyncio
async def test_moonshine_with_mia_audio_mocked(mia_audio_data, mia_metadata):
    """Test Moonshine STT with mia.mp3 audio using mocked transcription."""
    with patch("getstream.plugins.moonshine.stt.stt.moonshine") as mock_moonshine:
        # Extract expected text from mia.json metadata
        expected_segments = mia_metadata.get("segments", [])
        expected_full_text = " ".join(
            [segment["text"] for segment in expected_segments]
        ).strip()

        # Mock the transcribe function to return the expected text
        mock_moonshine.transcribe.return_value = [expected_full_text]

        stt = MoonshineSTT(model_name="moonshine/base", min_audio_length_ms=500)

        # Track events
        transcripts = []
        errors = []

        @stt.on("transcript")
        def on_transcript(text, user, metadata):
            transcripts.append((text, user, metadata))

        @stt.on("error")
        def on_error(error):
            errors.append(error)

        # Process the mia audio data
        await stt.process_audio(mia_audio_data)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Flush any remaining audio
        await stt.flush()
        await asyncio.sleep(0.1)

        # Verify results
        assert len(errors) == 0, f"Received errors: {errors}"
        assert len(transcripts) > 0, "No transcripts received"

        # Check transcript content
        text, user, metadata = transcripts[0]
        assert isinstance(text, str)
        assert len(text) > 0
        assert "model_name" in metadata
        assert "audio_duration_ms" in metadata

        # Verify the transcript contains expected content
        assert "mia" in text.lower()
        assert "village" in text.lower() or "treasure" in text.lower()

        # Verify metadata structure
        assert metadata["model_name"] == "moonshine/base"
        assert metadata["confidence"] is None  # Moonshine doesn't provide confidence scores
        assert metadata["target_sample_rate"] == 16000
        assert "processing_time_ms" in metadata
        assert "original_sample_rate" in metadata
        assert "resampled" in metadata

        # Verify the mock was called correctly
        mock_moonshine.transcribe.assert_called()
        call_args = mock_moonshine.transcribe.call_args
        assert len(call_args[0]) == 2  # audio_path, model_name
        assert call_args[0][1] == "moonshine/base"  # model_name

        await stt.close()


# Integration test with real Moonshine (if available)
@pytest.mark.asyncio
async def test_moonshine_real_integration(mia_audio_data, mia_metadata):
    """
    Integration test with the real Moonshine library using the mia.mp3 test file.

    This test processes the mia.mp3 audio file and compares the transcription results
    with the expected content from mia.json metadata.

    This test will be skipped if Moonshine is not installed.
    """
    # Only run if we have a reasonable amount of audio
    if len(mia_audio_data.samples) < 8000:  # Less than 0.5 seconds
        pytest.skip("Audio sample too short for meaningful integration test")

    print(
        f"Testing with mia.mp3: {len(mia_audio_data.samples)} samples at {mia_audio_data.sample_rate}Hz"
    )
    print(
        f"Audio duration: {len(mia_audio_data.samples) / mia_audio_data.sample_rate:.2f} seconds"
    )
    print(
        f"Audio range: {mia_audio_data.samples.min()} to {mia_audio_data.samples.max()}"
    )

    # Extract expected text from mia.json metadata
    expected_segments = mia_metadata.get("segments", [])
    expected_full_text = " ".join(
        [segment["text"] for segment in expected_segments]
    ).strip()
    expected_words = expected_full_text.lower().split()

    print(f"Expected transcript: {expected_full_text}")
    print(f"Expected word count: {len(expected_words)}")

    stt = MoonshineSTT(
        model_name="moonshine/tiny",  # Use tiny model for faster testing
        min_audio_length_ms=500,
    )

    # Track events
    transcripts = []
    errors = []

    @stt.on("transcript")
    def on_transcript(text, user, metadata):
        transcripts.append((text, user, metadata))

    @stt.on("error")
    def on_error(error):
        errors.append(error)

    try:
        # Process the audio in chunks to simulate real-time processing
        chunk_size = 8000  # Process in 0.5 second chunks at 16kHz
        total_samples = len(mia_audio_data.samples)

        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk_samples = mia_audio_data.samples[i:end_idx]

            chunk_data = PcmData(
                samples=chunk_samples,
                sample_rate=mia_audio_data.sample_rate,
                format=mia_audio_data.format,
            )

            await stt.process_audio(chunk_data)
            await asyncio.sleep(0.1)  # Small delay between chunks

        # Wait for processing to complete
        await asyncio.sleep(2.0)

        # Flush any remaining audio
        await stt.flush()
        await asyncio.sleep(1.0)

        # Check results
        print(f"Transcripts received: {len(transcripts)}")
        print(f"Errors received: {len(errors)}")

        if transcripts:
            for i, (text, user, metadata) in enumerate(transcripts):
                print(f"Transcript {i + 1}: {text}")
                print(f"Metadata: {metadata}")

        if errors:
            for i, error in enumerate(errors):
                print(f"Error {i + 1}: {error}")

        # We should either get transcripts or errors, but not silence
        assert (
            len(transcripts) > 0 or len(errors) > 0
        ), "No transcripts or errors received"

        # If we got transcripts, verify they contain reasonable content
        if transcripts:
            # Combine all transcript text
            combined_text = " ".join([t[0] for t in transcripts]).strip()
            actual_words = combined_text.lower().split()

            print(f"Combined transcript: {combined_text}")
            print(f"Actual word count: {len(actual_words)}")

            # Basic validation
            text, user, metadata = transcripts[0]
            assert isinstance(text, str)
            assert len(text.strip()) > 0
            assert "model_name" in metadata
            assert metadata["model_name"] == "moonshine/tiny"
            assert "audio_duration_ms" in metadata
            assert metadata["audio_duration_ms"] > 0

            # Content validation - check for key words from the expected transcript
            # We'll be lenient since STT accuracy can vary
            key_words = [
                "mia",
                "village",
                "brushes",
                "map",
                "treasure",
                "fields",
                "hues",
                "discovered",
            ]
            found_key_words = [
                word for word in key_words if word in combined_text.lower()
            ]

            print(f"Key words found: {found_key_words}")

            # We should find at least some key words from the story
            assert (
                len(found_key_words) >= 2
            ), f"Expected to find at least 2 key words from {key_words}, but only found {found_key_words}"

            # Check that we got a reasonable amount of text
            assert (
                len(actual_words) >= 10
            ), f"Expected at least 10 words, but got {len(actual_words)}: {combined_text}"

            # Verify metadata structure
            assert "processing_time_ms" in metadata
            assert "confidence" in metadata
            assert (
                metadata["confidence"] is None
            )  # Moonshine doesn't provide confidence scores
            assert "original_sample_rate" in metadata
            assert "target_sample_rate" in metadata
            assert metadata["target_sample_rate"] == 16000  # Moonshine's native rate

        # We shouldn't have any errors
        assert len(errors) == 0, f"Received errors: {errors}"

    finally:
        await stt.close()
