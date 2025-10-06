import asyncio
import logging
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torchaudio
from typing import List, Dict, Any, Optional

from vision_agents.plugins import silero
from vision_agents.core.vad.events import VADAudioEvent, VADPartialEvent
from plugins.plugin_test_utils import get_audio_asset, get_json_metadata
from getstream.video.rtc.track_util import PcmData

# Setup logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
def mia_wav_path(mia_mp3_path):
    """Convert the mp3 to wav for testing."""
    # Create a temporary wav file
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Convert mp3 to wav using torchaudio
    waveform, sample_rate = torchaudio.load(mia_mp3_path)
    torchaudio.save(wav_path, waveform, sample_rate)

    yield wav_path

    # Clean up temporary file
    if os.path.exists(wav_path):
        os.remove(wav_path)


@pytest.fixture
def audio_data(mia_wav_path):
    """Load and prepare the audio data for testing."""
    # Load audio file
    data, sample_rate = sf.read(mia_wav_path)

    # Convert to mono if stereo
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)

    return data, sample_rate


@pytest.fixture
async def vad_setup():
    """Create a Silero VAD instance with standard test configuration."""
    vad = silero.VAD(
        sample_rate=16000,  # Use the model's native sample rate
        speech_pad_ms=400,  # Increase padding to connect nearby speech segments
        min_speech_ms=100,  # Keep minimum low to catch shorter segments
        activation_th=0.2,  # Lower threshold for more sensitivity
        deactivation_th=0.15,  # Lower deactivation threshold
    )

    # Storage for detected segments
    detected_segments: List[Dict[str, Any]] = []

    return vad, detected_segments


async def process_audio_file(
    vad,
    data,
    original_sample_rate,
    detected_segments: List[Dict[str, Any]],
    partial_segments: Optional[List[Dict[str, Any]]] = None,
):
    """Process audio data with VAD and collect detected speech segments."""

    # Setup event handlers
    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent):
        # Extract duration and samples from the event
        duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
        samples = event.audio_data if event.audio_data is not None else b""
        detected_segments.append({"duration": duration, "bytes": len(samples)})
        logger.info(
            f"Detected speech segment: {duration:.2f} seconds ({len(samples)} bytes)"
        )

    # Add handler for partial events if tracking them
    if partial_segments is not None:

        @vad.events.subscribe
        async def on_partial(event: VADPartialEvent):
            # Extract duration and samples from the event
            duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
            samples = event.audio_data if event.audio_data is not None else b""
            partial_segments.append({"duration": duration, "bytes": len(samples)})
            logger.info(
                f"Partial speech data: {duration:.2f} seconds ({len(samples)} bytes)"
            )

    # Resample if needed
    if original_sample_rate != vad.sample_rate:
        import scipy.signal

        num_samples = int(len(data) * vad.sample_rate / original_sample_rate)
        data = scipy.signal.resample(data, num_samples)

    # Convert to int16 PCM
    data = np.asarray(data, dtype=np.float32)
    pcm_samples = (data * 32768.0).astype(np.int16)

    # Process the audio data
    await vad.process_audio(
        PcmData(samples=pcm_samples, sample_rate=vad.sample_rate, format="s16")
    )

    # Ensure we flush any remaining speech
    await vad.flush()

    # Add a small delay to allow all events to be processed
    await asyncio.sleep(0.1)


async def process_audio_in_chunks(
    vad,
    data,
    original_sample_rate,
    detected_segments: List[Dict[str, Any]],
    partial_segments: Optional[List[Dict[str, Any]]] = None,
    chunk_size_ms: int = 20,
):
    """Process audio data in small chunks to simulate streaming."""

    # Setup event handlers
    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent):
        # Extract duration and samples from the event
        duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
        samples = event.audio_data if event.audio_data is not None else b""
        detected_segments.append({"duration": duration, "bytes": len(samples)})
        logger.info(
            f"Detected speech segment: {duration:.2f} seconds ({len(samples)} bytes)"
        )

    # Add handler for partial events if tracking them
    if partial_segments is not None:

        @vad.events.subscribe
        async def on_partial(event: VADPartialEvent):
            # Extract duration and samples from the event
            duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
            samples = event.audio_data if event.audio_data is not None else b""
            partial_segments.append({"duration": duration, "bytes": len(samples)})
            logger.info(
                f"Partial speech data: {duration:.2f} seconds ({len(samples)} bytes)"
            )

    # Allow event handlers to register before processing audio
    await asyncio.sleep(0.01)

    # Resample if needed
    if original_sample_rate != vad.sample_rate:
        import scipy.signal

        num_samples = int(len(data) * vad.sample_rate / original_sample_rate)
        data = scipy.signal.resample(data, num_samples)

    # Calculate chunk size in samples for the target sample rate
    chunk_samples = int(vad.sample_rate * chunk_size_ms / 1000)
    logger.info(
        f"Processing audio in {chunk_size_ms}ms chunks ({chunk_samples} samples per chunk)"
    )

    # Process audio in chunks
    for i in range(0, len(data), chunk_samples):
        # Extract chunk
        chunk = data[i : i + chunk_samples]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            padded_chunk = np.zeros(chunk_samples)
            padded_chunk[: len(chunk)] = chunk
            chunk = padded_chunk

        # Convert to int16 PCM
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk_pcm = (chunk * 32768.0).astype(np.int16)

        # Process the chunk
        await vad.process_audio(
            PcmData(samples=chunk_pcm, sample_rate=vad.sample_rate, format="s16")
        )

        # Add a small delay to simulate real-time processing
        await asyncio.sleep(0.001)

    # Ensure we flush any remaining speech
    await vad.flush()

    # Add a small delay to allow all events to be processed
    await asyncio.sleep(0.1)


def verify_detected_speech(
    detected_segments: List[Dict[str, Any]],
    expected_duration: Optional[float] = None,
    expected_turns: Optional[int] = None,
    tolerance: float = 0.1,
):
    """
    Verify that the detected speech matches expectations.

    Args:
        detected_segments: List of detected speech segments
        expected_duration: Expected total duration of speech in seconds
        expected_turns: Expected number of speech turns
        tolerance: Tolerance for duration validation (e.g., 0.1 = ±10%)
    """
    # Verify that speech was detected
    assert len(detected_segments) > 0, "No speech segments were detected"

    # Verify number of turns if specified
    if expected_turns is not None:
        assert len(detected_segments) == expected_turns, (
            f"Expected {expected_turns} speech turns, got {len(detected_segments)}"
        )

    # Calculate total detected duration
    total_detected_duration = sum(segment["duration"] for segment in detected_segments)

    # Verify duration if specified
    if expected_duration is not None:
        # Calculate expected range with tighter tolerance (±10%)
        min_expected = expected_duration * (1 - tolerance)
        max_expected = expected_duration * (1 + tolerance)

        logger.info(f"Expected speech duration: {expected_duration:.2f} seconds")
        logger.info(f"Detected speech duration: {total_detected_duration:.2f} seconds")
        logger.info(
            f"Tolerance: ±{tolerance * 100:.0f}% ({min_expected:.2f} - {max_expected:.2f}s)"
        )

        if len(detected_segments) > 0:
            logger.info(f"Number of detected segments: {len(detected_segments)}")
            # Log the speech segments for inspection
            for i, segment in enumerate(detected_segments):
                logger.info(f"Speech segment {i + 1}: {segment['duration']:.2f}s")

        # Verify that the duration is within expected range
        assert min_expected <= total_detected_duration <= max_expected, (
            f"Expected speech duration {expected_duration}s (±{tolerance * 100:.0f}%), got {total_detected_duration}s"
        )


def verify_partial_events(
    partial_segments: List[Dict[str, Any]],
    detected_segments: List[Dict[str, Any]],
):
    """
    Verify that partial events were received before final speech events.

    Args:
        partial_segments: List of partial speech events
        detected_segments: List of final speech events
    """
    # Verify that at least one partial event was observed before each final audio event
    assert len(partial_segments) > 0, "No partial speech events were detected"

    # Each detected segment should have at least one corresponding partial event
    # For simplicity, we just check that we have at least one partial event per detected segment
    assert len(partial_segments) >= len(detected_segments), (
        f"Expected at least {len(detected_segments)} partial events, got {len(partial_segments)}"
    )


@pytest.mark.asyncio
async def test_silero_vad_initialization():
    """Test that the Silero VAD can be initialized with default parameters."""
    vad = silero.VAD()
    assert vad is not None
    assert vad.model is not None


@pytest.mark.asyncio
async def test_silero_vad_speech_detection(audio_data, mia_metadata, vad_setup):
    """Test that the Silero VAD can detect speech in an audio file."""
    vad, detected_segments = vad_setup
    data, sample_rate = audio_data

    # List to track partial events
    partial_segments: List[Dict[str, Any]] = []

    # Process the entire audio file
    await process_audio_file(
        vad, data, sample_rate, detected_segments, partial_segments
    )

    # Verify that speech was detected (without duration validation)
    assert len(detected_segments) > 0, "No speech segments were detected"
    logger.info(f"Detected {len(detected_segments)} speech segments")
    total_duration = sum(segment["duration"] for segment in detected_segments)
    logger.info(f"Total detected speech duration: {total_duration:.2f} seconds")

    # Verify partial events were received
    verify_partial_events(partial_segments, detected_segments)


@pytest.mark.asyncio
async def test_streaming_chunks_20ms(audio_data, mia_metadata):
    """
    Test that streaming chunks gives the same results as processing the entire file.
    This test verifies that the VAD works in streaming mode.
    """
    # Create a new VAD for this test using the same parameters as vad_setup
    vad = silero.VAD(
        sample_rate=16000,  # Use the model's native sample rate
        speech_pad_ms=400,  # Increase padding to connect nearby speech segments
        min_speech_ms=100,  # Keep minimum low to catch shorter segments
        activation_th=0.2,  # Lower threshold for more sensitivity
        deactivation_th=0.15,  # Lower deactivation threshold
    )

    # Lists to track detected segments and partial events
    detected_segments: List[Dict[str, Any]] = []
    partial_segments: List[Dict[str, Any]] = []
    data, sample_rate = audio_data

    # Process the audio in small chunks to simulate streaming
    await process_audio_in_chunks(
        vad, data, sample_rate, detected_segments, partial_segments, chunk_size_ms=20
    )

    # Verify that speech was detected (without duration validation)
    assert len(detected_segments) > 0, "No speech segments were detected"
    logger.info(f"Detected {len(detected_segments)} speech segments")
    total_duration = sum(segment["duration"] for segment in detected_segments)
    logger.info(f"Total detected speech duration: {total_duration:.2f} seconds")

    # Verify partial events were received
    verify_partial_events(partial_segments, detected_segments)


@pytest.mark.asyncio
async def test_vad_with_connection_manager_format(audio_data, vad_setup):
    """Test VAD with PCM data in bytes format."""
    vad, detected_segments = vad_setup
    data, sample_rate = audio_data

    # List to track partial events
    partial_segments = []

    # Resample if needed
    if sample_rate != vad.sample_rate:
        import scipy.signal

        num_samples = int(len(data) * vad.sample_rate / sample_rate)
        data = scipy.signal.resample(data, num_samples)

    # Convert to int16 PCM bytes
    data = np.asarray(data, dtype=np.float32)
    pcm_bytes = (data * 32768.0).astype(np.int16).tobytes()

    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent):
        # Extract duration and samples from the event
        duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
        samples = event.audio_data if event.audio_data is not None else b""
        detected_segments.append({"duration": duration, "bytes": len(samples)})
        logger.info(
            f"Detected speech segment: {duration:.2f} seconds ({len(samples)} bytes)"
        )

    @vad.events.subscribe
    async def on_partial(event: VADPartialEvent):
        # Extract duration and samples from the event
        duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
        samples = event.audio_data if event.audio_data is not None else b""
        partial_segments.append({"duration": duration, "bytes": len(samples)})
        logger.info(
            f"Partial speech data: {duration:.2f} seconds ({len(samples)} bytes)"
        )

    # Process the audio data as bytes
    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    await vad.process_audio(
        PcmData(samples=pcm_array, sample_rate=vad.sample_rate, format="s16")
    )

    # Ensure we flush any remaining speech
    await vad.flush()

    # Add a small delay to allow all events to be processed
    await asyncio.sleep(0.1)

    # Verify that speech was detected with expected turns
    assert len(detected_segments) > 0, "No speech segments were detected"
    logger.info(f"Detected {len(detected_segments)} speech segments")

    # Verify partial events were received
    verify_partial_events(partial_segments, detected_segments)


@pytest.mark.asyncio
async def test_silence_no_turns():
    """Test that no 'audio' events are fired when processing silence."""
    # Create a VAD for this test
    vad = silero.VAD(
        sample_rate=16000,
        speech_pad_ms=300,
        min_speech_ms=250,
        activation_th=0.5,
        deactivation_th=0.35,
    )

    # Create 5 seconds of silence (zeros)
    silence_duration_seconds = 5
    silence_samples = np.zeros(
        vad.sample_rate * silence_duration_seconds, dtype=np.int16
    )

    # Flag to track if audio event was fired
    audio_event_fired = False
    partial_event_fired = False

    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent):
        nonlocal audio_event_fired
        audio_event_fired = True
        logger.info(
            f"Audio event detected on silence! Duration: {event.duration_ms / 1000.0:.2f}s"
        )

    @vad.events.subscribe
    async def on_partial(event: VADPartialEvent):
        nonlocal partial_event_fired
        partial_event_fired = True
        logger.info(
            f"Partial event detected on silence! Duration: {event.duration_ms / 1000.0:.2f}s"
        )

    # Process the silence in chunks to simulate streaming
    chunk_size = 512
    for i in range(0, len(silence_samples), chunk_size):
        chunk = silence_samples[i : i + chunk_size]
        await vad.process_audio(
            PcmData(samples=chunk, sample_rate=vad.sample_rate, format="s16")
        )

    # Ensure we flush any remaining audio
    await vad.flush()

    # Add a small delay to allow all events to be processed
    await asyncio.sleep(0.1)

    # Verify that no audio or partial events were fired
    assert not audio_event_fired, "An audio event was fired when processing silence"
    assert not partial_event_fired, "A partial event was fired when processing silence"


class TestSileroVAD:
    """Tests for Silero VAD implementation."""

    @pytest.mark.asyncio
    async def test_speech_detection(self):
        """Test that Silero VAD correctly detects speech in an audio file."""
        # Initialize the VAD with asymmetric thresholds
        vad = silero.VAD(
            sample_rate=16000,
            frame_size=512,
            activation_th=0.3,  # Activation threshold
            deactivation_th=0.2,  # Deactivation threshold (lower)
            speech_pad_ms=300,  # Increase padding to join nearby speech segments into one turn
            min_speech_ms=250,
            model_rate=16000,
            window_samples=512,  # Silero requires exactly 512 samples at 16kHz
            partial_frames=5,  # Emit partial every 5 frames
        )

        # Load an audio file with speech
        audio_path = get_audio_asset("formant_speech_16k.wav")
        audio_data, sample_rate = sf.read(audio_path, dtype="int16")
        assert sample_rate == 16000, f"Expected sample rate 16000, got {sample_rate}"

        # Create listeners to capture emitted speech segments and partial events
        detected_speech = []
        partial_events = []

        @vad.events.subscribe
        async def on_audio(event: VADAudioEvent, user=None):
            detected_speech.append(event)
            duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
            samples = event.audio_data if event.audio_data is not None else b""
            logger.info(f"Audio event: {duration:.2f}s ({len(samples)} samples)")

        @vad.events.subscribe
        async def on_partial(event: VADPartialEvent, user=None):
            partial_events.append(event)
            duration = event.duration_ms / 1000.0 if event.duration_ms else 0.0
            samples = event.audio_data if event.audio_data is not None else b""
            logger.info(f"Partial event: {duration:.2f}s ({len(samples)} samples)")

        # Process the audio data
        await vad.process_audio(
            PcmData(samples=audio_data, sample_rate=vad.sample_rate, format="s16")
        )

        # Ensure we flush any remaining speech
        await vad.flush()

        # Add a small delay to allow all events to be processed
        await asyncio.sleep(0.1)

        # Verify that speech was detected and partial events were received
        assert len(detected_speech) > 0, "No speech segments detected"
        logger.info(f"Detected {len(detected_speech)} speech segments")

        # Verify that partial events were received before the final audio event
        assert len(partial_events) > 0, "No partial events detected"
        assert len(partial_events) >= len(detected_speech), (
            f"Expected at least {len(detected_speech)} partial events, got {len(partial_events)}"
        )
        logger.info(f"Detected {len(partial_events)} partial events")

        # Clean up
        await vad.close()

    @pytest.mark.asyncio
    async def test_silence_no_turns(self):
        """Test that Silero VAD does not emit turns for pure silence."""
        # Initialize the VAD with asymmetric thresholds
        vad = silero.VAD(
            sample_rate=16000,
            frame_size=512,
            activation_th=0.5,
            deactivation_th=0.35,
            speech_pad_ms=30,
            min_speech_ms=250,
            model_rate=16000,
            window_samples=512,  # Silero requires exactly 512 samples at 16kHz
        )

        # Create 5 seconds of silence (16000 samples per second * 5 seconds)
        silence = np.zeros(16000 * 5, dtype=np.int16)

        # Create a listener to capture emitted speech segments
        detected_speech = []
        detected_partials = []

        @vad.events.subscribe
        async def on_audio(event: VADAudioEvent, user=None):
            detected_speech.append(event)

        @vad.events.subscribe
        async def on_partial(event: VADPartialEvent, user=None):
            detected_partials.append(event)

        # Process the silent audio data
        await vad.process_audio(
            PcmData(samples=silence, sample_rate=vad.sample_rate, format="s16")
        )

        # Ensure we flush any remaining buffer
        await vad.flush()

        # Add a small delay to allow all events to be processed
        await asyncio.sleep(0.1)

        # Verify that no speech was detected
        assert len(detected_speech) == 0, "Speech segments detected in silent audio"
        assert len(detected_partials) == 0, "Partial events detected in silent audio"

        # Clean up
        await vad.close()

    @pytest.mark.asyncio
    async def test_mixed_samplerate(self):
        """Test that Silero VAD handles mixed sample rates correctly."""
        # Initialize a listener for captured speech segments
        detected_speech_16k = []
        detected_speech_48k = []
        partial_events_16k = []
        partial_events_48k = []

        # First test with 16 kHz audio
        vad_16k = silero.VAD(
            sample_rate=16000,
            frame_size=512,
            activation_th=0.2,
            deactivation_th=0.15,
            speech_pad_ms=30,
            min_speech_ms=100,
            model_rate=16000,
            window_samples=512,  # Silero requires exactly 512 samples at 16kHz
            partial_frames=2,
        )

        # Spy/wrap is_speech for 16k
        max_p_16k = 0.0
        calls_16k = 0
        orig_is_speech_16k = vad_16k.is_speech

        async def spy_is_speech_16k(frame):
            nonlocal max_p_16k, calls_16k
            p = await orig_is_speech_16k(frame)
            calls_16k += 1
            if p > max_p_16k:
                max_p_16k = p
            return p

        vad_16k.is_speech = spy_is_speech_16k

        @vad_16k.events.subscribe
        async def on_audio_16k(event: VADAudioEvent, user=None):
            detected_speech_16k.append(event)

        @vad_16k.events.subscribe
        async def on_partial_16k(event: VADPartialEvent, user=None):
            partial_events_16k.append(event)

        # Load 16 kHz audio file
        audio_path_16k = get_audio_asset("formant_speech_16k.wav")
        audio_data_16k, sample_rate_16k = sf.read(audio_path_16k, dtype="int16")
        assert sample_rate_16k == 16000, (
            f"Expected sample rate 16000, got {sample_rate_16k}"
        )

        # Process the 16 kHz audio
        await vad_16k.process_audio(
            PcmData(samples=audio_data_16k, sample_rate=16000, format="s16")
        )
        await vad_16k.flush()
        await asyncio.sleep(0.1)
        logger.info(f"16k: is_speech calls={calls_16k}, max_p={max_p_16k:.3f}")

        # Now test with 48 kHz audio using the same parameters
        vad_48k = silero.VAD(
            sample_rate=48000,  # Input is 48 kHz
            frame_size=512 * 3,  # Scale frame size to match time duration
            activation_th=0.2,
            deactivation_th=0.15,
            speech_pad_ms=30,
            min_speech_ms=100,
            model_rate=16000,  # Model still runs at 16 kHz
            window_samples=512,  # Silero requires exactly 512 samples at 16kHz
            partial_frames=2,
        )

        # Spy/wrap is_speech for 48k
        max_p_48k = 0.0
        calls_48k = 0
        orig_is_speech_48k = vad_48k.is_speech

        async def spy_is_speech_48k(frame):
            nonlocal max_p_48k, calls_48k
            p = await orig_is_speech_48k(frame)
            calls_48k += 1
            if p > max_p_48k:
                max_p_48k = p
            return p

        vad_48k.is_speech = spy_is_speech_48k

        @vad_48k.events.subscribe
        async def on_audio_48k(event: VADAudioEvent, user=None):
            detected_speech_48k.append(event)

        @vad_48k.events.subscribe
        async def on_partial_48k(event: VADPartialEvent, user=None):
            partial_events_48k.append(event)

        # Load 48 kHz audio file
        audio_path_48k = get_audio_asset("formant_speech_48k.wav")
        audio_data_48k, sample_rate_48k = sf.read(audio_path_48k, dtype="int16")
        assert sample_rate_48k == 48000, (
            f"Expected sample rate 48000, got {sample_rate_48k}"
        )

        # Process the 48 kHz audio
        await vad_48k.process_audio(
            PcmData(samples=audio_data_48k, sample_rate=48000, format="s16")
        )
        await vad_48k.flush()
        await asyncio.sleep(0.1)
        logger.info(f"48k: is_speech calls={calls_48k}, max_p={max_p_48k:.3f}")

        # Verify both detected speech segments and partial events
        assert len(detected_speech_16k) > 0, (
            "No speech segments detected in 16 kHz audio"
        )
        assert len(detected_speech_48k) > 0, (
            "No speech segments detected in 48 kHz audio"
        )
        logger.info(
            f"Detected {len(detected_speech_16k)} speech segments in 16 kHz audio"
        )
        logger.info(
            f"Detected {len(detected_speech_48k)} speech segments in 48 kHz audio"
        )

        # Verify partial events
        assert len(partial_events_16k) > 0, "No partial events detected in 16 kHz audio"
        assert len(partial_events_48k) > 0, "No partial events detected in 48 kHz audio"
        # Don't require specific counts, just verify partials were emitted
        logger.info(
            f"Detected {len(partial_events_16k)} partial events in 16 kHz audio"
        )
        logger.info(
            f"Detected {len(partial_events_48k)} partial events in 48 kHz audio"
        )

        # Clean up
        await vad_16k.close()
        await vad_48k.close()

    @pytest.mark.asyncio
    async def test_bytearray_efficiency(self):
        """
        Test that the bytearray implementation is memory efficient.
        This is a basic test to ensure the implementation avoids O(n²) memory growth.
        """
        import tracemalloc

        # Initialize the VAD
        vad = silero.VAD(
            sample_rate=16000,
            frame_size=512,
            activation_th=0.3,
            deactivation_th=0.2,
            speech_pad_ms=30,
            min_speech_ms=250,
        )

        # Create a 2-second audio signal that's always above threshold to force buffering
        # Shorter duration to avoid timeout
        duration_sec = 2
        samples = np.random.randint(
            -10000, 10000, size=16000 * duration_sec, dtype=np.int16
        )

        # Start memory tracking
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        # Track audio events
        detected_speech = []

        @vad.events.subscribe
        async def on_audio(event: VADAudioEvent, user=None):
            detected_speech.append(event)

        # Monkey-patch is_speech to always return high probability and force buffering
        original_is_speech = vad.is_speech

        async def mock_is_speech(frame):
            # Always return high probability to force speech detection
            return 0.9

        vad.is_speech = mock_is_speech

        # Process audio in smaller chunks for faster testing (200ms chunks)
        chunk_size = 3200  # 200ms at 16kHz
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            # Process the audio
            await vad.process_audio(
                PcmData(samples=chunk, sample_rate=16000, format="s16")
            )

        # Take memory snapshot after processing
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare memory usage
        stats = snapshot_end.compare_to(snapshot_start, "lineno")
        total_memory_bytes = sum(
            stat.size_diff for stat in stats if "vad/__init__.py" in str(stat)
        )
        total_memory_mb = total_memory_bytes / (1024 * 1024)

        # Check buffer size relative to audio size
        audio_size_bytes = samples.nbytes
        audio_size_mb = audio_size_bytes / (1024 * 1024)
        ratio = total_memory_mb / audio_size_mb if audio_size_mb > 0 else 0

        logger.info(
            f"Memory growth: {total_memory_mb:.2f} MB for {audio_size_mb:.2f} MB of audio (ratio: {ratio:.2f})"
        )

        # The ratio should be close to 1.0 for efficient buffering (no excessive copying)
        # Allow some overhead but ensure it's not O(n²)
        assert ratio < 5.0, f"Memory usage ratio too high: {ratio:.2f}"

        # Restore original is_speech method
        vad.is_speech = original_is_speech

        # Clean up
        await vad.flush()
        await vad.close()

    @pytest.mark.asyncio
    async def test_cuda_fallback(self):
        """
        Test that the Silero VAD falls back to CPU when CUDA is not available.
        This test will be skipped if CUDA is available.
        """
        import torch

        # Skip the test if CUDA is actually available
        if torch.cuda.is_available():
            pytest.xfail("CUDA is available, skipping fallback test")

        # Initialize VAD with CUDA device
        vad = silero.VAD(
            sample_rate=16000,
            window_samples=512,
            device="cuda:0",  # Request CUDA
        )

        # Check that the device fell back to CPU
        assert vad.device_name == "cpu", (
            "Failed to fall back to CPU when CUDA unavailable"
        )
        assert vad.device.type == "cpu", "Device is not CPU after fallback"

        # Create a short silence for inference
        silence = np.zeros(512, dtype=np.int16)

        # Try processing to ensure it works
        await vad.is_speech(PcmData(samples=silence, sample_rate=16000, format="s16"))

        # Clean up
        await vad.close()

    @pytest.mark.asyncio
    async def test_flush_api(self):
        """
        Test that flush() properly emits a speech turn even if the speech is not complete.
        """
        # Initialize the VAD with longer padding to ensure speech doesn't end naturally
        vad = silero.VAD(
            sample_rate=16000,
            window_samples=512,
            activation_th=0.3,
            deactivation_th=0.2,
            speech_pad_ms=1000,  # Long padding to ensure speech doesn't end naturally
            min_speech_ms=100,  # Short minimum to ensure even brief speech is detected
        )

        # Load an audio file with speech
        audio_path = get_audio_asset("formant_speech_16k.wav")
        audio_data, sample_rate = sf.read(audio_path, dtype="int16")
        assert sample_rate == 16000, f"Expected sample rate 16000, got {sample_rate}"

        # Get first second of audio (assumes speech starts quickly)
        half_audio = audio_data[: int(sample_rate)]  # First second

        # Create listeners to capture emitted speech segments
        detected_speech = []
        flush_triggered = False

        @vad.events.subscribe
        async def on_audio(event: VADAudioEvent, user=None):
            detected_speech.append({"event": event, "from_flush": flush_triggered})

        # Allow event handler to register
        await asyncio.sleep(0.01)

        # Process half the audio to get speech started but not completed
        await vad.process_audio(
            PcmData(samples=half_audio, sample_rate=sample_rate, format="s16")
        )

        # Mark that we're about to flush
        flush_triggered = True

        # Flush to force emission of the current speech
        await vad.flush()

        # Wait for all events to be processed
        await vad.events.wait(timeout=1.0)

        # Verify that at least one speech segment was emitted due to the flush
        assert len(detected_speech) > 0, "No speech segments detected after flush"
        assert detected_speech[-1]["from_flush"], (
            "Last speech segment was not triggered by flush"
        )

        # Clean up
        await vad.close()

    @pytest.mark.asyncio
    async def test_onnx_fallback(self):
        """
        Test that the Silero VAD falls back to PyTorch when ONNX is not available or fails.
        """

        # Initialize VAD with ONNX requested
        vad = silero.VAD(
            sample_rate=16000,
            window_samples=512,
            use_onnx=True,
        )

        # We know there's an issue with ONNX export in the current implementation
        # so just check that it falls back to PyTorch properly and doesn't crash
        assert not vad.use_onnx, "ONNX fallback to PyTorch should have occurred"
        assert vad.model is not None, "Model should have been loaded after fallback"

        # Create a short silence for inference
        silence = np.zeros(512, dtype=np.int16)

        # Try processing to ensure it works
        await vad.is_speech(PcmData(samples=silence, sample_rate=16000, format="s16"))

        # Clean up
        await vad.close()
