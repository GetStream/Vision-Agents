"""
Root conftest.py - Shared fixtures for all tests.

Pytest automatically discovers fixtures defined here and makes them
available to all tests in the project, including plugin tests.
"""

import asyncio
import logging
import os
from typing import Iterator

import numpy as np
import pytest
from blockbuster import BlockBuster, blockbuster_ctx
from dotenv import load_dotenv
from getstream.video.rtc.track_util import AudioFormat, PcmData
from torchvision.io.video import av
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import (
    STTErrorEvent,
    STTPartialTranscriptEvent,
    STTTranscriptEvent,
)

load_dotenv()


def skip_blockbuster(func_or_class):
    """Decorator to skip blockbuster checks for a test function or class.

    Use this decorator when testing code that makes unavoidable blocking calls
    (e.g., third-party SDKs like boto3, fish-audio-sdk).

    Examples:
        @skip_blockbuster
        async def test_aws_function():
            # boto3 makes blocking calls we can't fix
            pass

        @skip_blockbuster
        class TestAWSIntegration:
            # All tests in this class skip blockbuster
            pass
    """
    return pytest.mark.skip_blockbuster(func_or_class)


@pytest.fixture(autouse=True)
def blockbuster(request) -> Iterator[BlockBuster | None]:
    """Blockbuster fixture that detects blocking calls in async code.

    Can be disabled for specific tests using the @skip_blockbuster decorator.
    """
    # Check if test is marked to skip blockbuster
    if request.node.get_closest_marker("skip_blockbuster") or request.config.getoption(
        "--skip-blockbuster"
    ):
        yield None
    else:
        # Always allow blocking calls inside "Agent.__init__".
        # Agent.__init__ is called once before any processing, so it's ok for it to be blocking.
        from vision_agents.core import Agent

        agent_cls_file = Agent.__module__.replace(".", "/") + ".py"

        with blockbuster_ctx() as bb:
            for func in bb.functions.values():
                func.can_block_in(agent_cls_file, "__init__")

            # Allow Python's standard logging which is inherently synchronous.
            if "io.TextIOWrapper.write" in bb.functions:
                bb.functions["io.TextIOWrapper.write"].deactivate()
            if "io.BufferedWriter.write" in bb.functions:
                bb.functions["io.BufferedWriter.write"].deactivate()

            yield bb


class STTSession:
    """Helper class for testing STT implementations.

    Automatically subscribes to transcript and error events,
    collects them, and provides a convenient wait method.
    """

    def __init__(self, stt):
        """Initialize STT session with an STT object.

        Args:
            stt: STT implementation to monitor
        """
        self.stt = stt
        self.transcripts = []
        self.partial_transcripts = []
        self.errors = []
        self.logger = logging.getLogger("STTSession")
        self._event = asyncio.Event()

        # Subscribe to events
        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            self.logger.info(f"Received transcript event: {event}")
            self.transcripts.append(event)
            self._event.set()

        @stt.events.subscribe
        async def on_partial_transcript(event: STTPartialTranscriptEvent):
            self.logger.info("Partial transcript event: %s", event)
            self.partial_transcripts.append(event)

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            self.errors.append(event.error)
            self._event.set()

        self._on_transcript = on_transcript
        self._on_error = on_error

    async def wait_for_result(self, timeout: float = 30.0):
        """Wait for either a transcript or error event.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            asyncio.TimeoutError: If no result received within timeout
        """
        # Allow event subscriptions to be processed
        await asyncio.sleep(0.01)

        # Wait for an event
        await asyncio.wait_for(self._event.wait(), timeout=timeout)

    def get_full_transcript(self) -> str:
        """Get full transcription text from all transcript events.

        Returns:
            Combined text from all transcripts
        """
        return " ".join(t.text for t in self.transcripts)


def get_assets_dir():
    """Get the test assets directory path."""
    return os.path.join(os.path.dirname(__file__), "tests", "test_assets")


def _mp3_to_pcm(path: str, target_rate: int) -> PcmData:
    # Load audio file using PyAV
    container = av.open(path)
    audio_stream = container.streams.audio[0]
    original_sample_rate = audio_stream.sample_rate

    # Create resampler if needed
    resampler = None
    if original_sample_rate != target_rate:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)

    # Read all audio frames
    samples = []
    for frame in container.decode(audio_stream):
        # Resample if needed
        if resampler:
            frame = resampler.resample(frame)[0]

        # Convert to numpy array
        frame_array = frame.to_ndarray()
        if len(frame_array.shape) > 1:
            # Convert stereo to mono
            frame_array = np.mean(frame_array, axis=0)
        samples.append(frame_array)

    # Concatenate all samples
    samples = np.concatenate(samples)

    # Convert to int16
    samples = samples.astype(np.int16)
    container.close()

    # Create PCM data
    pcm = PcmData(samples=samples, sample_rate=target_rate, format=AudioFormat.S16)

    return pcm


@pytest.fixture(scope="session")
def assets_dir():
    """Fixture providing the test assets directory path."""
    return get_assets_dir()


@pytest.fixture
def participant():
    """Create a test participant for STT testing."""
    return Participant({}, user_id="test-user")


@pytest.fixture
def mia_audio_16khz():
    """Load mia.mp3 and convert to 16kHz PCM data."""
    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")
    pcm = _mp3_to_pcm(audio_file_path, 16000)
    return pcm


@pytest.fixture
def describe_what_you_see_audio_16khz():
    """Load describe_what_you_see.mp3 and convert to 16kHz PCM data."""
    audio_file_path = os.path.join(get_assets_dir(), "describe_what_you_see.mp3")
    pcm = _mp3_to_pcm(audio_file_path, 16000)
    return pcm


@pytest.fixture
def mia_audio_48khz():
    """Load mia.mp3 and convert to 48kHz PCM data."""
    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")
    pcm = _mp3_to_pcm(audio_file_path, 48000)
    return pcm


@pytest.fixture
def silence_2s_48khz():
    """Generate 2 seconds of silence at 48kHz PCM data."""
    sample_rate = 48000
    duration_seconds = 2.0

    # Calculate number of samples for 2 seconds
    num_samples = int(sample_rate * duration_seconds)

    # Create silence (zeros) as int16
    samples = np.zeros(num_samples, dtype=np.int16)

    # Create PCM data
    pcm = PcmData(samples=samples, sample_rate=sample_rate, format=AudioFormat.S16)

    return pcm


@pytest.fixture
def silence_1s_16khz():
    """Generate 1 seconds of silence at 16kHz PCM data."""
    sample_rate = 16000

    # Create silence (zeros) as int16
    samples = np.zeros(sample_rate, dtype=np.int16)

    # Create PCM data
    pcm = PcmData(samples=samples, sample_rate=sample_rate, format=AudioFormat.S16)

    return pcm


@pytest.fixture
def mia_audio_48khz_chunked():
    """Load mia.mp3 and yield 48kHz PCM data in 20ms chunks."""
    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")

    # Load audio file using PyAV
    container = av.open(audio_file_path)
    audio_stream = container.streams.audio[0]
    original_sample_rate = audio_stream.sample_rate
    target_rate = 48000

    # Create resampler if needed
    resampler = None
    if original_sample_rate != target_rate:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)

    # Read all audio frames
    samples = []
    for frame in container.decode(audio_stream):
        # Resample if needed
        if resampler:
            frame = resampler.resample(frame)[0]

        # Convert to numpy array
        frame_array = frame.to_ndarray()
        if len(frame_array.shape) > 1:
            # Convert stereo to mono
            frame_array = np.mean(frame_array, axis=0)
        samples.append(frame_array)

    # Concatenate all samples
    samples = np.concatenate(samples)

    # Convert to int16
    samples = samples.astype(np.int16)
    container.close()

    # Calculate chunk size for 20ms at 48kHz
    chunk_size = int(target_rate * 0.020)  # 960 samples per 20ms

    # Yield chunks of audio
    chunks = []
    for i in range(0, len(samples), chunk_size):
        chunk_samples = samples[i : i + chunk_size]

        # Create PCM data for this chunk
        pcm_chunk = PcmData(
            samples=chunk_samples, sample_rate=target_rate, format=AudioFormat.S16
        )
        chunks.append(pcm_chunk)

    return chunks


@pytest.fixture
def golf_swing_image():
    """Load golf_swing.png image and return as bytes."""
    image_file_path = os.path.join(get_assets_dir(), "golf_swing.png")

    with open(image_file_path, "rb") as f:
        image_bytes = f.read()

    return image_bytes


@pytest.fixture
async def bunny_video_track():
    """Create RealVideoTrack from video file."""
    from aiortc import VideoStreamTrack

    video_file_path = os.path.join(get_assets_dir(), "bunny_3s.mp4")

    class RealVideoTrack(VideoStreamTrack):
        def __init__(self, video_path, max_frames=None):
            super().__init__()
            self.container = av.open(video_path)
            self.video_stream = self.container.streams.video[0]
            self.frame_count = 0
            self.max_frames = max_frames
            self.frame_duration = 1.0 / 15.0  # 15 fps

        async def recv(self):
            if self.max_frames is not None and self.frame_count >= self.max_frames:
                raise asyncio.CancelledError("No more frames")

            try:
                for frame in self.container.decode(self.video_stream):
                    if frame is None:
                        raise asyncio.CancelledError("End of video stream")

                    self.frame_count += 1
                    frame = frame.to_rgb()
                    await asyncio.sleep(self.frame_duration)
                    return frame

                raise asyncio.CancelledError("End of video stream")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if "End of file" in str(e) or "avcodec_send_packet" in str(e):
                    raise asyncio.CancelledError("End of video stream")
                else:
                    print(f"Error reading video frame: {e}")
                    raise asyncio.CancelledError("Video read error")

    track = RealVideoTrack(video_file_path, max_frames=None)
    try:
        yield track
    finally:
        track.container.close()


@pytest.fixture
async def audio_track_48khz():
    """Create audio track that produces 48kHz audio frames."""
    from getstream.video.rtc.audio_track import AudioStreamTrack

    audio_file_path = os.path.join(get_assets_dir(), "formant_speech_48k.wav")

    class TestAudioTrack(AudioStreamTrack):
        def __init__(self, audio_path):
            super().__init__()
            self.container = av.open(audio_path)
            self.audio_stream = self.container.streams.audio[0]
            self.decoder = self.container.decode(self.audio_stream)

        async def recv(self):
            try:
                frame = next(self.decoder)
                return frame
            except StopIteration:
                raise asyncio.CancelledError("End of audio stream")

    track = TestAudioTrack(audio_file_path)
    try:
        yield track
    finally:
        track.container.close()


def pytest_addoption(parser):
    parser.addoption(
        "--skip-blockbuster",
        "--skip-bb",
        action="store_true",
        default=False,
        help="Skip BlockBuster blocking calls detection for the test run",
    )
