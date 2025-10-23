"""
Root conftest.py - Shared fixtures for all tests.

Pytest automatically discovers fixtures defined here and makes them
available to all tests in the project, including plugin tests.
"""

import asyncio
import os

import numpy as np
import pytest
from dotenv import load_dotenv
from torchvision.io.video import av

from vision_agents.core.edge.types import PcmData
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent

load_dotenv()


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
        self.errors = []
        self._event = asyncio.Event()
        
        # Subscribe to events
        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            self.transcripts.append(event)
            self._event.set()
        
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


@pytest.fixture(scope="session")
def assets_dir():
    """Fixture providing the test assets directory path."""
    return get_assets_dir()


@pytest.fixture
def mia_audio_16khz():
    """Load mia.mp3 and convert to 16kHz PCM data."""
    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")
    
    # Load audio file using PyAV
    container = av.open(audio_file_path)
    audio_stream = container.streams.audio[0]
    original_sample_rate = audio_stream.sample_rate
    target_rate = 16000

    # Create resampler if needed
    resampler = None
    if original_sample_rate != target_rate:
        resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=target_rate
        )

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
    pcm = PcmData(
        samples=samples,
        sample_rate=target_rate,
        format="s16"
    )

    return pcm


@pytest.fixture
def mia_audio_48khz():
    """Load mia.mp3 and convert to 48kHz PCM data."""
    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")
    
    # Load audio file using PyAV
    container = av.open(audio_file_path)
    audio_stream = container.streams.audio[0]
    original_sample_rate = audio_stream.sample_rate
    target_rate = 48000

    # Create resampler if needed
    resampler = None
    if original_sample_rate != target_rate:
        resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=target_rate
        )

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
    pcm = PcmData(
        samples=samples,
        sample_rate=target_rate,
        format="s16"
    )

    return pcm


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

