"""
Root conftest.py - Shared fixtures for all tests.

Pytest automatically discovers fixtures defined here and makes them
available to all tests in the project, including plugin tests.
"""

import asyncio
import os

import numpy as np
import pytest
from torchvision.io.video import av

from vision_agents.core.edge.types import PcmData


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

