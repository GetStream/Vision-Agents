import asyncio
import os

import numpy as np
import pytest
from torchvision.io.video import av

from stream_agents.core.edge.types import PcmData


class BaseTest:
    @pytest.fixture
    def mia_audio_16khz(self):
        audio_file_path = os.path.join(os.path.dirname(__file__), "test_assets/mia.mp3")
        """Load mia.mp3 and convert to 16kHz PCM data"""
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

        # Convert to int16 (PyAV already gives us int16, but ensure it's the right type)
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
    def bunny_video_track(self):
        """Create RealVideoTrack from video file"""
        from aiortc import VideoStreamTrack
        video_file_path = os.path.join(os.path.dirname(__file__), "test_assets/bunny_3s.mp4")

        class RealVideoTrack(VideoStreamTrack):
            def __init__(self, video_path):
                super().__init__()
                self.container = av.open(video_path)
                self.video_stream = self.container.streams.video[0]
                self.frame_count = 0
                self.max_frames = 10  # Limit to first 10 frames for testing

            async def recv(self):
                if self.frame_count >= self.max_frames:
                    raise asyncio.CancelledError("No more frames")

                try:
                    # Read frame from video
                    for frame in self.container.decode(self.video_stream):
                        self.frame_count += 1
                        # Convert to RGB
                        frame = frame.to_rgb()
                        return frame
                except Exception as e:
                    print(f"Error reading video frame: {e}")
                    raise asyncio.CancelledError("Video read error")

        return RealVideoTrack(video_file_path)