import asyncio
import os

import numpy as np
import pytest
from torchvision.io.video import av

from getstream.video.rtc.track_util import PcmData, AudioFormat


class BaseTest:
    @property
    def assets_dir(self):
        """Get the test assets directory path."""
        return os.path.join(os.path.dirname(__file__), "test_assets")

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
                format=AudioFormat.S16, layout="mono", rate=target_rate
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
        pcm = PcmData(samples=samples, sample_rate=target_rate, format=AudioFormat.S16)

        return pcm

    @pytest.fixture
    def bunny_video_track(self):
        """Create RealVideoTrack from video file"""
        from aiortc import VideoStreamTrack

        video_file_path = os.path.join(
            os.path.dirname(__file__), "test_assets/bunny_3s.mp4"
        )

        class RealVideoTrack(VideoStreamTrack):
            def __init__(self, video_path, max_frames=None):
                super().__init__()
                self.container = av.open(video_path)
                self.video_stream = self.container.streams.video[0]
                self.frame_count = 0
                self.max_frames = max_frames  # None means no limit
                self.frame_duration = 1.0 / 15.0  # 15 fps source video

            async def recv(self):
                if self.max_frames is not None and self.frame_count >= self.max_frames:
                    raise asyncio.CancelledError("No more frames")

                try:
                    # Read frame from video
                    for frame in self.container.decode(self.video_stream):
                        if frame is None:
                            raise asyncio.CancelledError("End of video stream")

                        self.frame_count += 1
                        # Convert to RGB
                        frame = frame.to_rgb()

                        # Sleep for realistic video timing
                        await asyncio.sleep(self.frame_duration)

                        return frame

                    # If we get here, we've exhausted all frames in the stream
                    raise asyncio.CancelledError("End of video stream")

                except asyncio.CancelledError:
                    # Re-raise CancelledError as-is
                    raise
                except Exception as e:
                    # Check if it's an end-of-file error
                    if "End of file" in str(e) or "avcodec_send_packet" in str(e):
                        raise asyncio.CancelledError("End of video stream")
                    else:
                        print(f"Error reading video frame: {e}")
                        raise asyncio.CancelledError("Video read error")

        return RealVideoTrack(video_file_path, max_frames=None)
