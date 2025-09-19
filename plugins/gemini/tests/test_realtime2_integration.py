import asyncio
import os
import pytest
import numpy as np
import wave
import av
from dotenv import load_dotenv

from stream_agents.plugins.gemini.realtime2 import Realtime2
from getstream.video.rtc.track_util import PcmData

# Load environment variables
load_dotenv()


class TestRealtime2Integration:
    """Integration tests for Realtime2 connect flow"""

    
    @pytest.fixture
    def realtime2(self):
        """Create Realtime2 instance with API key"""
        return Realtime2(
            model="gemini-2.5-flash-exp-native-audio-thinking-dialog",
        )
    
    @pytest.fixture
    def audio_file_path(self):
        """Get path to test audio file"""
        return os.path.join(os.path.dirname(__file__), "../../../tests/test_assets/mia.mp3")
    
    @pytest.fixture
    def mia_audio_16khz(self, audio_file_path):
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

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime2):
        """Test sending a simple text message and receiving response"""
        await realtime2.connect()

        try:
            # Send a simple message
            events = []
            realtime2.on("audio", lambda x: events.append(x))
            await realtime2.simple_response("Hello, can you hear me?")

            # Wait for response
            await asyncio.sleep(3.0)
            assert len(events) > 0

            # Verify we have a session and it's active
            assert realtime2._session is not None

        finally:
            await realtime2.close()


    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime2, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        
        try:
            print(f"Loaded real audio file: {len(mia_audio_16khz.samples)} samples at {mia_audio_16khz.sample_rate}Hz")
            
            print("Sending real audio data...")
            # Send audio data (already at 16kHz from fixture)
            await realtime2.send_audio_pcm(mia_audio_16khz)

            
            # Wait a moment to ensure processing
            await asyncio.sleep(10.0)
            assert len(events) > 0
            
            # Verify connection is still active
            assert realtime2._session is not None
            print("Real audio sending completed successfully")
            
        finally:
            await realtime2.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_sending_flow(self, realtime2):
        """Test sending real video data and verify connection remains stable"""
        await realtime2.connect()
        
        try:
            # Load real video file
            video_file_path = os.path.join(os.path.dirname(__file__), "../../../tests/test_assets/test_video_3s.mp4")
            
            # Create a video track from real video file
            from aiortc import VideoStreamTrack
            
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
            
            real_track = RealVideoTrack(video_file_path)
            
            print("Starting real video sender...")
            # Start video sender with low FPS to avoid overwhelming the connection
            await realtime2.start_video_sender(real_track, fps=1)
            
            # Let it run for a few seconds
            await asyncio.sleep(3.0)
            
            # Stop video sender
            await realtime2.stop_video_sender()
            
            # Verify connection is still active
            assert realtime2._session is not None
            print("Real video sending completed successfully")
            
        finally:
            await realtime2.close()

