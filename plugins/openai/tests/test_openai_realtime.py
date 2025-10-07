import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.plugins.openai import Realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent

# Load environment variables
load_dotenv()


class TestOpenAIRealtime:
    """Integration tests for OpenAI Realtime API"""

    @pytest.fixture
    async def realtime(self):
        """Create and manage Realtime connection lifecycle"""
        realtime = Realtime(
            model="gpt-realtime",
            voice="alloy",
        )
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        
        # Wait for connection to be fully established
        await asyncio.sleep(2.0)
        
        # Convert 16kHz audio to 48kHz for OpenAI realtime
        # OpenAI expects 48kHz PCM audio
        import numpy as np
        from scipy import signal
        from vision_agents.core.edge.types import PcmData
        
        # Resample from 16kHz to 48kHz
        samples_16k = mia_audio_16khz.samples
        num_samples_48k = int(len(samples_16k) * 48000 / 16000)
        samples_48k = signal.resample(samples_16k, num_samples_48k).astype(np.int16)
        
        # Create new PcmData with 48kHz
        audio_48khz = PcmData(
            samples=samples_48k,
            sample_rate=48000,
            format="s16"
        )
        
        await realtime.simple_response("Listen to the following audio and tell me what you hear")
        await asyncio.sleep(5.0)
        
        # Send the resampled audio
        await realtime.simple_audio_response(audio_48khz)

        # Wait for response
        await asyncio.sleep(10.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_video_sending_flow(self, realtime, bunny_video_track):
        """Test sending real video data and verify connection remains stable"""
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Describe what you see in this video please")
        await asyncio.sleep(10.0)
        # Start video sender with low FPS to avoid overwhelming the connection
        await realtime._watch_video_track(bunny_video_track)
        
        # Let it run for a few seconds
        await asyncio.sleep(10.0)
        
        # Stop video sender
        await realtime._stop_watching_video_track()
        assert len(events) > 0

