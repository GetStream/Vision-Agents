import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.core.tts.manual_test import play_pcm_with_ffplay
from vision_agents.plugins.gemini import Realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent
from getstream.video.rtc import PcmData, AudioFormat

# Load environment variables
load_dotenv()


@pytest.fixture
async def realtime():
    """Create and manage Realtime connection lifecycle"""
    realtime = Realtime(
        model="gemini-2.0-flash-exp",
    )
    try:
        yield realtime
    finally:
        await realtime.close()


class TestGeminiRealtime:
    """Integration tests for Gemini Realtime connect flow"""

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        pcm = PcmData(sample_rate=24000, format=AudioFormat.S16)

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
            pcm.append(event.data)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

        # play the generated audio
        await play_pcm_with_ffplay(pcm)

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        await realtime.simple_response(
            "Listen to the following story, what is Mia looking for?"
        )
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
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
        await realtime.watch_video_track(bunny_video_track)

        # Let it run for a few seconds
        await asyncio.sleep(10.0)

        # Stop video sender
        await realtime._stop_watching_video_track()
        assert len(events) > 0
