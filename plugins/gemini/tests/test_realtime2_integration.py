import asyncio
import os
import pytest
from dotenv import load_dotenv

from stream_agents.plugins.gemini.realtime2 import Realtime2
from tests.base_test import BaseTest

# Load environment variables
load_dotenv()


class TestRealtime2Integration(BaseTest):
    """Integration tests for Realtime2 connect flow"""

    @pytest.fixture
    async def realtime2(self):
        """Create and manage Realtime2 connection lifecycle"""
        realtime2 = Realtime2(
            model="gemini-2.5-flash-exp-native-audio-thinking-dialog",
        )
        try:
            yield realtime2
        finally:
            await realtime2.close()
    

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime2):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        await realtime2.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime2, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        
        await realtime2.simple_response("Listen to the following story, what is Mia looking for?")
        await asyncio.sleep(10.0)
        await realtime2.send_audio_pcm(mia_audio_16khz)

        # Wait a moment to ensure processing
        await asyncio.sleep(10.0)
        assert len(events) > 0


    @pytest.mark.integration
    async def test_video_sending_flow(self, realtime2, bunny_video_track):
        """Test sending real video data and verify connection remains stable"""
        print("Starting real video sender...")
        events = []
        realtime2.on("audio", lambda x: events.append(x))
        await realtime2.connect()
        # Start video sender with low FPS to avoid overwhelming the connection
        await realtime2.start_video_sender(bunny_video_track, fps=1)
        
        # Let it run for a few seconds
        await asyncio.sleep(3.0)
        
        # Stop video sender
        await realtime2.stop_video_sender()
        
        # Verify connection is still active
        assert realtime2._session is not None
        print("Real video sending completed successfully")

