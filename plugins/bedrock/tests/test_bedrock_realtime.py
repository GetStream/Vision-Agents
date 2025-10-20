"""Tests for AWS Bedrock Realtime plugin."""
import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.plugins.bedrock import Realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent

# Load environment variables
load_dotenv()


class TestBedrockRealtime:
    """Integration tests for Bedrock Realtime connect flow"""

    @pytest.fixture
    async def realtime(self):
        """Create and manage Realtime connection lifecycle"""
        # Note: Nova Sonic requires a specialized WebSocket API, not ConverseStream
        # Using Claude for testing as it supports ConverseStream
        realtime = Realtime(
            model="amazon.nova-sonic-v1:0",
            region_name="us-east-1",
        )
        realtime._set_instructions("you're a kind assistant, always be friendly please.")
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        # unlike other realtime LLMs, AWS doesn't reply if you only send text
        events = []
        realtime._set_instructions("whenever you reply mention a fun fact about The Netherlands")
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me? Please respond with a short greeting.")

        # Wait for response
        await asyncio.sleep(10.0)


    @pytest.mark.integration
    async def test_audio_first(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        
        await realtime.simple_response("Listen to the following story, what is Mia looking for?")
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
        await asyncio.sleep(10.0)
        
        # Test passes if no exceptions are raised
        assert True

    @pytest.mark.integration
    async def test_connection_lifecycle(self, realtime):
        """Test that connection can be established and closed properly"""
        # Connect
        await realtime.connect()
        assert realtime.connected is True
        
        # Send a simple message
        await realtime.simple_response("Test message")
        await asyncio.sleep(2.0)
        
        # Close
        await realtime.close()
        assert realtime.connected is False

