"""Tests for AWS Realtime plugin."""

import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.agent_types import AgentOptions
from vision_agents.core.instructions import Instructions

from vision_agents.core.tts.manual_test import play_pcm_with_ffplay
from vision_agents.plugins.aws import Realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent
from getstream.video.rtc import PcmData, AudioFormat

# Load environment variables
load_dotenv()


class TestBedrockRealtime:
    """Integration tests for AWS Bedrock Realtime connect flow"""

    @pytest.fixture
    async def realtime(self, tmp_path):
        """Create and manage Realtime connection lifecycle"""
        # Using AWS Nova Sonic model for testing
        realtime = Realtime(
            model="amazon.nova-sonic-v1:0",
            region_name="us-east-1",
        )

        realtime.options = AgentOptions(model_dir=tmp_path.as_posix())
        await realtime.warmup()
        realtime.set_instructions(
            Instructions("you're a kind assistant, always be friendly please.")
        )
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        # unlike other realtime LLMs, AWS doesn't reply if you only send text
        events = []
        realtime.set_instructions(
            Instructions("whenever you reply mention a fun fact about The Netherlands")
        )

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response(
            "Hello, can you hear me? Please respond with a short greeting."
        )

        # Wait for response
        await asyncio.sleep(10.0)

    @pytest.mark.integration
    async def test_audio_first(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        pcm = PcmData(sample_rate=24000, format=AudioFormat.S16)

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
            pcm.append(event.data)

        await asyncio.sleep(0.01)
        await realtime.connect()

        await realtime.simple_response(
            "Listen to the following story, what is Mia looking for?"
        )
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
        await asyncio.sleep(10.0)

        # play the generated audio
        await play_pcm_with_ffplay(pcm)

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

    @pytest.mark.integration
    async def test_function_calling(self, realtime):
        """Test that function calling works with AWS Bedrock Realtime"""

        # Register a test function
        @realtime.register_function(
            name="get_test_data", description="Get test data for a given key"
        )
        def get_test_data(key: str) -> dict:
            """Get test data.

            Args:
                key: The key to look up

            Returns:
                Test data for the key
            """
            test_data = {
                "weather": {"value": "sunny", "temp": 72},
                "time": {"value": "12:00 PM"},
            }
            return test_data.get(key, {"error": "Key not found"})

        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        # Ask the model to use the function
        await realtime.simple_response(
            "Please use the get_test_data function to get data for the weather key and tell me what you find."
        )

        # Wait for response
        await asyncio.sleep(15.0)

        # Verify we got some audio output (meaning the model responded after function call)
        assert len(events) > 0


class TestNova2Realtime:
    @pytest.fixture
    async def realtime(self, tmp_path):
        """Create and manage Realtime connection lifecycle"""
        # Using AWS Nova Sonic model for testing
        realtime = Realtime(
            model="amazon.nova-2-sonic-v1:0",
            region_name="us-east-1",
        )

        realtime.options = AgentOptions(model_dir=tmp_path.as_posix())
        await realtime.warmup()
        realtime.set_instructions("you're a kind assistant, always be friendly please.")
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        # unlike other realtime LLMs, AWS doesn't reply if you only send text
        events = []
        realtime.set_instructions(
            "whenever you reply mention a fun fact about The Netherlands"
        )

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response(
            "Hello, can you hear me? Please respond with a short greeting."
        )

        # Wait for response
        await asyncio.sleep(10.0)

    @pytest.mark.integration
    async def test_audio_first(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        pcm = PcmData(sample_rate=24000, format=AudioFormat.S16)

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
            pcm.append(event.data)

        await asyncio.sleep(0.01)
        await realtime.connect()

        await realtime.simple_response(
            "Listen to the following story, what is Mia looking for?"
        )
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
        await asyncio.sleep(10.0)

        # play the generated audio
        await play_pcm_with_ffplay(pcm)

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

    @pytest.mark.integration
    async def test_function_calling(self, realtime):
        """Test that function calling works with AWS Bedrock Realtime"""

        # Register a test function
        @realtime.register_function(
            name="get_test_data", description="Get test data for a given key"
        )
        def get_test_data(key: str) -> dict:
            """Get test data.

            Args:
                key: The key to look up

            Returns:
                Test data for the key
            """
            test_data = {
                "weather": {"value": "sunny", "temp": 72},
                "time": {"value": "12:00 PM"},
            }
            return test_data.get(key, {"error": "Key not found"})

        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        # Ask the model to use the function
        await realtime.simple_response(
            "Please use the get_test_data function to get data for the weather key and tell me what you find."
        )

        # Wait for response
        await asyncio.sleep(15.0)

        # Verify we got some audio output (meaning the model responded after function call)
        assert len(events) > 0
