import asyncio
import os

import pytest
from dotenv import load_dotenv

from getstream.video.rtc import PcmData, AudioFormat
from vision_agents.core.llm.events import RealtimeAudioOutputEvent
from vision_agents.plugins.xai import Realtime

load_dotenv()


@pytest.fixture
async def realtime():
    """Create and manage Realtime connection lifecycle."""
    realtime = Realtime(
        api_key=os.getenv("XAI_API_KEY"),
        voice="Ara",
    )
    try:
        yield realtime
    finally:
        await realtime.close()


class TestXAIRealtime:
    """Integration tests for xAI Realtime."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving audio response."""
        events = []
        pcm = PcmData(sample_rate=24000, format=AudioFormat.S16)

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
            pcm.append(event.data)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me? Say yes briefly.")

        # Wait for response
        await asyncio.sleep(5.0)
        assert len(events) > 0, "Expected audio output events"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable."""
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        # Send instruction and audio
        await realtime.simple_response(
            "Listen to the following audio and describe what you hear briefly."
        )
        await asyncio.sleep(3.0)

        # Send audio in chunks
        chunk_size = realtime.sample_rate // 10  # 100ms chunks
        samples = mia_audio_16khz.samples
        for i in range(0, len(samples), chunk_size):
            chunk_samples = samples[i : i + chunk_size]
            if len(chunk_samples) > 0:
                chunk_pcm = PcmData(
                    samples=chunk_samples,
                    sample_rate=mia_audio_16khz.sample_rate,
                    format=mia_audio_16khz.format,
                )
                await realtime.simple_audio_response(chunk_pcm)
            await asyncio.sleep(0.05)

        # Wait for processing and response
        await asyncio.sleep(8.0)
        assert len(events) > 0, "Expected audio output events after sending audio"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_function_calling(self, realtime):
        """Test function calling with xAI realtime."""

        @realtime.register_function(description="Get the current weather")
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"The weather in {location} is sunny and 72 degrees."

        await realtime.connect()
        await realtime.simple_response("What is the weather in San Francisco?")

        # Wait for function call and response
        await asyncio.sleep(8.0)
        # The test passes if no exceptions are raised during function calling


class TestXAIRealtimeConfiguration:
    """Tests for xAI Realtime configuration options."""

    async def test_default_configuration(self):
        """Test that default configuration is set correctly."""
        realtime = Realtime(api_key="test-key")
        assert realtime.voice == "Ara"
        assert realtime.sample_rate == 48000
        assert realtime.turn_detection == "server_vad"
        assert realtime.provider_name == "xai"
        # Web search and X search enabled by default
        assert realtime.web_search is True
        assert realtime.x_search is True
        assert realtime.x_search_allowed_handles is None

    async def test_custom_configuration(self):
        """Test custom configuration options."""
        realtime = Realtime(
            api_key="test-key",
            voice="Rex",
            turn_detection=None,
        )
        assert realtime.voice == "Rex"
        assert realtime.sample_rate == 48000  # Always 48kHz
        assert realtime.turn_detection is None

    async def test_search_tools_can_be_disabled(self):
        """Test that web_search and x_search can be disabled."""
        realtime = Realtime(
            api_key="test-key",
            web_search=False,
            x_search=False,
        )
        assert realtime.web_search is False
        assert realtime.x_search is False

    async def test_x_search_allowed_handles(self):
        """Test that X search allowed handles can be configured."""
        realtime = Realtime(
            api_key="test-key",
            x_search_allowed_handles=["elonmusk", "xai"],
        )
        assert realtime.x_search_allowed_handles == ["elonmusk", "xai"]

    async def test_api_key_required(self):
        """Test that API key is required."""
        # Temporarily unset the environment variable
        original_key = os.environ.pop("XAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="XAI API key is required"):
                Realtime()
        finally:
            if original_key:
                os.environ["XAI_API_KEY"] = original_key

    async def test_instructions_setting(self):
        """Test that instructions can be set."""
        realtime = Realtime(api_key="test-key")
        realtime.set_instructions("You are a helpful assistant.")
        assert realtime._instructions == "You are a helpful assistant."
