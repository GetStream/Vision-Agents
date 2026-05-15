import asyncio
import os

import pytest
from dotenv import load_dotenv
from getstream.video.rtc import PcmData
from vision_agents.core.llm.realtime import RealtimeAudioOutput
from vision_agents.plugins.xai import Realtime

load_dotenv()


class TestXAIRealtimeConfiguration:
    """Unit tests for xAI Realtime configuration options."""

    async def test_default_configuration(self):
        """Test that default configuration is set correctly."""
        realtime = Realtime(api_key="test-key")
        assert realtime.model == "grok-voice-think-fast-1.0"
        assert realtime.voice == "ara"
        # xAI realtime emits PCM at 24 kHz natively.
        assert realtime.sample_rate == 24000
        assert realtime.turn_detection == "server_vad"
        assert realtime.provider_name == "xai_realtime"
        # VAD interrupt defaults to False to avoid mic-echo cancellation.
        assert realtime.vad_interrupt_response is False
        # Web search and X search enabled by default
        assert realtime.web_search is True
        assert realtime.x_search is True
        assert realtime.x_search_allowed_handles is None

    async def test_custom_configuration(self):
        """Test custom configuration options."""
        realtime = Realtime(
            api_key="test-key",
            voice="rex",
            turn_detection=None,
            vad_interrupt_response=True,
        )
        assert realtime.voice == "rex"
        assert realtime.sample_rate == 24000
        assert realtime.turn_detection is None
        assert realtime.vad_interrupt_response is True

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


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
class TestXAIRealtimeIntegration:
    """End-to-end tests against the live xAI Realtime API."""

    @pytest.fixture
    async def realtime(self):
        rt = Realtime(api_key=os.getenv("XAI_API_KEY"), voice="ara")
        try:
            await rt.connect()
            yield rt
        finally:
            await rt.close()

    async def test_simple_response_flow(self, realtime):
        """A text prompt produces an audio response."""
        async for _ in realtime.simple_response(
            "Hello, can you hear me? Say yes briefly."
        ):
            pass

        await asyncio.sleep(5.0)
        audio = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAudioOutput)
        ]
        assert len(audio) > 0, "Expected audio output events"

    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Audio chunks sent to xAI produce an audio response."""
        async for _ in realtime.simple_response(
            "Listen to the following audio and describe what you hear briefly."
        ):
            pass
        await asyncio.sleep(3.0)

        # Send audio in 100 ms chunks at the source sample rate.
        chunk_size = realtime.sample_rate // 10
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

        await asyncio.sleep(8.0)
        audio = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAudioOutput)
        ]
        assert len(audio) > 0, "Expected audio output events after sending audio"

    @pytest.fixture
    async def realtime_with_tools(self):
        """Realtime with get_weather registered, then connected."""
        rt = Realtime(api_key=os.getenv("XAI_API_KEY"), voice="ara")

        @rt.register_function(description="Get the current weather")
        async def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"The weather in {location} is sunny and 72 degrees."

        try:
            await rt.connect()
            yield rt
        finally:
            await rt.close()

    async def test_function_calling(self, realtime_with_tools):
        """Function calling completes without raising."""
        async for _ in realtime_with_tools.simple_response(
            "What is the weather in San Francisco?"
        ):
            pass

        await asyncio.sleep(8.0)
