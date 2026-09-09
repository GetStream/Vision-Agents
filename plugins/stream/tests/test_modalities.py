import asyncio
import os

import pytest
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import CUSTOMER_ENV, URL_ENV

STT_TARGET = os.getenv("STREAM_ACCELERATION_STT", "en-low-latency")
TTS_TARGET = os.getenv("STREAM_ACCELERATION_TTS", "en-low-latency")
LLM_TARGET = os.getenv("STREAM_ACCELERATION_LLM", "llm-fast")


def _require_router() -> None:
    if not os.getenv(URL_ENV) or not os.getenv(CUSTOMER_ENV):
        pytest.fail(
            f"These tests need a running acceleration router. Set {URL_ENV} and "
            f"{CUSTOMER_ENV} in the environment or in a .env file before running tests "
            "marked with @pytest.mark.integration.",
            pytrace=False,
        )


@pytest.mark.integration
class TestModalityStreams:
    @pytest.fixture(autouse=True)
    def router(self) -> None:
        _require_router()

    async def test_speech_comes_back_as_audio_that_can_be_played(self):
        tts = stream.TTS(TTS_TARGET)
        await tts.start()
        try:
            chunks = [
                chunk
                async for chunk in tts.send_iter("Hello from the acceleration router.")
                if chunk.data is not None
            ]
        finally:
            await tts.close()

        assert chunks
        first = chunks[0].data
        assert isinstance(first, PcmData)
        assert first.sample_rate > 0
        assert len(first.samples) > 0

    async def test_a_completion_comes_back_as_text(self):
        llm = stream.LLM(LLM_TARGET)
        await llm.start()
        try:
            answers = [answer async for answer in llm.simple_response("Say hello.")]
        finally:
            await llm.close()

        assert answers
        assert answers[-1].text

    async def test_a_name_routes_to_the_modality_that_serves_it(self):
        # resolve asks the router which modality serves the name and blocks on the answer,
        # since it is what builds an agent rather than something a call waits on.
        routed = await asyncio.to_thread(stream.Router().resolve, TTS_TARGET)

        assert isinstance(routed, stream.TTS)
