import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import inworld

load_dotenv()


@pytest.mark.skipif(
    os.getenv("INWORLD_API_KEY") is None, reason="INWORLD_API_KEY not set"
)
@pytest.mark.integration
class TestInworldTTSIntegration:
    @pytest.fixture
    async def tts(self) -> inworld.TTS:
        return inworld.TTS()

    async def test_inworld_tts_convert_text_to_audio(self, tts: inworld.TTS):
        text = "Hello from Inworld AI."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final

    async def test_stop_audio_terminates_in_flight_stream(self, tts: inworld.TTS):
        long_text = (
            "This is a fairly long sentence that the server should "
            "synthesize across many audio chunks before completing. " * 4
        )

        stream = await tts.stream_audio(long_text)
        chunks = []
        async for pcm in stream:
            chunks.append(pcm)
            if len(chunks) == 2:
                await tts.stop_audio()

        assert len(chunks) >= 2

        follow_up = await tts.stream_audio("Hello again.")
        follow_up_chunks = [pcm async for pcm in follow_up]
        assert len(follow_up_chunks) > 0
