import pytest
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded
from vision_agents.plugins import deepgram


class TestDeepgramSTT:
    """Integration tests for Deepgram STT"""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture(params=[True, False], ids=["eager", "no_eager"])
    async def deepgram_stt(self, request):
        stt = deepgram.STT(eager_turn_detection=request.param)
        await stt.start()
        yield stt
        await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, deepgram_stt, mia_audio_48khz, silence_2s_48khz, participant
    ):
        await deepgram_stt.process_audio(mia_audio_48khz, participant=participant)
        # Send 2 seconds of silence to trigger end of turn
        await deepgram_stt.process_audio(silence_2s_48khz, participant=participant)

        items = await deepgram_stt.output.collect(timeout=10.0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert transcripts, "No Transcript emitted by Deepgram STT"
        finals = [t for t in transcripts if t.final]
        assert finals, "No final Transcript emitted by Deepgram STT"
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
        assert transcripts[0].participant == participant

        assert any(isinstance(i, TurnEnded) for i in items)

    async def test_close_closes_http_client(self):
        stt = deepgram.STT(api_key="fake")
        httpx_client = stt.client._client_wrapper.httpx_client.httpx_client

        assert httpx_client.is_closed is False
        await stt.close()
        assert httpx_client.is_closed is True
