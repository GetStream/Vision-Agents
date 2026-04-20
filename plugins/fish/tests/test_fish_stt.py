import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.plugins import fish

load_dotenv()


class TestFishSTT:
    """Integration tests for Fish Audio STT"""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    async def stt(self):
        stt = fish.STT()
        yield stt
        await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, stt, mia_audio_16khz, participant):
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz, participant):
        await stt.process_audio(mia_audio_48khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
