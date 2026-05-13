import asyncio

import pytest
from dotenv import load_dotenv
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.plugins import mistral

load_dotenv()


class TestMistralSTT:
    """Integration tests for Mistral Voxtral STT."""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    async def stt(self):
        stt = mistral.STT()
        await stt.start()
        yield stt
        await stt.close()

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(
        self, stt, mia_audio_48khz_chunked, participant
    ):
        """Test transcription with chunked audio (simulates real-time streaming)."""
        # Send audio in chunks like real-time streaming
        for chunk in mia_audio_48khz_chunked:
            await stt.process_audio(chunk, participant)
            await asyncio.sleep(
                0.001
            )  # Simulate real-time pacing, allow receive task to run

        items = await stt.output.collect(5.0)
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
