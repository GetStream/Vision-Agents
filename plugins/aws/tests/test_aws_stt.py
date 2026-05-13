import asyncio

import pytest
from dotenv import load_dotenv
from vision_agents.core.stt import Transcript
from vision_agents.plugins import aws

load_dotenv()


class TestTranscribeSTT:
    def test_partial_static_credentials_rejected(self):
        with pytest.raises(ValueError, match="provided together"):
            aws.STT(aws_access_key_id="AKIA...")
        with pytest.raises(ValueError, match="provided together"):
            aws.STT(aws_secret_access_key="secret")


@pytest.mark.integration
class TestTranscribeSTTIntegration:
    @pytest.fixture
    async def stt(self):
        stt = aws.STT(language_code="en-US")
        await stt.start()
        yield stt
        await stt.close()

    async def test_transcribe_mia_audio_16khz(
        self, stt, mia_audio_16khz_chunked, participant
    ):
        # AWS Transcribe expects audio at real-time pace; without pacing the
        # silence-detection window never fires before the 15s idle timeout.
        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.002)

        items = await stt.output.collect(timeout=5.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        full_transcript = " ".join(t.text for t in transcripts if t.final).lower()
        assert any(
            word in full_transcript for word in ["village", "quiet", "mia", "treasures"]
        )

    async def test_transcribe_mia_audio_48khz(
        self, stt, mia_audio_48khz_chunked, participant
    ):
        for chunk in mia_audio_48khz_chunked:
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.002)

        items = await stt.output.collect(timeout=5.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        full_transcript = " ".join(t.text for t in transcripts if t.final).lower()
        assert any(
            word in full_transcript for word in ["village", "quiet", "mia", "treasures"]
        )
        assert transcripts[0].participant == participant

    async def test_partial_transcripts_emitted(
        self, stt, mia_audio_16khz_chunked, participant
    ):
        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.002)

        items = await stt.output.collect(timeout=5.0)
        partials = [
            i for i in items if isinstance(i, Transcript) and i.mode == "replacement"
        ]
        assert partials
