import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.plugins import assemblyai
from conftest import STTSession


class TestAssemblyAISTT:
    """Integration tests for AssemblyAI STT."""

    @pytest.fixture
    async def stt(self):
        """Create and manage AssemblyAI STT lifecycle."""
        stt = assemblyai.STT()
        try:
            await stt.start()
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, stt, mia_audio_48khz, silence_2s_48khz
    ):
        session = STTSession(stt)

        await stt.process_audio(
            mia_audio_48khz, participant=Participant({}, user_id="hi", id="hi")
        )

        await stt.process_audio(
            silence_2s_48khz, participant=Participant({}, user_id="hi", id="hi")
        )

        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert full_transcript is not None
        assert "forgotten treasures" in full_transcript.lower()

        assert session.transcripts[0].participant.user_id == "hi"

    async def test_prompt_and_keyterms_exclusive(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            assemblyai.STT(
                api_key="test",
                prompt="test prompt",
                keyterms_prompt=["term1"],
            )

    async def test_default_configuration(self):
        stt = assemblyai.STT(api_key="test-key")
        assert stt._speech_model == "u3-rt-pro"
        assert stt._sample_rate == 16000
        assert stt.turn_detection is True
        assert stt.provider_name == "assemblyai"

    async def test_reconnect_defaults(self):
        stt = assemblyai.STT(api_key="test-key")
        assert stt._max_reconnect_attempts == 3
        assert stt._reconnect_backoff_initial_s == 0.5
        assert stt._reconnect_backoff_max_s == 4.0

    async def test_custom_reconnect_config(self):
        stt = assemblyai.STT(
            api_key="test-key",
            max_reconnect_attempts=5,
            reconnect_backoff_initial_s=1.0,
            reconnect_backoff_max_s=8.0,
        )
        assert stt._max_reconnect_attempts == 5
        assert stt._reconnect_backoff_initial_s == 1.0
        assert stt._reconnect_backoff_max_s == 8.0
