import os
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from vision_agents.plugins import aws as aws_plugin
from vision_agents.core.tts.testing import TTSSession
from vision_agents.core.tts.manual_test import manual_tts_to_wav


load_dotenv()


def _has_aws_creds() -> bool:
    return any(
        os.environ.get(k)
        for k in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
    )


class TestAWSPollyTTS:
    @pytest_asyncio.fixture
    async def tts(self) -> aws_plugin.TTS:  # type: ignore[name-defined]
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Polly TTS tests")
        # Region can be overridden via AWS_REGION/AWS_DEFAULT_REGION
        return aws_plugin.TTS(voice_id=os.environ.get("AWS_POLLY_VOICE", "Joanna"))

    @pytest.mark.integration
    async def test_aws_polly_tts_speech(self, tts: aws_plugin.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)

        await tts.send("Hello from AWS Polly TTS")

        result = await session.wait_for_result(timeout=30.0)
        assert not result.errors
        assert len(result.speeches) > 0

    @pytest.mark.integration
    async def test_aws_polly_tts_manual_wav(self, tts: aws_plugin.TTS):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
