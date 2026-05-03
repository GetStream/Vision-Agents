import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import aws

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


@pytest.mark.integration
class TestAWSPollyTTSIntegration:
    @pytest.fixture
    async def tts(self) -> aws.TTS:
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Polly TTS tests")
        # Region can be overridden via AWS_REGION/AWS_DEFAULT_REGION
        return aws.TTS(voice_id=os.environ.get("AWS_POLLY_VOICE", "Joanna"))

    async def test_aws_polly_tts_speech(self, tts: aws.TTS):
        out = []
        async for item in tts.send_iter("Hello from AWS Polly TTS"):
            out.append(item)
        assert len(out) > 0
