"""Tests for AWS plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.aws.aws_llm import BedrockLLM

from tests.utils import collect_simple_response

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
class TestBedrockLLMIntegration:
    @pytest.fixture
    async def llm(self):
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Bedrock LLM tests")
        llm = BedrockLLM(model="amazon.nova-lite-v1:0", region_name="us-east-1")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        yield llm
        await llm.close()

    async def test_simple_response(self, llm: BedrockLLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Explain quantum computing in 1 paragraph")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: BedrockLLM):
        await collect_simple_response(
            llm.simple_response("There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_instruction_following(self):
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Bedrock LLM tests")
        llm = BedrockLLM(model="qwen.qwen3-32b-v1:0", region_name="us-east-1")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        llm.set_instructions("only reply in 2 letter country shortcuts")
        try:
            _, final = await collect_simple_response(
                llm.simple_response(
                    "Which country is rainy, protected from water with dikes and below sea level?"
                )
            )
            assert "nl" in final.text.lower()
        finally:
            await llm.close()

    async def test_function_calling_live_roundtrip(self, llm: BedrockLLM):
        calls: list[str] = []

        @llm.register_function(
            description="Probe tool that records invocation and returns a marker string"
        )
        async def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "You MUST call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string and nothing else."
        )

        _, final = await collect_simple_response(llm.simple_response(prompt))

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert isinstance(final.text, str)
        assert "probe_ok:pong" in final.text
