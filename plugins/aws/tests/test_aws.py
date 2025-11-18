"""Tests for AWS plugin."""

import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.agents.conversation import Message
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.aws.aws_llm import BedrockLLM

load_dotenv()


class TestBedrockLLM:
    """Test suite for BedrockLLM class with real API calls."""

    def assert_response_successful(self, response):
        """
        Utility method to verify a response is successful.

        A successful response has:
        - response.text is set (not None and not empty)
        - response.exception is None

        Args:
            response: LLMResponseEvent to check
        """
        assert response.text is not None, "Response text should not be None"
        assert len(response.text) > 0, "Response text should not be empty"
        assert not hasattr(response, "exception") or response.exception is None, (
            f"Response should not have an exception, got: {getattr(response, 'exception', None)}"
        )

    @pytest.fixture
    async def llm(self) -> BedrockLLM:
        """Test BedrockLLM initialization with a provided client."""
        llm = BedrockLLM(model="qwen.qwen3-32b-v1:0", region_name="us-east-1")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    @pytest.mark.asyncio
    async def test_message(self):
        messages = BedrockLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    @pytest.mark.asyncio
    async def test_advanced_message(self):
        advanced = {
            "role": "user",
            "content": [{"text": "Explain quantum entanglement in simple terms."}],
        }
        messages2 = BedrockLLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.mark.integration
    async def test_simple(self, llm: BedrockLLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        self.assert_response_successful(response)

    @pytest.mark.integration
    async def test_native_api(self, llm: BedrockLLM):
        response = await llm.converse(
            messages=[{"role": "user", "content": [{"text": "say hi"}]}],
        )

        self.assert_response_successful(response)

    @pytest.mark.integration
    async def test_stream(self, llm: BedrockLLM):
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        await llm.converse_stream(
            messages=[
                {"role": "user", "content": [{"text": "Explain magma to a 5 year old"}]}
            ]
        )
        # Wait for all events in queue to be processed
        await llm.events.wait()

        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: BedrockLLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )

        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: BedrockLLM):
        await llm.converse(
            messages=[
                {"role": "user", "content": [{"text": "There are 2 dogs in the room"}]}
            ],
        )
        response = await llm.converse(
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "How many paws are there in the room?"}],
                }
            ],
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_image_description(self, golf_swing_image):
        # Use a vision-capable model (Claude 3 Haiku supports images and is widely available)
        vision_llm = BedrockLLM(
            model="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1"
        )

        image_bytes = golf_swing_image
        response = await vision_llm.converse(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                        {"text": "What sport do you see in this image?"},
                    ],
                }
            ]
        )

        self.assert_response_successful(response)
        assert "golf" in response.text.lower()

    @pytest.mark.integration
    async def test_instruction_following(self, llm: BedrockLLM):
        llm = BedrockLLM(
            model="qwen.qwen3-32b-v1:0",
            region_name="us-east-1",
        )
        llm.set_instructions("only reply in 2 letter country shortcuts")

        response = await llm.simple_response(
            text="Which country is rainy, protected from water with dikes and below sea level?",
        )

        self.assert_response_successful(response)
        assert "nl" in response.text.lower()
