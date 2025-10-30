"""Tests for OpenRouter LLM plugin."""

import os

import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import Message, InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
)
from vision_agents.plugins.openrouter import LLM

load_dotenv()


class TestOpenRouterLLM:
    """Test suite for OpenRouter LLM class."""

    def assert_response_successful(self, response):
        """Utility method to verify a response is successful.

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

    def test_message(self):
        """Test basic message normalization."""
        messages = LLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        """Test advanced message format with image."""
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

        advanced = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what do you see in this image?"},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ]
        messages = LLM._normalize_message(advanced)
        assert messages[0].original is not None

    @pytest.fixture
    async def llm(self) -> LLM:
        """Fixture for OpenRouter LLM with z-ai/glm-4.6 model."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        
        llm = LLM(model="anthropic/claude-haiku-4.5")
        llm._conversation = InMemoryConversation("be friendly", [])
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: LLM):
        """Test simple response generation."""
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        self.assert_response_successful(response)

    @pytest.mark.integration
    async def test_native_api(self, llm: LLM):
        """Test native OpenAI-compatible API."""
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )

        self.assert_response_successful(response)
        assert hasattr(response.original, "id")  # OpenAI-compatible response has id

    @pytest.mark.integration
    async def test_streaming(self, llm: LLM):
        """Test streaming response."""
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        await llm.events.wait()

        self.assert_response_successful(response)
        assert streamingWorks, "Streaming should have generated chunk events"

    @pytest.mark.integration
    async def test_memory(self, llm: LLM):
        """Test conversation memory using simple_response."""
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        
        self.assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_native_memory(self, llm: LLM):
        """Test conversation memory using native API."""
        await llm.create_response(
            input="There are 2 dogs in the room",
        )
        response = await llm.create_response(
            input="How many paws are there in the room?",
        )
        
        self.assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_instruction_following(self):
        """Test that the LLM follows system instructions."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY environment variable not set")

        pytest.skip("instruction following doesnt always work")
        llm = LLM(model="anthropic/claude-haiku-4.5")
        llm._set_instructions("Only reply in 2 letter country shortcuts")

        response = await llm.simple_response(
            text="Which country is rainy, protected from water with dikes and below sea level?",
        )

        self.assert_response_successful(response)
        assert "nl" in response.text.lower(), (
            f"Expected 'NL' in response, got: {response.text}"
        )

