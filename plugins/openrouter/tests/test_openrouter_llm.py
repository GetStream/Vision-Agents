"""Tests for OpenRouter LLM plugin."""

import os

import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import Message, InMemoryConversation
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.openrouter import LLM

load_dotenv()


def assert_response_successful(response):
    """Verify a response is successful (has text and no exception)."""
    assert response.text, "Response text should not be None or empty"
    assert response.exception is None, f"Unexpected exception: {response.exception}"


def skip_without_api_key():
    """Skip test if OPENROUTER_API_KEY is not set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY environment variable not set")


class TestOpenRouterLLM:
    """Test suite for OpenRouter LLM class."""

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

    async def test_strict_mode_for_non_openai(self):
        """Non-OpenAI models should have strict mode enabled for tools with required params."""
        llm = LLM(model="google/gemini-2.0-flash-001")
        tools = [
            {
                "name": "test_tool",
                "description": "A test",
                "parameters": {
                    "type": "object",
                    "properties": {"foo": {"type": "string"}},
                    "required": ["foo"],
                },
            }
        ]
        converted = llm._convert_tools_to_chat_completions_format(tools)
        func = converted[0]["function"]
        assert func.get("strict") is True
        assert func["parameters"].get("additionalProperties") is False

    async def test_no_strict_mode_for_openai(self):
        """OpenAI models should NOT have strict mode (breaks with optional params)."""
        llm = LLM(model="openai/gpt-4o")
        tools = [
            {
                "name": "test_tool",
                "description": "A test",
                "parameters": {
                    "type": "object",
                    "properties": {"foo": {"type": "string"}},
                    "required": ["foo"],
                },
            }
        ]
        converted = llm._convert_tools_to_chat_completions_format(tools)
        func = converted[0]["function"]
        assert func.get("strict") is None
        assert func["parameters"].get("additionalProperties") is None

    @pytest.fixture
    async def llm(self) -> LLM:
        """Fixture for OpenRouter LLM with conversation."""
        skip_without_api_key()
        llm = LLM(model="anthropic/claude-haiku-4.5")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: LLM):
        """Test simple response generation."""
        response = await llm.simple_response("Explain quantum computing in 1 paragraph")
        assert_response_successful(response)

    @pytest.mark.integration
    async def test_native_api(self, llm: LLM):
        """Test native OpenAI-compatible API."""
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )
        assert_response_successful(response)
        assert hasattr(response.original, "id")

    @pytest.mark.integration
    async def test_streaming(self, llm: LLM):
        """Test streaming response."""
        streaming_works = False

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        response = await llm.simple_response("Explain quantum computing in 1 paragraph")
        await llm.events.wait()

        assert_response_successful(response)
        assert streaming_works, "Streaming should have generated chunk events"

    @pytest.mark.integration
    async def test_memory(self, llm: LLM):
        """Test conversation memory using simple_response."""
        await llm.simple_response(text="There are 2 dogs in the room")
        response = await llm.simple_response(
            text="How many paws are there in the room?"
        )

        assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_native_memory(self, llm: LLM):
        """Test conversation memory using native API."""
        await llm.create_response(input="There are 2 dogs in the room")
        response = await llm.create_response(
            input="How many paws are there in the room?"
        )

        assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_instruction_following(self):
        """Test that the LLM follows system instructions."""
        skip_without_api_key()
        llm = LLM(model="anthropic/claude-haiku-4.5")
        llm.set_instructions(Instructions("Only reply in 2 letter country shortcuts"))

        response = await llm.simple_response(
            text="Which country is rainy, flat, famous for windmills and tulips, protected from water with dykes and below sea level?"
        )

        assert_response_successful(response)
        assert "nl" in response.text.lower(), (
            f"Expected 'NL' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_function_calling_openai(self):
        """Test function calling with OpenAI model."""
        skip_without_api_key()
        llm = LLM(model="openai/gpt-4o-mini")

        calls: list[str] = []

        @llm.register_function(description="Probe tool that records invocation")
        def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "You MUST call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string."
        )
        response = await llm.create_response(input=prompt)

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in response.text, (
            f"Expected 'probe_ok:pong', got: {response.text}"
        )

    @pytest.mark.integration
    async def test_function_calling_gemini(self):
        """Test function calling with Gemini model."""
        skip_without_api_key()
        llm = LLM(model="google/gemini-2.0-flash-001")

        calls: list[str] = []

        @llm.register_function(description="Probe tool that records invocation")
        def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "You MUST call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string."
        )
        response = await llm.create_response(input=prompt)

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in response.text, (
            f"Expected 'probe_ok:pong', got: {response.text}"
        )
