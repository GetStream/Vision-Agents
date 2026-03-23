"""Tests for OpenRouter LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.openrouter import LLM

load_dotenv()


def assert_response_successful(response):
    """Verify a response is successful (has text and no exception)."""
    assert response.text, "Response text should not be None or empty"
    assert response.exception is None, f"Unexpected exception: {response.exception}"


@pytest.fixture()
async def llm_factory():
    """Fixture for OpenRouter LLM with conversation."""

    def factory(
        model: str = "anthropic/claude-haiku-4.5",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(model=model, max_tokens=max_tokens)
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


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

    async def test_strict_mode_for_non_openai(self, llm_factory):
        """Non-OpenAI models should have strict mode enabled for tools with required params."""
        llm = llm_factory(model="google/gemini-3-flash-preview")
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

    async def test_no_strict_mode_for_openai(self, llm_factory):
        """OpenAI models should NOT have strict mode (breaks with optional params)."""
        llm = llm_factory(model="openai/gpt-4o")
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


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
@pytest.mark.integration
class TestOpenRouterLLMIntegration:
    async def test_simple(self, llm_factory):
        """Test simple response generation."""
        llm = llm_factory()
        response = await llm.simple_response("Greet the user")
        assert_response_successful(response)

    async def test_native_api(self, llm_factory):
        """Test native OpenAI-compatible API."""
        llm = llm_factory()
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )
        assert_response_successful(response)
        assert hasattr(response.original, "id")

    async def test_streaming(self, llm_factory):
        """Test streaming response."""
        streaming_works = False
        llm = llm_factory()

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        response = await llm.simple_response("Greet the user")
        await llm.events.wait()

        assert_response_successful(response)
        assert streaming_works, "Streaming should have generated chunk events"

    async def test_memory(self, llm_factory):
        """Test conversation memory using simple_response."""
        llm = llm_factory()
        await llm.simple_response(text="There are 2 dogs in the room")
        response = await llm.simple_response(
            text="How many paws are there in the room?"
        )

        assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    async def test_native_memory(self, llm_factory):
        """Test conversation memory using native API."""
        llm = llm_factory()
        await llm.create_response(input="There are 2 dogs in the room")
        response = await llm.create_response(
            input="How many paws are there in the room?"
        )

        assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    async def test_instruction_following(self, llm_factory):
        """Test that the LLM follows system instructions."""
        llm = llm_factory(model="anthropic/claude-haiku-4.5")
        llm.set_instructions(Instructions("Only reply in 2 letter country shortcuts"))

        response = await llm.simple_response(
            text="Which country is rainy, flat, famous for windmills and tulips, protected from water with dykes and below sea level?"
        )

        assert_response_successful(response)
        assert "nl" in response.text.lower(), (
            f"Expected 'NL' in response, got: {response.text}"
        )

    async def test_function_calling_openai(self, llm_factory):
        """Test function calling with OpenAI model."""
        llm = llm_factory(model="openai/gpt-4o-mini", max_tokens=512)

        calls: list[str] = []

        @llm.register_function(description="Probe tool that records invocation")
        async def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "Call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string."
        )
        response = await llm.simple_response(prompt)

        await llm.events.wait()

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in response.text, (
            f"Expected 'probe_ok:pong', got: {response.text}"
        )
