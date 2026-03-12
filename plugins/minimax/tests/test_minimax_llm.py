"""Tests for MiniMax LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.minimax import LLM

load_dotenv()


def skip_without_api_key():
    """Skip test if MINIMAX_API_KEY is not set."""
    if not os.environ.get("MINIMAX_API_KEY"):
        pytest.skip("MINIMAX_API_KEY environment variable not set")


class TestMiniMaxLLM:
    """Test suite for MiniMax LLM class."""

    def test_default_model(self):
        """Test that default model is MiniMax-M2.5."""
        llm = LLM()
        assert llm.model == "MiniMax-M2.5"

    def test_custom_model(self):
        """Test setting a custom model."""
        llm = LLM(model="MiniMax-M2.5-highspeed")
        assert llm.model == "MiniMax-M2.5-highspeed"

    def test_tool_conversion(self):
        """Test converting ToolSchema to Chat Completions format."""
        llm = LLM()
        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        converted = llm._convert_tools_to_provider_format(tools)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["description"] == "Get the weather"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_input_to_messages_string(self):
        """Test converting string input to messages."""
        llm = LLM()
        messages = llm._input_to_messages("hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_input_to_messages_with_instructions(self):
        """Test that instructions are included as system message."""
        llm = LLM()
        llm.set_instructions("Be helpful")
        messages = llm._input_to_messages("hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"

    def test_input_to_messages_list(self):
        """Test converting list input to messages."""
        llm = LLM()
        input_list = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        messages = llm._input_to_messages(input_list)
        assert len(messages) == 2
        assert messages[0]["content"] == "hello"
        assert messages[1]["content"] == "hi there"

    def test_build_model_request_empty(self):
        """Test building model request with no state."""
        llm = LLM()
        messages = llm._build_model_request()
        assert messages == []

    def test_build_model_request_with_instructions(self):
        """Test building model request with instructions."""
        llm = LLM()
        llm.set_instructions("Be concise")
        messages = llm._build_model_request()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"

    @pytest.fixture
    async def llm(self) -> LLM:
        """Fixture for MiniMax LLM with conversation."""
        skip_without_api_key()
        llm = LLM(model="MiniMax-M2.5")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: LLM):
        """Test simple response generation."""
        response = await llm.simple_response("Say hello in one sentence")
        assert response.text, "Response text should not be empty"

    @pytest.mark.integration
    async def test_streaming(self, llm: LLM):
        """Test streaming response."""
        streaming_works = False

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        response = await llm.simple_response("Explain AI in one paragraph")
        await llm.events.wait()

        assert response.text, "Response text should not be empty"
        assert streaming_works, "Streaming should have generated chunk events"

    @pytest.mark.integration
    async def test_create_response(self, llm: LLM):
        """Test create_response with input parameter."""
        response = await llm.create_response(
            messages=[{"role": "user", "content": "say hi"}]
        )
        assert response.text, "Response text should not be empty"

    @pytest.mark.integration
    async def test_memory(self, llm: LLM):
        """Test conversation memory."""
        await llm.simple_response(text="There are 2 dogs in the room")
        response = await llm.simple_response(
            text="How many paws are there in the room?"
        )

        assert response.text, "Response text should not be empty"
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )
