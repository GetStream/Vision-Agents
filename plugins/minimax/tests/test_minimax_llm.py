"""Tests for MiniMax LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.minimax import LLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture()
async def llm_factory():
    """Fixture for MiniMax LLM with conversation."""

    def factory(
        model: str = "MiniMax-M3",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("MINIMAX_API_KEY") or "test",
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestMiniMaxLLM:
    """Test suite for MiniMax LLM class."""

    def test_default_model_is_m3(self):
        """Default model should be MiniMax-M3 (latest flagship)."""
        llm = LLM(api_key="test")
        assert llm.model == "MiniMax-M3"

    def test_provider_name(self):
        """provider_name should be 'minimax'."""
        llm = LLM(api_key="test")
        assert llm.provider_name == "minimax"

    def test_custom_model(self):
        """Custom model should override the default."""
        llm = LLM(model="MiniMax-M2.7", api_key="test")
        assert llm.model == "MiniMax-M2.7"

    async def test_convert_tools_to_provider_format(self, llm_factory):
        """Tools should be converted to Chat Completions function format."""
        llm = llm_factory()
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
        converted = llm._convert_tools_to_provider_format(tools)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "test_tool"
        assert converted[0]["function"]["description"] == "A test"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    async def test_default_temperature_is_one(self, llm_factory):
        """Default temperature passed to the API must be 1.0 (MiniMax rejects 0)."""
        llm = llm_factory()

        captured: dict = {}

        class _Stream:
            id = "fake-id"

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class _FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _Stream()

        class _FakeChat:
            completions = _FakeCompletions()

        llm._client.chat = _FakeChat()

        async for _ in llm._create_response_internal(messages=[]):
            pass

        assert captured.get("temperature") == 1.0

    async def test_build_model_request_includes_instructions(self, llm_factory):
        """System instructions must be the first message in the request."""
        llm = llm_factory()
        llm.set_instructions("be terse")
        messages = await llm._build_model_request()
        assert messages and messages[0]["role"] == "system"
        assert messages[0]["content"] == "be terse"


@pytest.mark.skipif(not os.getenv("MINIMAX_API_KEY"), reason="MINIMAX_API_KEY not set")
@pytest.mark.integration
class TestMiniMaxLLMIntegration:
    async def test_simple_response(self, llm_factory):
        """Test simple response yields deltas and a final."""
        llm = llm_factory()
        deltas, final = await collect_simple_response(
            llm.simple_response("Greet the user")
        )
        assert deltas, "Streaming should yield deltas"
        assert final.text

    async def test_memory(self, llm_factory):
        """Test conversation memory using simple_response."""
        llm = llm_factory()
        await collect_simple_response(
            llm.simple_response("There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )

        assert "8" in final.text or "eight" in final.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {final.text}"
        )

    async def test_m27_still_works(self, llm_factory):
        """Switching to MiniMax-M2.7 should still work after the M3 upgrade."""
        llm = llm_factory(model="MiniMax-M2.7", max_tokens=128)
        _, final = await collect_simple_response(
            llm.simple_response("Reply with the single word: pong")
        )
        assert "pong" in final.text.lower(), (
            f"Expected 'pong' in response, got: {final.text}"
        )

    async def test_function_calling(self, llm_factory):
        """Test function calling with the MiniMax LLM."""
        llm = llm_factory(max_tokens=512)
        calls: list[str] = []

        @llm.register_function(description="Probe tool that records invocation")
        async def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "Call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string."
        )
        _, final = await collect_simple_response(llm.simple_response(prompt))

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in final.text, (
            f"Expected 'probe_ok:pong', got: {final.text}"
        )
