"""Tests for Requesty LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.requesty import LLM

from vision_agents.testing import collect_simple_response

load_dotenv()


def _require_requesty_api_key() -> str:
    api_key = os.getenv("REQUESTY_API_KEY")
    if not api_key:
        pytest.fail(
            "Requesty integration tests require REQUESTY_API_KEY. "
            "Set REQUESTY_API_KEY in the environment or in a .env file before "
            "running tests marked with @pytest.mark.integration.",
            pytrace=False,
        )
    return api_key


@pytest.fixture()
async def llm_factory():
    """Fixture for Requesty LLM with conversation."""

    def factory(
        model: str = "anthropic/claude-sonnet-4-5",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("REQUESTY_API_KEY") or "test",
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestRequestyLLM:
    """Test suite for Requesty LLM class."""

    async def test_strict_mode_for_non_openai(self, llm_factory):
        """Non-OpenAI models should have strict mode enabled for tools with required params."""
        llm = llm_factory(model="google/gemini-2.5-flash")
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
        converted = llm._convert_tools_to_provider_format(tools)
        func = converted[0]["function"]
        assert func.get("strict") is None
        assert func["parameters"].get("additionalProperties") is None

    async def test_convert_tools_does_not_mutate_input_schema(self, llm_factory):
        """Converting tools must not write strict-mode keys back into the shared schema."""
        llm = llm_factory(model="google/gemini-2.5-flash")
        schema = {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
            "required": ["foo"],
        }
        tools = [{"name": "test_tool", "description": "A test", "parameters": schema}]
        llm._convert_tools_to_provider_format(tools)
        assert "additionalProperties" not in schema


@pytest.mark.integration
class TestRequestyLLMIntegration:
    @pytest.fixture(autouse=True)
    def require_api_key(self) -> str:
        return _require_requesty_api_key()

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

    async def test_instruction_following(self, llm_factory):
        """Test that the LLM follows system instructions."""
        llm = llm_factory(model="anthropic/claude-sonnet-4-5")
        llm.set_instructions("Only reply in 2 letter country shortcuts")

        _, final = await collect_simple_response(
            llm.simple_response(
                "Which country is rainy, flat, famous for windmills and tulips, "
                "protected from water with dykes and below sea level?"
            )
        )

        assert "nl" in final.text.lower(), (
            f"Expected 'NL' in response, got: {final.text}"
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
        _, final = await collect_simple_response(llm.simple_response(prompt))

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in final.text, (
            f"Expected 'probe_ok:pong', got: {final.text}"
        )
