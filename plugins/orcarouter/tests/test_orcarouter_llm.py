"""Tests for OrcaRouter LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.instructions import Instructions
from vision_agents.plugins.orcarouter import LLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture()
async def llm_factory():
    """Fixture for OrcaRouter LLM with conversation."""

    def factory(
        model: str = "openai/gpt-4o-mini",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("ORCAROUTER_API_KEY") or "test",
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestOrcaRouterLLM:
    """Test suite for OrcaRouter LLM class."""

    async def test_strict_mode_for_non_openai(self, llm_factory):
        """Non-OpenAI models should have strict mode enabled for tools with required params."""
        llm = llm_factory(model="anthropic/claude-opus-4.8")
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

    async def test_auto_router_is_not_the_default(self, llm_factory):
        """The default model is a pinned model, not the auto router."""
        llm = llm_factory()
        assert llm._is_auto_model() is False
        assert llm._is_auto_model("orcarouter/auto") is True

    async def test_auto_router_sends_tool_capable_fallbacks(self, llm_factory):
        """The auto router needs a fallback list so tool calls keep working."""
        llm = llm_factory(model="orcarouter/auto")
        body = llm._tool_fallback_body()
        assert body["route"] == "fallback"
        assert body["models"], "Fallback list must not be empty"


@pytest.mark.integration
class TestOrcaRouterLLMIntegration:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        """Fail loudly rather than skip when the opted-in secret is missing."""
        if not os.environ.get("ORCAROUTER_API_KEY"):
            pytest.fail(
                "ORCAROUTER_API_KEY is required to run the OrcaRouter integration "
                "tests. Get a key at https://www.orcarouter.ai/console and export it, "
                "or deselect these tests with -m 'not integration'."
            )

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
        llm = llm_factory()
        llm.set_instructions(Instructions("Only reply in 2 letter country shortcuts"))

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
