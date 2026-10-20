"""Tests for OpenRouter LLM plugin."""

import os
from typing import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS
from vision_agents.core.instructions import Instructions
from vision_agents.plugins.openrouter import LLM

from vision_agents.testing import answer, collect_simple_response

load_dotenv()

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


@pytest.fixture
async def openrouter_api() -> AsyncIterator[tuple[str, list[dict]]]:
    """A real server standing in for OpenRouter: its base URL, and every request body sent.

    Every answer is an empty stream: the tests are about requests.
    """
    bodies: list[dict] = []

    async def stream(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", stream)
    server = TestServer(app)
    await server.start_server()
    yield str(server.make_url("/v1")), bodies
    await server.close()


@pytest.fixture
async def openrouter_llm(
    openrouter_api: tuple[str, list[dict]],
) -> AsyncIterator[LLM]:
    base_url, _ = openrouter_api
    llm = LLM(model="test", api_key="test", base_url=base_url)
    yield llm
    await llm.close()


@pytest.fixture()
async def llm_factory():
    """Fixture for OpenRouter LLM with conversation."""

    def factory(
        model: str = "anthropic/claude-haiku-4.5",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("OPENROUTER_API_KEY") or "test",
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestOpenRouterLLM:
    """Test suite for OpenRouter LLM class."""

    async def test_who_spoke_is_named_once_a_second_person_has(
        self, openrouter_llm: LLM, openrouter_api: tuple[str, list[dict]]
    ):
        _, bodies = openrouter_api
        llm = openrouter_llm
        conversation = InMemoryConversation("", [])
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")

        await answer(llm, conversation, ALICE, "Hi")
        await answer(llm, conversation, ALICE, "Are you there?")
        # One person: what the base sha (9653cb696) sent for the same two turns.
        assert bodies[-1]["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok"},
            {"role": "user", "content": "Are you there?"},
        ]

        await answer(llm, conversation, BOB, "Me too")
        await answer(llm, conversation, ALICE, "Who is that?")
        assert bodies[-1]["messages"][0] == {
            "role": "system",
            "content": "be brief\n\n" + SEVERAL_SPEAKERS,
        }
        assert bodies[-1]["messages"][3:] == [
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "content": "Ok"},
            {"role": "user", "content": "Bob: Me too"},
            {"role": "assistant", "content": "Ok"},
            {"role": "user", "content": "Alice: Who is that?"},
        ]

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


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
@pytest.mark.integration
class TestOpenRouterLLMIntegration:
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
        llm = llm_factory(model="anthropic/claude-haiku-4.5")
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
