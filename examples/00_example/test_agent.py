"""Integration tests for the 00_example agent.

These tests use the testing framework to verify agent behavior
in text-only mode with a real LLM (no audio/video/edge needed).

Run:
    uv run py.test examples/00_example/test_agent.py -m integration
"""

import os

import pytest
from dotenv import load_dotenv

from vision_agents.plugins import gemini
from vision_agents.testing import TestEval

load_dotenv()

pytestmark = pytest.mark.integration

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-2.5-flash-lite")


def _skip_if_no_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


async def test_greeting():
    """Agent responds with a friendly greeting."""
    _skip_if_no_key()

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise.",
    ) as session:
        response = await session.simple_response("Hello!")
        await response.judge(intent="Responds with a friendly greeting")
        response.no_more_events()


async def test_grounding():
    """Agent does not hallucinate personal information it cannot know."""
    _skip_if_no_key()

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise.",
    ) as session:
        response = await session.simple_response("What city was I born in?")
        await response.judge(
            intent="Does NOT claim to know the user's birthplace. "
            "Instead asks for clarification or says it doesn't have that info.",
        )


async def test_concise_response():
    """Agent keeps responses short as instructed."""
    _skip_if_no_key()

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise.",
    ) as session:
        response = await session.simple_response("Explain what Python is")
        await response.judge(
            intent="Gives a brief, concise explanation of Python "
            "(not a long multi-paragraph essay).",
        )


async def test_function_call():
    """Agent calls a registered tool with correct arguments."""
    _skip_if_no_key()

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict:
        return {"temperature": 22, "condition": "sunny", "unit": "celsius"}

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise. "
        "Use the get_weather tool when asked about weather.",
    ) as session:
        response = await session.simple_response("What's the weather in Tokyo?")
        response.function_called("get_weather", arguments={"location": "Tokyo"})
        await response.judge(
            intent="Reports weather for Tokyo including temperature"
        )
        response.no_more_events()


async def test_function_call_error_handling():
    """Agent handles tool errors gracefully."""
    _skip_if_no_key()

    from vision_agents.testing import mock_tools

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict:
        return {"temperature": 22, "condition": "sunny"}

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise.",
    ) as session:
        with mock_tools(llm, {"get_weather": lambda location: (_ for _ in ()).throw(RuntimeError("Service unavailable"))}):
            response = await session.simple_response("What's the weather in Paris?")

        await response.judge(
            intent="Informs the user that it could not get the weather "
            "or that something went wrong.",
        )


async def test_multi_turn_conversation():
    """Agent maintains context across multiple turns."""
    _skip_if_no_key()

    llm = gemini.LLM(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="You're a helpful voice assistant. Be concise.",
    ) as session:
        response = await session.simple_response("My name is Alice")
        await response.judge(
            intent="Acknowledges the user's name (Alice)"
        )

        response = await session.simple_response("What's my name?")
        await response.judge(
            intent="Correctly recalls that the user's name is Alice",
        )
