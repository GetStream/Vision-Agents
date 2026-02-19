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
        await session.user_says("Hello!")
        await session.agent_responds(intent="Responds with a friendly greeting")
        session.no_more_events()


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
        await session.user_says("What city was I born in?")
        await session.agent_responds(
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
        await session.user_says("Explain what Python is")
        await session.agent_responds(
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
        await session.user_says("What's the weather in Tokyo?")
        session.agent_calls("get_weather", arguments={"location": "Tokyo"})
        await session.agent_responds(
            intent="Reports weather for Tokyo including temperature"
        )
        session.no_more_events()


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
            await session.user_says("What's the weather in Paris?")

        await session.agent_responds(
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
        await session.user_says("My name is Alice")
        await session.agent_responds(
            intent="Acknowledges the user's name (Alice)"
        )

        await session.user_says("What's my name?")
        await session.agent_responds(
            intent="Correctly recalls that the user's name is Alice",
        )
