"""Integration tests for the 01_simple_agent_example.

Run:
    uv run py.test examples/01_simple_agent_example/test_simple_agent.py -m integration
"""

import os

import pytest
from dotenv import load_dotenv

from .simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.plugins import gemini
from vision_agents.testing import LLMJudge, TestSession

load_dotenv()

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-2.5-flash-lite")


def _skip_if_no_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


@pytest.mark.integration
async def test_greeting():
    """Agent gives a friendly, short greeting."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge = LLMJudge(gemini.LLM(MODEL))

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        response = await session.simple_response("Hey there!")
        response.function_not_called("get_weather")
        event = response.assistant_message()
        verdict = await judge.evaluate(event, intent="Friendly, short greeting")
        assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_weather_tool_call():
    """Agent calls get_weather with the right location and reports back."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge = LLMJudge(gemini.LLM(MODEL))

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        response = await session.simple_response("What's the weather like in Berlin?")
        response.function_called("get_weather", arguments={"location": "Berlin"})
        response.function_called_times("get_weather", 1)
        event = response.assistant_message()
        verdict = await judge.evaluate(
            event, intent="Reports current weather for Berlin"
        )
        assert verdict.success, verdict.reason
