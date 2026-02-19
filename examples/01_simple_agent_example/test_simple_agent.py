"""Integration tests for the 01_simple_agent_example.

Run:
    uv run py.test examples/01_simple_agent_example/test_simple_agent.py -m integration
"""

import os

import pytest
from dotenv import load_dotenv

from .simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.plugins import gemini
from vision_agents.testing import TestEval

load_dotenv()

pytestmark = pytest.mark.integration

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-2.5-flash-lite")


def _skip_if_no_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


async def test_greeting():
    """Agent gives a friendly, short greeting."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(llm=llm, judge=judge_llm, instructions=INSTRUCTIONS) as session:
        await session.user_says("Hey there!")
        await session.agent_responds(intent="Friendly, short greeting")
        session.no_more_events()


async def test_weather_tool_call():
    """Agent calls get_weather with the right location and reports back."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestEval(llm=llm, judge=judge_llm, instructions=INSTRUCTIONS) as session:
        await session.user_says("What's the weather like in Berlin?")
        session.agent_calls("get_weather", arguments={"location": "Berlin"})
        await session.agent_responds(
            intent="Reports current weather for Berlin"
        )
        session.no_more_events()
