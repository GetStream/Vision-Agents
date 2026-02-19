"""Integration tests for the 01_simple_agent_example.

Run:
    uv run py.test examples/01_simple_agent_example/test_simple_agent.py -m integration
"""

import os

import pytest
from dotenv import load_dotenv

from .simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.plugins import gemini
from vision_agents.testing import TestSession

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

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        result = await session.run("Hey there!")

        await (
            result.expect
            .next_event(type="message")
            .judge(judge_llm, intent="Friendly, short greeting")
        )
        result.expect.no_more_events()


async def test_weather_tool_call():
    """Agent calls get_weather with the right location and reports back."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge_llm = gemini.LLM(MODEL)

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        result = await session.run("What's the weather like in Berlin?")

        result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Berlin"}
        )
        result.expect.next_event().is_function_call_output()
        await (
            result.expect
            .next_event()
            .is_message(role="assistant")
            .judge(judge_llm, intent="Reports current weather for Berlin")
        )
        result.expect.no_more_events()
