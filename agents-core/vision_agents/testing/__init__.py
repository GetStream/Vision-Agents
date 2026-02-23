"""Testing framework for Vision-Agents.

Provides text-only testing of agents without requiring audio/video
infrastructure or edge connections.

Usage:

Verify a greeting::

    async def test_greeting():
        judge = LLMJudge(gemini.LLM(MODEL))
        async with TestSession(llm=llm, instructions="Be friendly") as session:
            response = await session.simple_response("Hello")
            event = response.assistant_message()
            verdict = await judge.evaluate(event, intent="Friendly greeting")
            assert verdict.success, verdict.reason

Verify tool calls::

    async def test_weather():
        judge = LLMJudge(gemini.LLM(MODEL))
        async with TestSession(llm=llm, instructions="...") as session:
            response = await session.simple_response("Weather in Tokyo?")
            response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
            event = response.assistant_message()
            verdict = await judge.evaluate(event, intent="Reports weather for Tokyo")
            assert verdict.success, verdict.reason

Key exports:
    TestSession: async context manager that wraps an LLM for testing.
    TestResponse: returned by ``simple_response()`` — carries events and assertions.
    Judge: protocol for intent evaluation strategies.
    JudgeVerdict: dataclass returned by ``Judge.evaluate()``.
    LLMJudge: default judge backed by an LLM instance.
    mock_tools: context manager to temporarily replace tool implementations.
    RunEvent: union of ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent.
"""

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._judge import Judge, JudgeVerdict, LLMJudge
from vision_agents.testing._mock_tools import mock_tools
from vision_agents.testing._run_result import TestResponse
from vision_agents.testing._session import TestSession

__all__ = [
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "TestSession",
    "TestResponse",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "mock_tools",
]
